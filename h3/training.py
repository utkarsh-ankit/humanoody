import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import os
from torch.cuda.amp import GradScaler, autocast

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
LR = 3e-4
HIDDEN_SIZE = 256
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 1000000
ENV_NAME = 'Humanoid-v4'
MAX_EPISODE_STEPS = 3000
SAVE_INTERVAL = 1000

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

# Efficient Data Loading
class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Define networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.mean = nn.Linear(HIDDEN_SIZE, action_dim)
        self.log_std = nn.Linear(HIDDEN_SIZE, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.q_value = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_value(x)
        return q_value

class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = nn.DataParallel(Actor(state_dim, action_dim)).to(device)
        self.critic1 = nn.DataParallel(Critic(state_dim, action_dim)).to(device)
        self.critic2 = nn.DataParallel(Critic(state_dim, action_dim)).to(device)
        self.target_critic1 = nn.DataParallel(Critic(state_dim, action_dim)).to(device)
        self.target_critic2 = nn.DataParallel(Critic(state_dim, action_dim)).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, _ = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.module.sample(next_state)
            target_q1 = self.target_critic1.module(next_state, next_action)
            target_q2 = self.target_critic2.module(next_state, next_action)
            target_q = reward + (1 - done) * GAMMA * (torch.min(target_q1, target_q2) - ALPHA * next_log_prob)

        current_q1 = self.critic1.module(state, action)
        current_q2 = self.critic2.module(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()

        # Mixed precision training for critics
        scaler.scale(critic1_loss).backward()
        scaler.scale(critic2_loss).backward()
        scaler.step(self.critic1_optimizer)
        scaler.step(self.critic2_optimizer)
        scaler.update()

        action, log_prob = self.actor.module.sample(state)
        q1 = self.critic1.module(state, action)
        q2 = self.critic2.module(state, action)
        actor_loss = (ALPHA * log_prob - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()

        # Mixed precision training for actor
        scaler.scale(actor_loss).backward()
        scaler.step(self.actor_optimizer)
        scaler.update()

        for target_param, param in zip(self.target_critic1.module.parameters(), self.critic1.module.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for target_param, param in zip(self.target_critic2.module.parameters(), self.critic2.module.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def save_checkpoint(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.module.state_dict(),
            'critic1_state_dict': self.critic1.module.state_dict(),
            'critic2_state_dict': self.critic2.module.state_dict(),
            'target_critic1_state_dict': self.target_critic1.module.state_dict(),
            'target_critic2_state_dict': self.target_critic2.module.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer.buffer,
            'replay_buffer_position': self.replay_buffer.position
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.module.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.module.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.module.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.module.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.module.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.replay_buffer.buffer = checkpoint['replay_buffer']
        self.replay_buffer.position = checkpoint['replay_buffer_position']

def compute_reward(state, action, next_state, done):
    forward_velocity = next_state[0]  # Assume the first state dimension represents forward velocity
    height = next_state[1]  # Assume the second state dimension represents the height of the torso
    energy_consumption = np.sum(np.square(action))  # Penalize large actions

    # Reward for moving forward
    reward = forward_velocity * 4.0  # Increase the incentive for moving forward significantly

    # Reward for maintaining height (encourages the agent to stay upright)
    reward += (height - 1.0) * 0.5

    # Penalize falling
    if done:
        reward -= 100

    # Penalize high energy consumption
    reward -= energy_consumption * 0.1

    return reward

# Main Training Loop
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = SACAgent(state_dim, action_dim)

num_episodes = 10000  # Increased number of episodes for longer training
max_steps = MAX_EPISODE_STEPS
old_checkpoint_path = '/home/uankit/checkpoints/sac_checkpoint2.pth'  # Path to the old checkpoint
checkpoint_dir = '/home/uankit/checkpoints/3'  # Directory to save new checkpoints

# Load the latest checkpoint if exists
latest_checkpoint = None
if os.path.exists(checkpoint_dir):
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('sac_checkpoint3_') and f.endswith('.pth')])
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])

if latest_checkpoint:
    agent.load_checkpoint(latest_checkpoint)
    print(f"Resuming from checkpoint: {latest_checkpoint}")
else:
    try:
        agent.load_checkpoint(old_checkpoint_path)
        print(f"Resuming from old checkpoint: {old_checkpoint_path}")
    except FileNotFoundError:
        print("No checkpoint found. Starting training from scratch.")

try:
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Compute the custom reward
            reward = compute_reward(state, action, next_state, done)

            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            agent.train()

            if done or truncated:
                break

        print(f"Episode {episode}, Reward: {episode_reward}")

        # Save checkpoint periodically
        if episode % SAVE_INTERVAL == 0:
            checkpoint_filename = f"sac_checkpoint3_{episode}.pth"
            checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)
            agent.save_checkpoint(checkpoint_filepath)
            print(f"Checkpoint saved at episode {episode}")

    # Final save at the end of training
    final_checkpoint_filepath = os.path.join(checkpoint_dir, "sac_checkpoint3_final.pth")
    agent.save_checkpoint(final_checkpoint_filepath)
    print("Final checkpoint saved.")

except KeyboardInterrupt:
    print("Training interrupted. Saving checkpoint...")
    interrupted_checkpoint_filepath = os.path.join(checkpoint_dir, f"sac_checkpoint3_interrupted_{episode}.pth")
    agent.save_checkpoint(interrupted_checkpoint_filepath)
    print("Checkpoint saved after interruption.")

env.close()

