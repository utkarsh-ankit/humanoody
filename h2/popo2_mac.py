import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

# Hyperparameters
HIDDEN_SIZE = 256
ENV_NAME = 'Humanoid-v4'
CHECKPOINT_PATH = '/Users/utkarshankit/Documents/mujo/sac_checkpoint2.pth'  # Path to your new checkpoint file

# Actor Network
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

# SAC Agent with only the actor
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])

# Main Rendering Code
env = gym.make(ENV_NAME, render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = SACAgent(state_dim, action_dim)

# Load checkpoint
agent.load_checkpoint(CHECKPOINT_PATH)
print("Checkpoint loaded successfully.")

# Run the trained agent
num_episodes = 10  # Number of episodes to render
max_steps = env.spec.max_episode_steps

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        env.render()
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        episode_reward += reward

        if done or truncated:
            break

    print(f"Episode {episode}, Reward: {episode_reward}")

env.close()
