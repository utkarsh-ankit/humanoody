import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

# Define the Actor network (same as used in training)
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

# Hyperparameters (make sure these match your training script)
HIDDEN_SIZE = 256
ENV_NAME = 'Humanoid-v4'
CHECKPOINT_PATH = '/Users/utkarshankit/Documents/mujo/sac_checkpoint3_3000.pth'  # Update the path as necessary

# Initialize the environment
env = gym.make(ENV_NAME, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize the Actor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor(state_dim, action_dim).to(device)
actor.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device)['actor_state_dict'])
actor.eval()

# Function to select action
def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action, _ = actor.sample(state)
    return action.detach().cpu().numpy()[0]

# Run the environment
state, _ = env.reset()
done = False
while not done:
    action = select_action(state)
    state, reward, done, truncated, _ = env.step(action)
    env.render()

env.close()
