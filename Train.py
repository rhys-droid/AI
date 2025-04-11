import gym
import simple_driving
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.97
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
NUM_EPISODES = 15000
MODEL_PATH = "trained_dqn.pth"

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True).unwrapped

n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

policy_net = DQN(state_dim, n_actions)
target_net = DQN(state_dim, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

epsilon = EPS_START

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        if epsilon < 0.2:
            epsilon = 0.2

        if random.random() < epsilon:
            action_probs = np.array([0.07]*3 + [0.03]*3 + [0.2, 0.2, 0.3]) #backward, turn only, forward
            action = np.random.choice(np.arange(n_actions), p=action_probs)
        else:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state)).argmax().item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done and not getattr(env, "reached_goal", False):
            reward -= 50

        memory.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            q_values = policy_net(states).gather(1, actions).squeeze()
            next_q_values = target_net(next_states).max(1)[0]
            expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, expected_q_values.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

torch.save(policy_net.state_dict(), MODEL_PATH)
print("✅ Model saved to", MODEL_PATH)