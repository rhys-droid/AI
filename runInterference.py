import gym
import simple_driving
import torch
import torch.nn as nn
import numpy as np

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

# Load environment
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True).unwrapped

# Load model
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
model = DQN(state_dim, n_actions)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

TOTAL_GOALS = 10
goals_attempted = 0
goals_reached = 0

env.keep_goal = False  # ensures new goal every time one is reached
print("🚗 Inference started with 10 sequential goals")

while goals_attempted < TOTAL_GOALS:
    state, _ = env.reset()
    env.keep_goal = True
    goal_success = False

    while True:
        with torch.no_grad():
            action = model(torch.FloatTensor(state)).argmax().item()

        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state

        # If agent reached goal
        goal_success = info.get("reached_goal", False)

            

        # If episode terminates (stuck or timeout)
        if terminated or truncated:
            break

    goals_attempted += 1
    if goal_success:
        goals_reached += 1
        print(f"🏁 Reached goal {goals_reached}/{goals_attempted}\n")
    else:
        print(f"❌ Failed to reach goal {goals_attempted}\n")

env.close()
print(f"✅ Finished reaching {goals_reached}/{TOTAL_GOALS} goals")
print(f"🎯 Success rate: {goals_reached / TOTAL_GOALS * 100:.1f}%")