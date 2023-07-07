import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

class FlowExpirationEnv(gym.Env):
    def __init__(self, flow_data):
        super(FlowExpirationEnv, self).__init__()
        self.flow_data = flow_data
        self.state = 10
        self.current_step = 0
        self.max_steps = len(self.flow_data)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([65335, 65335, 65335]), shape=(3,), dtype=np.int32)

    def _get_observation(self):
        packets = self.flow_data.loc[self.current_step, 'Packets']
        bytes = self.flow_data.loc[self.current_step, 'Bytes']
        return np.array([self.state, packets, bytes], dtype=np.int32)

    def step(self, action):
        self.state += action - 1
        reward = self._calculate_reward()
        self.current_step += 1
        done = self.current_step == self.max_steps - 1
        observation = self._get_observation()
        return observation, reward, done, {}

    def _calculate_reward(self):
        current_time = self.current_step
        expiration_time = self.state
        if expiration_time < current_time:
            return -1
        else:
            return 2

    def reset(self):
        self.current_step = 0
        self.max_steps = len(self.flow_data)
        self.state = 10
        return np.array([self.state, 0, 0], dtype=np.int32)


# Load flow data from CSV
data = pd.read_csv('flow_statistics5.csv')

# Get unique flow IDs
flow_ids = data['Flow_Id'].unique()

# Create a function to create environment instances for each flow
def make_env(flow_id):
    flow_data = data[data['Flow_Id'] == flow_id].reset_index()
    return FlowExpirationEnv(flow_data)

# Create a list of callable functions to create environment instances
envs = [lambda: make_env(flow_id) for flow_id in flow_ids]

# Vectorize the environments
vec_env = DummyVecEnv(envs)

# Create the PPO model
model = PPO('MlpPolicy', vec_env, verbose=1)

# Train the agent
model.learn(total_timesteps=20000)

# Evaluate the policy
mean_rewards, _ = evaluate_policy(model, vec_env, n_eval_episodes=10)

# Print mean rewards for each flow
for i, flow_id in enumerate(flow_ids):
    print(f"Flow ID: {flow_id}, Mean Reward: {mean_rewards[i]}")

# Testing the environment for each flow
    episodes = 5
    for episode in range(1, episodes + 1):
        observation = vec_env.reset()
        done = False
        score = 0

        while not done:
            action, _ = model.predict(observation)
            observation, reward, done, _ = vec_env.step(action)
            score += reward

        print(f"Flow ID: {flow_id}, Episode: {episode}, Score: {score}")