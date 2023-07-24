import gym
import numpy as np
import pandas as pd
import functools
from stable_baselines3 import PPO
from gym.spaces import Discrete, Box, Dict
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
        self.action_space = Discrete(20,start=-10) 
        self.observation_space = Dict({
            #'expiration time': gym.spaces.Box(low=0, high=65535, shape=(1,), dtype=np.int32),
            'ongoing': Discrete(2),  # Transmission ongoing (0 or 1)
            'expired': Discrete(2),  # Flow expired (0 or 1)
            'packets': Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),  # Packets
            'bytes': Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)  # Bytes
        })

    def _get_observation(self):
        current_time = self.current_step  # Assuming each step represents a unit of time
        expiration_time = self.state
        packets = self.flow_data.loc[self.current_step, 'Packets']
        bytes = self.flow_data.loc[self.current_step, 'Bytes']
        ongoing = self.flow_data.loc[self.current_step, 'State'] != 'PENDING_REMOVE' #packet transmission is ongoingn or not
        if ongoing and expiration_time < current_time:
            expired = True #packet transmission is on going and flow rule has expired
        else:
            expired = False
        return {
            #'expiration time' : expiration_time,
            'ongoing': ongoing,
            'expired': expired,
            'packets': np.array([packets], dtype=np.int32),
            'bytes': np.array([bytes], dtype=np.int32)
        }

    def step(self, action):
        # Apply action
        # 0 -1 = -1 
        # 1 -1 = 0 
        # 2 -1 = 1  
        self.state += action -10

        # Calculate the reward based on the expiration time
        reward = self._calculate_reward()

        self.current_step += 1
        
        # Check if the episode is done
        done = self.current_step == self.max_steps-1
        
        # Get the current observation
        observation = self._get_observation()
        #observation = np.array([self.state], dtype=np.int32) 
 
        return observation, reward, done, {'expiration time':self.state}

    def _calculate_reward(self):
        current_time = self.current_step  # Assuming each step represents a unit of time
        expiration_time = self.state
        time_penalty = (expiration_time - current_time) 
        #print('expiration_time',expiration_time)
        #print('current_time',current_time)
        #print('time_penalty==',time_penalty)
        if self.flow_data.loc[self.current_step, 'State'] != 'PENDING_REMOVE':
            if expiration_time < current_time:
                return time_penalty  # Flow expired, assign a negative reward
            else:
                return 20 - time_penalty # Default reward if flow has not expired
        elif expiration_time > current_time:
            return -time_penalty #Transmission done, flow not removed

    def reset(self):
        self.current_step = 0
        self.max_steps = len(self.flow_data)
        self.state = 10
        return self._get_observation()


# Load flow data from CSV
#data = pd.read_csv('filtered_data.csv')
data = pd.read_csv('flow_statistics.csv')

# Get unique flow IDs
flow_ids = data['Flow_Id'].unique()

# Create a function to create environment instances for each flow
def make_env(flow_id):
    flow_data = data[data['Flow_Id'] == flow_id].reset_index(drop=True)
    return FlowExpirationEnv(flow_data)

# Create a list of callable functions to create environment instances
#envs = [lambda: make_env(flow_id) for flow_id in flow_ids]
envs = [functools.partial(make_env, flow_id) for flow_id in flow_ids[:4]]

# Vectorize the environments
vec_env = DummyVecEnv(envs)

# Create the PPO model
#model = PPO('MlpPolicy', vec_env, verbose=1,learning_rate=0.001)
model = PPO('MultiInputPolicy', vec_env, verbose=1, learning_rate=0.001, gamma=0.99)

# Train the agent
model.learn(total_timesteps=20000)

# Evaluate the policy
evaluate_policy(model, vec_env, n_eval_episodes=50, render=True)

# Testing the environment for each flow
episodes = 5
for episode in range(1, episodes + 1):
    observations = vec_env.reset()
    dones = [False] * vec_env.num_envs
    scores = [0] * vec_env.num_envs

    while not all(dones):
        actions, _states = model.predict(observations)
        observations, rewards, dones, info = vec_env.step(actions)
        scores = [score + reward for score, reward in zip(scores, rewards)]
        #print("observations==",observations)
    print('Final Expiration time==', info)
    for i, score in enumerate(scores):
        print(f"Episode: {episode}, Flow ID: {flow_ids[i]}, Score: {score}")
vec_env.close()
    