import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import pandas as pd

class FlowExpirationEnv(Env):
    def __init__(self, csv_file):
        super(FlowExpirationEnv, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.state = 10 # flow expiration 
        self.current_step = 1
        self.max_steps = len(self.data)
        # Actions we can take, decrease, increase, no change
        self.action_space = Discrete(3)
        # Expiration time in seconds
        self.observation_space = Box(low=0, high=65535, shape=(1,) ,dtype=np.int32) 

    def _get_observation(self):
        # Extract relevant flow statistics from the current step
        packets = self.data.loc[self.current_step, 'Packets']
        bytes = self.data.loc[self.current_step, 'Bytes']
        # Return the observation as a NumPy array
        return [packets, bytes]  
        
    def step(self, action):
        # Apply action
        # 0 -1 = -1 
        # 1 -1 = 0 
        # 2 -1 = 1  
        self.state += action -1

        # Calculate the reward based on the expiration time
        reward = self._calculate_reward()

        self.current_step += 1
        
        # Check if the episode is done
        done = self.current_step == self.max_steps-1
        
        observation = np.array([self.state], dtype=np.int32)  # Ensure correct shape of observation
        return observation, reward, done, {}
    
    def _calculate_reward(self):
        current_time = self.current_step  # Assuming each step represents a unit of time
        expiration_time = self.state
        # Check if the flow has expired
        #print('expiration_time',expiration_time)
        #print('current_time',current_time)
        #print('current State',self.data.loc[self.current_step, 'State'])

        if self.data.loc[self.current_step, 'State'] != 'PENDING_REMOVE' and expiration_time < current_time:
            return -1  # Flow expired, assign a negative reward
        elif self.data.loc[self.current_step, 'State'] != 'PENDING_REMOVE' and expiration_time > current_time:
            return 1  # Default reward if flow has not expired
        else:
            return 0
        if self.flow_data.loc[self.current_step, 'State'] != 'PENDING_REMOVE':
            if expiration_time < current_time:
                # Penalize expired flows with a time-dependent penalty
                time_penalty = (current_time - expiration_time) / self.max_steps
                return -1 - time_penalty
            else:
                # Give a small positive shaping reward for flows that are still active and closer to expiration
                shaping_reward = (self.max_steps - expiration_time) / self.max_steps
                return shaping_reward
        else:
            return 0


    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        self.current_step = 0
        self.max_steps = len(self.data)
        self.state = 10
        return np.array([self.state], dtype=np.int32)
    
#env = FlowExpirationEnv('flow_statistics5.csv')
env = FlowExpirationEnv('flowstats.csv')

#env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)
model.learn(total_timesteps=20000)
evaluate_policy(model, env, n_eval_episodes=10, render=True)


episodes = 5
for episode in range(1, episodes+1):
    observation = env.reset()
    done = False
    score = 0 

    while not done:
        action, _states = model.predict(observation)
        #action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print('reward==', reward)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    #print('Final Expiration time==',observation)
env.close()