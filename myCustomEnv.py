import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
from stable_baselines3 import PPO,DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import pandas as pd

class FlowExpirationEnv(Env):
    def __init__(self, csv_file):
        super(FlowExpirationEnv, self).__init__()
        self.data = pd.read_csv(csv_file)
        flow_id = 32088147978799629
        self.flow_data = self.data[self.data['Flow_Id'] == flow_id].reset_index()
        self.state = 10 # flow expiration 
        self.current_step = 0
        self.max_steps = len(self.flow_data)
        # Actions we can take, decrease, increase, no change
        self.action_space = Discrete(3)
        # Expiration time in seconds
        self.observation_space = Box(low=np.array([0,0,0]), high=np.array([65335,65335,65335]), shape=(3,) ,dtype=np.int32) 

    def _get_observation(self):
        # Extract relevant flow statistics from the current step
        packets = self.data.loc[self.current_step, 'Packets']
        bytes = self.data.loc[self.current_step, 'Bytes']
        # Return the observation as a NumPy array
        return np.array([self.state, packets, bytes],dtype=np.int32)  
        
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
        
        # Get the current observation
        observation = self._get_observation()

        #observation = np.array([self.state], dtype=np.int32)  
        return observation, reward, done, {}
    
    def _calculate_reward(self):
        current_time = self.current_step  # Assuming each step represents a unit of time
        expiration_time = self.state
        # Check if the flow has expired
        #print('expiration_time',expiration_time)
        #print('current_time',current_time)
        #print('current State',self.data.loc[self.current_step, 'State'])

        if self.flow_data.loc[self.current_step, 'State'] != 'PENDING_REMOVE':
            if expiration_time < current_time:
                return -1  # Flow expired, assign a negative reward
            else:
                return 2  # Default reward if flow has not expired
        else:
            return 0 #flow is removed
        
    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        self.current_step = 0
        self.max_steps = len(self.flow_data)
        self.state = 10
        return np.array([self.state,0,0], dtype=np.int32)
    
#env = FlowExpirationEnv('flow_statistics5.csv')
env = FlowExpirationEnv('flow_statistics5.csv')

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