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
        self.state = 10 # flow expiration 
        self.current_step = 0
        self.max_steps = len(self.data)
        self.action_space = Discrete(10,start=-5) 
        self.observation_space = Dict({
            #'expiration time': gym.spaces.Box(low=0, high=65535, shape=(1,), dtype=np.int32),
            'ongoing': gym.spaces.Discrete(2),  # Transmission ongoing (0 or 1)
            'expired': gym.spaces.Discrete(2),  # Flow expired (0 or 1)
            'packets': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),  # Packets
            'bytes': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)  # Bytes
        })

    def _get_observation(self):
        current_time = self.current_step  # Assuming each step represents a unit of time
        expiration_time = self.state
        packets = self.data.loc[self.current_step, 'Packets']
        bytes = self.data.loc[self.current_step, 'Bytes']
        ongoing = self.data.loc[self.current_step, 'State'] != 'PENDING_REMOVE' #packet transmission is ongoingn or not
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
        self.state += action -5

        # Calculate the reward based on the expiration time
        reward = self._calculate_reward()

        self.current_step += 1
        
        # Check if the episode is done
        done = self.current_step == self.max_steps-1
        
        # Get the current observation
        observation = self._get_observation()
        return observation, reward, done, {'expiration time':self.state}
    
    def _calculate_reward(self):
        current_time = self.current_step  # Assuming each step represents a unit of time
        expiration_time = self.state
        # Check if the flow has expired
        #print('expiration_time',expiration_time)
        #print('current_time',current_time)
        #print('current State',self.data.loc[self.current_step, 'State'])
        time_penalty = (expiration_time - current_time) 
        #print('time_penalty==',time_penalty)
        if self.data.loc[self.current_step, 'State'] != 'PENDING_REMOVE':
            if expiration_time < current_time:
                return time_penalty  # Flow expired, assign a negative reward
            else:
                return 10 - time_penalty # Default reward if flow has not expired
        elif expiration_time > current_time:
            return -time_penalty #Transmission done, flow not removed
        
    def render(self):
        pass
    
    def reset(self):
        self.current_step = 0
        self.max_steps = len(self.data)
        self.state = 10
        return self._get_observation()
    
#env = FlowExpirationEnv('flow_statistics5.csv')
env = FlowExpirationEnv('filtered_data.csv')

#env = DummyVecEnv([lambda: env])
model = PPO('MultiInputPolicy', env, verbose=1, learning_rate=0.001, n_steps=2048, batch_size=64, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, vf_coef=0.5)
model.learn(total_timesteps=20000)
evaluate_policy(model, env, n_eval_episodes=50, render=True)


episodes = 5
for episode in range(1, episodes+1):
    observation = env.reset()
    done = False
    score = 0 

    while not done:
        # Get the predicted action probabilities
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        score+=reward

    print('Episode:{} Score:{}'.format(episode, score))
    print('Final Expiration time==', info)
env.close()