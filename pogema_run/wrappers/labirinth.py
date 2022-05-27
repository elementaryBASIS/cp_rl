import gym
import numpy as np


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.env.render()
        return next_state, reward, done, info