import gym
import numpy as np


class Wrapper(gym.Wrapper):
    
    
    '''
    action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=bool)
    '''
    
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = gym.spaces.Dict(dict(
            observation=gym.spaces.Box(0.0, 1.0, shape=(3, 11, 11), dtype='float32')
        ))
        self.action_space = gym.spaces.MultiDiscrete([5])
        #print(self.action_space)
        #self.observation_space =  gym.spaces.Box(0.0, 1.0, shape=(3, 11, 11), dtype='float32')
        self.env = env
    def _obs(self):
        obs = self.env._obs()
        return obs[0]
    
    def step(self, action):
        self.steps += 1
        #action = np.int0(action)
        next_state, reward, done, info = self.env.step(action)
        #next_state =next_state[0]
        next_state = {'observation': next_state[0]}
        rew = -np.linalg.norm(np.array(self.env.get_targets_xy()[0]) - np.array(self.env.get_agents_xy()[0])) + self.delta - self.steps * 0.2
        #print(rew)
        #self.env.render()
        return next_state, rew, done[0], info[0]


    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.steps = 0
        self.delta = np.linalg.norm(np.array(self.env.get_targets_xy()[0]) - np.array(self.env.get_agents_xy()[0]))
        return {'observation': state[0]}