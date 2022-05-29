import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from agent import Agent

from algorithms.rele import APPOHolder
from random import random

class Model:
    def __init__(self):
        self.agents = None
        self.actions = {tuple(GridConfig().MOVES[i]): i for i in
                        range(len(GridConfig().MOVES))}  # make a dictionary to translate coordinates of actions into id
        self.appo = APPOHolder('./weights/c164')

    def act(self, obs, dones, positions_xy, targets_xy) -> list:

        if self.agents is None:
            self.agents = [Agent() for _ in range(len(obs))]

        actions = []

        self.RLState = self.appo.act(obs)
        for k in range(len(obs)):
            if positions_xy[k] == targets_xy[k]:  # don't waste time on the agents that have already reached their goals
                actions.append(0)  # just add useless action to save the order and length of the actions
                continue
            
            step = self.agents[k].doStep(obs[k],positions_xy[k],targets_xy[k], self.RLState[k])
            actions.append(step if step != None else np.random.randint(1,5))
        
        return actions
