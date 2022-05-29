import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig

from astar import AStar
from rele import APPOHolder

from random import random

p = {
    'random': 0.0,
    'stuck': 5,
    'start': 10 # 0 даёт похожий результат
}

print(p)

last_positions = [[] for _ in range(256)]
count_of_rl_actions = count_of_rl_actions = [p['start'] for _ in range(256)]


class Model:
    def __init__(self):
        self.agents = None
        self.actions = {tuple(GridConfig().MOVES[i]): i for i in
                        range(len(GridConfig().MOVES))}  # make a dictionary to translate coordinates of actions into id
        self.appo = APPOHolder('./weights/c164')

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        
        global p

        # память последних действий
        global last_positions, count_of_rl_actions
        # if (len(last_positions) == 0):
        #     last_positions = [[] for _ in range(len(obs))]
        #     count_of_rl_actions = [10 for _ in range(len(obs))]
        
        if self.agents is None:
            self.agents = [AStar() for _ in range(len(obs))]  # create a planner for each of the agents
        actions = []
        rl_actions = self.appo.act(obs)
        for k in range(len(obs)):
            if positions_xy[k] == targets_xy[k]:  # don't waste time on the agents that have already reached their goals
                actions.append(0)  # just add useless action to save the order and length of the actions
                continue
            
            new_agents = np.zeros([11,11],dtype=float)

            # self.agents[k].update_obstacles(obs[k][0], obs[k][1], (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
            self.agents[k].update_obstacles(obs[k][0], new_agents, (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
            self.agents[k].compute_shortest_path(start=positions_xy[k], goal=targets_xy[k])
            next_node = self.agents[k].get_next_node()

            if (len(last_positions[k]) > 2 and last_positions[k][-1] == last_positions[k][-2] and random() >= p['random']):
                count_of_rl_actions[k] = p['stuck']
            # if (random() >= 0.5 or True):
            if (count_of_rl_actions[k] == 0):
                actor = 'A*'
                current_action = self.actions[(next_node[0] - positions_xy[k][0], next_node[1] - positions_xy[k][1])]
            else:
                count_of_rl_actions[k] -= 1
                actor = 'RL'
                current_action = rl_actions[k]
            
            actions.append(current_action)
            last_positions[k].append(positions_xy[k])

        return actions
