from pogema import GridConfig
from astar import AStar
import numpy as np
from random import random

class Model:
    def __init__(self):
        self.agents = None
        self.actions = {tuple(GridConfig().MOVES[i]): i for i in
                        range(len(GridConfig().MOVES))}  # make a dictionary to translate coordinates of actions into id

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        if self.agents is None:
            self.agents = [AStar() for _ in range(len(obs))]  # create a planner for each of the agents
        actions = []
        for k in range(len(obs)):
            if positions_xy[k] == targets_xy[k]:  # don't waste time on the agents that have already reached their goals
                actions.append(0)  # just add useless action to save the order and length of the actions
                continue

            new_agents = np.zeros([11,11],dtype=float)
            if (random() >= 0.5):
                self.agents[k].update_obstacles(obs[k][0], obs[k][1], (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
            else:
                self.agents[k].update_obstacles(obs[k][0], new_agents, (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
            
            self.agents[k].compute_shortest_path(start=positions_xy[k], goal=targets_xy[k])
            next_node = self.agents[k].get_next_node()
            actions.append(self.actions[(next_node[0] - positions_xy[k][0], next_node[1] - positions_xy[k][1])])
        return actions