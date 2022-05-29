import argparse
from pyexpat import model
import numpy as np
from stable_baselines3 import PPO
import torch as th
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
import random
import cv2 as cv
from time import sleep
from heapq import heappop, heappush
from random import random

class Node:
    def __init__(self, coord: (int, int) = (0, 0), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f or ((self.f == other.f) and (self.g < other.g))


class AStar:
    def __init__(self):
        self.start = (0, 0)
        self.goal = (0, 0)
        self.max_steps = 10000  # due to the absence of information about the map size we need some other stop criterion
        self.OPEN = list()
        self.CLOSED = dict()
        self.obstacles = set()
        self.other_agents = set()

    def compute_shortest_path(self, start, goal):
        self.start = start
        self.goal = goal
        self.CLOSED = dict()
        self.OPEN = list()
        heappush(self.OPEN, Node(self.start))
        u = Node()
        steps = 0
        while len(self.OPEN) > 0 and steps < self.max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            steps += 1
            for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n = (u.i+d[0], u.j + d[1])
                if n not in self.obstacles and n not in self.CLOSED and n not in self.other_agents:
                    h = abs(n[0] - self.goal[0]) + abs(n[1] - self.goal[1]) # Manhattan distance as a heuristic function
                    nac = 0
                    for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        point  = (n[0] + d[0], n[1] + d[1])
                        if(point in self.obstacles):
                            nac = -1
                        elif(point in self.other_agents):
                            nac = 1
                    h = h+nac
                    heappush(self.OPEN, Node(n, u.g + 1, h))
                    self.CLOSED[n] = (u.i, u.j)  # store information about the predecessor

    def get_next_node(self):
        next_node = self.start  # if path not found, current start position is returned
        if self.goal in self.CLOSED:  # if path found
            next_node = self.goal
            while self.CLOSED[next_node] != self.start:  # get node in the path with start node as a predecessor
                next_node = self.CLOSED[next_node]
        return next_node

    def update_obstacles(self, obs, other_agents, n):
        obstacles = np.transpose(np.nonzero(obs))  # get the coordinates of all obstacles in current observation
        for obstacle in obstacles:
            self.obstacles.add((n[0] + obstacle[0], n[1] + obstacle[1]))  # save them with correct coordinates
        self.other_agents.clear()  # forget previously seen agents as they move
        agents = np.transpose(np.nonzero(other_agents))  # get the coordinates of all agents that are seen
        for agent in agents:
            self.other_agents.add((n[0] + agent[0], n[1] + agent[1]))  # save them with correct coordinates

class Model:
    def __init__(self):
        """
        This should handle the logic of your model initialization like loading weights, setting constants, etc.
        """
        self.start = True
        pass

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        """
        Given current observations, Done flags, agents' current positions and their targets, produce actions for agents.
        """
        actions = []
        astar_actions = []
        if self.start:
            self.start = False
            self.init(len(dones))
        for i in range(len(dones)):


            k = i
            self.astars[k].update_obstacles(obs[k][0], obs[k][1], (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
            self.astars[k].compute_shortest_path(start=positions_xy[k], goal=targets_xy[k])
            next_node = self.astars[k].get_next_node()
            astar_action = (next_node[0] - positions_xy[k][0], next_node[1] - positions_xy[k][1])
            astar_action = {tuple(GridConfig().MOVES[i]): i for i in range(len(GridConfig().MOVES))
            }[astar_action]
            vec2targ = np.array(positions_xy[k]) - np.array(targets_xy[k])

            state = np.concatenate((obs[i][:2].flatten(), vec2targ, [astar_action]))
            
            observ = {'observation': state}


            
            if dones[i]:
                actions.append(0)
            else:
                if (random() >= 0.2):
                    action, _ = self.models[i].predict(observ)
                else:
                    action = astar_action
                actions.append(int(action))
        #return [np.random.randint(5) for _ in range(len(obs))]
        return actions

            
    def init(self, num):
        self.models = []
        self.astars = []
        for i in range(num):
            self.models.append(PPO.load("models/best_model.zip"))
            self.astars.append(AStar())
        
def render(grid):
        image = np.zeros((len(grid.obstacles), len(grid.obstacles[0]),3), np.uint8)
        for i in range(len(grid.obstacles)):
            for j in range(len(grid.obstacles)):
                if grid.obstacles[i][j]:
                    image[i][j] = [255, 0, 0]
        for i in grid.positions_xy[1:]:
            image[i[0]][i[1]] = [0, 25, 255]
        for i in grid.finishes_xy[1:]:
            image[i[0]][i[1]] = [0, 255, 25]
        
        cv.imshow("field", cv.resize(image, (400, 400), interpolation = cv.INTER_AREA))
        #cv.imshow("field", image)
        cv.waitKey(1)
        sleep(0.01)
def main():
    # Define random configuration
    grid_config = GridConfig(num_agents=128, # количество агентов на карте
                         size=64,      # размеры карты
                         density = 0.4,  # плотность препятствий
                         seed=100,       # сид генерации задания 
                         max_episode_steps=256,  # максимальная длина эпизода
                         obs_radius=5, # радиус обзора
                        )

    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()

    done = [False for k in range(len(obs))]
    solver = Model()
    steps = 0
    while not all(done):
        # Используем AStar
        obs, reward, done, info = env.step(solver.act(obs, done,
                                                      env.get_agents_xy_relative(),
                                                      env.get_targets_xy_relative()))
        steps += 1
        print(steps, np.sum(done))
        render(env.grid)
    # сохраняем анимацию и рисуем ее
    env.save_animation("render.svg", egocentric_idx=None)


main()