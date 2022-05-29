import gym
import numpy as np
import cv2 as cv
from pogema import GridConfig
from pogema.grid import Grid
from heapq import heappop, heappush
from model import Model
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

class Wrapper(gym.Wrapper):
    
    
    '''
    action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=bool)
    '''
    
    def __init__(self, env, headless = True):
        super().__init__(env)
        
        self.observation_space = gym.spaces.Dict(dict(
            observation=gym.spaces.Box(0.0, 1.0, shape=(245,), dtype='int8')
        ))
        self.action_space = gym.spaces.MultiDiscrete([5])
        #print(self.action_space)
        #self.observation_space =  gym.spaces.Box(0.0, 1.0, shape=(3, 11, 11), dtype='float32')
        self.env = env
        self.headless = headless
        self.prev_pos = [0, 0]
        self.prev_prev_pos = [0, 0]
        self.solver = Model()
        
    def compute_astar(self, map):

        self.astar.update_obstacles(map[0], map[1], (self.env.get_agents_xy_relative()[0][0] - 5, self.env.get_agents_xy_relative()[0][1] - 5))
        self.astar.compute_shortest_path(start=self.env.get_agents_xy_relative()[0], goal=self.env.get_targets_xy_relative()[0])
        next_node = self.astar.get_next_node()
        astar = {tuple(GridConfig().MOVES[i]): i for i in range(len(GridConfig().MOVES))
            }[(next_node[0] - self.env.get_agents_xy_relative()[0][0], next_node[1] - self.env.get_agents_xy_relative()[0][1])]
        return astar

    def step(self, action):
        self.steps += 1
        action = action[0]

        #actions = self.solver.act(self.state, [0] * len(self.state), self.env.get_agents_xy_relative(), self.env.get_targets_xy_relative())
        actions = [np.random.randint(0, 5) for i in range(30)]
        actions[0] = action
        next_state, reward, done, info = self.env.step(actions)
        self.state = next_state

        astar = [self.compute_astar( next_state[0])]
        vec2targ = np.array(self.env.get_targets_xy_relative()[0]) - np.array(self.env.get_agents_xy_relative()[0])
        state = np.concatenate((next_state[0][:2].flatten(), vec2targ, astar))
        # self.compute_astar( next_state[0])
        #print(state)
        observation = {'observation': state}

        #print(observation)
        #rew = -np.linalg.norm(np.array(self.env.get_targets_xy()[0]) - np.array(self.env.get_agents_xy()[0])) + self.delta - self.steps * 0.1
        #print(rew)
        rew = reward[0] * 100
        if action != astar[0]:
            rew -= 0.1
        if self.prev_pos[0] == self.env.get_agents_xy()[0][0] and self.prev_pos[1] == self.env.get_agents_xy()[0][1]:
            rew -= 0.1
        elif self.prev_prev_pos[0] == self.env.get_agents_xy()[0][0] and self.prev_prev_pos[1] == self.env.get_agents_xy()[0][1]:
            rew -= 0.1


        self.prev_prev_pos = self.prev_pos.copy()
        self.prev_pos = np.array(self.env.get_agents_xy()[0])
        
        #print(astar_move, action, astar_move == action)
        for i in range(len(info)):
            info[i]['is_success'] = reward[i]
        if not self.headless:
            self.render()
        return observation, rew, done[0], info[0]

    def render(self):
        self.check_reset()
        image = np.zeros((len(self.grid.obstacles), len(self.grid.obstacles[0]),3), np.uint8)
        for i in range(len(self.grid.obstacles)):
            for j in range(len(self.grid.obstacles)):
                if self.grid.obstacles[i][j]:
                    image[i][j] = [255, 0, 0]
        image[self.grid.positions_xy[0][0]][self.grid.positions_xy[0][1]] = [0, 0, 255]
        image[self.grid.finishes_xy[0][0]][self.grid.finishes_xy[0][1]] = [0, 255, 0]
        for i in self.grid.positions_xy[1:]:
            image[i[0]][i[1]] = [0, 25, 50]
        for i in self.grid.finishes_xy[1:]:
            image[i[0]][i[1]] = [0, 50, 25]
        
        cv.imshow("field", cv.resize(image, (1600, 1600), interpolation = cv.INTER_AREA))
        #cv.imshow("field", image)
        cv.waitKey(1)

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.state = state
        self.steps = 0
        self.delta = np.linalg.norm(np.array(self.env.get_targets_xy()[0]) - np.array(self.env.get_agents_xy()[0]))
        self.astar = AStar()
        vec2targ = np.array(self.env.get_targets_xy_relative()[0]) - np.array(self.env.get_agents_xy_relative()[0])
        state = np.concatenate((state[0][:2].flatten(), vec2targ, [0]))
        return {'observation': state}
