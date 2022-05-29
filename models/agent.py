from turtle import position
import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig

from algorithms.rele import APPOHolder
from algorithms.astar import AStar

import random

actions = {tuple(GridConfig().MOVES[i]): i for i in range(len(GridConfig().MOVES))}

class Agent:
    def __init__(self):
        #hyperparams
        '''
        #[INFO][MEAN][ 10 ] CSR:  0.58 , ISR:  0.9667437842184009
        self.rlIntervalStart = 120
        self.rlIntervalEnd = 180
        self.linearChangeToAStar = 1
        self.stuckDuration = 8
        self.repeatDuration = 8
        self.countNeigbhoursL1 = 2
        self.countNeigbhoursL2 = 5
        self.randomNeigL1 =0.2
        self.randomNeigL2 =0.1
        self.k_check = 5
        '''

        self.rlIntervalStart = 90
        self.rlIntervalEnd = 130
        self.linearChangeToAStar = 1.5
        self.stuckDuration = 2
        self.repeatDuration = 2
        self.activateFullMapDuration = 2
        self.countNeigbhoursL1 = 1
        self.countNeigbhoursL2 = 2
        self.randomNeigL1 =0.015
        self.randomNeigL2 =0.015
        self.k_check = 1



        self.isRandom = False

        self.AStarState = AStar()
        self.RLState = None
        self.AStarStep = None

        self.lastPosition = []
        self.lastStepAuthor = None
        self.position = None
        self.target = None
        self.obs = None
        self.currentStep = 0

        self.lastStart = None
        self.lastStartStep = 0

        self.isStuck = False
        self.stuckStep = -1
        self.isRepeat = False
        self.repeatStep = -1

        self.countNeigbhours = 0
        self.neigStep = -1
        self.isNeig = False
        self.neigLevel = 0

        self.activateFullMap = False
        self.activateFullMapStep = -1

        self.history = []

    def convertToAction(self, step):
        global actions
        return actions[(step[0]-self.position[0], step[1]-self.position[1])]

    def __AStarCalculateStep(self):

        if random.random() >= 0.050:
            self.AStarState.update_obstacles(self.obs[0], self.obs[1], (self.position[0] - 5, self.position[1] - 5))
        else:
            self.AStarState.update_obstacles(self.obs[0], np.zeros([11,11]), (self.position[0] - 5, self.position[1] - 5))
            self.AStarState.compute_shortest_path(start=self.position, goal=self.target)
            self.AStarStep = self.convertToAction(self.AStarState.get_next_node())

    def __UpdateRL(self, RLState):
        self.RLState = RLState

    #Analyse
    def stuckAnalyse(self):
        if len(self.lastPosition) < 2:
            self.isStuck = False
            return

        if self.position == self.lastPosition[-1] and self.position == self.lastPosition[-2] and self.history[-1] != 0:
            self.stuckStep = self.currentStep
            self.isStuck = True
        else:
            self.isStuck = False  


    def repeatAnalyse(self):
        if len(self.history) < 5:
            self.isRepeat = False
            return

        if self.history[-1] != self.history[-2] and self.history[-1] == self.history[-3] and self.history[-2] == self.history[-4] and ((self.history[-1] == 1 and self.history[-2] == 2) or (self.history[-1] == 3 and self.history[-2] == 4) or (self.history[-1] == 1 and self.history[-2] == 2) or (self.history[-1] == 3 and self.history[-2] == 4) or (self.history[-1] == 2 and self.history[-2] == 1) or (self.history[-1] == 4 and self.history[-2] == 3)):
            self.repeatStep = self.currentStep
            self.isRepeat = True
        else:
            self.isRepeat = False

    def findNeighbours(self):
        k_check = self.k_check
        self.countNeigbhours = np.sum(self.obs[1][5-k_check:5+k_check + 1, 5-k_check:5+k_check + 1])

        if self.countNeigbhours >= self.countNeigbhoursL1 and self.countNeigbhours <= self.countNeigbhoursL2:
            self.neigStep = self.currentStep
            self.isNeig = True
            self.neigLevel = 1
            self.activateFullMap = False
        elif self.countNeigbhours > self.countNeigbhoursL2:
            self.neigStep = self.currentStep
            self.isNeig = True
            self.neigLevel = 2
            self.activateFullMap = True
            self.activateFullMapStep = self.currentStep
        else:
            self.isNeig = False
            self.neigLevel = 0
            self.activateFullMap = False

    #End of Analyse

    def __analyseEnvironment(self):
        self.stuckAnalyse()
        self.repeatAnalyse()
        self.findNeighbours()

        return

    def __analyseMemory(self):
        return

    def getRLStep(self):
        self.lastStepAuthor = 'RL'
        return self.RLState

    def getAStarStep(self):
        self.lastStepAuthor = 'A'
        return self.AStarStep
        
    def __generate(self):

        #RL Triggers

        if self.currentStep >= self.rlIntervalStart and self.currentStep < self.rlIntervalEnd :
            return self.getRLStep()

        
        if self.currentStep >= 190 and self.currentStep < 220 :
            return self.getRLStep()

        if self.currentStep - self.stuckStep <= self.stuckDuration:
            if random.random() > (self.currentStep/255 *self.linearChangeToAStar):
                return self.getRLStep()

        if self.currentStep - self.neigStep <= 4:
            if self.neigLevel == 1:
                return self.getRLStep() if random.random() > self.randomNeigL1 else self.getAStarStep()
            elif self.neigLevel == 2:
                return self.getRLStep() if random.random() > self.randomNeigL2 else np.random.randint(1,5)

        if self.currentStep - self.repeatStep <= self.repeatDuration:
            if self.isStuck and self.lastStepAuthor == 'A':
                self.AStarState = AStar()
                return self.getRLStep()
        #End of RL Triggers

        return self.getAStarStep()

    def doStep(self, obs, position, target, RLState):
        if self.lastStart == None:
            self.lastStart = position

        if random.random() >= 0.4:
            self.isRandom = True

        self.position = position
        self.target = target
        self.obs = obs

        self.__UpdateRL(RLState)
        self.__AStarCalculateStep()

        self.__analyseEnvironment()
        self.__analyseMemory()

        action = self.__generate()

        self.history.append(action)
        self.lastPosition.append(self.position)

        if len(self.history) == 6:
            self.history.pop(0)

        if len(self.lastPosition) == 6:
            self.lastPosition.pop(0)

        self.currentStep+=1
        return action
