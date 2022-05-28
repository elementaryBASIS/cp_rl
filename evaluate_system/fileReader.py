import numpy as np
from pogema import GridConfig

class FileReader():
    def __init__(self, file = 'log.txt'):
        self.currentConfig = {}
        self.file = open(file, 'r')
        

    def __generateConfig(self, f):
        line = f.split(' ')

        return {
            'num_agents':line[0],
            'density':line[1],
            'seed': line[2],
            'size': line[3],
            'max_episode_steps': line[4],
            'obs_radius': line[5]
        }

    def next(self):

        line = self.file.readline()

        if line == '':
            return

        self.currentConfig = self.__generateConfig(line)

        gridConfig = GridConfig(num_agents=self.currentConfig['num_agents'],
                        size=self.currentConfig['size'],
                        density=self.currentConfig['density'],
                        seed=self.currentConfig['seed'],
                        max_episode_steps=self.currentConfig['max_episode_steps'],
                        obs_radius=self.currentConfig['obs_radius'],
                        )
        return gridConfig

    def evaluateCurrentTest(self, mark):
        return