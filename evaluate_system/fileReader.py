import numpy as np
from pogema import GridConfig

class FileReader():
    def __init__(self, file = 'evaluate_system/log.txt'):
        self.currentConfig = {}
        self.file = open(file, 'r')
        

    def __generateConfig(self, f):
        line = f.split(' ')
        
        return {
            'last_result':float(line[0]),
            'num_agents':int(line[1]),
            'size': int(line[2]),
            'density':float(line[3]),
            'seed': int(line[4]),
            'max_episode_steps': int(line[5]),
            'obs_radius': int(line[6])
        }

    def next(self):

        line = self.file.readline()

        if line == '':
            self.file.close()
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
        # print('[INFO][COMPARISON]', mark-self.currentConfig['last_result'])
        return