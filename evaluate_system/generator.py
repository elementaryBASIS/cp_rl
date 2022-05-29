import numpy as np
from pogema import GridConfig
import os

class Configurator:
    def __init__(self, 
                min_agents = 1, 
                max_agents = 32,
                min_density = 0.2,
                max_density = 0.4,
                min_size = 8,
                max_size = 64,
                obs_radius = 5,
                max_episode_steps = 256):

        self.min_agents = min_agents
        self.max_agents = max_agents if max_agents > min_agents else min_agents
        self.min_density = min_density
        self.max_density = max_density if max_density > min_density else min_density
        self.min_size = min_size
        self.max_size = max_size if max_size > min_size else min_size
        self.obs_radius = obs_radius
        self.max_episode_steps = max_episode_steps


class Generator():
    def __init__(self, configurator = Configurator()):
        self.configurator = configurator
        self.currentConfig = {}

        self.file = open('evaluate_system/log.txt', 'a')
        self.saveEvery = 10
        self.saveCounter = 0

    def __generateConfig(self):

        num_agents = np.random.randint(self.configurator.min_agents, self.configurator.max_agents+1)
        density = np.random.uniform(self.configurator.min_density, self.configurator.max_density+0.001)
        size = np.random.randint(self.configurator.min_size, self.configurator.max_size+1)
        
        if num_agents >= int(size*size*(1-density)):
            num_agents = size*size*(1-density)//np.random.randint(1,2)
            num_agents = num_agents if num_agents > 0 else 1 
        
        return {
            'num_agents':num_agents,
            'density':density,
            'seed': np.random.randint(100000),
            'size': size,
            'max_episode_steps': self.configurator.max_episode_steps,
            'obs_radius': self.configurator.obs_radius
        }

    def next(self):

        self.currentConfig = self.__generateConfig()

        gridConfig = GridConfig(num_agents=self.currentConfig['num_agents'],
                        size=self.currentConfig['size'],
                        density=self.currentConfig['density'],
                        seed=self.currentConfig['seed'],
                        max_episode_steps=self.currentConfig['max_episode_steps'],
                        obs_radius=self.currentConfig['obs_radius'],
                        )
        return gridConfig

    def evaluateCurrentTest(self, mark):
        self.file.write(str(mark) + ' ')
        self.file.write(str(self.currentConfig['num_agents']) + ' ')
        self.file.write(str(self.currentConfig['size']) + ' ')
        self.file.write(str(self.currentConfig['density']) + ' ')
        self.file.write(str(self.currentConfig['seed']) + ' ')
        self.file.write(str(self.currentConfig['max_episode_steps']) + ' ')
        self.file.write(str(self.currentConfig['obs_radius']) + '\n')

        if (self.saveCounter == self.saveEvery):
            print('[INFO][CONFIGURATOR] Save results')
            self.file.flush()
            os.fsync(self.file.fileno())
            self.saveCounter = 0
        
        self.saveCounter += 1