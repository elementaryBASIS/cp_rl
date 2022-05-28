import numpy as np
from pogema import GridConfig
import gym
from pogema.animation import AnimationMonitor
import statistics

class Config:
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


class Simulation:
    def __init__(self, model, config = Config()):
        self.modelClass = model
        self.config = config

    def __generateConfig(self):

        num_agents = np.random.randint(self.config.min_agents, self.config.max_agents+1)
        density = np.random.uniform(self.config.min_density, self.config.max_density+0.001)
        size = np.random.randint(self.config.min_size, self.config.max_size+1)
        
        if num_agents >= int(size*density):
            num_agents = size*density//np.random.randint(1,3)
            num_agents = num_agents if num_agents > 0 else 1 
        
        return GridConfig(num_agents=num_agents,
                        size=size,
                        density=density,
                        seed=np.random.randint(100000),
                        max_episode_steps=self.config.max_episode_steps,
                        obs_radius=self.config.obs_radius,
                        )

    def simulate(self):

        grid_config = 0

        while True:
            try:
                grid_config = self.__generateConfig()
                env = gym.make("Pogema-v0", grid_config=grid_config)
                env = AnimationMonitor(env)
                obs = env.reset()
                break
            except OverflowError:
                continue
        
        print('[INFO][EPOCH CONFIG]', grid_config)

        solver = self.modelClass()
        done = [False for k in range(len(obs))]
        steps = 0
        info = []

        while not all(done):
            obs, reward, done, info = env.step(solver.act(obs, done,
                                                            env.get_agents_xy_relative(),
                                                            env.get_targets_xy_relative()))
            steps += 1
        
        return info

    def evaluate(self, epochs):
        csr = []
        isr = []

        for i in range(epochs):

            while True:
                try:
                    info = self.simulate()
                    break
                except IndexError:
                    continue

            print('[INFO][EPOCH][',i+1,'] Result: ', info)
            csr.append(info[0]['metrics']['CSR'])
            isr.extend([info[i]['metrics']['ISR'] for i in range(len(info))])

        print('TOTAL: ')
        print('CSR:', statistics.fmean(csr))
        print('ISR: ', statistics.fmean(isr))

    