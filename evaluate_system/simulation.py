import numpy as np
from pogema import GridConfig
import gym
from pogema.animation import AnimationMonitor
import statistics
from generator import Generator

class Simulation:
    def __init__(self, model, input = Generator()):
        self.modelClass = model
        self.input = input
        self.stopFlag = False

    def simulate(self):

        grid_config = self.input.next()
        
        if grid_config == None:
            self.stopFlag = True
            return

        env = gym.make("Pogema-v0", grid_config=grid_config)
        obs = env.reset()
        
        #print('[INFO][EPOCH CONFIG]', grid_config)

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

            info = self.simulate()

            if self.stopFlag:
                print('Stop flag')
                return

            self.input.evaluateCurrentTest(statistics.fmean([info[i]['metrics']['ISR'] for i in range(len(info))]))

            #print('[INFO][EPOCH][',i+1,'] Result: ', info)

            csr.append(info[0]['metrics']['CSR'])
            isr.extend([info[i]['metrics']['ISR'] for i in range(len(info))])

            print('[INFO][MEAN][',i+1,'] CSR: ', statistics.fmean(csr), ', ISR: ', statistics.fmean(isr))

        print('TOTAL: ')
        print('CSR:', statistics.fmean(csr))
        print('ISR: ', statistics.fmean(isr))

    