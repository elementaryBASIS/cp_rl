import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from model import Model
from simulation import Config, Simulation

def main():

    testConfig = Config(min_agents=10, min_size=16, max_episode_steps=256)
    simulation = Simulation(Model, config=testConfig)
    simulation.evaluate(1000)


if __name__ == '__main__':
    main()
