import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from generator import Generator, Configurator
from model import Model
from simulation import Simulation

def main():

    simulation = Simulation(Model, Generator(configurator=Configurator(min_agents=1000, min_size=64, max_size=64)))
    simulation.evaluate(10)


if __name__ == '__main__':
    main()
