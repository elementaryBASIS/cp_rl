import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from generator import Generator, Configurator
from fileReader import FileReader
from model import Model
from simulation import Simulation

def main():

    simulation = Simulation(Model, input=FileReader())
    simulation.evaluate(10000)


if __name__ == '__main__':
    main()
