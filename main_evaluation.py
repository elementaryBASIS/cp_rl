import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from evaluate_system.generator import Generator, Configurator
from evaluate_system.fileReader import FileReader
from evaluate_system.simulation import Simulation
from model import Model

def main():

    simulation = Simulation(Model, input=FileReader())
    simulation.evaluate(10000)


if __name__ == '__main__':
    main()
