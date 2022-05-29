from model import Model


# from best import Model

import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
import random

def main():
    # Define random configuration
    grid_config = GridConfig(num_agents=128,  # количество агентов на карте
                             size=64,  # размеры карты
                             density=0.4,  # плотность препятствий
                             seed=None,  # сид генерации задания
                             max_episode_steps=256,  # максимальная длина эпизода
                             obs_radius=5,  # радиус обзора
                             )

    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()

    done = [False for k in range(len(obs))]
    solver = Model()
    steps = 0
    while not all(done):
        # Используем AStar
        obs, reward, done, info = env.step(solver.act(obs, done,
                                                      env.get_agents_xy_relative(),
                                                      env.get_targets_xy_relative()))
        steps += 1
        #print(steps, np.sum(done))

    # сохраняем анимацию и рисуем ее
    env.save_animation("render.svg", egocentric_idx=None)


if __name__ == '__main__':
    main()
