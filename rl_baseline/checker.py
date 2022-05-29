from mark_model import Model
#from rl_baseline_model import Model
from pogema.animation import AnimationMonitor
from pogema import GridConfig
import gym
import numpy as np
if __name__ == "__main__":
    
    # Define random configuration
    grid_config = GridConfig(num_agents=128,  # количество агентов на карте
                             size=64,  # размеры карты
                             density=0.4,  # плотность препятствий
                             seed=58,  # сид генерации задания
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
        print(f"({steps};{sum(done)})")
        steps += 1
        print(steps, np.sum(done))
        #render(env.grid)
    # сохраняем анимацию и рисуем ее
    env.save_animation("render.svg", egocentric_idx=None)

