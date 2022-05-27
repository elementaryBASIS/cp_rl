import gym
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.animation import AnimationMonitor
import numpy as np
import pogema
from pogema import GridConfig

# Define random configuration
grid_config = GridConfig(num_agents=2, # количество агентов на карте
                         size=8,      # размеры карты
                         density=0.3,  # плотность препятствий
                         seed=1,       # сид генерации задания 
                         max_episode_steps=256,  # максимальная длина эпизода
                         obs_radius=5, # радиус обзора
                        )

env = gym.make("Pogema-v0", grid_config=grid_config)
env = AnimationMonitor(env)

# обновляем окружение
obs = env.reset()

done = [False, ...]

while not all(done):
    # Используем случайную стратегию
    for i in range(1):
        obs, reward, done, info = env.step([np.random.randint(4) for _ in range(len(obs))])
        env.render()

# сохраняем анимацию и рисуем ее
#env.save_animation("render.svg", egocentric_idx=0)