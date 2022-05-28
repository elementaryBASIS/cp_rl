import argparse
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import torch as th

import gym
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.animation import AnimationMonitor
import pogema
from pogema import GridConfig
# Instantiate the parser
from time import sleep
def play_episode(episode_id, headless = False):
    obs = enjoy_env.reset()
    done = False
    episode_reward = 0.0
    episode_length = 0
    step = 0
    is_success = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = enjoy_env.step(action)
        is_success = info['is_success'] # get latest success status
        episode_reward += reward
        episode_length += 1
        step += 1
        dt = 1.0 / 120.0
        if not headless:
            sleep(dt)
    return episode_reward, episode_length, is_success

# Define random configuration
grid_config = GridConfig(num_agents=1, # количество агентов на карте
                         size=64,      # размеры карты
                         density = 0.35,  # плотность препятствий
                         seed=None,       # сид генерации задания 
                         max_episode_steps=100,  # максимальная длина эпизода
                         obs_radius=5, # радиус обзора
                        )

if __name__ == "__main__":

    from wrappers.labirinth import Wrapper as BASISwrapper


    env = gym.make("Pogema-v0", grid_config=grid_config)


    # Create the evaluation environment and callbacks
    #train_env = AnimationMonitor(env)
    enjoy_env = BASISwrapper(env, False)
    enjoy_env = Monitor(enjoy_env)

    enjoy_env.reset()

    model = PPO.load("models.zip", env=enjoy_env)

    try:
        # Use deterministic actions for evaluation
        episode_rewards, episode_lengths, episode_success = [], [], []
        for episode_id in range(100):
            episode_reward, episode_length, is_success = play_episode(episode_id, False)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_success.append(is_success)
            print(
                f"Episode {len(episode_rewards)} reward={episode_reward}, length={episode_length}, is success={str(is_success)}"
            )

           

        
    except KeyboardInterrupt:
        pass
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_success, std_success = np.mean(episode_success), np.std(episode_success)

    mean_len, std_len = np.mean(episode_lengths), np.std(episode_lengths)
    print("\n-------------------------")
    print("    ==== Results ====")
    print(f"Episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Episode_length={mean_len:.2f} +/- {std_len:.2f}")
    print(f"Episode_success={mean_success:.2f} +/- {std_success:.2f}")
    print("-------------------------")
        
    # Close process
    enjoy_env.close()