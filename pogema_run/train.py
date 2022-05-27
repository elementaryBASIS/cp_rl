import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch as th

import gym
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.animation import AnimationMonitor
import pogema
from pogema import GridConfig
# Instantiate the parser
parser = argparse.ArgumentParser(description='Trainer optional arguments')

parser.add_argument('--headless',
                    action='store_true',
                    help="do not use bullet GUI", 
                    default=False)

parser.add_argument('-t', '--tsteps', 
                    type=float, 
                    default=10e6,
                    help='number of timesteps (default: 1e6)')

# Define random configuration
grid_config = GridConfig(num_agents=1, # количество агентов на карте
                         size=2,      # размеры карты
                         density=0.3,  # плотность препятствий
                         seed=1,       # сид генерации задания 
                         max_episode_steps=256,  # максимальная длина эпизода
                         obs_radius=5, # радиус обзора
                        )

if __name__ == "__main__":

    args = parser.parse_args()
    print("-----------------------------------------\n")
    print("viz    :", not args.headless)
    print("tsteps :", args.tsteps)
    print("\n-----------------------------------------")

    N_TIMESTEPS = int(args.tsteps)
    from wrappers.labirinth import Wrapper as BASISwrapper


    env = gym.make("Pogema-8x8-easy-v0", integration="SampleFactory",)
    
    #from hyperparams import PPO1 as hyperparams
    from hyperparams import A2C1 as hyperparams
    if args is not None:
        headless = args.headless

    save_path = "models/"
    tensorboard_path = "tensorboard_log/"

    # Create the evaluation environment and callbacks
    #train_env = AnimationMonitor(env)
    train_env = BASISwrapper(env)
    train_env = Monitor(train_env)

    callbacks = [EvalCallback(train_env, best_model_save_path=save_path)]

    # Save a checkpoint every n steps
    callbacks.append(
        CheckpointCallback(
            save_freq=100000, save_path=save_path, name_prefix="rl_model"
        )
    )
    #n_actions = train_env.action_space.shape[0]

    train_env.reset()

    model = A2C('MultiInputPolicy', train_env, verbose=1, tensorboard_log=tensorboard_path, **hyperparams)

    try:
        model.learn(N_TIMESTEPS, callback=callbacks)
    except KeyboardInterrupt:
        pass
    print(f"Saving to {save_path}.zip")
    model.save(save_path)

    train_env.close()
