import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch as th
# Instantiate the parser
parser = argparse.ArgumentParser(description='Trainer optional arguments')

parser.add_argument('--headless',
                    action='store_true',
                    help="do not use bullet GUI", 
                    default=False)

parser.add_argument('-t', '--tsteps', 
                    type=float, 
                    default=1e6,
                    help='number of timesteps (default: 1e6)')

if __name__ == "__main__":

    args = parser.parse_args()
    print("-----------------------------------------\n")
    print("viz    :", not args.headless)
    print("tsteps :", args.tsteps)
    print("\n-----------------------------------------")

    RL_ALGORITHM = args.algo
    N_TIMESTEPS = int(args.tsteps)
    from envs.labirinth import Labirinth as env
    from hyperparams import PPO1 as hyperparams
    if args is not None:
        headless = args.headless

    save_path = "models/"
    tensorboard_path = "tensorboard_log/"

    # Create the evaluation environment and callbacks
    train_env = Monitor(env(headless))

    callbacks = [EvalCallback(train_env, best_model_save_path=save_path)]

    # Save a checkpoint every n steps
    callbacks.append(
        CheckpointCallback(
            save_freq=100000, save_path=save_path, name_prefix="rl_model"
        )
    )


    n_actions = env.action_space.shape[0]

    train_env.reset()
    model = PPO('MultiInputPolicy', train_env, verbose=1, tensorboard_log=tensorboard_path, **hyperparams)

    try:
        model.learn(N_TIMESTEPS, callback=callbacks)
    except KeyboardInterrupt:
        pass
    print(f"Saving to {save_path}.zip")
    model.save(save_path)

    train_env.close()
