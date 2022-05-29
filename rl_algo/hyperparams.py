import torch as th
# Tuned hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
PPO1 =  dict(
        batch_size=64,
        gamma=0.99,
        n_steps=512,
        learning_rate= 2.5e-4,
        ent_coef=0,
        clip_range=0.2,
        n_epochs=10,
        gae_lambda=0.95,
    )
A2C1 = dict()