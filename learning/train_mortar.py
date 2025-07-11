import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from learning.mortar_env import MortarEnv
from torch import nn


def make_env():
    def _init():
        env = Monitor(MortarEnv())
        return env
    return _init


def train(total_timesteps=100000, num_envs=4):
    os.makedirs("models_sac_mortar/checkpoints", exist_ok=True)
    os.makedirs("logs_sac_mortar/tensorboard", exist_ok=True)
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        policy_kwargs={"net_arch": [256, 256], "activation_fn": nn.ReLU},
        tensorboard_log="logs_sac_mortar/tensorboard",
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="models_sac_mortar/checkpoints",
        name_prefix="sac_mortar",
    )
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])
    model.save("models_sac_mortar/sac_mortar_final")


if __name__ == "__main__":
    train()
