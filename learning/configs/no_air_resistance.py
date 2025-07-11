from torch import nn
from learning.env import ProjectileEnv
from stable_baselines3.common.monitor import Monitor


def make_env(simulation_config=None):
    def _init():
        env = Monitor(ProjectileEnv(simulation_config=simulation_config))
        return env
    return _init

config = {
    "env": {
        "generater": make_env,
        "count": 6,
        "simulation_config": None
    },
    "model": {
        "policy": "MlpPolicy",
        "verbose": 1,
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "batch_size": 256,
        "learning_starts": 1000,
        "train_freq": 1,
        "gradient_steps": 1,
        "target_update_interval": 1,
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": "auto",
        "target_entropy": "auto",
        "use_sde": False,
        "policy_kwargs": {
            "net_arch": [256, 256],
            "activation_fn": nn.ReLU
        },
        "tensorboard_log": "logs_sac/tensorboard/"
    },
    "checkpoint": {
        "save_freq": 12288,
        "save_model_path": "models_sac/checkpoints/",
        "name_prefix": "sac_projectile"
    },
    "train": {
        "total_timesteps_per_iter": 100_000,
        "num_iterations": 1
    }
}
