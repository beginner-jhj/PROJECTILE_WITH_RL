from torch import nn
from stable_baselines3.common.monitor import Monitor
from learning.mortar_env import MortarEnv

# Helper to create environments for the mortar simulation

def make_mortar_env():
    def _init():
        env = Monitor(MortarEnv())
        return env
    return _init

config = {
    "save": {
        "save_freq": 20000,
        "save_model_path": "models_sac_mortar/checkpoints",
        "name_prefix": "sac_mortar"
    },
    "env": {
        "generater": make_mortar_env,
        "count": 4,
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
        "tensorboard_log": "logs_sac_mortar/tensorboard"
    },
    "checkpoint": {
        "save_freq": 20000,
        "save_model_path": "models_sac_mortar/checkpoints",
        "name_prefix": "sac_mortar"
    },
    "train": {
        "total_timesteps_per_iter": 100_000,
        "num_iterations": 5
    }
}
