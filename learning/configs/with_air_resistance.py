from torch import nn
from learning.learn import make_env

config = {
    "env": {
        "generater": make_env,
        "count": 6,
        "simulation_config": {
            "apply_air_resistance": True
        }
    },
    "save": {
        "save_freq": 20000,
        "save_model_path": "models_sac_with_air_resistance/checkpoints/",
        "name_prefix": "sac_projectile_with_air_resistance"
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
        "tensorboard_log": "logs_sac_with_air_resistance/tensorboard/"
    },
    "checkpoint": {
        "save_freq": 10000,
        "save_model_path": "models_sac_with_air_resistance/checkpoints/",
        "name_prefix": "sac_projectile_with_air_resistance"
    },
    "train": {
        "total_timesteps_per_iter": 100_000,
        "num_iterations": 4
    }
}
