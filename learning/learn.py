import os
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from learning.env import ProjectileEnv
import learning.configs.mortar 
import learning.configs.with_air_resistance 
import learning.configs.no_air_resistance 

mortar_config = learning.configs.mortar.config
with_air_resistance_config = learning.configs.with_air_resistance.config
no_air_resistance_config = learning.configs.no_air_resistance.config


def make_env(simulation_config=None):
    def _init():
        env = Monitor(ProjectileEnv(simulation_config=simulation_config))
        return env
    return _init


def deep_merge_dicts(base, override):
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result



def learn(config=None):
    from stable_baselines3.common.vec_env import SubprocVecEnv

    default_config = {
        "env": {
            "generater": make_env,
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
            "tensorboard_log": "logs_sac/tensorboard/"
        },
        "checkpoint": {
            "save_freq": 20000,
            "save_model_path": "models_sac/checkpoints/",
            "name_prefix": "sac_projectile"
        },
        "train": {
            "total_timesteps_per_iter": 100_000,
            "num_iterations": 5
        }
    }

    if config is None:
        config = {}

    merged_config = deep_merge_dicts(default_config, config)

    os.makedirs(merged_config['checkpoint']['save_model_path'], exist_ok=True)
    os.makedirs(merged_config['model']['tensorboard_log'], exist_ok=True)

    env = SubprocVecEnv([
        merged_config['env']['generater'](simulation_config=merged_config['env']['simulation_config'])
        for _ in range(merged_config['env']['count'])
    ])

    model = SAC(
        policy=merged_config['model']['policy'],
        env=env,
        **{k: v for k, v in merged_config['model'].items() if k != "policy"}
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=merged_config['checkpoint']['save_freq'],
        save_path=merged_config['checkpoint']['save_model_path'],
        name_prefix=merged_config['checkpoint']['name_prefix']
    )

    total_timesteps_per_iter = merged_config['train']['total_timesteps_per_iter']
    num_iterations = merged_config['train']['num_iterations']

    print("=== START ===")
    print(f"TOTAL ITERATIONS: {num_iterations}")
    print(f"TIMESTEPS PER ITERATION: {total_timesteps_per_iter}")
    print(f"TOTAL TIMESTEPS: {total_timesteps_per_iter * num_iterations}")

    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1}/{num_iterations} ===")

        model.learn(
            total_timesteps=total_timesteps_per_iter,
            reset_num_timesteps=False,
            callback=[checkpoint_callback],
            tb_log_name=f"SAC_iter_{i+1}"
        )

        print(f"ITERATION {i+1}/{num_iterations} COMPLETED")

    final_model_path = os.path.join(merged_config['checkpoint']['save_model_path'], "sac_final_model")
    model.save(final_model_path)
    print(f"\nModel saved to {final_model_path}")




if __name__ == "__main__":
    for config in [ {"config":no_air_resistance_config, "model_name":"no_air_resistance"},{"config":with_air_resistance_config, "model_name":"with_air_resistance"},{"config":mortar_config, "model_name":"mortar"}]:
        print(f"\n=== {config['model_name']} ===")
        learn(config=config['config'])

