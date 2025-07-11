if __name__ == "__main__":
    import os
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from learning.env import ProjectileEnv
    from torch import nn

    os.makedirs("models_sac", exist_ok=True)
    os.makedirs("logs_sac", exist_ok=True)

    def make_env():
        def _init():
            env = Monitor(ProjectileEnv())
            return env
        return _init

    env = SubprocVecEnv([make_env() for _ in range(4)])

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        learning_starts=1000,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        target_entropy='auto',
        use_sde=False,
        policy_kwargs={
            "net_arch": [256, 256],
            "activation_fn": nn.ReLU
        },
        tensorboard_log="logs_sac/tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="models_sac/checkpoints/",
        name_prefix="sac_projectile"
    )

    total_timesteps_per_iter = 100_000
    num_iterations = 10

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
