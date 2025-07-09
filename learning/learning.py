from stable_baselines3 import TD3
from learning.env import ProjectileEnv
import os

os.makedirs("models", exist_ok=True)


env = ProjectileEnv()

model = TD3(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=100_000,
    batch_size=64,
    train_freq=1,
    gradient_steps=1,
    learning_starts=1000,
    policy_kwargs={"net_arch": [64, 64]}
)


total_timesteps_per_iter = 100_000
num_iterations = 5

for i in range(num_iterations):
    print(f"=== Iteration {i+1} ===")
    model.learn(total_timesteps=total_timesteps_per_iter, reset_num_timesteps=False)
    model.save(f"models/td3_projectile_checkpoint_{(i+1)*total_timesteps_per_iter}")
