from stable_baselines3 import PPO
from learning.env import ProjectileEnv

env = ProjectileEnv()
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.05)

model.learn(total_timesteps=10**5)

obs = env.reset()

action, _ = model.predict(obs)
obs, reward, done, info = env.step(action)

print(f"Action taken: {action}, Reward: {reward}, Done: {done}")