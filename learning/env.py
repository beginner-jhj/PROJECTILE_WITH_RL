from gym import spaces, Env
from simulation.simul import Simulation
import numpy as np

class ProjectileEnv(Env):
    def __init__(self):
        super().__init__()
        self.min_angle = 10.0
        self.max_angle = 80.0
        self.max_steps = 10 
        self.current_step = 0
        self.total_reward = 0.0
        self.last_angle = 45.0
        self.last_reward = 0.0

        self.action_space = spaces.Box(
            low=np.array([self.min_angle], dtype=np.float32),
            high=np.array([self.max_angle], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: [angle_norm, reward_norm, step_ratio]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.simulation = Simulation(1500, 500)
        self.state = self._get_state()

    def _get_state(self):
        angle_norm = self.last_angle / 90.0
        reward_norm = self.last_reward / 10.0
        step_ratio = self.current_step / self.max_steps
        return np.array([angle_norm, reward_norm, step_ratio], dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.total_reward = 0.0
        self.last_angle = 45.0
        self.last_reward = 0.0
        self.state = self._get_state()
        return self.state

    def step(self, action):
        angle = float(action[0])
        self.last_angle = angle

        simulation_result = self.simulation.run(
            angle=angle,
            speed=500,
            apply_air_resistance=True,
            step=1/120,
            record_trajectory=False  
        )

        landing_x = simulation_result["landing_x"]
        reward = np.log1p(landing_x)
        self.last_reward = reward

        self.total_reward += reward
        self.current_step += 1
        done = self.current_step >= self.max_steps

        self.state = self._get_state()

        if done:
            print(f"Last Reward received: {reward}, Last State: {self.state}, Total Reward: {self.total_reward}")

        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass
