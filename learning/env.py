from gym import spaces, Env
from simulation.simul import Simulation
import numpy as np

class ProjectileEnv(Env):
    def __init__(self):
        super().__init__()
        self.min_angle = 10.0
        self.max_angle = 80.0
        self.last_angle = 45.0
        self.last_landing_x = 0.0
        self.min_speed = 480.0
        self.max_speed = 520.0
        self.last_speed = 500.0
        self.target_x = 1400.0

        self.max_steps = 5
        self.current_step = 0
        self.cumulative_reward = 0.0

        self.action_space = spaces.Box(
            low=np.array([self.min_angle, self.min_speed], dtype=np.float32),
            high=np.array([self.max_angle, self.max_speed], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0, -1, 0.0], dtype=np.float32),
            high=np.array([1.0, 0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.simulation = Simulation(1500, 500)
        self.state = self._get_state()

    def _get_state(self):
        angle_norm = (self.last_angle-self.min_angle) / (self.max_angle-self.min_angle)
        error_norm = -abs(self.target_x-self.last_landing_x)/ self.target_x
        speed_norm = (self.last_speed - 480) / 40.0
        return np.array([angle_norm, error_norm, speed_norm], dtype=np.float32)

    def reset(self):
        self.last_angle = np.random.uniform(self.min_angle, self.max_angle)
        self.last_speed = np.random.uniform(self.min_speed, self.max_speed)
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.state = self._get_state()
        return self.state

    def step(self, action):
        angle = float(action[0])
        speed = float(action[1])
        self.last_angle = angle
        self.last_speed = speed

        simulation_result = self.simulation.run(
            angle=angle,
            speed=self.last_speed,
            apply_air_resistance=False,
            step=1/60,
            record_trajectory=False
        )

        self.last_landing_x = simulation_result["landing_x"]
        error = abs(self.last_landing_x - self.target_x)
        reward = np.exp(-(error / 100))

        self.cumulative_reward += reward
        self.current_step += 1
        self.state = self._get_state()
        done = self.current_step >= self.max_steps

        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass
