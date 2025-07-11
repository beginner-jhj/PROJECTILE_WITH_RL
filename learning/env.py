from gymnasium import spaces, Env
from simulation.simul import Simulation
import numpy as np

class ProjectileEnv(Env):
    def __init__(self,simulation_config=None):
        super().__init__()
        self.min_angle = 10.0
        self.max_angle = 80.0
        self.min_speed = 480.0
        self.max_speed = 520.0
        self.target_x = 1400.0
        
        self.last_angle = 45.0
        self.last_speed = 500.0
        self.last_landing_x = 0.0
        self.last_error = float('inf')
        self.prev_error = float('inf')
        
        self.max_steps = 10 
        self.current_step = 0
        self.best_error_in_episode = float('inf')
        
        # Action space: [angle, speed]
        self.action_space = spaces.Box(
            low=np.array([self.min_angle, self.min_speed], dtype=np.float32),
            high=np.array([self.max_angle, self.max_speed], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: [angle_norm, speed_norm, landing_ratio, error_norm, improvement]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.simulation = Simulation(1500, 500)
        self.state = self._get_state()

        self.default_simulation_config = {
            "apply_air_resistance": False,
            "record_trajectory": False,
            "step": 1/60,
            "log_last_position": False
        }

        if simulation_config is None:
            simulation_config = {}

        self.merged_simulation_config = {**self.default_simulation_config, **simulation_config}
        
        self.episode_rewards = []
        self.episode_errors = []

    def _get_state(self):
        angle_norm = (self.last_angle - self.min_angle) / (self.max_angle - self.min_angle)
        
        speed_norm = (self.last_speed - self.min_speed) / (self.max_speed - self.min_speed)
        
        landing_ratio = max(0, min(2.0, self.last_landing_x / self.target_x))
        
        max_possible_error = self.target_x  
        error_norm = min(1.0, abs(self.last_error) / max_possible_error)
        
        if self.prev_error == float('inf'):
            improvement = 0.0
        else:
            improvement = np.tanh((self.prev_error - self.last_error) / 100.0)
        
        return np.array([angle_norm, speed_norm, landing_ratio, error_norm, improvement], dtype=np.float32)

    def _calculate_reward(self, error):
        accuracy_reward = max(0, 1 - (error / self.target_x))
        
        improvement_reward = 0.0
        if self.prev_error != float('inf'):
            if error < self.prev_error:
                improvement_reward = 0.5 * (1 - error / self.prev_error)
            else:
                improvement_reward = -0.2 
        
        if error < 50:
            proximity_reward = 2.0 * (1 - error / 50)
        elif error < 100:
            proximity_reward = 1.0 * (1 - error / 100)
        else:
            proximity_reward = 0.0
        
        record_reward = 0.0
        if error < self.best_error_in_episode:
            record_reward = 0.3
            self.best_error_in_episode = error
        
        total_reward = accuracy_reward + improvement_reward + proximity_reward + record_reward
        return total_reward

    def reset(self,*, seed=None, options=None):
        super().reset(seed=seed)
        self.last_angle = np.random.uniform(self.min_angle, self.max_angle)  
        self.last_speed = np.random.uniform(self.min_speed, self.max_speed)  
        
        simulation_result = self.simulation.run(
            angle=self.last_angle,
            speed=self.last_speed,
            apply_air_resistance=self.merged_simulation_config["apply_air_resistance"],
            step=self.merged_simulation_config["step"],
            record_trajectory=self.merged_simulation_config["record_trajectory"],
            log_last_position=self.merged_simulation_config["log_last_position"]
        )
        
        self.last_landing_x = simulation_result["landing_x"]
        self.last_error = abs(self.last_landing_x - self.target_x)
        self.prev_error = float('inf')
        
        self.current_step = 0
        self.best_error_in_episode = self.last_error
        
        self.state = self._get_state()
        return self.state, {}

    def step(self, action, max_steps=10):
        angle = float(np.clip(float(action[0]), self.min_angle, self.max_angle))
        speed = float(np.clip(float(action[1]), self.min_speed, self.max_speed))
        
        self.prev_error = self.last_error
        
        self.last_angle = angle
        self.last_speed = speed
        
        simulation_result = self.simulation.run(
            angle=angle,
            speed=speed,
            apply_air_resistance=self.merged_simulation_config["apply_air_resistance"],
            step=self.merged_simulation_config["step"],
            record_trajectory=self.merged_simulation_config["record_trajectory"],
            log_last_position=self.merged_simulation_config["log_last_position"]
        )
        
        self.last_landing_x = simulation_result["landing_x"]
        self.last_error = abs(self.last_landing_x - self.target_x)
        
        reward = self._calculate_reward(self.last_error)
        
        self.current_step += 1
        self.state = self._get_state()
        
        done = (self.current_step >= max_steps) or (self.last_error < 20)  
        terminated = done
        truncated = False
        
        info = {
            'error': self.last_error,
            'landing_x': self.last_landing_x,
            'angle': angle,
            'speed': speed,
            'best_error': self.best_error_in_episode,
            'trajectory': simulation_result["trajectory"] if self.merged_simulation_config["record_trajectory"] else None
        }
        
        return self.state, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}, Error: {self.last_error:.2f}, "
                  f"Angle: {self.last_angle:.2f}, Speed: {self.last_speed:.2f}")
        
    def get_episode_stats(self):
        return {
            'best_error': self.best_error_in_episode,
            'final_error': self.last_error,
            'steps_taken': self.current_step
        }