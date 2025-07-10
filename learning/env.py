from gymnasium import spaces, Env
from simulation.simul import Simulation
import numpy as np

class ProjectileEnv(Env):
    def __init__(self):
        super().__init__()
        self.min_angle = 10.0
        self.max_angle = 80.0
        self.min_speed = 480.0
        self.max_speed = 520.0
        self.target_x = 1400.0
        
        # 이전 상태 추적을 위한 변수들
        self.last_angle = 45.0
        self.last_speed = 500.0
        self.last_landing_x = 0.0
        self.last_error = float('inf')
        self.prev_error = float('inf')
        
        # 에피소드 관리
        self.max_steps = 10  # 더 많은 step으로 학습 기회 증가
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
        
        # 성능 추적
        self.episode_rewards = []
        self.episode_errors = []

    def _get_state(self):
        """더 풍부한 상태 정보 제공"""
        # 각도 정규화 (0~1)
        angle_norm = (self.last_angle - self.min_angle) / (self.max_angle - self.min_angle)
        
        # 속도 정규화 (0~1)
        speed_norm = (self.last_speed - self.min_speed) / (self.max_speed - self.min_speed)
        
        # 착지 위치 비율 (target 대비)
        landing_ratio = max(0, min(2.0, self.last_landing_x / self.target_x))
        
        # 에러 정규화 (0~1, 0이 최고)
        max_possible_error = self.target_x  # 최대 가능 에러
        error_norm = min(1.0, abs(self.last_error) / max_possible_error)
        
        # 개선 여부 (-1~1, 양수면 개선)
        if self.prev_error == float('inf'):
            improvement = 0.0
        else:
            improvement = np.tanh((self.prev_error - self.last_error) / 100.0)
        
        return np.array([angle_norm, speed_norm, landing_ratio, error_norm, improvement], dtype=np.float32)

    def _calculate_reward(self, error):
        """개선된 보상 함수"""
        # 1. 기본 정확도 보상 (선형적으로 감소)
        accuracy_reward = max(0, 1 - (error / self.target_x))
        
        # 2. 개선 보상 (이전 step 대비 개선되면 추가 보상)
        improvement_reward = 0.0
        if self.prev_error != float('inf'):
            if error < self.prev_error:
                improvement_reward = 0.5 * (1 - error / self.prev_error)
            else:
                improvement_reward = -0.2  # 악화 시 패널티
        
        # 3. 목표 근접 보상 (매우 가까우면 큰 보상)
        if error < 50:
            proximity_reward = 2.0 * (1 - error / 50)
        elif error < 100:
            proximity_reward = 1.0 * (1 - error / 100)
        else:
            proximity_reward = 0.0
        
        # 4. 에피소드 최고 기록 보상
        record_reward = 0.0
        if error < self.best_error_in_episode:
            record_reward = 0.3
            self.best_error_in_episode = error
        
        total_reward = accuracy_reward + improvement_reward + proximity_reward + record_reward
        return total_reward

    def reset(self,*, seed=None, options=None):
        super().reset(seed=seed)
        """환경 초기화"""
        # 랜덤 초기화 대신 좀 더 합리적인 범위에서 시작
        self.last_angle = np.random.uniform(30, 60)  # 더 좁은 범위
        self.last_speed = np.random.uniform(490, 510)  # 더 좁은 범위
        
        # 초기 시뮬레이션 실행
        simulation_result = self.simulation.run(
            angle=self.last_angle,
            speed=self.last_speed,
            apply_air_resistance=False,
            step=1/60,
            record_trajectory=False
        )
        
        self.last_landing_x = simulation_result["landing_x"]
        self.last_error = abs(self.last_landing_x - self.target_x)
        self.prev_error = float('inf')
        
        self.current_step = 0
        self.best_error_in_episode = self.last_error
        
        self.state = self._get_state()
        return self.state, {}

    def step(self, action):
        """한 스텝 실행"""
        # 액션 클리핑
        angle = float(np.clip(float(action[0]), self.min_angle, self.max_angle))
        speed = float(np.clip(float(action[1]), self.min_speed, self.max_speed))
        
        # 이전 에러 저장
        self.prev_error = self.last_error
        
        # 액션 적용
        self.last_angle = angle
        self.last_speed = speed
        
        # 시뮬레이션 실행
        simulation_result = self.simulation.run(
            angle=angle,
            speed=speed,
            apply_air_resistance=False,
            step=1/60,
            record_trajectory=False
        )
        
        self.last_landing_x = simulation_result["landing_x"]
        self.last_error = abs(self.last_landing_x - self.target_x)
        
        # 보상 계산
        reward = self._calculate_reward(self.last_error)
        
        # 상태 업데이트
        self.current_step += 1
        self.state = self._get_state()
        
        # 종료 조건
        done = (self.current_step >= self.max_steps) or (self.last_error < 20)  # 매우 정확하면 조기 종료
        terminated = done
        truncated = False
        
        # 정보 수집
        info = {
            'error': self.last_error,
            'landing_x': self.last_landing_x,
            'angle': angle,
            'speed': speed,
            'best_error': self.best_error_in_episode
        }
        
        return self.state, reward, terminated, truncated, info

    def render(self, mode='human'):
        """렌더링 (현재는 패스)"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Error: {self.last_error:.2f}, "
                  f"Angle: {self.last_angle:.2f}, Speed: {self.last_speed:.2f}")
        
    def get_episode_stats(self):
        """에피소드 통계 반환"""
        return {
            'best_error': self.best_error_in_episode,
            'final_error': self.last_error,
            'steps_taken': self.current_step
        }