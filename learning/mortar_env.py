import numpy as np  # 수치 계산을 위한 numpy 사용
from gymnasium import Env, spaces  # Gymnasium의 Env, spaces 불러오기
from simulation.advanced_engine import AdvancedSimulationEngine  # 공기 저항 등을 포함한 시뮬레이션 엔진
from simulation.advanced_simul import simulate

class MortarEnv(Env):  # 강화학습을 위한 맞춤형 환경 클래스
    """Gymnasium environment for learning optimal mortar angles."""  # 간단한 설명

    def __init__(self, speed=500.0):  # 초기화 시 포탄 속도를 인자로 받음
        super().__init__()  # Gymnasium 기본 초기화
        self.min_angle = 10.0  # 발사각 최소값(도)
        self.max_angle = 80.0  # 발사각 최대값(도)
        self.speed = speed  # 초기 속도(m/s)

        # State ranges
        self.wind_range = (-10.0, 10.0)  # 풍속 범위(m/s)
        self.temp_range = (260.0, 320.0)  # 기온 범위(K)
        self.pressure_range = (90000.0, 110000.0)  # 기압 범위(Pa)
        self.distance_range = (500.0, 1500.0)  # 목표 거리(px)
        self.alt_range = (-50.0, 50.0)  # 고도차(px)

        low = np.array([  # 상태 하한값
            self.wind_range[0], self.wind_range[0],  # 풍속 x,y 최소값
            self.temp_range[0], self.pressure_range[0],  # 기온, 기압 최소값
            self.distance_range[0], self.alt_range[0]  # 거리 및 고도차 최소값
        ], dtype=np.float32)
        high = np.array([  # 상태 상한값
            self.wind_range[1], self.wind_range[1],  # 풍속 x,y 최대값
            self.temp_range[1], self.pressure_range[1],  # 기온, 기압 최대값
            self.distance_range[1], self.alt_range[1]  # 거리 및 고도차 최대값
        ], dtype=np.float32)
        # 관측 공간 정의
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # 행동 공간은 단일 발사각
        self.action_space = spaces.Box(
            low=np.array([self.min_angle], dtype=np.float32),  # 최소 발사각
            high=np.array([self.max_angle], dtype=np.float32),  # 최대 발사각
            dtype=np.float32,
        )

        self.state = None  # 현재 환경 상태 저장
        self.engine = None  # 시뮬레이션 엔진 인스턴스
        self.target_x = None  # 목표 지점 x 좌표

    def _sample_scenario(self):  # 무작위 환경 파라미터 생성
        wind = (np.random.uniform(*self.wind_range), np.random.uniform(*self.wind_range))  # 풍속 (x, y)
        temp = np.random.uniform(*self.temp_range)  # 기온
        pressure = np.random.uniform(*self.pressure_range)  # 기압
        dist = np.random.uniform(*self.distance_range)  # 목표 거리
        alt = np.random.uniform(*self.alt_range)  # 고도차
        return wind, temp, pressure, dist, alt

    def reset(self, *, seed=None, options=None):  # 에피소드 초기화
        super().reset(seed=seed)  # Gymnasium 초기화
        wind, temp, pressure, dist, alt = self._sample_scenario()  # 무작위 시나리오
        self.engine = AdvancedSimulationEngine(  # 물리 엔진 생성
            wind=wind, temperature=temp, pressure=pressure, ground_y=400-alt  # 고도차에 따른 지면 위치 조정
        )
        self.target_x = dist  # 목표 거리 저장
        self.state = np.array([wind[0], wind[1], temp, pressure, dist, alt], dtype=np.float32)  # 상태 저장
        return self.state, {}  # 관측과 추가정보 반환

    def step(self, action):  # 에이전트의 행동 실행
        angle = float(np.clip(action[0], self.min_angle, self.max_angle))  # 범위 내 발사각 확보
        result = simulate(self.engine, angle, self.speed)  # 시뮬레이션 실행
        landing_x = result["landing_x"]  # 착지 지점 x 좌표
        error = abs(landing_x - self.target_x)  # 목표와의 오차 계산
        reward = -error  # 오차가 작을수록 보상 증가
        terminated = True  # 한 발로 에피소드 종료
        truncated = False  # 타임아웃 없음
        info = {"landing_x": landing_x, "error": error, "target_x": self.target_x}  # 기록용 정보
        return self.state, reward, terminated, truncated, info  # 다음 상태(없음), 보상 등 반환

    def render(self, mode="human"):  # 시각화 필요 시 구현
        pass
