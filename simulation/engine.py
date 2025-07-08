import pymunk
import math

class SimulationEngine:
    def __init__(self, width, ground_y, gravity=(0, 900)):
        self.space = pymunk.Space()
        self.space.gravity = gravity
        self.width = width
        self.ground_y = ground_y
        self.add_ground()

    def add_ground(self, friction=5.0):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (self.width / 2, self.ground_y)  # 바닥 위치 설정
        shape = pymunk.Segment(body, (-self.width // 2, 0), (self.width // 2, 0), 5)  # 선분 형태 바닥
        shape.friction =  friction  # 마찰력 (구르는 걸 방지)
        self.space.add(body, shape)  # space에 바닥 추가

    def create_projectile(self, angle_deg, speed, mass=1, radius=5, friction=0.5, elasticity=0.3):
        angle_rad = math.radians(angle_deg)  # 각도를 라디안으로 변환
        inertia = pymunk.moment_for_circle(mass, 0, radius)  # 회전 관성 계산
        body = pymunk.Body(mass, inertia)  # 바디 생성
        body.position = (100, self.ground_y)  # 시작 위치 설정
        velocity = speed * pymunk.Vec2d(math.cos(angle_rad), -math.sin(angle_rad))  # 초기 속도 벡터
        body.velocity = velocity  # 속도 적용
        shape = pymunk.Circle(body, radius)  # 원형 모양의 shape 설정
        shape.friction = friction  # 마찰력
        shape.elasticity = elasticity  # 탄성 (튕기는 정도)
        shape.color = (255, 0, 0)  # 색상 설정 (사용되지 않지만 나중에 커스터마이징 가능)
        self.space.add(body, shape)  # space에 투사체 추가
        return body  # 바디 반환
    
    def apply_air_resistance(self, body, k=0.002):
        v = body.velocity
        if v.length > 0:  # 속도가 0이 아닐 때만 적용
            drag = -k * v.length**2 * v.normalized()  # 공기 저항력 = -k * v^2 * 방향
            body.apply_force_at_local_point(drag)  # 힘 적용