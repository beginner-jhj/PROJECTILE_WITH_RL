import pymunk
import math

class AdvancedSimulationEngine:
    def __init__(self, width=2000, ground_y=400, gravity=(0, 900), wind=(0.0, 0.0),
                 temperature=293.15, pressure=101325.0, drag_coefficient=0.47,
                 area=0.01):
        """Create a 2D simulation world using pymunk."""
        self.space = pymunk.Space()
        self.space.gravity = gravity
        self.width = width
        self.ground_y = ground_y
        self.wind = pymunk.Vec2d(*wind)
        self.temperature = temperature
        self.pressure = pressure
        self.drag_coefficient = drag_coefficient
        self.area = area
        self.add_ground()

    def add_ground(self, friction=5.0):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (self.width / 2, self.ground_y)
        shape = pymunk.Segment(body, (-self.width // 2, 0), (self.width // 2, 0), 5)
        shape.friction = friction
        self.space.add(body, shape)

    def create_projectile(self, angle_deg, speed, mass=1.0, radius=5):
        angle_rad = math.radians(angle_deg)
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)
        body.position = (100, self.ground_y)
        velocity = speed * pymunk.Vec2d(math.cos(angle_rad), -math.sin(angle_rad))
        body.velocity = velocity
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.5
        self.space.add(body, shape)
        return body

    # Ideal gas constant for dry air (J/(kg*K))
    R_AIR = 287.05

    def _air_density(self):
        return self.pressure / (self.R_AIR * self.temperature)

    def _apply_air_drag(self, body):
        rel_v = body.velocity - self.wind
        if rel_v.length > 0:
            drag_mag = 0.5 * self._air_density() * self.drag_coefficient * self.area * (rel_v.length ** 2)
            drag_force = -drag_mag * rel_v.normalized()
            body.apply_force_at_local_point(drag_force)

    def step(self, dt):
        for body in self.space.bodies:
            if not body.body_type == pymunk.Body.STATIC:
                self._apply_air_drag(body)
        self.space.step(dt)

    def simulate(self, angle, speed, dt=1/60.0, max_time=10.0, record=False):
        projectile = self.create_projectile(angle, speed)
        t = 0.0
        trajectory = []
        direction_changed = False
        while t < max_time:
            self.step(dt)
            if record:
                trajectory.append(projectile.position)
            if projectile.velocity.y > 0 and not direction_changed:
                direction_changed = True
            if abs(projectile.position.y - self.ground_y) < 10 and direction_changed:
                break
            t += dt
        return {
            "landing_x": projectile.position.x,
            "trajectory": trajectory if record else None
        }
