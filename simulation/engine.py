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
        body.position = (self.width / 2, self.ground_y)  
        shape = pymunk.Segment(body, (-self.width // 2, 0), (self.width // 2, 0), 5) 
        shape.friction =  friction 
        self.space.add(body, shape)  

    def create_projectile(self, angle_deg, speed, mass=1, radius=5, friction=0.5, elasticity=0.0):
        angle_rad = math.radians(angle_deg) 
        inertia = pymunk.moment_for_circle(mass, 0, radius)  
        body = pymunk.Body(mass, inertia)  
        body.position = (0, self.ground_y)  
        velocity = speed * pymunk.Vec2d(math.cos(angle_rad), -math.sin(angle_rad))  
        body.velocity = velocity  
        shape = pymunk.Circle(body, radius)  
        shape.friction = friction  
        shape.elasticity = elasticity 
        self.space.add(body, shape)  
        return body  
    
    def apply_air_resistance(self, body, k=0.002):
        v = body.velocity
        if v.length > 0:
            drag = -k * v.length**2 * v.normalized()  
            body.apply_force_at_local_point(drag)  