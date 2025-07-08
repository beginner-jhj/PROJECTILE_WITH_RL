from simulation.engine import SimulationEngine

class Simulation:
    def __init__(self, width, height):
        self.engine = SimulationEngine(width, height - 50)

    def run(self, angle=45, speed=400, apply_air_resistance=True, step=1/120, record_trajectory=False):
        body = self.engine.create_projectile(angle_deg=angle, speed=speed)
        trajectory = [] if record_trajectory else None

        while body.position.y >= self.engine.ground_y:
            if apply_air_resistance:
                self.engine.apply_air_resistance(body)
            self.engine.space.step(step)
            if record_trajectory:
                trajectory.append(body.position)

        return {
            "landing_x": body.position.x,
            "trajectory": trajectory if record_trajectory else None
        }
