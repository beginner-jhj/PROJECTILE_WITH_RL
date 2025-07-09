from simulation.engine import SimulationEngine

class Simulation:
    def __init__(self, width, height):
        self.width = width
        self.height = height - 50

    def run(self, angle=45, speed=400, apply_air_resistance=True, step=1/120, record_trajectory=False, log_last_position=False):
        engine = SimulationEngine(self.width, self.height)
        body = engine.create_projectile(angle_deg=angle, speed=speed, friction=5)

        trajectory = [] if record_trajectory else None
        direction_changed = False

        while True:
            if apply_air_resistance:
                engine.apply_air_resistance(body)
            engine.space.step(step)
            if record_trajectory:
                trajectory.append(body.position)

            if body.velocity.y > 0 and not direction_changed:
                direction_changed = True

            if abs(body.position.y - engine.ground_y) < 10 and direction_changed:
                if log_last_position:
                    print(f"Final position: ({body.position.x}, {body.position.y})")
                break

        return {
            "landing_x": body.position.x,
            "trajectory": trajectory if record_trajectory else None
        }
