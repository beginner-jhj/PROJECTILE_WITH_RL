from simulation.advanced_engine import AdvancedSimulationEngine


def simulate(engine: AdvancedSimulationEngine, angle: float, speed: float,
             dt: float = 1/60.0, max_time: float = 10.0,
             record_trajectory: bool = False):
    """Run a projectile simulation using ``engine``.

    Parameters
    ----------
    engine : AdvancedSimulationEngine
        Configured simulation engine instance.
    angle : float
        Launch angle in degrees.
    speed : float
        Launch speed.
    dt : float, optional
        Time step for the simulation.
    max_time : float, optional
        Maximum simulation time.
    record_trajectory : bool, optional
        If True, return a list of projectile positions over time.
    """
    projectile = engine.create_projectile(angle, speed)
    t = 0.0
    trajectory = []
    direction_changed = False

    while t < max_time:
        engine.step(dt)
        if record_trajectory:
            trajectory.append(projectile.position)
        if projectile.velocity.y > 0 and not direction_changed:
            direction_changed = True
        if abs(projectile.position.y - engine.ground_y) < 10 and direction_changed:
            break
        t += dt

    return {
        "landing_x": projectile.position.x,
        "trajectory": trajectory if record_trajectory else None,
    }
