from simulation.simul import Simulation
from simulation.render import Renderer

simulation_result = Simulation(400, 400).run(
    angle=45, record_trajectory=True, log_last_position=True)

renderer = Renderer(400, 400)
renderer.render(simulation_result["trajectory"], log=False)