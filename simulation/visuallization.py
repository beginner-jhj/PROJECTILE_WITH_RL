from simulation.simul import Simulation
from simulation.render import Renderer

simulation_result = Simulation(1500, 500).run(
    angle=50,record_trajectory=True, log_last_position=True)

renderer = Renderer(1100, 500)
renderer.render(simulation_result["trajectory"], log=False)