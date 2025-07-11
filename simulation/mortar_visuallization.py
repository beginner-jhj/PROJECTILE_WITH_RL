from simulation.advanced_engine import AdvancedSimulationEngine
from simulation.advanced_simul import simulate
from simulation.render import Renderer

engine = AdvancedSimulationEngine()
simulation_result = simulate(engine, angle=45, speed=500, record_trajectory=True)

print(simulation_result)

renderer = Renderer(400, 400)
renderer.render(simulation_result["trajectory"], log=False)
