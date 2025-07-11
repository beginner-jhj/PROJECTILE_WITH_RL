import numpy as np

from simulation.simul import Simulation

record = []

for angle in np.arange(30, 61, 0.1):


    simulation_result = Simulation(400, 400).run(
        angle=angle, speed=550)
    
    record.append({
        "angle": angle,
        "landing_x": simulation_result['landing_x']
    })

record.sort(key=lambda x: x['landing_x'], reverse=True)




for result in record:
    print(result)
    
    
    



