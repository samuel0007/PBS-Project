import taichi as ti
from simulation import Simulation
ti.init(arch=ti.gpu)

simulation = Simulation(1000, 100., bounds=1., frame_export=False, debug=True)

simulation.run()
simulation.save("results.npy")