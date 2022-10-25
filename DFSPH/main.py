import taichi as ti
import math
from src.simulator.simulation import Simulation
from src.renderer.renderer import Renderer

RESULT_DIR = "results/example/"
BOUNDS = 1.
MASS = 0.01
REST_DENSITY = 1000
SUPPORT_RADIUS = 0.065
NUM_PARTICLES = 1000
MAX_TIME = 1.

RADIUS = pow(MASS/(REST_DENSITY*4./3.*math.pi), 1./3.)


# Run Simulation
ti.init(arch=ti.gpu)
simulation = Simulation(NUM_PARTICLES, MAX_TIME, mass=MASS, bounds=BOUNDS, is_frame_export=True, debug=True, result_dir=RESULT_DIR)
simulation.run()

# Render Simulation
renderer = Renderer(bounds=BOUNDS, result_dir=RESULT_DIR, radius=RADIUS)
renderer.show()

