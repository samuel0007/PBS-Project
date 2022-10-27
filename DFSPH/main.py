import taichi as ti
import math
from src.simulator.simulation import Simulation
from src.renderer.renderer import Renderer

RESULT_DIR = "results/run_6/"
BOUNDS = 1
REST_DENSITY = 1000
RADIUS = 0.025
SUPPORT_RADIUS = 4*RADIUS

# Mass from density
MASS = (4/3*math.pi*RADIUS**3) *REST_DENSITY 
NUM_PARTICLES = 1000
MAX_TIME = 1

# Run Simulation
ti.init(arch=ti.gpu, debug=False)
simulation = Simulation(NUM_PARTICLES, MAX_TIME, mass=MASS, support_radius=SUPPORT_RADIUS, bounds=BOUNDS, is_frame_export=True, debug=True, result_dir=RESULT_DIR)
simulation.run()

# Render Simulation
renderer = Renderer(bounds=BOUNDS, result_dir=RESULT_DIR, radius=RADIUS*0.99, SHOW=True, render_boundary=False, render_density=False, mass=MASS, start_frame=0)
renderer.render()

