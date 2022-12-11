import taichi as ti
import math
from src.simulator.simulation import Simulation
from src.renderer.renderer import Renderer
# from src.renderer.pyvista_renderer import Renderer

RESULT_DIR = "results/run_flask_3/" # directory has to exist, otherwise crash
BOUNDS = 4. # If particles are falling down, change this to int.
REST_DENSITY = 300
RADIUS = 0.025
SUPPORT_RADIUS = 4*RADIUS
MAX_DT = 1e-3

# Mass from density
MASS = ((4./3.)*math.pi*(RADIUS**3)) *REST_DENSITY
NUM_PARTICLES = 10**3
MAX_TIME = 5.

# Those are upwards velocities
INITIAL_FLUID_VELOCITY = 0.
EMISSION_VELOCITY = 0.

PARTICLES_PER_SECOND = 1500

MU = 1500
B_MU = 500

# Run Simulation
ti.init(arch=ti.cpu, debug=False, cpu_max_num_threads=8)
# simulation = Simulation(NUM_PARTICLES, MAX_TIME, max_dt=MAX_DT, mass=MASS, rest_density=REST_DENSITY, support_radius=SUPPORT_RADIUS, mu=MU, b_mu=B_MU, bounds=BOUNDS, is_frame_export=True, debug=True, result_dir=RESULT_DIR,\
#     pointData_file=r"src/pointDataFiles/erlenmayer_quickerstart.npy", boundary_pointData_file = r"src/pointDataMetaFiles/flask_on_plane.txt", is_uniform_export=True,\
#     initial_fluid_velocity = INITIAL_FLUID_VELOCITY, emission_velocity = EMISSION_VELOCITY,\
#     particles_per_second = PARTICLES_PER_SECOND)
# simulation.run()

# Render Simulation
renderer = Renderer(bounds=BOUNDS, result_dir=RESULT_DIR, radius=RADIUS*0.99, SHOW=True, render_boundary=True, render_density=False, mass=MASS, start_frame=1100)
renderer.render()
