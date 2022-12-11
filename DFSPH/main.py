import taichi as ti
import math
from src.simulator.simulation import Simulation
from src.renderer.renderer import Renderer
# from src.renderer.pyvista_renderer import Renderer

RESULT_DIR = "results/run_flask_1/" # directory has to exist, otherwise crash
BOUNDS = 4. 
REST_DENSITY = 300
RADIUS = 0.025
SUPPORT_RADIUS = 4*RADIUS
MAX_DT = 1e-3

# Mass from density
MASS = ((4./3.)*math.pi*(RADIUS**3)) *REST_DENSITY
NUM_PARTICLES = 7**3
MAX_TIME = 5.

MU = 1000
B_MU = 100

GAMMA = 1e-1 # If too high, might expode do to euler explicit integration. Max tested working value: 1e-1

# Those are upwards velocities
INITIAL_FLUID_VELOCITY = 50.
EMISSION_VELOCITY = 50.
PARTICLES_PER_SECOND = 2000

MU = 500
B_MU = 100

# Run Simulation
ti.init(arch=ti.cpu, debug=False, cpu_max_num_threads=8)
simulation = Simulation(NUM_PARTICLES, MAX_TIME, max_dt=MAX_DT, mass=MASS, rest_density=REST_DENSITY, support_radius=SUPPORT_RADIUS, mu=MU, b_mu=B_MU,  gamma=GAMMA, bounds=BOUNDS, is_frame_export=True, debug=True, result_dir=RESULT_DIR,\
    pointData_file=r"src/pointDataFiles/erlenmayer_quickerstart.npy", boundary_pointData_file = r"src/pointDataMetaFiles/flask_on_plane.txt", is_uniform_export=False,\
    initial_fluid_velocity = INITIAL_FLUID_VELOCITY, emission_velocity = EMISSION_VELOCITY,\
    particles_per_second = PARTICLES_PER_SECOND)
simulation.run()

STARTFRAME = 0
FRAMESTEP = 1

# Render Simulation
renderer = Renderer(result_dir=RESULT_DIR, radius=RADIUS*0.99, SHOW=True, render_boundary=True, render_density=False, render_temperature=True, mass=MASS, start_frame=STARTFRAME, framestep=FRAMESTEP, render_uniform=False)
# renderer = Renderer(bounds=BOUNDS, result_dir=RESULT_DIR, radius=RADIUS*0.99, SHOW=True, render_boundary=True, render_density=False, mass=MASS, start_frame=120)
renderer.render()
