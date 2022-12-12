import taichi as ti
import math
from src.simulator.simulation import Simulation
# from src.renderer.renderer import Renderer
from src.renderer.pyvista_renderer import Renderer

RESULT_DIR = "results/run_flask_7/" # directory has to exist, otherwise crash
BOUNDS = 4. 
REST_DENSITY = 300
RADIUS = 0.025
SUPPORT_RADIUS = 4*RADIUS
MAX_DT = 1e-3

# Mass from density
MASS = ((4./3.)*math.pi*(RADIUS**3)) *REST_DENSITY
NUM_PARTICLES = 7**3
MAX_TIME = 2.

MU = 1000
# B_MU = 100

GAMMA = 1e-1 # If too high, might expode do to euler explicit integration. Max tested working value: 1e-1

# Those are upwards velocities
INITIAL_FLUID_VELOCITY = 10.
EMISSION_VELOCITY = 10.
PARTICLES_PER_SECOND = 2500
EMITTER_POS = [2., 0.2, 2.]
EMITTER_RADIUS = 0.07

MU = 2500
B_MU_FLASK = 0
B_MU_GROUND = 25000

# Run Simulation
ti.init(arch=ti.cpu, debug=False, cpu_max_num_threads=12)
# simulation = Simulation(NUM_PARTICLES, MAX_TIME, max_dt=MAX_DT, mass=MASS, rest_density=REST_DENSITY, support_radius=SUPPORT_RADIUS, mu=MU, b_mu=[B_MU_FLASK, B_MU_GROUND],  gamma=GAMMA, bounds=BOUNDS, is_frame_export=True, debug=True, result_dir=RESULT_DIR,\
#     pointData_file=r"src/pointDataFiles/erlenmayer_quickerstart.npy", boundary_pointData_file = r"src/pointDataMetaFiles/flask_on_plane.txt", is_uniform_export=False,\
#     initial_fluid_velocity = INITIAL_FLUID_VELOCITY, emission_velocity = EMISSION_VELOCITY,\
#     particles_per_second = PARTICLES_PER_SECOND, emitter_pos = EMITTER_POS, emitter_radius = EMITTER_RADIUS)
# simulation.run()

STARTFRAME = 0
FRAMESTEP = 1
RESOLUTION = [2560,1440]

# Render Simulation
renderer = Renderer(result_dir=RESULT_DIR, radius=RADIUS*0.99, SHOW=True, render_boundary=True, render_density=False, render_temperature=True, mass=MASS, start_frame=STARTFRAME, framestep=FRAMESTEP, render_uniform=False, resolution =  RESOLUTION)
# renderer = Renderer(bounds=BOUNDS, result_dir=RESULT_DIR+r"\data", radius=RADIUS*0.99, SHOW=True, render_boundary=False, render_density=False, mass=MASS, start_frame=80)
renderer.render()
