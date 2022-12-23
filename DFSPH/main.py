import taichi as ti
import math
from src.simulator.simulation import Simulation
# from src.renderer.renderer import Renderer
from src.renderer.pyvista_renderer import Renderer
import os

RESULT_DIR = "results/run_1/"
# create result directory if it doesn't exist
os.makedirs(RESULT_DIR, exist_ok=True)

BOUNDS = 4. 
UNIFORM_EXPORT = False
SURFACE_RENDER = False
REST_DENSITY = 300
RADIUS = 0.025
SUPPORT_RADIUS = 4*RADIUS
MAX_DT = 1e-3

# Mass from density
MASS = ((4./3.)*math.pi*(RADIUS**3)) *REST_DENSITY
NUM_PARTICLES = 7**3
MAX_TIME = 4.

GAMMA = 1e-1 # If too high, might expode do to euler explicit integration. Max tested working value: 1e-1

# This one may not be a good idea because it takes thousands of particles to fill
# INITIAL_FLUID = r"src\pointDataMetaFiles\empty.txt"
# BOUNDARY = r"src\pointDataMetaFiles\big_tube_on_plane.txt"

# This one is our favorite
INITIAL_FLUID = r"src\pointDataFiles\erlenmayer_half_full.npy"
# INITIAL_FLUID = r"src\pointDataFiles\erlenmayer_quickstart.npy"
# INITIAL_FLUID = r"src\pointDataFiles\erlenmayer_quickerstart.npy"
BOUNDARY = r"src\pointDataMetaFiles\flask_on_plane.txt"

# This one can handle higher viscosity and emission rates
# INITIAL_FLUID = r"src\pointDataMetaFiles\empty.txt"
# BOUNDARY = r"src\pointDataMetaFiles\small_tube_on_plane.txt"

GRAVITY = -5
# Those are upwards velocities
INITIAL_FLUID_VELOCITY = 10.
EMISSION_VELOCITY = 10.
PARTICLES_PER_SECOND = 3000
PPS_SLOWDOWN = 1500
EMITTER_POS = [2., 0.2, 2.]
EMITTER_RADIUS = 0.07


# In our metafiles, the first is the flask and the second is the ground. 
# It is recommended to choose a lower viscosity for the flask than the ground.
B_MU = [2500, 25000]

T_ROOM = 25
ROOM_RADIATION_HALF_TIME = 2 # If too low, may explode. Min tested working value: 0.01
EMISSION_T = 300
INIT_T = 250
# t_to_mu: Temperature -> Viscosity
@ti.func
def T_TO_MU(t: ti.f32) -> ti.f32:
    return t*10

# Doesn't do anything if the viscosity is updated according to temperature    
MU = 2500

# Run Simulation
ti.init(arch=ti.cpu, debug=False, cpu_max_num_threads=6)
simulation = Simulation(NUM_PARTICLES, MAX_TIME, max_dt=MAX_DT, mass=MASS, rest_density=REST_DENSITY, support_radius=SUPPORT_RADIUS, mu=MU, b_mu=B_MU,  gamma=GAMMA, bounds=BOUNDS, is_frame_export=True, debug=True, result_dir=RESULT_DIR,
    pointData_file=INITIAL_FLUID, boundary_pointData_file=BOUNDARY, is_uniform_export=UNIFORM_EXPORT, gravity=GRAVITY,
    initial_fluid_velocity=INITIAL_FLUID_VELOCITY, emission_velocity=EMISSION_VELOCITY,
    particles_per_second=PARTICLES_PER_SECOND, pps_slowdown=PPS_SLOWDOWN, t_room=T_ROOM, room_radiation_half_time=ROOM_RADIATION_HALF_TIME,
    emitter_pos=EMITTER_POS, emitter_radius=EMITTER_RADIUS, t_to_mu=T_TO_MU, emission_t=EMISSION_T, init_t=INIT_T)
simulation.run()


STARTFRAME = 0
FRAMESTEP = 1
RESOLUTION = [1280,720]

# Render Simulation
renderer = Renderer(result_dir=RESULT_DIR, radius=RADIUS*0.99, SHOW=True, render_boundary=True, render_density=False, render_temperature=True, mass=MASS, start_frame=STARTFRAME, framestep=FRAMESTEP, render_uniform=SURFACE_RENDER, resolution=RESOLUTION)
# renderer = Renderer(bounds=BOUNDS, result_dir=RESULT_DIR+r"\data", radius=RADIUS*0.99, SHOW=True, render_boundary=False, render_density=False, mass=MASS, start_frame=80)
renderer.render()
