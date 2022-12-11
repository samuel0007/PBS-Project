import taichi as ti
import math
from src.simulator.simulation import Simulation
# from src.renderer.renderer import Renderer
from src.renderer.pyvista_renderer import Renderer

RESULT_DIR = "results/run_temp_1/" # directory has to exist, otherwise crash
BOUNDS = 2. 
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

# Run Simulation
# ti.init(arch=ti.cpu, debug=False, cpu_max_num_threads=8)
# simulation = Simulation(NUM_PARTICLES, MAX_TIME, max_dt=MAX_DT, mass=MASS, rest_density=REST_DENSITY, support_radius=SUPPORT_RADIUS, mu=MU, b_mu=B_MU, bounds=BOUNDS, is_frame_export=True, debug=True, result_dir=RESULT_DIR, pointData_file=r"src/pointDataMetaFiles/test.txt", boundary_pointData_file = r"src/pointDataFiles/BigDragonSurface.vtk")
# simulation = Simulation(NUM_PARTICLES, MAX_TIME, max_dt=MAX_DT, mass=MASS, rest_density=REST_DENSITY, support_radius=SUPPORT_RADIUS, mu=MU, b_mu=B_MU, gamma=GAMMA, bounds=BOUNDS, is_frame_export=True, debug=True, result_dir=RESULT_DIR, pointData_file=r"", boundary_pointData_file = r"", is_uniform_export=False)
# simulation.run()

STARTFRAME = 0
FRAMESTEP = 1

# Render Simulation
renderer = Renderer(result_dir=RESULT_DIR, radius=RADIUS*0.99, SHOW=True, render_boundary=True, render_density=False, render_temperature=True, mass=MASS, start_frame=STARTFRAME, framestep=FRAMESTEP, render_uniform=False)
renderer.render()
