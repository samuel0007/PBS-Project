# DFSPH + Weiler2018: SPH Based Elephant Toothpaste Simulation

## Installation

From an environment with Python 3.8.5 installed:

```bash
pip install -r requirements.txt
```

## Usage

Simply run the main script:

```bash
python main.py
```

You can comment out the simulation.run() and run a second process to do rendering concurrently by rerunning the same script. This will render the frames in the `frames` folder.

## Configuration

The following parameters can be changed in `main.py`:

```python
# Simulation parameters
BOUNDS = 4. # Size of the simulation box
UNIFORM_EXPORT = False # If true, will export the density field on an uniform grid inside the bounds, necessary if we want to do surface rendering
SURFACE_RENDER = False # If true, will render the isosurface of the density field.
REST_DENSITY = 300
RADIUS = 0.025 # Particle radius
SUPPORT_RADIUS = 4*RADIUS # Kernel support radius
MAX_DT = 1e-3 # Max time step, clips the CFL condition

MASS = ((4./3.)*math.pi*(RADIUS**3)) *REST_DENSITY # Particle mass
NUM_PARTICLES = 7**3 # Initial number of particles if no initial_fluid file is provided
MAX_TIME = 4. # Max simulation time

GAMMA = 1e-1 # Temperature diffusion coefficient. If too high, might expode do to euler explicit integration. Max tested working value: 1e-1

INITIAL_FLUID = r"src\pointDataFiles\erlenmayer_half_full.npy" # Initial condition file. See under in initial condition section for more detail.
BOUNDARY = r"src\pointDataMetaFiles\flask_on_plane.txt" # Boundary file. See under in initial condition section for more detail.


GRAVITY = -5
# Those are upwards velocities
INITIAL_FLUID_VELOCITY = 10. # Initial velocity of the fluid
EMISSION_VELOCITY = 10. # Velocity of the emitted particles
PARTICLES_PER_SECOND = 3000
PPS_SLOWDOWN = 1500 # Slowdown  of the emission rate per second
EMITTER_POS = [2., 0.2, 2.] # Emitter position
EMITTER_RADIUS = 0.07 # Emitter radius, circular emission

B_MU = [2500, 25000] # Boundary viscosity, give different boundary visocities to different boundaries. has to be an array with the same length as the number of boundary files. See under in boundary condition section for more detail.

T_ROOM = 25
ROOM_RADIATION_HALF_TIME = 2 # If too low, may explode. Min tested working value: 0.01
EMISSION_T = 300 # Emitted particles temperature
INIT_T = 250 # Initial temperature of the fluid
# t_to_mu: Temperature -> Viscosity. For constant viscosity, simply return a scalar value
@ti.func
def T_TO_MU(t: ti.f32) -> ti.f32:
    return t*10
```

### Initial condition and boundary condition

The initial condition and boundary conditions can either be defined by a 3D numpy array of particles positions in a .npy file or by a meta file, see `src/pointDataMetaFiles/flask_on_plane.txt` for an example.

If the initial_condition file is leaved empty, you can define the number of particles to be generated in the simulation box by setting the `NUM_PARTICLES` parameter and it will create a cube of particles.

If the boundary_condition file is leaved empty, the simulation will run with a box boundary (without the top plane)

The meta files are text files with the following format:

```txt
file_name.vtk // Name of the file containing the particles positions (has to be in pointDataFiles)
1 // Alignment mode
0. 0. 0. // offset

file_name_2.vtk
2
0. 0. 0.
```
For each file named in the metafile, first the alignmode is aplied, and then the offset.
The alignment mode can be either 1, 2 or 3.

0. Do nothing for alignment.
1. Make it so the for each coordinate, the lowest value is 0.
2. Make it so the center of the particles is the coordinate origin.
3. The lowest y-coordinate is 0, and the x and z coordinates are centered around the origin.

## Logger

During the simulation, the following values will be output to terminal:

T: current time

dt: current timestep

B_cnt_avg: average number of boundary neighbors for fluid particles

F_cnt_avg: average number of fluid neighbors for fluid particles

d_avg: average density

cnt: number of active particles

oob: number of particles that went out of bounds

## Rendering

ffmpeg command to convert frames to video:

```bash
ffmpeg -framerate 24 -i frames/%06d.png -pix_fmt yuv420p output.mp4
```
The videos in presentation_results were made with -framerate 120

## Shown results

The above parameters were used to make the video flask_video.mp4. flask_video2.mp4 uses the same parameters except that B_MU=[4000,25000] (i.e., higher flask viscosity),\
 causing the foam to stick more to the flask. tube_video.mp4 showcases a higher emission rate (9000 particles_per_second, with 0 slowdown), which is enabled by the different geometry.\
 old_videos showcases previous steps of our progress.
