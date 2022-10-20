import taichi as ti
import math
import numpy as np

ti.init(arch=ti.gpu)

# Simulation constants
EXPORT_COLOR_FIELD = False
EXPORT_FRAMES = False
N = 2000
resolution = 24
dt = 5e-4
r_tol = 1e-8
epsilon = 1e-5
substeps = int(1 / 180 // dt)
print(substeps)
total_frames = 2000
camera_pos = [7, 3, 7]
camera_look_at = [0, 3, 0]

x_bound = 0.75
y_bound = 6
z_bound = 0.75

bottom_heat_bound = 0.1
top_heat_bound = 0.45

heating_coefficient = 0.
cooling_coefficient = 0.

result_dir = "./results"

# Global Physical Constants
# Hint: It's quite hard to find constants s.t. the diffusion works stabily. If the simulation turns all black, reduce support radius
K = 2.
mu = 50.
surface_constant = 0.
buoyancy_coefficient = 0.
gravity = ti.Vector([0, -9.81, 0])
boundary_damping_coeff = 0.
mass = 0.012
radius = 0.018
support_radius = 0.065
alpha = 130000
diffusion_coeff = 0.
max_temp = 50
amb_temp = 15

# Data orientied particles informations
X = ti.Vector.field(3, dtype=ti.f32, shape=N)
V = ti.Vector.field(3, dtype=ti.f32, shape=N)
M = ti.field(dtype=ti.f32, shape=N)
T = ti.field(dtype=ti.f32, shape=N)
colors = ti.Vector.field(3, dtype=ti.f32, shape=N)
rest_density = ti.field(dtype=ti.f32, shape=N)
density = ti.field(dtype=ti.f32, shape=N)
pressure = ti.field(dtype=ti.f32, shape=N)
color_field = ti.field(dtype=ti.f32, shape=N)
color_field_gradient = ti.Vector.field(3, dtype=ti.f32, shape=N)
color_field_laplacian = ti.field(dtype=ti.f32, shape=N)
a = ti.Vector.field(3, dtype=ti.f32, shape=N)
lambdas = ti.field(dtype=ti.f32, shape=N)
density_contraint = ti.field(dtype=ti.f32, shape=N)


# SPH Poly6 Kernel function
@ti.func
def W(r_len, h):
    return  (315. / (64. * math.pi * h**9)) * (h**2 - r_len**2)**3

# SPH Pressure Spiky Kernel Gradient
@ti.func
def gradW(r, r_len, h):
    return -(45. / (math.pi * h**6)) * (h - r_len)**2 * r / r_len

# SPH Viscosity Kernel Laplacian
@ti.func
def laplW(r_len, h):
    return (45. / math.pi * h**6) * (h - r_len)

# SPH Poly6 laplacian
@ti.func
def laplW_poly6(r_len, h):
    return (-945. / (32. * math.pi * h**9)) * (h**2 - r_len**2)*(3*h**2 - 7*r_len**2)
        
@ti.kernel
def initialize_fluid_particles():
    for particle in X:
        X[particle] = [ti.random()*0.5, ti.random()*0.5, ti.random()*0.5]
        V[particle] = [0, 0, 0]

        # half of the particles have some random velocity to the left
        M[particle] = mass
        # T[particle] = ti.random()*max_temp
        T[particle] = 10
        # rest_density[particle] = alpha/ (T[particle]+273.15)
        rest_density[particle] = 800
        density[particle] = rest_density[particle]
        a[particle] = [0, 0, 0]


@ti.func
def update_pressure(particle):
    new_pressure = K * (density[particle] - rest_density[particle])
    pressure[particle] = new_pressure if new_pressure < 0 else 0

@ti.func
def update_density(particle):
    density[particle] = 0
    for other in range(N):
        r = X[particle] - X[other]
        r_len = r.norm()
        if r_tol < r_len and r_len <= support_radius:
            density[particle] += M[other] * W(r_len, support_radius)
    if density[particle] < 100:
        density[particle] = 100

@ti.func
def update_color_field(particle):
    color_field[particle] = 0.
    color_field_gradient[particle] = ti.Vector([0., 0., 0.], ti.f32)
    color_field_laplacian[particle] = 0.
    for other in range(N):
        r = X[particle] - X[other]
        r_len = r.norm()
        if r_tol < r_len and r_len <= support_radius:
            color_field[particle] += M[other] / density[other] * W(r_len, support_radius)
            color_field_gradient[particle] += M[other] / density[other] * gradW(r, r_len, support_radius)
            color_field_laplacian[particle] += M[other] / density[other] * laplW_poly6(r_len, support_radius)
        

# function returning the updated temperature attribute of each particle
@ti.func
def update_temperature(particle):
    derivate_temperature = 0.
    for other in range(N):
        r = X[particle] - X[other]
        r_len = r.norm()
        # Avoid division by zero
        if r_tol < r_len and r_len <= support_radius:
            derivate_temperature += M[other] * (T[other] - T[particle])/density[other] * laplW(r_len, support_radius)
    derivate_temperature *= diffusion_coeff
    T[particle] += derivate_temperature * dt

@ti.func
def update_boundary_temperature(particle):
    # Heat particle if at the boundary on the center of the bottom, and cool it if on the top
    if X[particle][1] < bottom_heat_bound and X[particle][0] < 0.3 and X[particle][0] > 0.2 and X[particle][2] < 0.3 and X[particle][2] > 0.2:
        if T[particle] < max_temp:
            T[particle] += heating_coefficient * dt
    elif X[particle][1] > top_heat_bound:
        if T[particle] > cooling_coefficient * dt + 1e-5:
            T[particle] -= cooling_coefficient * dt

@ti.func
def update_rest_density(particle):
    # rest_density[particle] = alpha / (T[particle]+273.15)
    rest_density[particle] = 800

# function updating color according to temperature
@ti.func
def update_color(particle):
    # Map temperature to 0-1 range
    colors[particle] = [T[particle]/max_temp, 0, 0]

@ti.func
def update_lambda(particle):
    lambdas[particle] = 1. - density[particle] / rest_density[particle] 
    gradient_sum = 0.
    for other in range(N):
        if(other != particle):
            r = X[particle] - X[other]
            r_len = r.norm()
            if r_tol < r_len and r_len <= support_radius:
                gradient_sum -= gradW(r, r_len, support_radius)
        r = X[particle] - X[other]
        r_len = r.norm()
        if r_len <= support_radius:
            lambdas[particle] += (M[other] / density[other]) * (gradW(r, r_len, support_radius) @ color_field_gradient[particle])
    lambdas[particle] = -1 / (color_field_laplacian[particle] + lambdas[particle])

@ti.func
def update_attributes(i):
    # -- Should be optimized by only iterating once through the particles --
    # Update density
    update_density(i)
    # Update pressure for each particle
    update_pressure(i)
    # Update color field
    update_color_field(i)
    # diffusion of attributes
    update_temperature(i)
    update_rest_density(i)
    update_color(i)
    # update_lambda(i)

# function returning the force on each particle
@ti.func
def force(particle: ti.i32) -> ti.types.vector(2, ti.f32):
    force = ti.Vector([0., 0., 0.], dt=ti.f32)
    for other in range(N):
        if other != particle:
            r = X[particle] - X[other]
            r_len = r.norm()
            if r_tol < r_len and r_len <= support_radius:
                # pressure force
                force -= M[other] * (pressure[particle] + pressure[other]) / (2.*density[other]) * gradW(r, r_len, support_radius)
                # viscosity force
                force += mu * M[other] * (V[other] - V[particle]) / density[other] * laplW(r_len, support_radius)    
    # add surface tension force
    color_field_gradient_norm = color_field_gradient[particle].norm()
    if color_field_gradient_norm > 1e-3:
        force -= surface_constant * color_field_laplacian[particle] * color_field_gradient[particle] / color_field_gradient_norm                

    # add buoyancy force
    # force += buoyancy_coefficient * (amb_temp - T[particle]) * gravity
    # add gravity
    force += gravity * density[particle]
    return force

@ti.func
def enforce_boundary(i):
    if X[i][0] < 0:
        X[i][0] = 0 + epsilon*ti.random()
        V[i][0] *= -boundary_damping_coeff
    if X[i][0] > x_bound:
        X[i][0] = x_bound + epsilon*ti.random()
        V[i][0] *= -boundary_damping_coeff
    if X[i][1] < 0:
        X[i][1] = 0 + epsilon*ti.random()
        V[i][1] *= -boundary_damping_coeff
    if X[i][1] > y_bound:
        X[i][1] = y_bound + epsilon*ti.random()
        V[i][1] *= -boundary_damping_coeff
    if X[i][2] < 0:
        X[i][2] = 0 + epsilon*ti.random()
        V[i][2] *= -boundary_damping_coeff
    if X[i][2] > z_bound:
        X[i][2] = z_bound + epsilon*ti.random()
        V[i][2] *= -boundary_damping_coeff

@ti.kernel
def substep():
    # The for loop iterates over all elements of the field v
    for i in X:
        # Update attributes
        update_attributes(i)

        # leapfrog integration of the force
        new_acc = force(i) / density[i]
        V[i] += 0.5 * (a[i] + new_acc) * dt
        X[i] += V[i] * dt + 0.5 * a[i] * dt**2
        a[i] = new_acc
        
        enforce_boundary(i)
        update_boundary_temperature(i)

@ti.func
def evaluate_color_field(r) -> ti.f32:
    color_field_value = 0.
    for particle in range(N):
        r_len = (r - X[particle]).norm()
        if r_len <= support_radius:
            color_field_value += M[particle] / density[particle] * W(r_len, support_radius)
    return color_field_value

@ti.kernel
def compute_color_field():
    for i, j, k in color_field_values:
        color_field_values[i, j, k] = evaluate_color_field(color_field_mesh[i, j, k])

@ti.kernel
def build_mesh():
    for i, j, k in color_field_mesh:
        color_field_mesh[i, j, k] = ti.Vector([i/resolution * x_bound, j/resolution * y_bound, k/resolution * z_bound], ti.f32)

def export_mesh():
    np.save("results/frames_data/color_field_mesh.npy", color_field_mesh.to_numpy())

def export_color_field(frame_idx: ti.i32):
    compute_color_field()
    np.save("results/frames_data/color_field_{:04d}.npy".format(frame_idx), color_field_values.to_numpy())


window = ti.ui.Window("Taichi Fluid Particle Simulation", (1024, 1024),
                      vsync=True)


canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

current_t = 0.0

# draw bounds
bounds = ti.Vector.field(3, ti.f32, 24)
bounds[0] = [0, 0, 0]
bounds[1] = [x_bound, 0, 0]
bounds[2] = [x_bound, 0, 0]
bounds[3] = [x_bound, y_bound, 0]
bounds[4] = [x_bound, y_bound, 0]
bounds[5] = [0, y_bound, 0]
bounds[6] = [0, y_bound, 0]
bounds[7] = [0, 0, 0]

bounds[8] = [0, 0, z_bound]
bounds[9] = [x_bound, 0, z_bound]
bounds[10] = [x_bound, 0, z_bound]
bounds[11] = [x_bound, y_bound, z_bound]
bounds[12] = [x_bound, y_bound, z_bound]
bounds[13] = [0, y_bound, z_bound]
bounds[14] = [0, y_bound, z_bound]
bounds[15] = [0, 0, z_bound]

bounds[16] = [0, 0, 0]
bounds[17] = [0, 0, z_bound]
bounds[18] = [0, y_bound, 0]
bounds[19] = [0, y_bound, z_bound]
bounds[20] = [x_bound, 0, 0]
bounds[21] = [x_bound, 0, z_bound]
bounds[22] = [x_bound, y_bound, 0]
bounds[23] = [x_bound, y_bound, z_bound]


initialize_fluid_particles()

color_field_mesh = ti.Vector.field(3, ti.f32, shape=(10, 10, 10))
color_field_values = ti.field(ti.f32, shape=(10, 10, 10))

if EXPORT_COLOR_FIELD:
    build_mesh()
    export_mesh()

# while window.running:
for l in range(total_frames):
    for i in range(substeps):
        substep()
        current_t += dt
    camera.position(*camera_pos)
    camera.lookat(*camera_look_at)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    # Draw particles with each a different color
    scene.particles(centers=X, radius=radius, per_vertex_color=colors)
    scene.lines(bounds, 1)
    # Plot color field on a grid
    # color_particle_pos = ti.Vector.field(3, ti.f32, 10*10*10)
    # color_particle_color = ti.Vector.field(3, ti.f32, 10*10*10)

    if EXPORT_COLOR_FIELD:
        export_color_field(l)
    print(np.average(T.to_numpy()), np.average(density.to_numpy()), np.average(rest_density.to_numpy()))
    canvas.scene(scene)
    if EXPORT_FRAMES:
        video_manager.write_frame(window.get_image_buffer_as_numpy())
        print(f'\rFrame {l+1}/{total_frames} is recorded')
    window.show()

# print('Exporting .mp4 video...')
# this is somehow not working on my system, probably comes from ffmpeg
# video_manager.make_video(mp4=True)
# print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')

    

