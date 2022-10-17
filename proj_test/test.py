import taichi as ti
import math
from scipy.integrate import solve_ivp
import numpy as np
ti.init(arch=ti.gpu)

N = 2000

# Global constants
# Hint: It's quite hard to find constants s.t. the diffusion works stabily. If the simulation turns all black, reduce support radius
K = 8.
mu = 50.
surface_constant = 2
mass = 0.012
support_radius = 0.045
radius = 0.015
# rest_density = 1000
alpha = 10000
diffusion_coeff = 7e12
dt = 6e-4
gravity = ti.Vector([0, -30, 0])
substeps = int(1 / 180 // dt)
total_frames = 1000

result_dir = "./results"
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

# x, v, m, T, density, a
X = ti.Vector.field(3, dtype=ti.f32, shape=N)
V = ti.Vector.field(3, dtype=ti.f32, shape=N)
M = ti.field(dtype=ti.f32, shape=N)
T = ti.field(dtype=ti.f32, shape=N)
max_temp = 100
colors = ti.Vector.field(3, dtype=ti.f32, shape=N)
rest_density = ti.field(dtype=ti.f32, shape=N)
density = ti.field(dtype=ti.f32, shape=N)
pressure = ti.field(dtype=ti.f32, shape=N)
color_field = ti.field(dtype=ti.f32, shape=N)
color_field_gradient = ti.Vector.field(3, dtype=ti.f32, shape=N)
color_field_laplacian = ti.field(dtype=ti.f32, shape=N)

a = ti.Vector.field(3, dtype=ti.f32, shape=N)
# fill temperature with random values

# SPH Poly6 Kernel function
@ti.func
def W(r_len, h):
    return  (315. / (64. * math.pi * h**9)) * (h**2 - r_len**2)**3

# SPH Pressure Spiky Kernel Gradient
@ti.func
def gradW(r_len, h):
    value = -(90. / (math.pi * h**6)) * (h - r_len)**2 * r_len
    return ti.Vector([value, value, value], ti.f32)

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
        X[particle] = [ti.random()*0.5, 1 + ti.random()*0.5, ti.random()*0.5]
        V[particle] = [0, 0, 0]

        # half of the particles have some random velocity to the left
        # V[particle] = [ti.random()*0.1-0.1, 0, 0]
        M[particle] = mass
        T[particle] = ti.random()*max_temp
        rest_density[particle] = alpha/T[particle]
        density[particle] = rest_density[particle]
        a[particle] = [0, 0, 0]


# function reutrning the density of each particle
@ti.func
def update_pressure(particle):
    return K * (density[particle] - rest_density[particle])

@ti.func
def update_density(particle):
    density[particle] = 0
    for other in range(N):
        r = X[particle] - X[other]
        r_len = r.norm()
        if r_len <= support_radius:
            density[particle] += M[other] * W(r_len, support_radius)
    return density[particle]

@ti.func
def update_color_field(particle):
    color_field[particle] = 0.
    color_field_gradient[particle] = ti.Vector([0., 0., 0.], ti.f32)
    color_field_laplacian[particle] = 0.
    for other in range(N):
        r = X[particle] - X[other]
        r_len = r.norm()
        if r_len <= support_radius:
            color_field[particle] += M[other] / density[other] * W(r_len, support_radius)
            color_field_gradient[particle] += M[other] / density[other] * gradW(r_len, support_radius)
            color_field_laplacian[particle] += M[other] / density[other] * laplW_poly6(r_len, support_radius)
        

# function returning the updated temperature attribute of each particle
@ti.func
def update_temperature(particle):
    derivate_temperature = 0.
    for other in range(N):
        r = X[particle] - X[other]
        r_len = r.norm()
        if r_len <= support_radius:
            derivate_temperature += M[other] * (T[other] - T[particle])/density[other] * laplW(r_len, support_radius)
    derivate_temperature *= diffusion_coeff
    T[particle] += derivate_temperature * dt

@ti.func
def update_rest_density(particle):
    rest_density[particle] = alpha / T[particle]
# function updating color according to temperature
@ti.func
def update_color(particle):
    # Map temperature to 0-1 range
    colors[particle] = [T[particle] / max_temp, 0, 0]



# function returning the force on each particle
@ti.func
def force(particle: ti.i32) -> ti.types.vector(2, ti.f32):
    force = ti.Vector([0., 0., 0.], dt=ti.f32)
    for other in range(N):
        if other != particle:
            r = X[particle] - X[other]
            r_len = r.norm()
            if r_len <= support_radius:
                # pressure force
                force -= M[other] * (pressure[particle] + pressure[other]) / (2.*density[other]) * gradW(r_len, support_radius)
                # viscosity force
                force += mu * M[other] * (V[other] - V[particle]) / density[other] * laplW(r_len, support_radius)    

    # add surface tension force
    color_field_gradient_norm = color_field_gradient[particle].norm()
    if color_field_gradient_norm > 1e-6:
        force -= surface_constant * color_field_laplacian[particle] * color_field_gradient[particle] / color_field_gradient_norm                

    # add gravity
    force += gravity * density[particle]
    return force

@ti.kernel
def substep():
    # The for loop iterates over all elements of the field v
    for i in X:
        # Update density
        density[i] = update_density(i)
        # Update pressure for each particle
        pressure[i] = update_pressure(i)
        # Update color field
        update_color_field(i)

        # diffusion of attributes
        update_temperature(i)
        update_rest_density(i)
        update_color(i)

        # leapfrog integration of the force
        new_acc = force(i) / density[i]
        V[i] += 0.5 * (a[i] + new_acc) * dt
        X[i] += V[i] * dt + 0.5 * a[i] * dt**2
        a[i] = new_acc
        
        # Check for boundaries in 3d
        if X[i][0] < 0:
            X[i][0] = 0
            V[i][0] *= -1
        if X[i][0] > 0.75:
            X[i][0] = 0.75
            V[i][0] *= -1
        if X[i][1] < 0:
            X[i][1] = 0
            V[i][1] *= -1
        if X[i][1] > 2:
            X[i][1] = 2
            V[i][1] *= -1
        if X[i][2] < 0:
            X[i][2] = 0
            V[i][2] *= -1
        if X[i][2] > 0.75:
            X[i][2] = 0.75
            V[i][2] *= -1

@ti.func
def evaluate_color_field(r) -> ti.f32:
    color_field_value = 0.
    for particle in range(N):
        r_len = (r - X[particle]).norm()
        if r_len <= support_radius:
            color_field_value += M[particle] / density[particle] * W(r_len, support_radius)
    return color_field_value




window = ti.ui.Window("Taichi Fluid Particle Simulation", (1024, 1024),
                      vsync=True)

canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0

# draw bounds
bounds = ti.Vector.field(3, ti.f32, 24)
bounds[0] = [0, 0, 0]
bounds[1] = [0.75, 0, 0]
bounds[2] = [0.75, 0, 0]
bounds[3] = [0.75, 1, 0]
bounds[4] = [0.75, 1, 0]
bounds[5] = [0, 1, 0]
bounds[6] = [0, 1, 0]
bounds[7] = [0, 0, 0]
bounds[8] = [0, 0, 0]
bounds[9] = [0, 0, 0.75]
bounds[10] = [0, 0, 0.75]
bounds[11] = [0.75, 0, 0.75]
bounds[12] = [0.75, 0, 0.75]
bounds[13] = [0.75, 1, 0.75]
bounds[14] = [0.75, 1, 0.75]
bounds[15] = [0, 1, 0.75]
bounds[16] = [0, 1, 0.75]
bounds[17] = [0, 1, 0]
bounds[18] = [0, 1, 0]
bounds[19] = [0.75, 0, 0.75]
bounds[20] = [0.75, 0, 0.75]
bounds[21] = [0.75, 1, 0.75]
bounds[22] = [0.75, 1, 0.75]
bounds[23] = [0, 1, 0.75]

initialize_fluid_particles()
camera_pos = [1, 3, 3]

resolution = 5
color_field_mesh = ti.Vector.field(3, ti.f32, shape=(10, 10, 10))
color_field_values = ti.Vector.field(3, ti.f32, shape=(10, 10, 10))

# build mesh
@ti.kernel
def build_mesh():
    resolution_double = ti.cast(resolution, ti.f32)
    for i, j, k in color_field_mesh:
        color_field_mesh[i, j, k] = ti.Vector([i/(resolution*2), j/resolution, k/(resolution*2)], ti.f32)

build_mesh()

@ti.kernel
def compute_color_field():
    for i, j, k in color_field_values:
        r = color_field_mesh[i, j, k]
        value = 1-evaluate_color_field(r)
        color_field_values[i, j, k] = ti.Vector([value, value, value], ti.f32)


# while window.running:
for l in range(total_frames):
    if current_t > 100:
        # Reset
        initialize_fluid_particles()
        current_t = 0

    for i in range(substeps):
        substep()
        current_t += dt
    
    # update_vertices()

    # Camera rotates
    # camera_pos[0] = 3 + 3 * math.sin(current_t)

    camera.position(*camera_pos)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    # Draw particles with each a different color
    # print(color_field_laplacian)
    # print(T)
    # print(T_normalized)
    scene.particles(centers=X, radius=radius, per_vertex_color=colors)
    # Plot color field on a grid
    # color_particle_pos = ti.Vector.field(3, ti.f32, 10*10*10)
    # color_particle_color = ti.Vector.field(3, ti.f32, 10*10*10)

    # print("color field computations...")
    # compute_color_field()
    # print("finished color field computations")

    # print("flatten...")
    # # flatten color field values and mesh
    # color_field_mesh_flatten = ti.Vector.field(3, ti.f32, shape=(resolution*resolution*resolution))
    # color_field_values_flatten = ti.Vector.field(3, ti.f32, shape=(resolution*resolution*resolution))
    # for i in range(resolution):
    #     for j in range(resolution):
    #         for k in range(resolution):
    #             color_field_mesh_flatten[i*resolution*resolution + j*resolution + k] = color_field_mesh[i, j, k]
    #             color_field_values_flatten[i*resolution*resolution + j*resolution + k] = color_field_values[i, j, k]
    # print("end flatten...")
    
    # scene.particles(centers=color_field_mesh_flatten, radius=0.01, per_vertex_color=color_field_values_flatten)
    scene.lines(bounds, 1)
    canvas.scene(scene)
    video_manager.write_frame(window.get_image_buffer_as_numpy())

    window.show()
    print(f'\rFrame {l+1}/{total_frames} is recorded')

print('Exporting .mp4 video...')
# this is somehow not working on my system, probably comes from ffmpeg
# video_manager.make_video(mp4=True)
# print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')

    

