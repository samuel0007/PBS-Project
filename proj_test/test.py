import taichi as ti
import math
from scipy.integrate import solve_ivp
import numpy as np
ti.init(arch=ti.gpu)

N = 1000

# Global constants
# Hint: It's quite hard to find constants s.t. the diffusion works stabily. If the simulation turns all black, reduce support radius
K = 20.
mu = 50.
mass = 0.012
support_radius = 0.03
radius = 0.01
rest_density = 1000
diffusion_coeff = 1e12
dt = 4e-2 / N
gravity = ti.Vector([0, -9.8, 0])
substeps = int(1 / 180 // dt)
total_frames = 100

result_dir = "./results"
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)


# x, v, m, T, density, a
X = ti.Vector.field(3, dtype=ti.f32, shape=N)
V = ti.Vector.field(3, dtype=ti.f32, shape=N)
M = ti.field(dtype=ti.f32, shape=N)
T = ti.field(dtype=ti.f32, shape=N)
max_temp = 400
colors = ti.Vector.field(3, dtype=ti.f32, shape=N)
density = ti.field(dtype=ti.f32, shape=N)
pressure = ti.field(dtype=ti.f32, shape=N)
a = ti.Vector.field(3, dtype=ti.f32, shape=N)
# fill temperature with random values

# SPH Kernel function
@ti.func
def W(r_vec: ti.types.vector(3, ti.f32), h):
    r = r_vec.norm()
    value = 0.
    if r < h:
        value = 315 / (64 * math.pi * h**9) * (h**2 - r**2)**3
    return value

# SPH Kernel gradient
@ti.func
def gradW(r: ti.types.vector(3, ti.f32), h):
    q = r.norm() / h
    value = ti.Vector([0, 0, 0], ti.f32)
    if q < 1:
        value = -45. / (math.pi * h**6) * (h - r.norm())**2 * r.normalized()
    return value

# SPH Kernel Laplacian
@ti.func
def laplW(r, h):
    q = r / h
    value = 0.
    if q < 1:
        value = (45. / math.pi * h**6) * (h - r)
    return value

@ti.kernel
def initialize_fluid_particles():
    for particle in X:
        X[particle] = [ti.random()*0.25, ti.random(), 1]
        # half of the particles have some random velocity to the left
        # V[particle] = [0, 0]

        V[particle] = [ti.random()*0.1-0.1, 0, 0]
        M[particle] = mass
        T[particle] = ti.random()*max_temp
        density[particle] = rest_density
        pressure[particle] = K*(density[particle] - rest_density)
        a[particle] = [0, 0, 0]


# function reutrning the density of each particle
@ti.func
def update_pressure(particle):
    return K * (density[particle] - rest_density)

@ti.func
def update_density(particle):
    density[particle] = 0
    for other in range(N):
        r = X[particle] - X[other]
        density[particle] += M[other] * W(r, support_radius)
    return density[particle]
# function returning the updated temperature attribute of each particle
@ti.func
def update_temperature(particle):
    derivate_temperature = 0.
    for other in range(N):
        r = X[particle] - X[other]
        r_len = r.norm()
        derivate_temperature += M[other] * (T[other] - T[particle])/density[other] * laplW(r_len, support_radius)
    derivate_temperature *= diffusion_coeff
    T[particle] += derivate_temperature * dt
# function updating color according to temperature
@ti.func
def update_color(particle):
    # Map temperature to 0-1 range
    # colors[particle] = [T_normalized[particle], 0, 0]
    colors[particle] = [T[particle] / max_temp, 0, 0]


# function returning the force on each particle
@ti.func
def force(particle: ti.i32) -> ti.types.vector(2, ti.f32):
    force = ti.Vector([0., 0., 0.], dt=ti.f32)
    for other in range(N):
        if other != particle:
            r = X[particle] - X[other]
            r_len = r.norm()
            if r_len < support_radius:
                # pressure force
                force -= M[other] * (pressure[particle] + pressure[other]) / (2.*density[other]) * gradW(r, support_radius)
                # viscosity force
                force += mu * M[other] * (V[other] - V[particle]) / density[other] * laplW(r_len, support_radius)
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

        # diffusion of attributes
        update_temperature(i)
        update_color(i)

        # leapfrog integration of the force
        X[i] += V[i] * dt + 0.5 * a[i] * dt**2
        new_acc = force(i) / density[i]
        V[i] += 0.5 * (a[i] + new_acc) * dt
        a[i] = new_acc
        
        # Check for boundaries in 3d
        if X[i][0] < 0:
            X[i][0] = 0
            V[i][0] = 0
        if X[i][0] > 0.5:
            X[i][0] = 0.5
            V[i][0] = 0
        if X[i][1] < 0:
            X[i][1] = 0
            V[i][1] = 0
        if X[i][1] > 0.5:
            X[i][1] = 0.5
            V[i][1] = 0
        if X[i][2] < 0:
            X[i][2] = 0
            V[i][2] = 0
        if X[i][2] > 0.5:
            X[i][2] = 0.5
            V[i][2] = 0


window = ti.ui.Window("Taichi Fluid Particle Simulation", (1024, 1024),
                      vsync=True)

canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0

initialize_fluid_particles()
camera_pos = [0.5, 0.5, 2]
# while window.running:
for j in range(total_frames):
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
    # scene.mesh(vertices,
    #            indices=indices,
    #            per_vertex_color=colors,
    #            two_sided=True)

    # Draw particles with each a different color
    # print(colors)
    # print(T)
    # print(T_normalized)
    scene.particles(centers=X, radius=radius, per_vertex_color=colors)
    canvas.scene(scene)
    video_manager.write_frame(window.get_image_buffer_as_numpy())
    window.show()
    print(f'\rFrame {j+1}/{total_frames} is recorded', end='')

print('Exporting .mp4 video...')
# this is somehow not working on my system, probably comes from ffmpeg
# video_manager.make_video(mp4=True)
# print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')

    

