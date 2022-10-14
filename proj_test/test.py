import taichi as ti
import math
from scipy.integrate import solve_ivp

ti.init(arch=ti.gpu)

N = 1000

# Global constants
K = 20.
mu = 50.
support_radius = 0.045
radius = 0.01
rest_density = 1000
dt = 4e-2 / N
gravity = ti.Vector([0, -9.8])
substeps = int(1 / 60 // dt)

# x, v, m, T, density, a
X = ti.Vector.field(2, dtype=ti.f32, shape=N)
V = ti.Vector.field(2, dtype=ti.f32, shape=N)
M = ti.field(dtype=ti.f32, shape=N)
T = ti.field(dtype=ti.f32, shape=N)
density = ti.field(dtype=ti.f32, shape=N)
pressure = ti.field(dtype=ti.f32, shape=N)
a = ti.Vector.field(2, dtype=ti.f32, shape=N)

# SPH Kernel function
@ti.func
def W(r_vec, h):
    r = r_vec.norm()
    value = 0.
    if r < h:
        value = 315 / (64 * math.pi * h**9) * (h**2 - r**2)**3
    return value

# SPH Kernel gradient
@ti.func
def gradW(r: ti.types.vector(2, ti.f32), h):
    q = r.norm() / h
    value = ti.Vector([0, 0], ti.f32)
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
        X[particle] = [ti.random()*0.5, ti.random()]
        # half of the particles have some random velocity to the left
        # V[particle] = [0, 0]

        V[particle] = [ti.random()*0.1-0.1, 0]

        M[particle] = 0.012
        T[particle] = 0
        density[particle] = rest_density
        pressure[particle] = K*(density[particle] - rest_density)
        a[particle] = [0, 0]


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

# function returning the force on each particle
@ti.func
def force(particle: ti.i32) -> ti.types.vector(2, ti.f32):
    force = ti.Vector([0., 0.], dt=ti.f32)
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
        V[i] += force(i) / density[i] * dt
        # Check for boundaries
        if X[i][0] < radius and V[i][0] < 0:
            V[i][0] *= -0.5
        if X[i][0] > 1 - radius and V[i][0] > 0:
            V[i][0] *= -0.5
        if X[i][1] < radius and V[i][1] < 0:
            V[i][1] *= -0.5
        if X[i][1] > 1 - radius and V[i][1] > 0:
            V[i][1] *= -0.5
        
        X[i] += V[i] * dt



window = ti.ui.Window("Taichi Fluid Particle Simulation", (1024, 1024),
                      vsync=True)

canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0

initialize_fluid_particles()

while window.running:
    if current_t > 40:
        # Reset
        initialize_fluid_particles()
        current_t = 0

    for i in range(substeps):
        substep()
        current_t += dt
    
    # update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    # scene.mesh(vertices,
    #            indices=indices,
    #            per_vertex_color=colors,
    #            two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(X, radius=radius, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()

