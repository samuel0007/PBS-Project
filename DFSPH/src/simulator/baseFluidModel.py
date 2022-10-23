import numpy
import taichi as ti
from .kernel import Poly6, Spiky

@ti.data_oriented
class FluidModel:
    def __init__(self, num_particles: ti.i32, density0: ti.f32, support_radius: ti.f32, mass: ti.f32):
        self.num_particles = num_particles
        self.density0 = density0
        self.support_radius = support_radius
        self.mass = mass

        self.X = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_particles))
        self.V = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_particles))
        self.density = ti.field(dtype=ti.f32, shape=(self.num_particles))
        self.neighbors = ti.field(dtype=ti.i32, shape=(self.num_particles, self.num_particles))
        self.poly6 = Poly6(self.support_radius)
        self.spiky = Spiky(self.support_radius)

    @ti.kernel
    def CFL_condition(self) -> ti.f32:
        max_V = 0.
        for i in range(self.num_particles):
            max_V = max(max_V, self.V[i].norm())
        return 0.4 * self.support_radius / max_V

    @ti.kernel
    def explicit_update_position(self, dt: ti.f32):
        for i in range(self.num_particles):
            self.X[i] += self.V[i] * dt

    # Dumb O(n^2) neighbor search, replace with grid based later
    @ti.kernel
    def update_neighbors(self):
        for i in range(self.num_particles):
            local_pos = self.X[i]
            for j in range(self.num_particles):
                if (local_pos - self.X[j]).norm() < self.support_radius:
                    self.neighbors[i, j] = 1
                else:
                    self.neighbors[i, j] = 0
        
    @ti.kernel
    def update_density(self):
        for i in range(self.num_particles):
            density = 0.
            local_pos = self.X[i]
            # This is the dumbest datastructure ever, but for now... let it be
            for j in range(self.num_particles):
                if self.neighbors[i, j] == 1:
                    density += self.mass * self.poly6.W(local_pos - self.X[j])
            self.density[i] = density