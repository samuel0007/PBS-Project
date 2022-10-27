import taichi as ti
from .kernel import CubicSpline

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
        self.f_neighbors = ti.field(dtype=ti.i32, shape=(self.num_particles, self.num_particles))
        self.f_number_of_neighbors = ti.field(dtype=ti.i32, shape=(self.num_particles))
        self.b_number_of_neighbors = ti.field(dtype=ti.i32, shape=(self.num_particles))
        self.number_of_neighbors = ti.field(dtype=ti.i32, shape=(self.num_particles))
        self.kernel = CubicSpline(self.support_radius)

    def set_boundary_particles(self, b_X, b_M):
        self.b_num_particles = b_X.shape[0]
        self.b_X = b_X
        self.b_M = b_M
        self.b_neighbors = ti.field(dtype=ti.i32, shape=(self.num_particles, self.b_num_particles))

    @ti.kernel
    def CFL_condition(self) -> ti.f32:
        max_V = 0.
        for i in range(self.num_particles):
            max_V = max(max_V, self.V[i].norm())
        return min(5e-5, 0.4 * self.support_radius / max_V)

    @ti.kernel
    def explicit_update_position(self, dt: ti.f32):
        for i in range(self.num_particles):
            self.X[i] += self.V[i] * dt

    # Dumb O(n^2) neighbor search, replace with grid based later
    @ti.kernel
    def update_neighbors(self):
        for i in range(self.num_particles):
            local_pos = self.X[i]
            f_count = 0
            for j in range(self.num_particles):
                if (local_pos - self.X[j]).norm() < self.support_radius:
                    self.f_neighbors[i, j] = 1
                    f_count += 1
                else:
                    self.f_neighbors[i, j] = 0
            self.f_number_of_neighbors[i] = f_count
            
            b_count = 0
            for j in range(self.b_num_particles):
                if (local_pos - self.b_X[j]).norm() < self.support_radius:
                    self.b_neighbors[i, j] = 1
                    b_count += 1
                else:
                    self.b_neighbors[i, j] = 0
            self.b_number_of_neighbors[i] = b_count

            self.number_of_neighbors[i] = f_count + b_count
            
    @ti.kernel
    def update_density(self):
        for i in range(self.num_particles):
            density = 0.
            local_pos = self.X[i]

            # If the particle is out of bound, set its density to rest_density s.t. it doesn't penalize the solver
            if local_pos[0] < -0.5 or local_pos[0] > 1.5 or local_pos[1] < -0.5 or local_pos[1] > 1.5 or local_pos[2] < -0.5 or local_pos[2] > 1.5:
                density = self.density0

            else:            
                # This is the dumbest datastructure ever, but for now... let it be
                for j in range(self.num_particles):
                    if self.f_neighbors[i, j] == 1:
                        density += self.mass * self.kernel.W(local_pos - self.X[j])
                for j in range(self.b_num_particles):
                    if self.b_neighbors[i, j] == 1:
                        density += self.b_M[j] * self.kernel.W(local_pos - self.b_X[j])

            self.density[i] = density