import taichi as ti

class DensityAndPressureSolver:
    def __init__(self, num_particles: ti.i32):
        self.num_particles = num_particles
        self.divergenceSolver = DivergenceSolver(num_particles)
        self.densitySolver = DensitySolver(num_particles)
    
    def update_alpha_i(self):
        # compute alpha_i ...
        self.alpha_i = ti.field(dtype=ti.f32, shape=(self.num_particles))

        # set alpha_i to both solvers
        self.divergenceSolver.alpha_i = self.alpha_i
        self.densitySolver.alpha_i = self.alpha_i

class DivergenceSolver:
    def __init__(self, num_particles: ti.i32):
        self.num_particles = num_particles
        self.alpha_i = ti.field(dtype=ti.f32, shape=(self.num_particles))

    def solve(self, V):
        return V
    
class DensitySolver:
    def __init__(self, num_particles: ti.i32):
        self.num_particles = num_particles
        self.alpha_i = ti.field(dtype=ti.f32, shape=(self.num_particles))

    def solve(self, V):
        return V