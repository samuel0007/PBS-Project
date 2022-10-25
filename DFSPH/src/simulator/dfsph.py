import taichi as ti
from .kernel import Poly6, Spiky

@ti.data_oriented
class DensityAndPressureSolver:
    def __init__(self, num_particles: ti.i32, support_radius: ti.f32):
        self.num_particles = num_particles
        self.divergenceSolver = DivergenceSolver(num_particles)
        self.densitySolver = DensitySolver(num_particles)
        self.alpha_i = ti.field(dtype=ti.f32, shape=(self.num_particles))
        self.poly6 = Poly6(support_radius)
        self.spiky = Spiky(support_radius)

        self.eps = 1e-6

    @ti.kernel    
    def update_alpha_i(self, f_X: ti.template(), f_M: ti.f32, f_density: ti.template(), f_neighbors: ti.template(), b_X: ti.template(), b_M: ti.template(), b_neighbors: ti.template()):
        for i in range(self.num_particles):
            denom_grad_sum = ti.Vector([0., 0., 0.], ti.f32)
            denom_norm_sum = 0.
            local_pos = f_X[i]
            for j in range(self.num_particles):
                if f_neighbors[i, j] == 1:
                    value = f_M * self.spiky.W_grad(local_pos - f_X[j])
                    denom_grad_sum += value
                    denom_norm_sum += value.norm_sqr()
            
            b_num_particles = b_X.shape[0]
            for j in range(b_num_particles):
                if b_neighbors[i, j] == 1:
                    value = b_M[j] * self.spiky.W_grad(local_pos - b_X[j])
                    denom_grad_sum += value
                    denom_norm_sum += value.norm_sqr()
            
            self.alpha_i[i] = f_density[i] / (denom_grad_sum.norm_sqr() + denom_norm_sum + self.eps)
        
        # set alpha_i in both solvers
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