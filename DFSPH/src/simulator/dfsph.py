import taichi as ti

from .baseFluidModel import FluidModel
from .kernel import CubicSpline

@ti.data_oriented
class DensityAndPressureSolver:
    def __init__(self, num_particles: ti.i32, support_radius: ti.f32):
        self.num_particles = num_particles
        self.divergenceSolver = DivergenceSolver(num_particles, support_radius)
        self.densitySolver = DensitySolver(num_particles, support_radius)

        # set alpha_i reference in both solvers
        self.alpha_i = ti.field(dtype=ti.f32, shape=(self.num_particles))
        self.divergenceSolver.alpha_i = self.alpha_i
        self.densitySolver.alpha_i = self.alpha_i

        self.kernel = CubicSpline(support_radius)

        self.eps = 1e-5

    @ti.kernel    
    def update_alpha_i(self, f_X: ti.template(), f_M: ti.f32, f_density: ti.template(), f_neighbors: ti.template(), b_X: ti.template(), b_M: ti.template(), b_neighbors: ti.template()):
        for i in range(self.num_particles):
            denom_grad_sum = ti.Vector([0., 0., 0.], ti.f32)
            denom_norm_sum = 0.
            local_pos = f_X[i]
            for j in range(self.num_particles):
                if f_neighbors[i, j] == 1:
                    value = f_M * self.kernel.W_grad(local_pos - f_X[j])
                    denom_grad_sum += value
                    denom_norm_sum += value.norm_sqr()
            
            b_num_particles = b_X.shape[0]
            for j in range(b_num_particles):
                if b_neighbors[i, j] == 1:
                    value = b_M[j] * self.kernel.W_grad(local_pos - b_X[j])
                    denom_grad_sum += value
                    denom_norm_sum += value.norm_sqr()

            denom_norm_sum += denom_grad_sum.norm_sqr()
            if denom_norm_sum < self.eps:
                self.alpha_i[i] = 0.
            self.alpha_i[i] = (f_density[i]*f_density[i]) / (denom_norm_sum)

@ti.data_oriented
class DivergenceSolver:
    def __init__(self, num_particles: ti.i32, support_radius: ti.f32):
        self.num_particles = num_particles
        self.alpha_i = ti.field(dtype=ti.f32, shape=(self.num_particles))
        self.density_adv = ti.field(dtype=ti.f32, shape=(self.num_particles))
        self.kappa_v_i = ti.field(dtype=ti.f32, shape=(self.num_particles))
        self.factor_i = ti.field(dtype=ti.f32, shape=(self.num_particles))

        self.kernel = CubicSpline(support_radius)

        self.min_iter = 1
        self.max_iter = 100

        self.tol = 1e-4
        self.eps = 1e-5

    def solve(self, fluid: FluidModel, dt):
        # print(fluid.V)
        self.compute_density_adv(fluid.mass, fluid.X, fluid.V, fluid.f_neighbors, fluid.b_X, fluid.b_M, fluid.b_neighbors, fluid.number_of_neighbors)

        density_adv_avg = self.compute_field_average(self.density_adv)
        # print("Initial density_adv_avg: ", density_adv_avg)
        iteration = 0
        eta = self.tol * fluid.density0 

        while (density_adv_avg*dt > eta or iteration < self.min_iter) and not iteration > self.max_iter:
            self.compute_density_adv(fluid.mass, fluid.X, fluid.V, fluid.f_neighbors, fluid.b_X, fluid.b_M, fluid.b_neighbors, fluid.number_of_neighbors)
            self.compute_kappa_v_i(fluid.density, dt)
            self.update_velocity(fluid.mass, fluid.X, fluid.V, fluid.f_neighbors, fluid.b_M, fluid.b_X, fluid.b_neighbors, dt)
            density_adv_avg = self.compute_field_average(self.density_adv)
            iteration += 1
        return density_adv_avg - eta, iteration

    @ti.kernel
    def update_velocity(self, M: ti.f32, X: ti.template(), V: ti.template(), f_neighbors: ti.template(), b_M: ti.template(), b_X: ti.template(), b_neighbors: ti.template(), dt: ti.f32):
        for i in range(self.num_particles):
            vel_sum = ti.Vector([0., 0., 0.], ti.f32)
            local_pos = X[i]
            local_factor = self.factor_i[i]
            for j in range(self.num_particles):
                if f_neighbors[i, j] == 1:
                    ksum = local_factor + self.factor_i[j]
                    if(ksum > self.eps):
                        vel_sum += M * ksum * self.kernel.W_grad(local_pos - X[j])
            
            # neighbors
            b_num_particles = b_X.shape[0]
            for j in range(b_num_particles):
                if b_neighbors[i, j] == 1:
                    if local_factor > self.eps:
                        vel_sum += b_M[j] * local_factor * self.kernel.W_grad(local_pos - b_X[j])
            
            if vel_sum.norm() > self.eps:
                V[i] -= vel_sum

    @ti.kernel
    def compute_kappa_v_i(self, density: ti.template(), dt: ti.f32):
        for i in range(self.num_particles):
            if self.density_adv[i] > self.eps and self.alpha_i[i] > self.eps:
                self.kappa_v_i[i] = self.alpha_i[i] * self.density_adv[i]
                self.factor_i[i] = self.kappa_v_i[i] / (density[i]*density[i])
            else:
                self.kappa_v_i[i] = 0.
                self.factor_i[i] = 0.
        
    @ti.kernel
    def compute_density_adv(self, M: ti.f32, X: ti.template(), V: ti.template(), f_neighbors: ti.template(), b_X: ti.template(), b_M: ti.template(), b_neighbors: ti.template(), number_of_neighbors: ti.template()):
        for i in range(self.num_particles):
            if number_of_neighbors[i] < 20:
                self.density_adv[i] = 0.
                continue
            local_sum = 0.
            local_pos = X[i]
            local_vel = V[i]
            for j in range(self.num_particles):
                if f_neighbors[i, j] == 1:
                    local_sum += M * (local_vel - V[j]).dot(self.kernel.W_grad(local_pos - X[j]))

            #neighbors
            b_num_particles = b_X.shape[0]
            for j in range(b_num_particles):
                if b_neighbors[i, j] == 1:
                    local_sum += b_M[j] * local_vel.dot(self.kernel.W_grad(local_pos - b_X[j]))

            self.density_adv[i] = max(local_sum, 0.) # Only correct positive divergence

    @ti.kernel
    def compute_field_average(self, field: ti.template()) -> ti.f32:
        local_sum = 0.
        for i in range(self.num_particles):
            local_sum += field[i]
        return local_sum / self.num_particles

@ti.data_oriented 
class DensitySolver:
    def __init__(self, num_particles: ti.i32, support_radius: ti.f32):
        self.num_particles = num_particles
        self.support_radius = support_radius
        self.alpha_i = ti.field(dtype=ti.f32, shape=(self.num_particles))
        self.kappa_i = ti.field(dtype=ti.f32, shape=(self.num_particles))

        self.warm_factor_i = ti.field(dtype=ti.f32, shape=(self.num_particles))
        self.warmed = False

        self.factor_i = ti.field(dtype=ti.f32, shape=(self.num_particles))
        self.density_predict = ti.field(dtype=ti.f32, shape=(self.num_particles))

        self.kernel = CubicSpline(support_radius)

        self.eps = 1e-5
        self.tol = 1e-5
        self.max_iter = 100
        self.min_iter = -1

    def solve(self, fluid: FluidModel, dt: ti.f32):
        density_avg = self.compute_density_avg(fluid.density)
        # print("Original Density Avg: ", density_avg)
        iteration = 0
        eta = self.tol * fluid.density0
        if self.warmed:
            self.warm_start()
        else:
            self.warmed = True

        while (abs(density_avg - fluid.density0) > eta or iteration < self.min_iter) and not iteration > self.max_iter:
            self.predict_density(fluid.X, fluid.V, fluid.density, fluid.mass, fluid.f_neighbors, fluid.b_X, fluid.b_M, fluid.b_neighbors, fluid.density0, dt)
            density_avg = self.compute_density_avg(self.density_predict)
            self.compute_kappa_i(fluid.density, fluid.density0, dt)
            self.update_velocity(fluid.X, fluid.V, fluid.mass, fluid.f_neighbors, fluid.b_X, fluid.b_M, fluid.b_neighbors, dt)
            iteration += 1
        return density_avg - fluid.density0, iteration

    @ti.kernel
    def compute_density_avg(self, density: ti.template()) -> ti.f32:
        density_avg = 0.
        for i in range(self.num_particles):
            density_avg += density[i]
        return density_avg / self.num_particles

    @ti.kernel
    def warm_start(self):
        for i in range(self.num_particles):
            self.factor_i[i] = self.warm_factor_i[i]

    @ti.kernel
    def predict_density(self, X: ti.template(), V: ti.template(), f_density: ti.template(), f_M: ti.f32, f_neighbors: ti.template(), b_X: ti.template(), b_M: ti.template(), b_neighbors: ti.template(), density_0: ti.f32, dt: ti.f32):
        for i in range(self.num_particles):
            value = 0.
            local_pos = X[i]
            local_vel = V[i]
            for j in range(self.num_particles):
                if f_neighbors[i, j] == 1:
                    value += f_M * (local_vel - V[j]).dot(self.kernel.W_grad(local_pos - X[j]))
            
            # Boundary
            b_num_particles = b_X.shape[0]
            for j in range(b_num_particles):
                if b_neighbors[i, j] == 1:
                    value += b_M[j] * local_vel.dot(self.kernel.W_grad(local_pos - b_X[j]))

            self.density_predict[i] = max(f_density[i] + dt*value, density_0)

    @ti.kernel
    def compute_kappa_i(self, f_density: ti.template(), rest_density: ti.f32, dt: ti.f32):
        for i in range(self.num_particles):
            if (self.density_predict[i] - rest_density) > self.eps:
                self.kappa_i[i] = self.alpha_i[i] * ((self.density_predict[i] - rest_density) / (dt))
                self.factor_i[i] = self.kappa_i[i] / (f_density[i]*f_density[i])
                self.warm_factor_i[i] += self.factor_i[i]
            else:
                self.kappa_i[i] = 0.
                self.factor_i[i] = 0.

    @ti.kernel
    def update_velocity(self, f_X:ti.template(), f_V: ti.template(), f_mass: ti.f32, f_neighbors: ti.template(), b_X: ti.template(), b_M: ti.template(), b_neighbors: ti.template(), dt: ti.f32):
        for i in range(self.num_particles):
            vel_diff = ti.Vector([0., 0., 0.], ti.f32)
            local_pos = f_X[i]
            local_factor= self.factor_i[i]
            for j in range(self.num_particles):
                if f_neighbors[i, j] == 1:
                    vel_diff += f_mass * (local_factor + self.factor_i[j]) * self.kernel.W_grad(local_pos - f_X[j])

            b_num_particles = b_X.shape[0]
            for j in range(b_num_particles):
                if b_neighbors[i, j] == 1:
                    # kappa_j for boundary particles is 0 as their density = rest_density
                    vel_diff += b_M[j] * local_factor * self.kernel.W_grad(local_pos - b_X[j]) 
            if vel_diff.norm() > self.eps:
                f_V[i] -= vel_diff
