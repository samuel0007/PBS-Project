import taichi as ti
from .baseFluidModel import FluidModel

@ti.data_oriented
class TemperatureSolver:
    def __init__(self, gamma: ti.f32, t_room: ti.f32, room_radiation_half_time: ti.f32, fluid: FluidModel):
        """Temperature solver based on the paper by Weiler et al. (2018)"""
        self.gamma = gamma
        self.fluid = fluid
        self.t_room = t_room
        self.radiation_half_time = room_radiation_half_time

        self.laplacian = ti.field(dtype=ti.f32, shape=(self.fluid.max_num_particles))
        self.room_contribution = ti.field(dtype=ti.f32, shape=(self.fluid.max_num_particles))
        self.eps = 1e-5
    
    @ti.kernel
    def compute_laplacian(self):
        for i in range(self.fluid.num_particles[None]):
            if not self.fluid.active[i]: continue
            laplacian = 0.
            local_pos = self.fluid.X[i]
            local_temp = self.fluid.T[i]

            for l in range(self.fluid.get_num_neighbors_i(i)):
                j = self.fluid.neighbor_list[i,l]
                x_ij = local_pos - self.fluid.X[j]
                x_ij_norm_sqr = x_ij.norm_sqr()
                if x_ij_norm_sqr > self.eps:
                    laplacian += (self.fluid.mass/self.fluid.density[j])*(local_temp - self.fluid.T[j]) * ti.math.dot(x_ij, self.fluid.kernel.W_grad(x_ij)) / (x_ij_norm_sqr + 0.01*self.fluid.support_radius**2)
            self.laplacian[i] = 2*laplacian
        
    @ti.kernel
    def compute_room_contribution(self) -> ti.f32:
        counter = 0
        for i in range(self.fluid.num_particles[None]):
            if not self.fluid.active[i]: continue
            if self.fluid.normals_norm[i] > 5:
                self.room_contribution[i] = (self.fluid.T[i] - self.t_room) / self.radiation_half_time 
                counter += 1
            else:
                self.room_contribution[i] = 0.
        return counter
    @ti.kernel
    def explicit_temperature_update(self, dt: ti.f32):
        for i in range(self.fluid.num_particles[None]):
            if not self.fluid.active[i]: continue
            self.fluid.T[i] += dt * (self.gamma * self.laplacian[i] - self.room_contribution[i])
            if self.fluid.T[i] < 0:
                self.fluid.T[i] = 0
    
   

    def update_temperature(self, dt: ti.f32):
        self.compute_laplacian()
        self.compute_room_contribution()
        # print("surface particles counter: ", self.compute_room_contribution())
        # print(" ")
        self.explicit_temperature_update(dt)