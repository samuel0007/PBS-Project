import numpy as np
import taichi as ti
from .kernel import Poly6, Spiky

@ti.data_oriented
class FluidModel:
    def __init__(self, num_particles: ti.i32, density0: ti.f32, support_radius: ti.f32, mass: ti.f32, x_min: ti.f32 = -1.0, x_max: ti.f32 = 1.0, \
                          y_min: ti.f32 = -1.0, y_max: ti.f32 = 1.0, z_min: ti.f32 = -1.0, z_max: ti.f32 = 1.0):
        self.num_particles = num_particles
        self.density0 = density0
        self.support_radius = support_radius
        self.mass = mass

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

        self.num_x_cells = int(np.ceil((self.x_max - self.x_min)/self.support_radius))
        self.num_y_cells = int(np.ceil((self.y_max - self.y_min)/self.support_radius))
        self.num_z_cells = int(np.ceil((self.z_max - self.z_min)/self.support_radius))
        print((self.num_x_cells, self.num_y_cells, self.num_z_cells))

        print(type(self.num_x_cells))

        self.X = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_particles))
        self.V = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_particles))
        self.density = ti.field(dtype=ti.f32, shape=(self.num_particles))
        self.neighbors = ti.field(dtype=ti.i32, shape=(self.num_particles, self.num_particles))

        #I couldn't find a way to use an array as a dtype. Now, there is a maximal number of particles that can occupy any cell.
        self.max_particles_per_cell = int(32)
        self.grid = ti.field(dtype = ti.i32, shape = (self.num_x_cells, self.num_y_cells, self.num_z_cells, self.max_particles_per_cell))
        self.particles_in_cell = ti.field(dtype = ti.i32, shape = (self.num_x_cells, self.num_y_cells, self.num_z_cells))
                                                                    

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
    def update_grid(self):
        """places particle indices into the cell according to their position"""
        for i,j,k in self.particles_in_cell:
            self.particles_in_cell[i,j,k] = 0
        for i in range(self.num_particles):
            pos = self.X[i]
            x_cell = np.floor( (pos[0] - self.x_min) / (self.x_max - self.x_min) * self.num_x_cells)
            y_cell = np.floor( (pos[1] - self.y_min) / (self.y_max - self.y_min) * self.num_y_cells)
            z_cell = np.floor( (pos[2] - self.z_min) / (self.z_max - self.z_min) * self.num_z_cells)

            if x_cell >= self.num_x_cells:
                x_cell = self.num_x_cells - 1
            
            if y_cell >= self.num_y_cells:
                y_cell = self.num_y_cells - 1
            
            if z_cell >= self.num_z_cells:
                z_cell = self.num_z_cells - 1
                
            self.grid[x_cell, y_cell, z_cell, self.particles_in_cell[x_cell, y_cell, z_cell]] = i
            self.particles_in_cell[x_cell, y_cell, z_cell] += 1
        

    @ti.kernel
    def get_neighbors(self,i: ti.i32):
        pass
        
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