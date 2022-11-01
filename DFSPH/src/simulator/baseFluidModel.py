import numpy as np
import taichi as ti
from typing import Tuple
from .kernel import CubicSpline

@ti.data_oriented
class FluidModel:
    def __init__(self, num_particles: ti.i32, density0: ti.f32, support_radius: ti.f32, mass: ti.f32, x_min: ti.f32 = -1.5, x_max: ti.f32 = 1.5, \
                          y_min: ti.f32 = -1.5, y_max: ti.f32 = 1.5, z_min: ti.f32 = -1.5, z_max: ti.f32 = 1.5):
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


        #I couldn't find a way to use an array as a dtype. Now, there is a maximal number of particles that can occupy any cell.
        self.max_particles_per_cell = int(128)
        #grid_shape = (self.num_x_cells, self.num_y_cells, self.num_z_cells, self.max_particles_per_cell)
        #grid_snode = ti.root.dense(ti.ijk, grid_shape)
         
        self.grid = ti.field(dtype = ti.i32)
        self.grid_snode = ti.root.pointer(ti.ijk, (self.num_x_cells,self.num_y_cells,self.num_z_cells))
        self.grid_structure = self.grid_snode.dynamic(ti.l, self.max_particles_per_cell)
        self.grid_structure.place(self.grid)

        self.b_grid = ti.field(dtype = ti.i32)
        self.b_grid_snode = ti.root.pointer(ti.ijk, (self.num_x_cells,self.num_y_cells,self.num_z_cells))
        self.b_grid_structure = self.b_grid_snode.dynamic(ti.l, self.max_particles_per_cell)
        self.b_grid_structure.place(self.b_grid)

        
        #may not be needed
        #self.particles_in_cell = ti.field(dtype = ti.i32)
        #grid_snode.place(self.particles_in_cell)
        self.max_neighbors = int(64)

        self.neighbor_list = ti.field(dtype = ti.i32)
        self.neighbor_snode = ti.root.pointer(ti.i, self.num_particles)
        self.neighbor_structure = self.neighbor_snode.dynamic(ti.j, self.max_neighbors)
        self.neighbor_structure.place(self.neighbor_list)

        self.b_neighbor_list = ti.field(dtype = ti.i32)
        self.b_neighbor_snode = ti.root.pointer(ti.i, self.num_particles)
        self.b_neighbor_structure = self.b_neighbor_snode.dynamic(ti.j, self.max_neighbors)
        self.b_neighbor_structure.place(self.b_neighbor_list)
                                     
        print("Num Cells:", (self.num_x_cells,self.num_y_cells,self.num_z_cells))

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
        return min(5e-4, 0.4 * self.support_radius / max_V)

    @ti.kernel
    def explicit_update_position(self, dt: ti.f32):
        for i in range(self.num_particles):
            self.X[i] += self.V[i] * dt

    def update_neighbors(self):
        self.update_neighbors_kernel() # For now, until new method is implemented everywhere
        self.update_grid()
        self.update_neighbor_list()
        self.update_b_neighbor_list()

    def update_b_neighbors(self):
        self.update_b_grid()
        self.update_b_neighbor_list()

    # Dumb O(n^2) neighbor search, replace with grid based later
    @ti.kernel
    def update_neighbors_kernel(self):
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
            
    @ti.func
    def get_cell(self, pos: ti.types.vector(3, ti.f32)) -> Tuple[int, int, int]:
        x_cell = int( (pos[0] - self.x_min) / (self.x_max - self.x_min) * self.num_x_cells)
        y_cell = int( (pos[1] - self.y_min) / (self.y_max - self.y_min) * self.num_y_cells)
        z_cell = int( (pos[2] - self.z_min) / (self.z_max - self.z_min) * self.num_z_cells)

        if x_cell >= self.num_x_cells:
            x_cell = self.num_x_cells - 1
            
        if y_cell >= self.num_y_cells:
            y_cell = self.num_y_cells - 1
            
        if z_cell >= self.num_z_cells:
            z_cell = self.num_z_cells - 1

        cell = (x_cell,y_cell,z_cell)
        return cell

    @ti.kernel
    def update_grid(self):
        """places particle indices into the cell according to their position"""
        """
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
        """

        for i,j,k in self.grid_snode:
            ti.deactivate(self.grid_snode, [i,j,k])

        for i in range(self.num_particles):
            pos = self.X[i]
            cell = self.get_cell(pos)
            
            if not ti.is_active(self.grid_snode, cell):
                ti.activate(self.grid_snode, cell)
            
            ti.append(self.grid_structure, cell, i)

    @ti.kernel
    def update_b_grid(self):
        for i,j,k in self.b_grid_snode:
            ti.deactivate(self.b_grid_snode, [i,j,k])

        for i in range(self.b_num_particles):
            pos = self.b_X[i]
            cell = self.get_cell(pos)
            
            if not ti.is_active(self.b_grid_snode, cell):
                ti.activate(self.b_grid_snode, cell)
            
            ti.append(self.b_grid_structure, cell, i)

    @ti.kernel
    def update_neighbor_list(self):
        for i in range(self.num_particles):
            ti.deactivate(self.neighbor_snode, i)
            ti.activate(self.neighbor_snode, i)

        h2 = self.support_radius * self.support_radius
        for i in range(self.num_particles):
            pos_i = self.X[i]
            cell = self.get_cell(pos_i)
            (x_cell, y_cell, z_cell) = cell
            for x_n in range(max(0,x_cell-1),min(self.num_x_cells,x_cell+2)):
                for y_n in range(max(0,y_cell-1),max(self.num_y_cells,y_cell+2)):
                    for z_n in range(max(0,z_cell-1),max(self.num_z_cells,z_cell+2)):
                        for l in range(ti.length(self.grid_structure,[x_n,y_n,z_n])):
                            j = self.grid[x_n,y_n,z_n,l]
                            pos_j = self.X[j]
                            dist = pos_i - pos_j
                            d2 = dist.norm_sqr()
                            if d2 < h2:
                                ti.append(self.neighbor_structure, i, j)

            m1 = ti.length(self.neighbor_structure, i)
            m2 = self.f_number_of_neighbors[i]
            """
            if m1 != m2:
                print("f_neighbor count not the same in both methods:")
                print((i,m1,m2))
                print(pos_i)
            else:
                print("f_neighbor count is the same in both methods:")
                print((i,m1,m2))
            """
            

    @ti.kernel
    def update_b_neighbor_list(self):
        for i in range(self.num_particles):
            ti.deactivate(self.b_neighbor_snode, i)
            ti.activate(self.b_neighbor_snode, i)

        h2 = self.support_radius * self.support_radius
        for i in range(self.num_particles):
            pos_i = self.X[i]
            cell = self.get_cell(pos_i)
            (x_cell, y_cell, z_cell) = cell
            for x_n in range(max(0,x_cell-1),min(self.num_x_cells,x_cell+2)):
                for y_n in range(max(0,y_cell-1),max(self.num_y_cells,y_cell+2)):
                    for z_n in range(max(0,z_cell-1),max(self.num_z_cells,z_cell+2)):
                        for l in range(ti.length(self.b_grid_structure,[x_n,y_n,z_n])):
                            j = self.b_grid[x_n,y_n,z_n,l]
                            pos_j = self.b_X[j]
                            dist = pos_i - pos_j
                            d2 = dist.norm_sqr()
                            if d2 < h2:
                                ti.append(self.b_neighbor_structure, i, j)
            m1 = ti.length(self.b_neighbor_structure, i)
            m2 = self.b_number_of_neighbors[i]
            """
            if m1 != m2:
                print("b_neighbor count not the same in both methods:")
                print((i,m1,m2))
                print(pos_i)
            
            else:
                print("b_neighbor count is the same in both methods:")
                print((i,m1,m2)) 
            """              

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
                """
                for j in range(self.num_particles):
                    if self.f_neighbors[i, j] == 1:
                        density += self.mass * self.kernel.W(local_pos - self.X[j])
                for j in range(self.b_num_particles):
                    if self.b_neighbors[i, j] == 1:
                        density += self.b_M[j] * self.kernel.W(local_pos - self.b_X[j])
                """
                for l in range(ti.length(self.neighbor_structure,i)):
                    j = self.neighbor_list[i,l]
                    density += self.mass * self.kernel.W(local_pos - self.X[j])

                for l in range(ti.length(self.b_neighbor_structure,i)):
                    j = self.neighbor_list[i,l]
                    density += self.mass * self.kernel.W(local_pos - self.b_X[j])

            self.density[i] = density