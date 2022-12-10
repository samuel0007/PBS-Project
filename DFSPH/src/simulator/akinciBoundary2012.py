import taichi as ti
from .kernel import CubicSpline
from.pointDatareader import readParticles
import numpy as np

@ti.data_oriented
class BoundaryModel:
    def __init__(self, bounds: ti.f32, support_radius: ti.f32, pointData_file = ""):
        self.pointData_file = pointData_file
        self.bounds = bounds
        self.support_radius = support_radius
        self.resolution = self.support_radius / 4
        # self.resolution = resolution
        self.num_particles_per_axis = int(bounds / self.resolution)
        self.num_particles_per_face = self.num_particles_per_axis *self.num_particles_per_axis 
        # self.num_particles = 6 * self.num_particles_per_face

        # self.m_X = ti.Vector.field(3, dtype=ti.f32, shape=(6, self.num_particles_per_axis, self.num_particles_per_axis))
        # self.m_M = ti.field(ti.f32, shape=(6, self.num_particles_per_axis, self.num_particles_per_axis))
        self.num_particles = 0
        self.particle_array = np.array([])

        if self.pointData_file == "":
            self.num_particles = 5 * self.num_particles_per_face
        else:
            self.particle_array = readParticles(self.pointData_file)
            self.num_particles = self.particle_array.shape[0]

        self.particle_field = ti.field(dtype = ti.f32, shape = (self.num_particles, 3))

        if pointData_file != "":
            self.particle_field.from_numpy(self.particle_array)

        self.m_X = ti.Vector.field(3, dtype=ti.f32, shape=(5, self.num_particles_per_axis, self.num_particles_per_axis))
        self.m_M = ti.field(ti.f32, shape=(5, self.num_particles_per_axis, self.num_particles_per_axis))
       
        self.X = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_particles))
        self.M = ti.field(ti.f32, shape=(self.num_particles))

        self.kernel = CubicSpline(self.support_radius)
        self.generate_boundary_particles()

    @ti.kernel
    def generate_boundary_particles(self):
        # Generate particles on the boundary
        if self.pointData_file == "":
            for face, x, y in self.m_X:
                if face == 0:
                    self.m_X[face, x, y] = [x * self.resolution, y * self.resolution, 0]
                elif face == 1:
                    self.m_X[face, x, y] = [x * self.resolution, y * self.resolution, self.bounds]
                elif face == 2:
                    self.m_X[face, x, y] = [0, x * self.resolution, y * self.resolution]
                elif face == 3:
                    self.m_X[face, x, y] = [self.bounds, x * self.resolution, y * self.resolution]
                elif face == 4:
                    self.m_X[face, x, y] = [x * self.resolution, 0, y * self.resolution]
                # elif face == 5:
                #     self.m_X[face, x, y] = [x * self.resolution, self.bounds, y * self.resolution]
        else:
            for i in range(self.num_particles):
                x = self.particle_field[i,0]
                y = self.particle_field[i,1]
                z = self.particle_field[i,2]
                self.X[i] = ti.Vector([x,y,z],ti.f32)
            

    def compute_M(self, density0: ti.f32):
        if self.pointData_file == "":
            self.compute_M_kernel(density0)
            self.X.from_numpy(self.m_X.to_numpy().reshape(-1, 3))
            self.M.from_numpy(self.m_M.to_numpy().reshape(-1))
        else:
            self.compute_M_kernel(density0)

    # Compute mass of each boundary particle, O(n^2) for now, but is only used at init.
    # Should be optimized when rest density of fluid is not constant
    @ti.kernel
    def compute_M_kernel(self, density0: ti.f32):
        coefficient = 10
        if self.pointData_file == "":
            for face, x, y in self.m_X:
                denom = 0.
                local_pos = self.m_X[face, x, y]
                for other_face in range(5):
                # for other_face in range(6):
                    for other_x in range(self.num_particles_per_axis):
                        for other_y in range(self.num_particles_per_axis):
                            other_pos = self.m_X[other_face, other_x, other_y]
                            denom += self.kernel.W(local_pos - other_pos)
                
                self.m_M[face, x, y] = (density0 / denom) * coefficient
        else:
            for i in range(self.num_particles):
                local_pos = self.X[i]
                denom = 0.
                for j in range(self.num_particles):
                    other_pos = self.X[j]
                    denom += self.kernel.W(local_pos - other_pos)
                self.M[i] = (density0/denom) * coefficient

