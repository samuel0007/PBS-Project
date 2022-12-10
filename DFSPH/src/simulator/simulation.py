import taichi as ti
import numpy as np
import time
from .baseFluidModel import FluidModel
from .dfsph import DensityAndPressureSolver
from .viscosity2018 import ViscositySolver
from .akinciBoundary2012 import BoundaryModel
from .pointDatareader import readParticles
from. emitter import Emitter


@ti.data_oriented
class Simulation:
    def __init__(self, num_particles: int, max_time: float, max_dt: float, bounds: float, mass: ti.f32, rest_density: ti.f32, support_radius: ti.f32, mu: ti.f32, b_mu: ti.f32, is_frame_export=False, debug=False, result_dir="results/example/", pointData_file = "", boundary_pointData_file = "", is_uniform_export = False):
        # This is a bit of a nuisance, but the value couldn't be modified otherwise
        # Now, num_particles can be accessed as self.num_particles[None]
        self.num_particles = ti.field(ti.i32, shape = ())
        self.max_num_particles = 10000
        self.particle_array = np.array([])
        self.pointData_file = pointData_file
        self.emitter = Emitter(2., 0.2, 2., 0.1)

        if pointData_file == "":
            self.num_particles[None] = num_particles
        else:
            self.particle_array = readParticles(pointData_file)
            points = self.particle_array
            print("shape of array:")
            print(points.shape)
            x_coords = points[0:,0]
            print("x_range: ")
            print((min(x_coords),max(x_coords)))

            y_coords = points[0:,1]
            print("y_range: ")
            print((min(y_coords),max(y_coords)))

            z_coords = points[0:,2]
            print("z_range: ")
            print((min(z_coords),max(z_coords)))
            # append to the particle array itself
            # self.particle_array = np.append(self.particle_array, self.particle_array, axis=0)
            self.num_particles[None] = self.particle_array.shape[0]

        self.particle_field = ti.field(dtype = ti.f32, shape = (self.max_num_particles, 3))

        if pointData_file != "":
            if self.max_num_particles > self.num_particles[None]:
                padding_length = self.max_num_particles - self.num_particles[None]
                padding = np.zeros((padding_length, 3), dtype = float)
                print(padding.shape)
                self.particle_array = np.append(self.particle_array, padding, axis = 0)
            self.particle_field.from_numpy(self.particle_array)

        self.max_time = max_time
        self.is_frame_export = is_frame_export
        self.max_dt = max_dt
        self.dt = self.max_dt
        self.current_time = 0.
        self.time_since_last_emit = 0.
        self.current_frame_id = 0
        self.time_since_last_frame_export = 0.

        self.bounds = bounds
        self.support_radius = support_radius
        self.rest_density = rest_density
        self.mass = mass
        self.mu = mu
        self.b_mu = b_mu

        self.radius = self.support_radius / 4

        self.fluid = FluidModel(
            num_particles=self.num_particles[None],
            max_num_particles=self.max_num_particles,
            max_dt=self.max_dt,
            density0=self.rest_density,
            support_radius=self.support_radius,
            mass=self.mass,
            x_min=0,
            x_max=self.bounds + 2,
            y_min=0,
            y_max=self.bounds*2,
            z_min=0,
            z_max=self.bounds,
        )
        self.boundary = BoundaryModel(self.bounds, self.fluid.support_radius, pointData_file = boundary_pointData_file)
        
        # may need self.max_num_particles
        self.densityAndPressureSolver = DensityAndPressureSolver(self.num_particles[None], self.max_num_particles, self.fluid)
        self.viscositySolver = ViscositySolver(self.num_particles[None], self.max_num_particles, self.mu, self.b_mu, self.fluid)

        self.non_pressure_forces = ti.Vector.field(3, dtype=ti.f32, shape=(self.max_num_particles))

        self.debug = debug
        self.result_dir = result_dir
        self.is_uniform_export = is_uniform_export

    def should_emit(self):
        return True

    def should_emit(self):
        return True

    def prolog(self):
        self.init_non_pressure_forces()
        self.set_initial_fluid_condition()

        self.boundary.compute_M(self.fluid.density0)

        self.fluid.set_boundary_particles(self.boundary.X, self.boundary.M)

        self.fluid.update_neighbors()
        self.fluid.update_b_neighbors()

        self.fluid.update_density()

        self.densityAndPressureSolver.update_alpha_i(self.fluid.X, self.fluid.mass, self.fluid.density, self.fluid.f_neighbors, self.fluid.b_X, self.fluid.b_M, self.fluid.b_neighbors)
       
        # Print initial density
        print("Initial Density Average: ", self.compute_field_average(self.fluid.density))
        print("Initial Density Max: ", self.compute_field_max(self.fluid.density))
        print("Initial Density Min: ", self.compute_field_min(self.fluid.density))
        print("Initial F_neighbors_avg", self.fluid.get_num_neighbors_avg())
        print("Initial B_neighbors_avg", self.fluid.get_num_b_neighbors_avg())
        # Print number of active particles
        print("Number of active particles: ", self.fluid.get_num_active_particles())

        np.save(self.result_dir + "boundary.npy", self.fluid.b_X.to_numpy())

        if self.is_uniform_export:
            self.fluid.generate_uniform_pos()
            np.save(self.result_dir + "uniform_pos.npy", self.fluid.uniform_pos.to_numpy())

    def step(self):
        # Explicitly Apply non pressure forces
        # compute non pressure forces
        # self.compute_non_pressure_forces()
        
        self.apply_non_pressure_forces()

        # Print max, min, avg of alpha_i of fluid
        # print("Alpha_i Max: ", self.compute_field_max(self.densityAndPressureSolver.alpha_i))
        # print("Alpha_i Min: ", self.compute_field_min(self.densityAndPressureSolver.alpha_i))
        # print("Alpha_i Average: ", self.compute_field_average(self.densityAndPressureSolver.alpha_i))

        # Constant Density Solver
        # print("Original Speed Average: ", np.average(self.fluid.V.to_numpy()))
        self.pressure_solve, self.pressure_iteration = self.densityAndPressureSolver.densitySolver.solve(self.fluid, self.dt)
        # print("After solver Speed Average: ", np.average(self.fluid.V.to_numpy()))
        self.fluid.explicit_update_position(self.dt)

        if self.should_emit():
            self.fluid.insert_particle(self.emitter.get_particle())            
            self.viscositySolver.increase_particles()
            self.densityAndPressureSolver.increase_particles()


        # Prepare Divergence Free Solver
        self.fluid.update_neighbors()
        self.fluid.update_density()

        # Print density
        # print("Density Average: ", self.compute_field_average(self.fluid.density))
        # print("Density Max: ", self.compute_field_max(self.fluid.density))
        # print("Density Min: ", self.compute_field_min(self.fluid.density))

        # Print speed
        # print("Speed Average: ", self.compute_field_norm_average(self.fluid.V))
        # print("Speed Max: ", self.compute_field_norm_max(self.fluid.V))
        # print("Speed Min: ", self.compute_field_norm_min(self.fluid.V))

        # self.boundary.compute_M(self.fluid.density0)

        self.densityAndPressureSolver.update_alpha_i(self.fluid.X, self.fluid.mass, self.fluid.density, self.fluid.f_neighbors, self.fluid.b_X, self.fluid.b_M, self.fluid.b_neighbors)

        # Divergence Free Solver
        self.divergence_solve, self.divergence_iteration = self.densityAndPressureSolver.divergenceSolver.solve(self.fluid, self.dt)

        # Implicit Viscosity Solver
        # print("Velocity Average: ", np.average(self.fluid.V.to_numpy()))
        self.viscosity_sucess = self.viscositySolver.solve(self.dt)
        # self.viscosity_sucess = 1
        # print("Velocity Average After Viscosity: ", np.average(self.fluid.V.to_numpy()))
      
        self.current_time += self.dt

        self.dt = self.fluid.CFL_condition()

        if self.debug:
            # pass
            self.log_state()

    @ti.kernel
    def init_non_pressure_forces(self):
        # this should probebly be self.max_num_particles 
        for i in range(self.max_num_particles):
            self.non_pressure_forces[i] = ti.Vector([0., -9.81, 0.])

    @ti.kernel
    def apply_non_pressure_forces(self):
        for i in self.fluid.V:
            if not self.fluid.active[i]: continue
            self.fluid.V[i] += self.dt * self.non_pressure_forces[i] / self.fluid.mass

    @ti.kernel
    def set_initial_fluid_condition(self):  
        make_grid = (self.pointData_file == "")
        if make_grid:
            delta = self.support_radius / 2.
            num_particles_x = int(self.num_particles[None]**(1. / 3.)) + 1
            offs = ti.Vector([2.72, 3.2, 1]) 
            # offs = ti.Vector([(self.bounds - num_particles_x * delta) * 0.5, (self.bounds - num_particles_x * delta) * 0.05, (self.bounds - num_particles_x * delta) * 0.5], ti.f32)
            for i in range(num_particles_x):
                for j in range(num_particles_x):
                    for k in range(num_particles_x):
                        self.fluid.X[i * num_particles_x * num_particles_x + j * num_particles_x + k] = ti.Vector([i, j, k], ti.f32) * delta + offs
                        # add velocity in z direction
                        # self.fluid.V[i * num_particles_x * num_particles_x + j * num_particles_x + k] = ti.Vector([10., 0., 10.], ti.f32)
        else:
            offset_1 = ti.Vector([0.,0.,0.],ti.f32)
            for i in range(self.num_particles[None]):
                x = self.particle_field[i,0]
                y = self.particle_field[i,1]
                z = self.particle_field[i,2]
                self.fluid.X[i] = ti.Vector([x,y,z],ti.f32) + offset_1
                # self.fluid.V[i] = ti.Vector([0.,50.,0.],ti.f32)

            # offset_2 = ti.Vector([2.,1.3,2.],ti.f32)
            # for i in range(self.num_particles[None]/2, self.num_particles[None]):
            #     x = self.particle_field[i,0]
            #     y = self.particle_field[i,1]
            #     z = self.particle_field[i,2]
            #     self.fluid.X[i] = ti.Vector([x,y,z],ti.f32) + offset_2
            #     self.fluid.V[i] = ti.Vector([0.,-50.,0.],ti.f32)

    def run(self):
        self.prolog()
        while(self.current_time < self.max_time):
            self.step()
            if self.is_frame_export:
                self.time_since_last_frame_export += self.dt
                if self.time_since_last_frame_export >= self.max_dt:
                    self.frame_export()
                    self.time_since_last_frame_export = 0.
                    self.current_frame_id += 1
        self.postlog()
    
    def postlog(self):
        self.save()


    @ti.kernel
    def compute_field_average(self, field: ti.template()) -> ti.f32:
        average = 0.
        count = 0
        for i in range(self.num_particles[None]):
            if not self.fluid.active[i]: continue
            average += field[i]
            count += 1
        return average / count

    @ti.kernel
    def compute_field_max(self, field: ti.template()) -> ti.f32:
        max = 0.
        for i in range(self.num_particles[None]):
            if not self.fluid.active[i]: continue
            if field[i] > max:
                max = field[i]
        return max

    @ti.kernel
    def compute_field_min(self, field: ti.template()) -> ti.f32:
        min = 1e10
        for i in range(self.num_particles[None]):
            if not self.fluid.active[i]: continue
            if field[i] < min:
                min = field[i]
        return min

    @ti.kernel
    def compute_field_norm_average(self, field: ti.template()) -> ti.f32:
        average = 0.
        count = 0
        for i in range(self.num_particles[None]):
            if not self.fluid.active[i]: continue
            average += field[i].norm()
            count += 1
        return average / count

    @ti.kernel
    def compute_field_norm_max(self, field: ti.template()) -> ti.f32:
        max = 0.
        for i in range(self.num_particles[None]):
            if not self.fluid.active[i]: continue
            if field[i].norm() > max:
                max = field[i].norm()
        return max
    
    @ti.kernel
    def compute_field_norm_min(self, field: ti.template()) -> ti.f32:
        min = 1e10
        for i in range(self.num_particles[None]):
            if not self.fluid.active[i]: continue
            if field[i].norm() < min:
                min = field[i].norm()
        return min

    @ti.kernel
    def compare_adj_matrix(self) -> ti.i32:
        value = 0
        for i in range(self.num_particles[None]):
            num_neighbors_i = self.fluid.get_num_neighbors_i(i)
            for j in range(self.num_particles[None]):
                if self.fluid.f_neighbors[i, j] == 1:
                    check = False
                    for l in range(num_neighbors_i):
                        if self.fluid.neighbor_list[i, l] == j:
                            check = True
                            break
                    if not check:
                        print("particle", i, "has neighbor", j, "but not in neighbor list")
                        # print particle position
                        print("particle position", self.fluid.X[i])
                        value = 1
        return value

    def log_state(self):
        d_avg = self.compute_field_average(self.fluid.density)
        B_cnt_avg = self.fluid.get_num_b_neighbors_avg()
        F_cnt_avg = self.fluid.get_num_neighbors_avg()
        num_active_particles = self.fluid.get_num_active_particles()

        print(f"[T]:{self.current_time:.6f},[dt]:{self.dt},[B_cnt_avg]:{B_cnt_avg:.1f},[F_cnt_avg]:{F_cnt_avg:.1f},[d_avg]:{d_avg:.1f},[cnt]:{num_active_particles}", end="\r")

        # print(f"[T]:{self.current_time:.6f},[dt]:{self.dt},[B_cnt_avg]:{B_cnt_avg:.1f},[F_cnt_avg]:{F_cnt_avg:.1f},[d_avg]:{d_avg:.1f},[P_SOL]:{(self.pressure_solve):.1f},[P_I]:{self.pressure_iteration},[D_SOL]:{self.divergence_solve:1f},[D_I]:{self.divergence_iteration},[V]:{self.viscosity_sucess}", end="\r")

    def frame_export(self):
        # no need to store inactive particles
        # (this wasn't done from the start, so some exported frames still have a bunch of inactive particles)
        np.save(self.result_dir + f"frame_{self.current_frame_id}.npy", self.fluid.X.to_numpy()[:self.num_particles[None],])
        np.save(self.result_dir + f"frame_density_{self.current_frame_id}.npy", self.fluid.density.to_numpy()[:self.num_particles[None],])
        if self.is_uniform_export:
            self.fluid.compute_uniform_field()
            np.save(self.result_dir + f"frame_uniform_{self.current_frame_id}.npy", self.fluid.uniform_field.to_numpy())


    def save(self):
        np.save(self.result_dir + "results.npy", self.fluid.X.to_numpy())

    

