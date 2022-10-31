import taichi as ti
import numpy as np
from .baseFluidModel import FluidModel
from .dfsph import DensityAndPressureSolver
from .viscosity2018 import ViscositySolver
from .akinciBoundary2012 import BoundaryModel


@ti.data_oriented
class Simulation:
    def __init__(self, num_particles: int, max_time: float, bounds: float, mass: ti.f32, support_radius: ti.f32, mu: ti.f32, is_frame_export=False, debug=False, result_dir="results/example/"):
        self.num_particles = num_particles
        self.max_time = max_time
        self.is_frame_export = is_frame_export
        self.dt = 1e-5
        self.current_time = 0.
        self.current_frame_id = 0
        self.time_since_last_frame_export = 0.

        self.bounds = bounds
        self.support_radius = support_radius
        self.rest_density = 1000
        self.mass = mass
        self.mu = mu

        self.radius = self.support_radius / 4

        self.fluid = FluidModel(
            num_particles=self.num_particles,
            density0=self.rest_density,
            support_radius=self.support_radius,
            mass=self.mass
        )
        self.boundary = BoundaryModel(self.bounds, self.fluid.support_radius)
        
        self.densityAndPressureSolver = DensityAndPressureSolver(num_particles, self.fluid.support_radius)
        self.viscositySolver = ViscositySolver(num_particles, self.mu, self.fluid.support_radius)

        self.non_pressure_forces = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_particles))
        self.number_of_neighbors = ti.field(ti.i32, self.num_particles)
        self.number_of_b_neighbors = ti.field(ti.i32, self.num_particles)

        self.debug = debug
        self.result_dir = result_dir

    def prolog(self):
        self.init_non_pressure_forces()
        self.set_initial_fluid_condition()

        self.boundary.compute_M(self.fluid.X, self.fluid.density0)
        self.boundary.expose()

        self.fluid.set_boundary_particles(self.boundary.X, self.boundary.M)

        #new
        self.fluid.update_b_grid()
        
        self.fluid.update_neighbors()
        self.fluid.update_density()
        print("B_neighbor count avg", np.average(self.fluid.b_number_of_neighbors.to_numpy()))

        self.densityAndPressureSolver.update_alpha_i(self.fluid.X, self.fluid.mass, self.fluid.density, self.fluid.f_neighbors, self.fluid.b_X, self.fluid.b_M, self.fluid.b_neighbors)
        np.save(self.result_dir + "boundary.npy", self.fluid.b_X.to_numpy())

    def step(self):
        # Explicitly Apply non pressure forces
        # compute non pressure forces
        # self.compute_non_pressure_forces()
        
        self.apply_non_pressure_forces()

        # Constant Density Solver
        # print("Original Speed Average: ", np.average(self.fluid.V.to_numpy()))
        self.pressure_solve, self.pressure_iteration = self.densityAndPressureSolver.densitySolver.solve(self.fluid, self.dt)
        # print("After solver Speed Average: ", np.average(self.fluid.V.to_numpy()))
        self.fluid.explicit_update_position(self.dt)
        # print("B_neighbor count avg", np.average(self.fluid.b_number_of_neighbors.to_numpy()))

        # Prepare Divergence Free Solver
        self.fluid.update_neighbors()

        #new
        self.fluid.update_grid()
        self.fluid.update_neighbor_list()
        self.fluid.update_b_neighbor_list()
        # print(self.fluid.number_of_neighbors)
        self.fluid.update_density()

        # self.boundary.compute_M(self.fluid.X, self.fluid.density0)
        # self.boundary.expose()

        self.densityAndPressureSolver.update_alpha_i(self.fluid.X, self.fluid.mass, self.fluid.density, self.fluid.f_neighbors, self.fluid.b_X, self.fluid.b_M, self.fluid.b_neighbors)

        # Divergence Free Solver
        self.divergence_solve, self.divergence_iteration = self.densityAndPressureSolver.divergenceSolver.solve(self.fluid, self.dt)

        # Implicit Viscosity Solver
        # print("Velocity Average: ", np.average(self.fluid.V.to_numpy()))
        self.viscosity_sucess = self.viscositySolver.solve(self.fluid, self.dt)
        # self.viscosity_sucess = 1
        # print("Velocity Average After Viscosity: ", np.average(self.fluid.V.to_numpy()))
      
        self.current_time += self.dt

        self.dt = self.fluid.CFL_condition()

        if self.debug:
            # pass
            self.log_state()

    @ti.kernel
    def init_non_pressure_forces(self):
        for i in range(self.num_particles):
            # self.non_pressure_forces[i] = ti.Vector([0., 0., 0.], ti.f32)
            self.non_pressure_forces[i] = ti.Vector([0.2, -9.8, 0.1], ti.f32)

    @ti.kernel
    def apply_non_pressure_forces(self):
        for i in self.fluid.V:
            self.fluid.V[i] += self.dt * self.non_pressure_forces[i] / self.fluid.mass

    @ti.kernel
    def set_initial_fluid_condition(self):  
        delta = self.support_radius / 2.
        num_particles_x = int(self.num_particles**(1. / 3.))
        offs = ti.Vector([(self.bounds - num_particles_x * delta) * 0.5, (self.bounds - num_particles_x * delta) * 0.1, (self.bounds - num_particles_x * delta) * 0.5], ti.f32)

        for i in range(num_particles_x):
            for j in range(num_particles_x):
                for k in range(num_particles_x):
                    self.fluid.X[i * num_particles_x * num_particles_x + j * num_particles_x + k] = ti.Vector([i, j, k], ti.f32) * delta + offs
                    # add velocity in z direction
                    self.fluid.V[i * num_particles_x * num_particles_x + j * num_particles_x + k] = ti.Vector([1., 0., 1.], ti.f32)

    def run(self):
        self.prolog()
        while(self.current_time < self.max_time):
            self.step()
            if self.is_frame_export:
                self.time_since_last_frame_export += self.dt
                if self.time_since_last_frame_export > 1e-3:
                    self.frame_export()
                    self.time_since_last_frame_export = 0.
                    self.current_frame_id += 1
        self.postlog()
    
    def postlog(self):
        self.save()

    def log_state(self):
        print(f"[T]:{self.current_time:.6f},[dt]:{self.dt},[B_cnt_avg]:{np.average(self.fluid.b_number_of_neighbors.to_numpy()):.1f},[d_avg]:{np.average(self.fluid.density.to_numpy()):.1f},[P_SOL]:{(self.pressure_solve):.1f},[P_I]:{self.pressure_iteration},[D_SOL]:{self.divergence_solve:1f},[D_I]:{self.divergence_iteration},[V]:{self.viscosity_sucess}", end="\r")

    def frame_export(self):
        np.save(self.result_dir + f"frame_{self.current_frame_id}.npy", self.fluid.X.to_numpy())
        np.save(self.result_dir + f"frame_density_{self.current_frame_id}.npy", self.fluid.density.to_numpy())

    def save(self):
        np.save(self.result_dir + "results.npy", self.fluid.X.to_numpy())

    

