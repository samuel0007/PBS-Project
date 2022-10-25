import taichi as ti
import numpy as np
from .baseFluidModel import FluidModel
from .dfsph import DensityAndPressureSolver
from .viscosity2018 import ViscositySolver
from .akinciBoundary2012 import BoundaryModel


@ti.data_oriented
class Simulation:
    def __init__(self, num_particles: int, max_time: float, bounds: float, mass: ti.f32, is_frame_export=False, debug=False, result_dir="results/example/"):
        self.num_particles = num_particles
        self.max_time = max_time
        self.is_frame_export = is_frame_export
        self.dt = 1e-3
        self.current_time = 0.
        self.current_frame_id = 0
        self.time_since_last_frame_export = 0.

        self.fluid = FluidModel(
            num_particles=num_particles,
            density0=1000,
            support_radius=0.065,
            mass=mass
        )
        self.boundary = BoundaryModel(bounds, self.fluid.support_radius)
        self.boundary.compute_M(self.fluid.density0)
        self.boundary.expose()

        self.densityAndPressureSolver = DensityAndPressureSolver(num_particles, self.fluid.support_radius)
        self.viscositySolver = ViscositySolver(num_particles)

        self.non_pressure_forces = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_particles))

        self.debug = debug
        self.result_dir = result_dir

    def prolog(self):
        self.init_non_pressure_forces()
        self.fluid.set_boundary_particles(self.boundary.X, self.boundary.M)

        self.fluid.update_neighbors()
        self.fluid.update_density()

        self.densityAndPressureSolver.update_alpha_i(self.fluid.X, self.fluid.mass, self.fluid.density, self.fluid.f_neighbors, self.fluid.b_X, self.fluid.b_M, self.fluid.b_neighbors)
        print(self.densityAndPressureSolver.alpha_i)
        print(self.densityAndPressureSolver.densitySolver.alpha_i)
    def step(self):
        # Explicitly Apply non pressure forces
        self.apply_non_pressure_forces()

        # Adaptive time step
        # self.dt = self.fluid.CFL_condition()

        # Constant Density Solver
        self.fluid.V = self.densityAndPressureSolver.densitySolver.solve(self.fluid.V)
        self.fluid.explicit_update_position(self.dt)

        # Prepare Divergence Free Solver
        self.fluid.update_neighbors()
        self.fluid.update_density()
        self.densityAndPressureSolver.update_alpha_i(self.fluid.X, self.fluid.mass, self.fluid.density, self.fluid.f_neighbors, self.fluid.b_X, self.fluid.b_M, self.fluid.b_neighbors)

        # Divergence Free Solver
        self.fluid.V = self.densityAndPressureSolver.divergenceSolver.solve(self.fluid.V)

        # Implicit Viscosity Solver
        self.fluid.V = self.viscositySolver.solve(self.fluid.V)

        self.current_time += self.dt

        if self.debug:
            self.log_state()

    @ti.kernel
    def init_non_pressure_forces(self):
        for i in range(self.num_particles):
            self.non_pressure_forces[i] = ti.Vector([0., -9.8, 0.], ti.f32)

    @ti.kernel
    def apply_non_pressure_forces(self):
        for i in self.fluid.V:
            self.fluid.V[i] += self.dt * self.non_pressure_forces[i] / self.fluid.mass
        
        
    def run(self):
        self.prolog()
        while(self.current_time < self.max_time):
            self.step()
            if self.is_frame_export:
                self.time_since_last_frame_export += self.dt
                if self.time_since_last_frame_export > 1. / 60.:
                    self.frame_export()
                    self.time_since_last_frame_export = 0.
                    self.current_frame_id += 1
        self.postlog()
    
    def postlog(self):
        self.save()

    def log_state(self):
        print(f"Current Time: {self.current_time}, dt: {self.dt}, {self.time_since_last_frame_export}", end="\r")

    def frame_export(self):
        np.save(self.result_dir + f"frame_{self.current_frame_id}.npy", self.fluid.X.to_numpy())
        np.save(self.result_dir + f"frame_{self.current_frame_id}_B.npy", self.fluid.b_X.to_numpy())

    def save(self):
        np.save(self.result_dir + "results.npy", self.fluid.X.to_numpy())

    

