import taichi as ti
from baseFluidModel import FluidModel
from dfsph import DensityAndPressureSolver
from viscosity2018 import ViscositySolver
from akinciBoundary2012 import BoundaryModel


@ti.data_oriented
class Simulation:
    def __init__(self, num_particles: int, max_time: float, bounds: float, frame_export=False, debug=False):
        self.num_particles = num_particles
        self.max_time = max_time
        self.frame_export = frame_export
        self.dt = 1e-3
        self.current_time = 0.

        self.fluid = FluidModel(
            num_particles=num_particles,
            density0=1000,
            support_radius=0.065,
            mass=0.1
        )
        self.boundary = BoundaryModel(bounds, self.fluid.support_radius)
        self.boundary.compute_mass(self.fluid.density0)

        self.densityAndPressureSolver = DensityAndPressureSolver(num_particles)
        self.viscositySolver = ViscositySolver(num_particles)

        self.non_pressure_forces = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_particles))

        self.debug = debug

    def prolog(self):
        self.init_non_pressure_forces()
        self.densityAndPressureSolver.update_alpha_i()

    def step(self):
        # Explicitly Apply non pressure forces
        self.apply_non_pressure_forces()

        # Adaptive time step
        self.dt = self.fluid.CFL_condition()

        # Constant Density Solver
        self.fluid.V = self.densityAndPressureSolver.densitySolver.solve(self.fluid.V)
        self.fluid.explicit_update_position(self.dt)

        # Prepare Divergence Free Solver
        self.fluid.update_neighbors()
        self.fluid.update_density()
        self.densityAndPressureSolver.update_alpha_i()

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
            if self.frame_export:
                self.frame_export()
    
    def log_state(self):
        print(f"Current Time: {self.current_time}, dt: {self.dt}", end="\r")

    def frame_export(self):
        pass

    def save(self, filename):
        pass

