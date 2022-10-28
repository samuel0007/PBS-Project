import taichi as ti
from .kernel import CubicSpline
from .baseFluidModel import FluidModel

@ti.data_oriented
class ViscositySolver:
    def __init__(self, num_particles: ti.i32, mu: ti.f32, support_radius: ti.f32):
        self.num_particles = num_particles
        self.support_radius = support_radius
        self.mu = mu
        self.eps = 1e-2
        self.eps2 = 1e-5
        self.d = 10.
        self.kernel = CubicSpline(support_radius)
        self.row_contribution = ti.field(dtype=ti.f32, shape=(self.num_particles))


    def solve(self, fluid: FluidModel, dt: ti.f32):
        matrix_entries = self.count_matrix_entries(fluid.f_number_of_neighbors)
        
        A_builder = ti.linalg.SparseMatrixBuilder(self.num_particles*3, self.num_particles*3, max_num_triplets=matrix_entries)

        self.fill_A(fluid.mass, fluid.density, fluid.X, fluid.f_neighbors, dt, A_builder)
        A = A_builder.build()

        b = fluid.V.to_numpy().reshape(-1)
        # self.build_RHS(fluid.V, b)

        # Solve sparse linear system, use ldlt as matrix is spd
        # Should be changed to a ConjugateGradient solver for efficiency... but for now this is not supported in Taichi
        # One then may want to directly call Eigen...

        solver = ti.linalg.SparseSolver(solver_type="LDLT")

        solver.compute(A)
        x = solver.solve(b)
        isSuccess = solver.info()
        if isSuccess: fluid.V.from_numpy(x.reshape(-1, 3))
        else: print("Viscosity solver failed")
        return isSuccess


    @ti.kernel
    def count_matrix_entries(self, f_number_of_neighbors: ti.template()) -> ti.i32:
        count = 0
        for i in range(self.num_particles):
            count += f_number_of_neighbors[i]
        # For each neigbor of a particle there is a 3x3 matrix block in the A matrix
        # Also add diagonal entries
        return count*9 + self.num_particles*3

    @ti.func
    def copy_V_to_b(self, V: ti.template(), b: ti.template()):
        for i in range(self.num_particles):
            for d in ti.static(range(3)):
                b[i*3+d] = V[i][d]

    @ti.kernel
    def build_RHS(self, V: ti.template(), b: ti.template()):
        self.copy_V_to_b(V, b)

        # Here on could take into account boundary viscosity

    @ti.kernel
    def fill_A(self, f_M: ti.f32, density: ti.template(), X: ti.template(), f_neighbors: ti.template(), dt: ti.f32, A_builder: ti.types.sparse_matrix_builder()):
        for i in range(self.num_particles):
            local_X = X[i]
            local_density = density[i]
            for j in range(self.num_particles):
                if f_neighbors[i, j] == 1:
                    x_ij = local_X - X[j]
                    norm_x_ij = x_ij.norm_sqr()
                    if norm_x_ij < self.eps2:
                        continue

                    factor = self.d*self.mu*f_M/(local_density*density[j])
                    if factor < self.eps2:
                        continue
                    grad_i = self.kernel.W_grad(x_ij)

                    # Tensor Product of grad_i and x_ij
                    for d in ti.static(range(3)):
                        for e in ti.static(range(3)):
                            A_builder[i*3+d, j*3+e] += dt*factor*grad_i[d]*x_ij[e]/(norm_x_ij + self.eps*dt*dt)

        # Add Identity on the diagonal
        for i in range(3*self.num_particles):
            A_builder[i, i] += 1.

