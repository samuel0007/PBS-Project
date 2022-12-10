import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve
import taichi as ti
from .kernel import CubicSpline
from .baseFluidModel import FluidModel
from typing import List, Tuple


@ti.data_oriented
class ViscositySolver:
    def __init__(self, num_particles: ti.i32, max_num_particles: ti.i32, mu: ti.f32, b_mu: ti.f32, fluid: FluidModel):
        self.num_particles = ti.field(ti.i32, shape = ())
        self.num_particles[None] = num_particles
        self.max_num_particles = max_num_particles
        self.fluid = fluid
        self.support_radius = self.fluid.support_radius
        self.mu = mu
        self.b_mu = b_mu
        self.eps = 1e-2
        self.eps2 = 1e-5
        self.d = 10.
        self.kernel = CubicSpline(self.fluid.support_radius)
        # I assume
        self.row_contribution = ti.field(dtype=ti.f32, shape=(self.max_num_particles))
        self.matrix_entries = ti.i32
        self.m_matrix_entries = ti.i32

    # For now I gave up on implementing the conjugate gradient solver, it somehow doesn't converge...
    # I think it would greatly improve the performance
    def solve(self, dt: ti.f32):
        matrix_entries = self.count_matrix_entries()
        # m_matrix_entries = self.count_m_matrix_entries()
        
        A_builder = ti.linalg.SparseMatrixBuilder(self.num_particles[None]*3, self.num_particles[None]*3, max_num_triplets=matrix_entries)
        # A_triplets = ti.Vector.field(3, dtype=ti.f32, shape=(matrix_entries))
        # M_triplets = ti.Vector.field(3, dtype=ti.f32, shape=(m_matrix_entries))

        # self.fill_A(self.fluid.mass, self.fluid.density, self.fluid.X, self.fluid.f_neighbors, dt, A_builder, A_triplets, M_triplets)
        self.fill_A(self.fluid.mass, self.fluid.density, self.fluid.X, self.fluid.f_neighbors, dt, A_builder)
        A = A_builder.build()

        # A_triplets = A_triplets.to_numpy()
        # data = A_triplets[:, 2]
        # row_idx = A_triplets[:, 0]
        # col_idx = A_triplets[:, 1]
        # A_scipy = coo_matrix((data, (row_idx, col_idx)), shape=(self.num_particles[None]*3, self.num_particles[None]*3)).tocsc()

        # M_triplets = M_triplets.to_numpy()
        # data = M_triplets[:, 2]
        # row_idx = M_triplets[:, 0]
        # col_idx = M_triplets[:, 1]
        # M_scipy = coo_matrix((data, (row_idx, col_idx)), shape=(self.num_particles[None]*3, self.num_particles[None]*3)).tocsc()
        
        # b_field = ti.field(dtype=ti.f32, shape=(self.num_particles[None], 3))
        b = self.fluid.V.to_numpy().reshape(-1)[0:self.num_particles[None] * 3]

        # Solve sparse linear system using scipy conjugate gradient
        # x_, info = cg(A_scipy, b, M=M_scipy)
        # preconditioner = spsolve(M_scipy, np.eye(self.num_particles[None]*3))
        # x_, info = cg(A_scipy, b, M=M_scipy)
        # print(info)

        # self.build_RHS(fluid.V, b)

        # Solve sparse linear system, use ldlt as matrix is spd
        # Should be changed to a ConjugateGradient solver for efficiency... but for now this is not supported in Taichi
        # One then may want to directly call Eigen...

        solver = ti.linalg.SparseSolver(solver_type="LDLT")

        solver.compute(A)
        x = solver.solve(b)
        isSuccess = solver.info()
        # print("Diff: ", np.linalg.norm(x-x_))
        if isSuccess: 
            padding_length = (self.max_num_particles - self.num_particles[None]) * 3
            padding = np.zeros(padding_length, dtype = float)
            x = np.append(x, padding)
            self.fluid.V.from_numpy(x.reshape(-1, 3))
        else: print("Viscosity solver failed")
        return isSuccess

    @ti.kernel
    def increase_particles(self):
        if self.num_particles[None] < self.max_num_particles:
            self.num_particles[None] = self.num_particles[None] + 1
        # print("increased particles in viscosity solver")   
            
    @ti.kernel
    def count_matrix_entries(self) -> ti.i32:
        count = 0
        for i in range(self.num_particles[None]):
            if not self.fluid.active[i]: continue
            count += self.fluid.get_num_neighbors_i(i)
        # For each neigbor of a particle there is a 3x3 matrix block in the A matrix
        # Also add diagonal entries
        return count*9 + self.num_particles[None]*3
    
    @ti.kernel
    def count_m_matrix_entries(self) -> ti.i32:
        count = 0
        for i in range(self.num_particles[None]):
            if not self.fluid.active[i]: continue
            count += 9
        return count + self.num_particles[None]*3

    @ti.func
    def copy_V_to_b(self, V: ti.template(), b: ti.template()):
        for i in range(self.num_particles[None]):
            if not self.fluid.active[i]:
                for d in ti.static(range(3)):
                    b[i*3+d] = 0.
                continue

            for d in ti.static(range(3)):
                b[i*3+d] = V[i][d]

    @ti.kernel
    def build_RHS(self, V: ti.template(), b: ti.template()):
        self.copy_V_to_b(V, b)

        # num_neighbors_b_i = self.fluid.get_num_b_neighbors_i(i)
          
        # for l in range(num_neighbors_b_i):
        #     k = self.fluid.b_neighbor_list[i,l]
        #     x_ik = local_X - X[k]

        # Here on could take into account boundary viscosity

    # def fill_A(self, f_M: ti.f32, density: ti.template(), X: ti.template(), f_neighbors: ti.template(), dt: ti.f32, A_builder: ti.types.sparse_matrix_builder(), A_triplets: ti.template(), M_triplets: ti.template()):
    @ti.kernel
    def fill_A(self, f_M: ti.f32, density: ti.template(), X: ti.template(), f_neighbors: ti.template(), dt: ti.f32, A_builder: ti.types.sparse_matrix_builder()):
        # triplet_idx = 0
        # m_triplet_idx = 0
        # m_block = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        for i in range(self.num_particles[None]):
            if not self.fluid.active[i]: continue

            local_X = X[i]
            local_density = density[i]

            sum_j = ti.Matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

            num_neighbors_i = self.fluid.get_num_neighbors_i(i)
            for l in range(num_neighbors_i):
                j = self.fluid.neighbor_list[i,l]
                if i == j: continue
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
                        value = dt*factor*grad_i[d]*x_ij[e]/(norm_x_ij)
                        A_builder[i*3+d, j*3+e] += value
                        # A_triplets[triplet_idx] = [i*3+d, j*3+e, value]
                        # triplet_idx += 1
                        sum_j[d, e] += value

            # A_[i, i] =  sum_j A_[i, j]
            # A[i, i] = I - dt*sum_j
            # Boundary contribution
            b_block = ti.Matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
            local_density2 = local_density*local_density
            num_b_neighbors_i = self.fluid.get_num_b_neighbors_i(i)
            for l in range(num_b_neighbors_i):
                k = self.fluid.b_neighbor_list[i,l]
                x_ik = local_X - self.fluid.b_X[k]
                norm_x_ik = x_ik.norm_sqr()
                if norm_x_ik < self.eps2:
                    continue

                factor = self.d*self.b_mu*self.fluid.mass/(local_density2)
                if factor < self.eps2:
                    continue
                grad_ik = self.kernel.W_grad(x_ik)

                # Tensor Product of grad_ik and x_ik
                for d in ti.static(range(3)):
                    for e in ti.static(range(3)):
                        b_block[d, e] += dt*factor*grad_ik[d]*x_ik[e]/(norm_x_ik)

            for d in ti.static(range(3)):
                for e in ti.static(range(3)):
                    A_builder[i*3+d, i*3+e] -= (sum_j[d, e] + b_block[d, e])
                    # A_triplets[triplet_idx] = [i*3+d, i*3+e, -sum_j[d, e]]
                    # triplet_idx += 1
                    # m_block[d, e] -= sum_j[d, e]


            # Inverse the m matrix
            # m_block_inv = m_block.inverse()
            # add it to the M matrix
            # for d in ti.static(range(3)):
                # for e in ti.static(range(3)):
                    # M_triplets[m_triplet_idx] = [i*3+d, i*3+e, m_block_inv[d, e]]
                    # m_triplet_idx += 1

        # Add Identity on the diagonal
        # we still have to do it for the inactive particles to be sure to have a spd matrix
        for i in range(3*self.num_particles[None]):
            A_builder[i, i] += 1.
            # A_triplets[triplet_idx] = [i, i, 1.]
            # triplet_idx += 1

        
        
           

