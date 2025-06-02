Numerical Methods
=================

This section details the numerical methods and algorithms implemented in SemiDGFEM for solving semiconductor device equations.

.. contents::
   :local:
   :depth: 3

Time Integration Schemes
-----------------------

Backward Euler Method
~~~~~~~~~~~~~~~~~~~~

For the time-dependent drift-diffusion equations:

.. math::
   \frac{\partial n}{\partial t} = F_n(n, p, \phi)

The backward Euler scheme is:

.. math::
   \frac{n^{k+1} - n^k}{\Delta t} = F_n(n^{k+1}, p^{k+1}, \phi^{k+1})

**Advantages:**
- Unconditionally stable
- Suitable for stiff problems
- Good for steady-state convergence

**Implementation:**

.. code-block:: cpp

   void BackwardEuler::solve_timestep(double dt) {
       // Newton iteration for implicit system
       for (int iter = 0; iter < max_iter; ++iter) {
           assemble_jacobian(dt);
           solve_linear_system();
           update_solution();
           
           if (check_convergence()) break;
       }
   }

BDF2 (Second-Order Backward Differentiation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For higher accuracy:

.. math::
   \frac{3n^{k+1} - 4n^k + n^{k-1}}{2\Delta t} = F_n(n^{k+1}, p^{k+1}, \phi^{k+1})

**Advantages:**
- Second-order accuracy
- A-stable for linear problems
- Better accuracy for transient simulations

Adaptive Time Stepping
~~~~~~~~~~~~~~~~~~~~~

**Error Control:**

.. math::
   \text{err} = \frac{\|n^{k+1}_{fine} - n^{k+1}_{coarse}\|}{\|n^{k+1}_{fine}\|}

**Step Size Control:**

.. math::
   \Delta t_{new} = \Delta t_{old} \left(\frac{\text{tol}}{\text{err}}\right)^{1/(p+1)}

where :math:`p` is the order of the method.

Nonlinear Solvers
-----------------

Newton-Raphson Method
~~~~~~~~~~~~~~~~~~~

For the nonlinear system :math:`\mathbf{F}(\mathbf{u}) = 0`:

.. math::
   \mathbf{J}^k \Delta \mathbf{u}^k = -\mathbf{F}(\mathbf{u}^k)

.. math::
   \mathbf{u}^{k+1} = \mathbf{u}^k + \Delta \mathbf{u}^k

**Jacobian Matrix:**

.. math::
   J_{ij} = \frac{\partial F_i}{\partial u_j}

**Convergence Criterion:**

.. math::
   \|\mathbf{F}(\mathbf{u}^k)\| < \epsilon_{abs} + \epsilon_{rel} \|\mathbf{u}^k\|

**Implementation:**

.. code-block:: cpp

   bool NewtonSolver::solve(Vector& solution) {
       for (int iter = 0; iter < max_iterations; ++iter) {
           // Assemble residual and Jacobian
           assemble_residual(solution, residual);
           assemble_jacobian(solution, jacobian);
           
           // Solve linear system
           linear_solver.solve(jacobian, delta_u, -residual);
           
           // Line search for robustness
           double alpha = line_search(solution, delta_u);
           solution += alpha * delta_u;
           
           // Check convergence
           if (residual.norm() < tolerance) return true;
       }
       return false;
   }

Gummel Iteration
~~~~~~~~~~~~~~~

Decoupled solution of the semiconductor equations:

**Algorithm:**

1. **Solve Poisson:** :math:`-\nabla \cdot (\epsilon \nabla \phi^{k+1}) = \rho(n^k, p^k)`
2. **Solve Electron:** :math:`\nabla \cdot \mathbf{J}_n^{k+1} = q(G - R)`
3. **Solve Hole:** :math:`\nabla \cdot \mathbf{J}_p^{k+1} = -q(G - R)`
4. **Check Convergence:** :math:`\|\phi^{k+1} - \phi^k\| < \epsilon`

**Advantages:**
- Robust for difficult problems
- Lower memory requirements
- Good initial guess for Newton

**Disadvantages:**
- Slower convergence than Newton
- May not converge for strongly coupled problems

**Implementation:**

.. code-block:: cpp

   bool GummelSolver::solve() {
       for (int iter = 0; iter < max_gummel_iter; ++iter) {
           // Store previous solution
           phi_old = phi;
           
           // Solve Poisson equation
           poisson_solver.solve(phi, n, p);
           
           // Solve electron continuity
           electron_solver.solve(n, phi, p);
           
           // Solve hole continuity  
           hole_solver.solve(p, phi, n);
           
           // Check convergence
           double error = (phi - phi_old).norm();
           if (error < gummel_tolerance) return true;
       }
       return false;
   }

Linear Solvers
--------------

Direct Solvers
~~~~~~~~~~~~~

**LU Decomposition:**

.. math::
   \mathbf{A} = \mathbf{L} \mathbf{U}

**Cholesky Decomposition (SPD matrices):**

.. math::
   \mathbf{A} = \mathbf{L} \mathbf{L}^T

**Sparse Direct Solvers:**
- UMFPACK
- PARDISO
- SuperLU

**Advantages:**
- Exact solution (within machine precision)
- Robust for ill-conditioned systems

**Disadvantages:**
- High memory requirements: :math:`O(n^{3/2})` for 2D
- Limited scalability

Iterative Solvers
~~~~~~~~~~~~~~~~

**Conjugate Gradient (CG):**

For symmetric positive definite systems:

.. math::
   \mathbf{r}_0 = \mathbf{b} - \mathbf{A} \mathbf{x}_0

.. math::
   \mathbf{p}_0 = \mathbf{r}_0

.. math::
   \alpha_k = \frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{p}_k^T \mathbf{A} \mathbf{p}_k}

.. math::
   \mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k

.. math::
   \mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k \mathbf{A} \mathbf{p}_k

.. math::
   \beta_k = \frac{\mathbf{r}_{k+1}^T \mathbf{r}_{k+1}}{\mathbf{r}_k^T \mathbf{r}_k}

.. math::
   \mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k

**GMRES (Generalized Minimal Residual):**

For non-symmetric systems:

.. math::
   \mathbf{x}_m = \mathbf{x}_0 + \mathbf{V}_m \mathbf{y}_m

where :math:`\mathbf{y}_m` minimizes :math:`\|\beta \mathbf{e}_1 - \mathbf{H}_m \mathbf{y}_m\|_2`.

**BiCGSTAB:**

For non-symmetric systems with better convergence:

.. math::
   \mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k + \omega_k \mathbf{s}_k

Preconditioning
~~~~~~~~~~~~~~

**Jacobi Preconditioner:**

.. math::
   \mathbf{M}^{-1} = \text{diag}(\mathbf{A})^{-1}

**Gauss-Seidel Preconditioner:**

.. math::
   \mathbf{M} = \mathbf{L} + \mathbf{D}

**Incomplete LU (ILU):**

.. math::
   \mathbf{A} \approx \mathbf{L} \mathbf{U}

with sparsity pattern constraint.

**Algebraic Multigrid (AMG):**

- **Coarsening:** Select coarse grid points
- **Interpolation:** Define prolongation operator
- **Galerkin:** :math:`\mathbf{A}_c = \mathbf{R} \mathbf{A}_f \mathbf{P}`

**Block Preconditioning:**

For coupled systems:

.. math::
   \mathbf{M}^{-1} = \begin{pmatrix}
   \mathbf{A}_{11}^{-1} & 0 \\
   0 & \mathbf{S}^{-1}
   \end{pmatrix}

where :math:`\mathbf{S} = \mathbf{A}_{22} - \mathbf{A}_{21} \mathbf{A}_{11}^{-1} \mathbf{A}_{12}` is the Schur complement.

Mesh Generation
---------------

Structured Meshes
~~~~~~~~~~~~~~~~~

**Cartesian Grids:**

.. math::
   x_i = x_0 + i \Delta x, \quad y_j = y_0 + j \Delta y

**Advantages:**
- Simple data structures
- Efficient algorithms
- Good cache performance

**Disadvantages:**
- Limited geometry flexibility
- Difficult boundary conforming

**Stretched Grids:**

.. math::
   x_i = x_0 + \sum_{k=0}^{i-1} \Delta x_k r^k

where :math:`r` is the stretching ratio.

Unstructured Meshes
~~~~~~~~~~~~~~~~~~

**Delaunay Triangulation:**

Maximizes the minimum angle of triangles.

**Advancing Front Method:**

Generates elements by advancing from boundaries.

**Mesh Quality Metrics:**

**Aspect Ratio:**

.. math::
   AR = \frac{h_{max}}{h_{min}}

**Skewness:**

.. math::
   S = \frac{\theta_{max} - 60°}{120°}

**Jacobian Determinant:**

.. math::
   J = \det\left(\frac{\partial \mathbf{x}}{\partial \boldsymbol{\xi}}\right) > 0

Adaptive Mesh Refinement
~~~~~~~~~~~~~~~~~~~~~~~~

**h-Refinement:**

Subdivide elements based on error indicators.

**p-Refinement:**

Increase polynomial order in elements.

**hp-Refinement:**

Combine h- and p-refinement optimally.

**Anisotropic Refinement:**

Refine preferentially in one direction for boundary layers.

**Load Balancing:**

Redistribute elements among processors to maintain load balance.

Quadrature Rules
---------------

Gaussian Quadrature
~~~~~~~~~~~~~~~~~~

**1D Gauss-Legendre:**

.. math::
   \int_{-1}^1 f(x) dx \approx \sum_{i=1}^n w_i f(x_i)

**2D Triangle Quadrature:**

.. math::
   \int_T f(x,y) dx dy \approx \sum_{i=1}^n w_i f(x_i, y_i) |J|

**High-Order Rules:**

- **7-point rule:** Exact for polynomials up to degree 5
- **13-point rule:** Exact for polynomials up to degree 7

**Adaptive Quadrature:**

Refine quadrature based on integrand smoothness.

Numerical Flux Functions
-----------------------

Central Flux
~~~~~~~~~~~

.. math::
   \hat{f} = \{f\}

**Properties:**
- Conservative
- Consistent
- May be unstable for convection

Upwind Flux
~~~~~~~~~~

.. math::
   \hat{f} = \begin{cases}
   f^- & \text{if } a \geq 0 \\
   f^+ & \text{if } a < 0
   \end{cases}

**Properties:**
- Stable for convection
- Dissipative
- First-order accurate

Lax-Friedrichs Flux
~~~~~~~~~~~~~~~~~~

.. math::
   \hat{f} = \{f\} + \frac{\alpha}{2}[u]

where :math:`\alpha = \max |a|` is the maximum wave speed.

**Properties:**
- Stable and robust
- More dissipative than upwind
- Easy to implement

Local Lax-Friedrichs Flux
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
   \hat{f} = \{f\} + \frac{\alpha_{local}}{2}[u]

where :math:`\alpha_{local}` is computed locally on each face.

Roe Flux
~~~~~~~

.. math::
   \hat{f} = \frac{1}{2}(f^- + f^+) - \frac{1}{2}|\tilde{A}|(u^+ - u^-)

where :math:`\tilde{A}` is the Roe matrix.

Error Estimation
---------------

A Posteriori Error Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Residual-Based Estimator:**

.. math::
   \eta_K^2 = h_K^2 \|R_K\|_{L^2(K)}^2 + h_K \sum_{e \subset \partial K} \|R_e\|_{L^2(e)}^2

**Recovery-Based Estimator:**

.. math::
   \eta_K = \|\nabla u_h - G_h(\nabla u_h)\|_{L^2(K)}

**Hierarchical Estimator:**

.. math::
   \eta_K = \|u_{h,p+1} - u_{h,p}\|_{L^2(K)}

Goal-Oriented Error Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a quantity of interest :math:`J(u)`:

.. math::
   J(u) - J(u_h) \approx \sum_K \eta_K \omega_K

where :math:`\omega_K` are dual weights from the adjoint problem.

Efficiency and Reliability
~~~~~~~~~~~~~~~~~~~~~~~~~

**Efficiency Index:**

.. math::
   I_{eff} = \frac{\eta}{\|u - u_h\|}

**Reliability Index:**

.. math::
   I_{rel} = \frac{\|u - u_h\|}{\eta}

Ideally, both indices should be close to 1.

Parallel Algorithms
------------------

Domain Decomposition
~~~~~~~~~~~~~~~~~~~

**Overlapping Schwarz:**

.. math::
   u_i^{k+1} = \mathcal{S}_i(u_1^k, \ldots, u_p^k)

**Additive Schwarz:**

.. math::
   u^{k+1} = u^k + \sum_{i=1}^p \mathcal{P}_i (\mathcal{S}_i^{-1} - I) u^k

**Multiplicative Schwarz:**

.. math::
   u^{k+1} = \mathcal{S}_p \circ \cdots \circ \mathcal{S}_1 (u^k)

Communication Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

**Message Aggregation:**

Combine multiple small messages into larger ones.

**Asynchronous Communication:**

Overlap computation and communication.

**Communication-Avoiding Algorithms:**

Minimize communication volume and latency.

Load Balancing
~~~~~~~~~~~~~

**Static Load Balancing:**

Partition based on problem characteristics.

**Dynamic Load Balancing:**

Repartition during computation based on runtime information.

**Graph Partitioning:**

Use METIS/ParMETIS for optimal partitioning.

Performance Metrics
~~~~~~~~~~~~~~~~~~

**Parallel Efficiency:**

.. math::
   E_p = \frac{T_1}{p \cdot T_p}

**Speedup:**

.. math::
   S_p = \frac{T_1}{T_p}

**Scalability:**

Strong scaling: fixed problem size, increase processors
Weak scaling: fixed problem size per processor

GPU Computing
------------

CUDA Programming Model
~~~~~~~~~~~~~~~~~~~~~

**Thread Hierarchy:**
- Grid → Blocks → Threads
- Warp: 32 threads executing in lockstep

**Memory Hierarchy:**
- Global memory: Large, high latency
- Shared memory: Fast, limited size
- Registers: Fastest, very limited

**Kernel Launch:**

.. code-block:: cuda

   dim3 grid(n_blocks_x, n_blocks_y);
   dim3 block(n_threads_x, n_threads_y);
   kernel<<<grid, block>>>(args);

DG Assembly on GPU
~~~~~~~~~~~~~~~~~

**Element-wise Parallelization:**

.. code-block:: cuda

   __global__ void assemble_elements(
       const double* coords,
       const int* connectivity,
       double* element_matrices,
       int n_elements) {
       
       int tid = blockIdx.x * blockDim.x + threadIdx.x;
       if (tid >= n_elements) return;
       
       // Load element data to shared memory
       __shared__ double local_coords[3][2];
       __shared__ double local_matrix[10][10];
       
       // Compute element matrix
       compute_element_matrix(tid, local_coords, local_matrix);
       
       // Store to global memory
       store_element_matrix(tid, local_matrix, element_matrices);
   }

**Memory Coalescing:**

Ensure adjacent threads access adjacent memory locations.

**Occupancy Optimization:**

Balance register usage, shared memory, and thread count.

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

**Memory Bandwidth:**

.. math::
   \text{Bandwidth} = \frac{\text{Bytes Transferred}}{\text{Time}}

**Arithmetic Intensity:**

.. math::
   AI = \frac{\text{FLOPs}}{\text{Bytes Transferred}}

**Roofline Model:**

.. math::
   \text{Performance} = \min(\text{Peak FLOP/s}, AI \times \text{Bandwidth})

**Optimization Strategies:**
- Maximize occupancy
- Minimize memory transfers
- Use shared memory effectively
- Optimize memory access patterns
