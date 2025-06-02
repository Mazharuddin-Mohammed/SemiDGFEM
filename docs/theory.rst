Theoretical Background
======================

This section provides the theoretical foundation for semiconductor device simulation using Discontinuous Galerkin methods in SemiDGFEM.

.. contents::
   :local:
   :depth: 3

Semiconductor Physics
---------------------

Fundamental Equations
~~~~~~~~~~~~~~~~~~~~~

The behavior of charge carriers in semiconductor devices is governed by the following fundamental equations:

**Maxwell's Equations (Electrostatics):**

.. math::
   \nabla \cdot \mathbf{D} = \rho

.. math::
   \mathbf{D} = \epsilon \mathbf{E} = -\epsilon \nabla \phi

**Charge Density:**

.. math::
   \rho = q(p - n + N_D^+ - N_A^-)

**Current Density Equations:**

.. math::
   \mathbf{J}_n = q \mu_n n \mathbf{E} + q D_n \nabla n

.. math::
   \mathbf{J}_p = q \mu_p p \mathbf{E} - q D_p \nabla p

**Continuity Equations:**

.. math::
   \frac{\partial n}{\partial t} + \frac{1}{q} \nabla \cdot \mathbf{J}_n = G - R

.. math::
   \frac{\partial p}{\partial t} - \frac{1}{q} \nabla \cdot \mathbf{J}_p = G - R

Carrier Statistics
~~~~~~~~~~~~~~~~~

**Boltzmann Statistics (Non-degenerate):**

.. math::
   n = N_c \exp\left(\frac{E_F - E_c}{k_B T}\right)

.. math::
   p = N_v \exp\left(\frac{E_v - E_F}{k_B T}\right)

**Fermi-Dirac Statistics (Degenerate):**

.. math::
   n = N_c F_{1/2}\left(\frac{E_F - E_c}{k_B T}\right)

.. math::
   p = N_v F_{1/2}\left(\frac{E_v - E_F}{k_B T}\right)

where :math:`F_{1/2}` is the Fermi-Dirac integral of order 1/2.

**Quasi-Fermi Levels:**

.. math::
   n = n_i \exp\left(\frac{q(\phi - \phi_n)}{k_B T}\right)

.. math::
   p = n_i \exp\left(\frac{q(\phi_p - \phi)}{k_B T}\right)

where :math:`\phi_n, \phi_p` are the quasi-Fermi potentials.

Recombination Mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~

**Shockley-Read-Hall (SRH) Recombination:**

.. math::
   R_{SRH} = \frac{np - n_i^2}{\tau_p(n + n_1) + \tau_n(p + p_1)}

where:

.. math::
   n_1 = n_i \exp\left(\frac{E_t - E_i}{k_B T}\right), \quad p_1 = n_i \exp\left(\frac{E_i - E_t}{k_B T}\right)

**Radiative Recombination:**

.. math::
   R_{rad} = B(np - n_i^2)

**Auger Recombination:**

.. math::
   R_{Auger} = (C_n n + C_p p)(np - n_i^2)

**Total Recombination:**

.. math::
   R = R_{SRH} + R_{rad} + R_{Auger}

Mobility Models
~~~~~~~~~~~~~~

**Caughey-Thomas Model:**

.. math::
   \mu(N) = \mu_{min} + \frac{\mu_{max} - \mu_{min}}{1 + (N/N_{ref})^\alpha}

**Temperature Dependence:**

.. math::
   \mu(T) = \mu(300K) \left(\frac{T}{300}\right)^{-\gamma}

**High-Field Mobility:**

.. math::
   \mu_{eff} = \frac{\mu_{low}}{1 + (\mu_{low} E / v_{sat})^\beta}

Discontinuous Galerkin Theory
-----------------------------

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~

**Function Spaces:**

Let :math:`\Omega \subset \mathbb{R}^d` be the computational domain with triangulation :math:`\mathcal{T}_h`.

**Broken Sobolev Spaces:**

.. math::
   H^s(\mathcal{T}_h) = \{v \in L^2(\Omega) : v|_K \in H^s(K) \, \forall K \in \mathcal{T}_h\}

**Discontinuous Finite Element Space:**

.. math::
   V_h^p = \{v \in L^2(\Omega) : v|_K \in P^p(K) \, \forall K \in \mathcal{T}_h\}

where :math:`P^p(K)` is the space of polynomials of degree at most :math:`p` on element :math:`K`.

Trace Operators
~~~~~~~~~~~~~~

**Average Operator:**

.. math::
   \{v\} = \begin{cases}
   \frac{1}{2}(v^+ + v^-) & \text{on interior faces} \\
   v & \text{on boundary faces}
   \end{cases}

**Jump Operator:**

.. math::
   [v] = \begin{cases}
   v^+ \mathbf{n}^+ + v^- \mathbf{n}^- & \text{on interior faces} \\
   v \mathbf{n} & \text{on boundary faces}
   \end{cases}

**Normal Jump:**

.. math::
   [v]_n = \begin{cases}
   v^+ - v^- & \text{on interior faces} \\
   v & \text{on boundary faces}
   \end{cases}

DG Formulation for Poisson Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strong Form:**

.. math::
   -\nabla \cdot (\epsilon \nabla \phi) = \rho \quad \text{in } \Omega

**Weak Form:**

Find :math:`\phi_h \in V_h^p` such that:

.. math::
   a_h(\phi_h, v_h) = l_h(v_h) \quad \forall v_h \in V_h^p

where:

.. math::
   a_h(u, v) = \sum_{K \in \mathcal{T}_h} \int_K \epsilon \nabla u \cdot \nabla v \, dx
   - \sum_{F \in \mathcal{F}_h} \int_F \{\epsilon \nabla u\} \cdot [v] \, ds
   - \sum_{F \in \mathcal{F}_h} \int_F \{\epsilon \nabla v\} \cdot [u] \, ds
   + \sum_{F \in \mathcal{F}_h} \int_F \frac{\sigma}{h_F} [u] \cdot [v] \, ds

.. math::
   l_h(v) = \sum_{K \in \mathcal{T}_h} \int_K \rho v \, dx

Stability and Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~

**Coercivity:**

The bilinear form :math:`a_h(\cdot, \cdot)` is coercive if the penalty parameter satisfies:

.. math::
   \sigma \geq \sigma_0 = C \frac{p^2}{h}

where :math:`C` is a constant depending on the mesh geometry.

**Error Estimates:**

For sufficiently smooth solutions:

.. math::
   \|\phi - \phi_h\|_{DG} \leq C h^{p+1} \|\phi\|_{H^{p+2}}

where :math:`\|\cdot\|_{DG}` is the DG norm:

.. math::
   \|v\|_{DG}^2 = \sum_{K} \|\nabla v\|_{L^2(K)}^2 + \sum_{F} \frac{\sigma}{h_F} \|[v]\|_{L^2(F)}^2

Drift-Diffusion DG Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Convection-Diffusion Form:**

The drift-diffusion equations can be written as:

.. math::
   \frac{\partial n}{\partial t} - \nabla \cdot (D_n \nabla n) + \nabla \cdot (\mu_n n \nabla \phi) = \frac{G - R}{q}

**DG Weak Form:**

.. math::
   \int_K \frac{\partial n_h}{\partial t} v_h \, dx
   + \int_K D_n \nabla n_h \cdot \nabla v_h \, dx
   - \int_{\partial K} \{D_n \nabla n_h\} \cdot [v_h] \, ds
   + \int_{\partial K} \frac{\sigma_n}{h} [n_h] [v_h] \, ds
   + \int_K \mu_n n_h \nabla \phi_h \cdot \nabla v_h \, dx
   - \int_{\partial K} \{\mu_n n_h \nabla \phi_h\} \cdot [v_h] \, ds
   = \int_K \frac{G - R}{q} v_h \, dx

Upwind Stabilization
~~~~~~~~~~~~~~~~~~~

For convection-dominated problems, upwind fluxes are used:

**Upwind Flux:**

.. math::
   \hat{\mathbf{J}}_n \cdot \mathbf{n} = \begin{cases}
   \mathbf{J}_n^+ \cdot \mathbf{n} & \text{if } \mathbf{v} \cdot \mathbf{n} \geq 0 \\
   \mathbf{J}_n^- \cdot \mathbf{n} & \text{if } \mathbf{v} \cdot \mathbf{n} < 0
   \end{cases}

where :math:`\mathbf{v} = -\mu_n \nabla \phi` is the drift velocity.

**Lax-Friedrichs Flux:**

.. math::
   \hat{\mathbf{J}}_n \cdot \mathbf{n} = \{\mathbf{J}_n\} \cdot \mathbf{n} + \frac{\alpha}{2} [n]_n

where :math:`\alpha = \max(|\mathbf{v} \cdot \mathbf{n}|)` is the maximum wave speed.

Adaptive Mesh Refinement
------------------------

A Posteriori Error Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Residual-Based Estimator:**

.. math::
   \eta_K^2 = h_K^2 \|R_K\|_{L^2(K)}^2 + \frac{h_K}{2} \sum_{F \subset \partial K} \|R_F\|_{L^2(F)}^2

where:

.. math::
   R_K = \rho + \nabla \cdot (\epsilon \nabla \phi_h)

.. math::
   R_F = [\epsilon \nabla \phi_h \cdot \mathbf{n}]

**Kelly Estimator:**

.. math::
   \eta_K^2 = \frac{h_K}{2} \sum_{F \subset \partial K} \|[\nabla \phi_h \cdot \mathbf{n}]\|_{L^2(F)}^2

**Gradient Recovery Estimator:**

.. math::
   \eta_K = \|\nabla \phi_h - G_h(\nabla \phi_h)\|_{L^2(K)}

where :math:`G_h` is a gradient recovery operator.

Refinement Strategies
~~~~~~~~~~~~~~~~~~~~

**Dörfler Marking:**

Mark elements :math:`\mathcal{M} \subset \mathcal{T}_h` such that:

.. math::
   \sum_{K \in \mathcal{M}} \eta_K^2 \geq \theta \sum_{K \in \mathcal{T}_h} \eta_K^2

where :math:`\theta \in (0,1)` is the marking parameter.

**Maximum Strategy:**

Mark element :math:`K` if:

.. math::
   \eta_K \geq \theta \max_{K' \in \mathcal{T}_h} \eta_{K'}

**Anisotropic Refinement:**

For problems with boundary layers, anisotropic refinement is used based on:

.. math::
   \mathbf{M} = |\nabla \phi_h| \nabla \phi_h \otimes \nabla \phi_h

Numerical Linear Algebra
------------------------

Matrix Properties
~~~~~~~~~~~~~~~~

**Sparsity Pattern:**

The DG discretization leads to a block-sparse matrix structure:

- **Diagonal blocks**: Element self-interactions
- **Off-diagonal blocks**: Face neighbor interactions

**Condition Number:**

For the Poisson equation with penalty parameter :math:`\sigma`:

.. math::
   \kappa(\mathbf{A}) = O\left(\frac{\sigma}{h^2}\right)

Iterative Solvers
~~~~~~~~~~~~~~~~~

**Conjugate Gradient (CG):**

For symmetric positive definite systems (Poisson equation):

.. math::
   \mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k \mathbf{A} \mathbf{p}_k

**GMRES:**

For non-symmetric systems (drift-diffusion):

.. math::
   \mathbf{x}_m = \mathbf{x}_0 + \mathbf{V}_m \mathbf{y}_m

where :math:`\mathbf{V}_m` spans the Krylov subspace.

Preconditioning
~~~~~~~~~~~~~~

**Block Jacobi:**

.. math::
   \mathbf{M}^{-1} = \text{diag}(\mathbf{A}_{11}^{-1}, \mathbf{A}_{22}^{-1}, \ldots, \mathbf{A}_{nn}^{-1})

**Multigrid:**

- **Smoother**: Block Gauss-Seidel
- **Restriction**: Injection or weighted restriction
- **Prolongation**: Linear interpolation
- **Coarse grid**: Geometric or algebraic coarsening

Parallel Implementation
----------------------

Domain Decomposition
~~~~~~~~~~~~~~~~~~~

**Overlapping Schwarz:**

.. math::
   \Omega = \bigcup_{i=1}^p \Omega_i, \quad \Omega_i \cap \Omega_j \neq \emptyset

**Non-overlapping Decomposition:**

.. math::
   \Omega = \bigcup_{i=1}^p \Omega_i, \quad \Omega_i \cap \Omega_j = \emptyset

Communication Patterns
~~~~~~~~~~~~~~~~~~~~~

**Ghost Elements:**

Elements in neighboring subdomains needed for face assembly.

**Message Passing:**

- **Point-to-point**: MPI_Send/MPI_Recv
- **Collective**: MPI_Allreduce for global operations

Load Balancing
~~~~~~~~~~~~~

**Graph Partitioning:**

Using METIS or ParMETIS for optimal load distribution:

.. math::
   \min \sum_{i=1}^p |W_i - \bar{W}|

where :math:`W_i` is the work load of processor :math:`i`.

Performance Optimization
------------------------

Vectorization
~~~~~~~~~~~~

**SIMD Instructions:**

- **AVX2**: 256-bit vectors (4 doubles)
- **AVX-512**: 512-bit vectors (8 doubles)
- **FMA**: Fused multiply-add operations

**Loop Optimization:**

.. code-block:: c++

   #pragma omp simd aligned(a,b,c:32)
   for (int i = 0; i < n; i += 4) {
       __m256d va = _mm256_load_pd(&a[i]);
       __m256d vb = _mm256_load_pd(&b[i]);
       __m256d vc = _mm256_fmadd_pd(va, vb, vc);
       _mm256_store_pd(&c[i], vc);
   }

Memory Optimization
~~~~~~~~~~~~~~~~~~

**Cache-Friendly Access:**

- **Structure of Arrays (SoA)**: Better vectorization
- **Array of Structures (AoS)**: Better locality

**Memory Bandwidth:**

.. math::
   \text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}

GPU Acceleration
~~~~~~~~~~~~~~~

**CUDA Kernel Structure:**

.. code-block:: cuda

   __global__ void dg_assembly_kernel(
       const double* coords,
       const int* connectivity,
       double* matrix,
       int n_elements) {
       
       int tid = blockIdx.x * blockDim.x + threadIdx.x;
       if (tid >= n_elements) return;
       
       // Element-wise assembly
       assemble_element(tid, coords, connectivity, matrix);
   }

**Memory Hierarchy:**

- **Global Memory**: 1-2 TB/s bandwidth
- **Shared Memory**: 10-20 TB/s bandwidth  
- **Registers**: Highest bandwidth

Validation and Verification
---------------------------

Method of Manufactured Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Exact Solution:**

.. math::
   \phi_{exact}(x,y) = \sin(\pi x) \sin(\pi y)

**Source Term:**

.. math::
   f(x,y) = 2\pi^2 \epsilon \sin(\pi x) \sin(\pi y)

**Convergence Rate:**

.. math::
   \text{Rate} = \frac{\log(e_h/e_{h/2})}{\log(2)}

Benchmark Problems
~~~~~~~~~~~~~~~~~

**1D P-N Junction:**

Analytical solution for abrupt junction:

.. math::
   \phi(x) = \begin{cases}
   \frac{qN_A}{2\epsilon}(x + x_p)^2 + \phi_p & x \in [-x_p, 0] \\
   -\frac{qN_D}{2\epsilon}(x - x_n)^2 + \phi_n & x \in [0, x_n]
   \end{cases}

**2D MOSFET:**

Comparison with commercial TCAD tools (Sentaurus, Silvaco).

Error Norms
~~~~~~~~~~

**L² Error:**

.. math::
   \|e\|_{L^2} = \sqrt{\int_\Omega (u - u_h)^2 \, dx}

**H¹ Error:**

.. math::
   \|e\|_{H^1} = \sqrt{\int_\Omega |\nabla(u - u_h)|^2 \, dx}

**DG Error:**

.. math::
   \|e\|_{DG} = \sqrt{\sum_K \|\nabla e\|_{L^2(K)}^2 + \sum_F \frac{\sigma}{h_F} \|[e]\|_{L^2(F)}^2}
