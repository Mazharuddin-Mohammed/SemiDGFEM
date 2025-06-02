Mathematical Formulation
========================

This section provides the complete mathematical formulation of the Discontinuous Galerkin Finite Element Method (DG-FEM) as implemented in SemiDGFEM for semiconductor device simulation.

.. contents::
   :local:
   :depth: 3

Overview
--------

SemiDGFEM employs high-order Discontinuous Galerkin methods to solve the coupled system of semiconductor device equations:

1. **Poisson Equation**: Electrostatic potential
2. **Drift-Diffusion Equations**: Carrier transport
3. **Continuity Equations**: Current conservation

The DG method provides several advantages for semiconductor simulation:

- **High-order accuracy** with P3 elements (10 DOFs per triangle)
- **Local conservation** properties
- **Excellent parallel scalability**
- **Natural handling of discontinuities** at material interfaces

Governing Equations
-------------------

Poisson Equation
~~~~~~~~~~~~~~~~

The electrostatic potential :math:`\phi` is governed by Poisson's equation:

.. math::
   -\nabla \cdot (\epsilon \nabla \phi) = \frac{q}{\epsilon_0}(p - n + N_D^+ - N_A^-)

where:

- :math:`\phi` = electrostatic potential (V)
- :math:`\epsilon` = material permittivity (F/m)
- :math:`\epsilon_0` = vacuum permittivity = 8.854 × 10⁻¹² F/m
- :math:`q` = elementary charge = 1.602 × 10⁻¹⁹ C
- :math:`n, p` = electron and hole concentrations (m⁻³)
- :math:`N_D^+, N_A^-` = ionized donor and acceptor concentrations (m⁻³)

Drift-Diffusion Equations
~~~~~~~~~~~~~~~~~~~~~~~~~

The carrier transport is described by the drift-diffusion equations:

**Electron Current Density:**

.. math::
   \mathbf{J}_n = q \mu_n n \nabla \phi + q D_n \nabla n

**Hole Current Density:**

.. math::
   \mathbf{J}_p = -q \mu_p p \nabla \phi + q D_p \nabla p

**Continuity Equations:**

.. math::
   \frac{\partial n}{\partial t} + \frac{1}{q} \nabla \cdot \mathbf{J}_n = G - R

.. math::
   \frac{\partial p}{\partial t} - \frac{1}{q} \nabla \cdot \mathbf{J}_p = G - R

where:

- :math:`\mu_n, \mu_p` = electron and hole mobilities (m²/V·s)
- :math:`D_n, D_p` = electron and hole diffusion coefficients (m²/s)
- :math:`G` = generation rate (m⁻³/s)
- :math:`R` = recombination rate (m⁻³/s)

Einstein Relations
~~~~~~~~~~~~~~~~~~

The diffusion coefficients are related to mobilities by Einstein relations:

.. math::
   D_n = \frac{k_B T}{q} \mu_n, \quad D_p = \frac{k_B T}{q} \mu_p

where:

- :math:`k_B` = Boltzmann constant = 1.381 × 10⁻²³ J/K
- :math:`T` = temperature (K)

Discontinuous Galerkin Formulation
----------------------------------

Weak Formulation
~~~~~~~~~~~~~~~~

For the Poisson equation, the DG weak formulation on element :math:`K` is:

.. math::
   \int_K \epsilon \nabla \phi_h \cdot \nabla v_h \, d\mathbf{x} 
   - \int_{\partial K} \{\epsilon \nabla \phi_h\} \cdot \mathbf{n} [v_h] \, ds
   - \int_{\partial K} \{\epsilon \nabla v_h\} \cdot \mathbf{n} [\phi_h] \, ds
   + \int_{\partial K} \frac{\sigma}{h} [\phi_h] [v_h] \, ds
   = \int_K \rho v_h \, d\mathbf{x}

where:

- :math:`\phi_h` = discrete potential
- :math:`v_h` = test function
- :math:`\{\cdot\}` = average operator: :math:`\{u\} = \frac{1}{2}(u^+ + u^-)`
- :math:`[\cdot]` = jump operator: :math:`[u] = u^+ \mathbf{n}^+ + u^- \mathbf{n}^-`
- :math:`\sigma` = penalty parameter
- :math:`h` = element size
- :math:`\rho = \frac{q}{\epsilon_0}(p - n + N_D^+ - N_A^-)` = charge density

Penalty Parameter
~~~~~~~~~~~~~~~~~

The penalty parameter :math:`\sigma` is crucial for stability and is chosen as:

.. math::
   \sigma = C \frac{p^2}{h}

where:

- :math:`C` = penalty constant (typically 10-100)
- :math:`p` = polynomial degree
- :math:`h` = element diameter

Basis Functions
~~~~~~~~~~~~~~~

SemiDGFEM uses hierarchical P3 basis functions on triangular elements:

**P3 Triangle (10 DOFs):**

.. math::
   \phi_h(\mathbf{x}) = \sum_{i=1}^{10} \phi_i N_i(\xi, \eta)

where :math:`N_i(\xi, \eta)` are the P3 Lagrange basis functions in reference coordinates.

**Reference Triangle Coordinates:**

.. math::
   \xi, \eta \in [0,1], \quad \xi + \eta \leq 1

**Basis Function Hierarchy:**

- **Vertices (3 DOFs)**: :math:`N_1, N_2, N_3`
- **Edges (6 DOFs)**: :math:`N_4, N_5, N_6, N_7, N_8, N_9`  
- **Interior (1 DOF)**: :math:`N_{10}`

Numerical Integration
~~~~~~~~~~~~~~~~~~~~

High-order Gaussian quadrature is used for accurate integration:

**Volume Integration (7-point rule):**

.. math::
   \int_K f(\mathbf{x}) \, d\mathbf{x} \approx \sum_{q=1}^{7} w_q f(\mathbf{x}_q) |J_q|

**Face Integration (3-point rule):**

.. math::
   \int_{\partial K} g(s) \, ds \approx \sum_{q=1}^{3} w_q g(s_q) |J_q|

where :math:`w_q` are quadrature weights and :math:`|J_q|` are Jacobian determinants.

Drift-Diffusion DG Formulation
------------------------------

Electron Transport
~~~~~~~~~~~~~~~~~

The DG formulation for electron transport:

.. math::
   \int_K \frac{\partial n_h}{\partial t} v_h \, d\mathbf{x}
   + \int_K \mathbf{J}_{n,h} \cdot \nabla v_h \, d\mathbf{x}
   - \int_{\partial K} \{\mathbf{J}_{n,h}\} \cdot \mathbf{n} [v_h] \, ds
   + \int_{\partial K} \frac{\sigma_n}{h} [n_h] [v_h] \, ds
   = \int_K (G - R) v_h \, d\mathbf{x}

Hole Transport
~~~~~~~~~~~~~

Similarly for hole transport:

.. math::
   \int_K \frac{\partial p_h}{\partial t} v_h \, d\mathbf{x}
   - \int_K \mathbf{J}_{p,h} \cdot \nabla v_h \, d\mathbf{x}
   + \int_{\partial K} \{\mathbf{J}_{p,h}\} \cdot \mathbf{n} [v_h] \, ds
   + \int_{\partial K} \frac{\sigma_p}{h} [p_h] [v_h] \, ds
   = \int_K (G - R) v_h \, d\mathbf{x}

Upwind Flux
~~~~~~~~~~~

For convection-dominated transport, upwind fluxes are used:

.. math::
   \{\mathbf{J}_n\} \cdot \mathbf{n} = \mathbf{J}_n^{upw} \cdot \mathbf{n}

where the upwind direction is determined by the electric field.

Boundary Conditions
-------------------

Dirichlet Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Dirichlet boundaries with prescribed potential :math:`\phi_D`:

.. math::
   \int_{\Gamma_D} \frac{\sigma}{h} (\phi_h - \phi_D) v_h \, ds

Neumann Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Neumann boundaries with prescribed flux :math:`g_N`:

.. math::
   \int_{\Gamma_N} g_N v_h \, ds

Ohmic Contacts
~~~~~~~~~~~~~

At ohmic contacts, the boundary conditions are:

.. math::
   \phi = V_{contact}

.. math::
   n = n_{eq}, \quad p = p_{eq}

where :math:`n_{eq}, p_{eq}` are equilibrium concentrations.

Self-Consistent Coupling
------------------------

Gummel Iteration
~~~~~~~~~~~~~~~

The coupled system is solved using Gummel iteration:

1. **Solve Poisson equation** for :math:`\phi^{(k+1)}` with fixed :math:`n^{(k)}, p^{(k)}`
2. **Solve electron equation** for :math:`n^{(k+1)}` with fixed :math:`\phi^{(k+1)}, p^{(k)}`
3. **Solve hole equation** for :math:`p^{(k+1)}` with fixed :math:`\phi^{(k+1)}, n^{(k+1)}`
4. **Check convergence**: :math:`\|\phi^{(k+1)} - \phi^{(k)}\| < \epsilon`

Newton-Raphson Method
~~~~~~~~~~~~~~~~~~~~

For better convergence, Newton-Raphson can be used:

.. math::
   \mathbf{J} \Delta \mathbf{u} = -\mathbf{F}(\mathbf{u}^{(k)})

where :math:`\mathbf{J}` is the Jacobian matrix and :math:`\mathbf{F}` is the residual vector.

Matrix Assembly
---------------

Element Matrix
~~~~~~~~~~~~~

For each element :math:`K`, the local matrix is:

.. math::
   A_{ij}^K = \int_K \epsilon \nabla N_i \cdot \nabla N_j \, d\mathbf{x}
   + \int_{\partial K} \frac{\sigma}{h} N_i N_j \, ds

Global Assembly
~~~~~~~~~~~~~~

The global system is assembled as:

.. math::
   \mathbf{A} \boldsymbol{\phi} = \mathbf{b}

where :math:`\mathbf{A}` includes volume, face, and penalty terms.

Adaptive Mesh Refinement
------------------------

Error Estimation
~~~~~~~~~~~~~~~

The Kelly error estimator is used:

.. math::
   \eta_K^2 = \frac{h_K}{2} \int_{\partial K} [\nabla \phi_h \cdot \mathbf{n}]^2 \, ds

Refinement Criterion
~~~~~~~~~~~~~~~~~~~

Elements are refined if:

.. math::
   \eta_K > \theta \max_j \eta_j

where :math:`\theta` is the refinement threshold (typically 0.3).

Implementation Details
---------------------

Data Structures
~~~~~~~~~~~~~~

**Element Connectivity:**
- Vertex indices for each triangle
- Neighbor information for face assembly
- Material properties per element

**DOF Mapping:**
- Global DOF numbering
- Local-to-global mapping
- Boundary condition flags

**Quadrature Data:**
- Quadrature points and weights
- Basis function values at quadrature points
- Jacobian transformations

Parallel Implementation
~~~~~~~~~~~~~~~~~~~~~~

**Domain Decomposition:**
- Elements partitioned among processors
- Ghost elements for inter-processor communication
- PETSc for parallel linear algebra

**Load Balancing:**
- Dynamic load balancing for AMR
- Communication minimization
- Scalability up to thousands of cores

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Layout:**
- Structure of Arrays (SoA) for vectorization
- Cache-friendly data access patterns
- Memory pools for dynamic allocation

**Vectorization:**
- SIMD instructions (AVX2/FMA)
- Loop unrolling and fusion
- Compiler optimization flags

GPU Acceleration
~~~~~~~~~~~~~~~

**CUDA Implementation:**
- Element-wise parallelization
- Shared memory optimization
- Coalesced memory access
- 10-20x speedup over CPU

References
----------

1. Hesthaven, J. S., & Warburton, T. (2007). *Nodal discontinuous Galerkin methods: algorithms, analysis, and applications*. Springer.

2. Cockburn, B., Karniadakis, G. E., & Shu, C. W. (2000). *Discontinuous Galerkin methods: theory, computation and applications*. Springer.

3. Selberherr, S. (1984). *Analysis and simulation of semiconductor devices*. Springer-Verlag.

4. Markowich, P. A., Ringhofer, C. A., & Schmeiser, C. (2012). *Semiconductor equations*. Springer Science & Business Media.

5. Jerome, J. W. (1996). *Analysis of charge transport: a mathematical study of semiconductor devices*. Springer-Verlag.
