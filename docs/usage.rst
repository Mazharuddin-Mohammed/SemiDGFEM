Usage Guide
===========

This comprehensive usage guide provides detailed instructions for using the SemiDGFEM framework for semiconductor device simulation.

.. contents:: Table of Contents
   :local:
   :depth: 3

Getting Started
---------------

Basic Workflow
~~~~~~~~~~~~~~

The typical SemiDGFEM simulation workflow follows these steps:

1. **Define Device Geometry**: Specify the physical structure
2. **Set Material Properties**: Define semiconductor materials and doping
3. **Create Mesh**: Generate computational mesh
4. **Choose Transport Model**: Select appropriate physics model
5. **Set Boundary Conditions**: Define electrical contacts and boundaries
6. **Solve System**: Execute the simulation
7. **Analyze Results**: Visualize and extract device parameters

Quick Start Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from semidgfem import *

   # Create a simple 2D device
   device = Device(length=2e-6, width=1e-6)  # 2μm × 1μm
   
   # Set up drift-diffusion solver
   solver = DriftDiffusionSolver(device)
   
   # Define doping profile
   x = np.linspace(0, 2e-6, 100)
   nd = np.where(x < 1e-6, 1e17, 1e20)  # n-type doping
   na = np.full_like(x, 1e16)           # p-type background
   
   solver.set_doping(nd, na)
   
   # Set boundary conditions
   bc = BoundaryConditions()
   bc.set_voltage('left', 0.0)    # Source contact
   bc.set_voltage('right', 1.0)   # Drain contact
   bc.set_voltage('top', 0.5)     # Gate contact
   bc.set_voltage('bottom', 0.0)  # Substrate contact
   
   # Solve
   results = solver.solve(bc)
   
   # Plot results
   plt.figure(figsize=(12, 4))
   
   plt.subplot(131)
   plt.contourf(results.x, results.y, results.potential)
   plt.colorbar(label='Potential (V)')
   plt.title('Electrostatic Potential')
   
   plt.subplot(132)
   plt.contourf(results.x, results.y, np.log10(results.electron_density))
   plt.colorbar(label='log₁₀(n) [cm⁻³]')
   plt.title('Electron Density')
   
   plt.subplot(133)
   plt.contourf(results.x, results.y, np.log10(results.hole_density))
   plt.colorbar(label='log₁₀(p) [cm⁻³]')
   plt.title('Hole Density')
   
   plt.tight_layout()
   plt.show()

Core Components
---------------

Device Definition
~~~~~~~~~~~~~~~~~

**Basic Device Creation:**

.. code-block:: python

   # 1D device
   device_1d = Device1D(length=1e-6)
   
   # 2D device
   device_2d = Device2D(length=2e-6, width=1e-6)
   
   # 3D device
   device_3d = Device3D(length=2e-6, width=1e-6, height=0.5e-6)

**Device with Complex Geometry:**

.. code-block:: python

   from semidgfem.geometry import *
   
   # Create geometry using primitives
   substrate = Rectangle(0, 0, 10e-6, 5e-6)
   gate = Rectangle(3e-6, 4e-6, 4e-6, 0.1e-6)
   source = Rectangle(1e-6, 4e-6, 1.5e-6, 0.5e-6)
   drain = Rectangle(7.5e-6, 4e-6, 1.5e-6, 0.5e-6)
   
   # Combine geometries
   geometry = GeometryBuilder()
   geometry.add_region(substrate, material='Si', name='substrate')
   geometry.add_region(gate, material='PolySi', name='gate')
   geometry.add_region(source, material='Si', name='source')
   geometry.add_region(drain, material='Si', name='drain')
   
   # Create device
   device = Device(geometry)

Material Properties
~~~~~~~~~~~~~~~~~~~

**Built-in Materials:**

.. code-block:: python

   from semidgfem.materials import *
   
   # Silicon properties
   si = Silicon(temperature=300)
   print(f"Bandgap: {si.bandgap:.3f} eV")
   print(f"Electron mobility: {si.electron_mobility:.0f} cm²/V·s")
   
   # Compound semiconductors
   gaas = GaAs(temperature=300)
   algaas = AlGaAs(al_fraction=0.3, temperature=300)
   gan = GaN(temperature=300)

**Custom Material Definition:**

.. code-block:: python

   class CustomSemiconductor(Material):
       def __init__(self, temperature=300):
           super().__init__(temperature)
           
       @property
       def bandgap(self):
           # Varshni model
           Eg0 = 1.5  # eV at 0K
           alpha = 5e-4  # eV/K
           beta = 600  # K
           return Eg0 - (alpha * self.temperature**2) / (self.temperature + beta)
       
       @property
       def electron_mobility(self):
           # Temperature dependence
           mu0 = 1000  # cm²/V·s at 300K
           return mu0 * (self.temperature / 300)**(-2.0)
       
       @property
       def hole_mobility(self):
           mu0 = 300  # cm²/V·s at 300K
           return mu0 * (self.temperature / 300)**(-2.2)

Doping Profiles
~~~~~~~~~~~~~~~

**Uniform Doping:**

.. code-block:: python

   # Uniform n-type doping
   doping = UniformDoping(donor_density=1e17)
   
   # Uniform p-type doping
   doping = UniformDoping(acceptor_density=1e16)

**Gaussian Doping Profile:**

.. code-block:: python

   # Gaussian implant profile
   doping = GaussianDoping(
       peak_concentration=1e20,
       peak_position=0.1e-6,
       straggle=0.05e-6,
       doping_type='n'
   )

**Custom Doping Profile:**

.. code-block:: python

   def custom_doping_function(x, y):
       """Custom 2D doping profile"""
       # Junction at x = 1μm
       if x < 1e-6:
           return 1e17, 0  # n-type
       else:
           return 0, 1e16  # p-type
   
   doping = CustomDoping(custom_doping_function)

**Analytical Doping Profiles:**

.. code-block:: python

   # Exponential profile
   doping = ExponentialDoping(
       surface_concentration=1e20,
       junction_depth=0.5e-6,
       characteristic_length=0.1e-6
   )
   
   # Error function profile
   doping = ErfcDoping(
       surface_concentration=1e19,
       junction_depth=0.3e-6,
       diffusion_length=0.08e-6
   )

Transport Models
----------------

Drift-Diffusion Model
~~~~~~~~~~~~~~~~~~~~~

**Basic Setup:**

.. code-block:: python

   from semidgfem.transport import DriftDiffusionSolver
   
   solver = DriftDiffusionSolver(device)
   
   # Set physical parameters
   solver.set_temperature(300)  # Kelvin
   solver.set_doping(nd_profile, na_profile)
   
   # Solver options
   solver.set_options({
       'max_iterations': 100,
       'tolerance': 1e-6,
       'damping_factor': 0.7,
       'use_scharfetter_gummel': True
   })

**Advanced Drift-Diffusion:**

.. code-block:: python

   # Include recombination mechanisms
   solver.enable_srh_recombination(tau_n=1e-6, tau_p=1e-6)
   solver.enable_auger_recombination(Cn=1e-31, Cp=1e-31)
   solver.enable_radiative_recombination(B=1e-10)
   
   # Field-dependent mobility
   solver.enable_field_dependent_mobility(
       vsat_n=1e7,  # cm/s
       vsat_p=8e6,  # cm/s
       beta=2.0
   )
   
   # Bandgap narrowing
   solver.enable_bandgap_narrowing()

Energy Transport Model
~~~~~~~~~~~~~~~~~~~~~

**Setup and Configuration:**

.. code-block:: python

   from semidgfem.transport import EnergyTransportSolver
   
   solver = EnergyTransportSolver(device)
   
   # Energy relaxation times
   solver.set_energy_relaxation_time(
       tau_wn=1e-12,  # seconds
       tau_wp=1e-12
   )
   
   # Thermal conductivity
   solver.set_thermal_conductivity(
       kappa_n=1.0,  # W/cm·K
       kappa_p=0.5
   )

**Hot Carrier Analysis:**

.. code-block:: python

   # Solve energy transport
   results = solver.solve(boundary_conditions)
   
   # Extract carrier temperatures
   Tn = results.electron_temperature
   Tp = results.hole_temperature
   
   # Calculate average energies
   avg_energy_n = 1.5 * k_B * Tn
   avg_energy_p = 1.5 * k_B * Tp
   
   # Plot temperature distribution
   plt.figure(figsize=(10, 4))
   
   plt.subplot(121)
   plt.contourf(results.x, results.y, Tn)
   plt.colorbar(label='Electron Temperature (K)')
   plt.title('Electron Temperature')
   
   plt.subplot(122)
   plt.contourf(results.x, results.y, Tp)
   plt.colorbar(label='Hole Temperature (K)')
   plt.title('Hole Temperature')
   
   plt.show()

Hydrodynamic Model
~~~~~~~~~~~~~~~~~

**Momentum Conservation:**

.. code-block:: python

   from semidgfem.transport import HydrodynamicSolver
   
   solver = HydrodynamicSolver(device)
   
   # Momentum relaxation times
   solver.set_momentum_relaxation_time(
       tau_pn=1e-13,  # seconds
       tau_pp=1e-13
   )
   
   # Solve hydrodynamic equations
   results = solver.solve(boundary_conditions)
   
   # Extract velocities
   vn_x = results.electron_velocity_x
   vn_y = results.electron_velocity_y
   vp_x = results.hole_velocity_x
   vp_y = results.hole_velocity_y
   
   # Calculate velocity magnitude
   vn_mag = np.sqrt(vn_x**2 + vn_y**2)
   vp_mag = np.sqrt(vp_x**2 + vp_y**2)

Non-Equilibrium Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~

**Quasi-Fermi Level Formulation:**

.. code-block:: python

   from semidgfem.transport import NonEquilibriumSolver
   
   solver = NonEquilibriumSolver(device)
   
   # Enable generation-recombination
   solver.enable_optical_generation(
       generation_rate=1e21,  # cm⁻³·s⁻¹
       absorption_coefficient=1e4  # cm⁻¹
   )
   
   # Solve with quasi-Fermi levels
   results = solver.solve(boundary_conditions)
   
   # Extract quasi-Fermi levels
   EFn = results.electron_quasi_fermi_level
   EFp = results.hole_quasi_fermi_level
   
   # Calculate splitting
   quasi_fermi_splitting = EFn - EFp

Boundary Conditions
-------------------

Electrical Contacts
~~~~~~~~~~~~~~~~~~~

**Ohmic Contacts:**

.. code-block:: python

   bc = BoundaryConditions()
   
   # Ideal ohmic contact
   bc.add_ohmic_contact('source', voltage=0.0, work_function=4.1)
   bc.add_ohmic_contact('drain', voltage=1.0, work_function=4.1)
   
   # Non-ideal ohmic contact with resistance
   bc.add_ohmic_contact('substrate', voltage=0.0, 
                       contact_resistance=1e-6)  # Ω·cm²

**Schottky Contacts:**

.. code-block:: python

   # Schottky barrier contact
   bc.add_schottky_contact('gate', 
                          voltage=0.5,
                          barrier_height=0.8,  # eV
                          ideality_factor=1.1)

**MIS Contacts:**

.. code-block:: python

   # Metal-Insulator-Semiconductor contact
   bc.add_mis_contact('gate',
                     voltage=2.0,
                     oxide_thickness=5e-9,  # m
                     oxide_permittivity=3.9,
                     work_function_difference=0.5)  # eV

Surface Conditions
~~~~~~~~~~~~~~~~~

**Surface Recombination:**

.. code-block:: python

   # Surface recombination velocity
   bc.add_surface_recombination('top_surface',
                               sn=1e5,  # cm/s
                               sp=1e5)  # cm/s

**Fixed Charge:**

.. code-block:: python

   # Interface fixed charge
   bc.add_fixed_charge('oxide_interface',
                      charge_density=1e11)  # cm⁻²

**Periodic Boundaries:**

.. code-block:: python

   # Periodic boundary conditions
   bc.add_periodic_boundary('left', 'right')

Advanced Features
-----------------

Adaptive Mesh Refinement
~~~~~~~~~~~~~~~~~~~~~~~~

**Error-Based Refinement:**

.. code-block:: python

   from semidgfem.mesh import AdaptiveMeshRefinement
   
   # Create adaptive mesh refiner
   amr = AdaptiveMeshRefinement(device)
   
   # Set refinement criteria
   amr.set_error_estimator('kelly')  # or 'residual', 'gradient'
   amr.set_refinement_fraction(0.3)  # Refine 30% of elements
   amr.set_coarsening_fraction(0.1)  # Coarsen 10% of elements
   amr.set_max_refinement_level(5)
   
   # Adaptive solution loop
   for cycle in range(10):
       # Solve on current mesh
       results = solver.solve(boundary_conditions)
       
       # Estimate error
       error_indicators = amr.estimate_error(results)
       
       # Refine mesh
       if amr.should_refine(error_indicators):
           amr.refine_mesh()
           solver.update_mesh(amr.get_mesh())
       else:
           break

**Solution-Based Refinement:**

.. code-block:: python

   # Refine based on solution gradients
   amr.set_gradient_threshold('potential', 1e6)  # V/m
   amr.set_gradient_threshold('electron_density', 1e23)  # cm⁻⁴
   
   # Refine in specific regions
   amr.add_refinement_region(x_min=0.9e-6, x_max=1.1e-6,
                           y_min=0, y_max=0.2e-6,
                           min_level=3)

Time-Dependent Simulation
~~~~~~~~~~~~~~~~~~~~~~~~

**Transient Analysis:**

.. code-block:: python

   from semidgfem.time import TransientSolver
   
   # Create transient solver
   transient = TransientSolver(solver)
   
   # Set time integration parameters
   transient.set_time_step(1e-12)  # seconds
   transient.set_final_time(1e-9)  # seconds
   transient.set_time_integrator('backward_euler')  # or 'rk4', 'bdf2'
   
   # Time-dependent boundary conditions
   def time_dependent_voltage(t):
       if t < 1e-10:
           return 0.0
       else:
           return 1.0 * (1 - np.exp(-(t - 1e-10) / 1e-11))
   
   bc.set_time_dependent_voltage('drain', time_dependent_voltage)
   
   # Solve transient problem
   time_points, solutions = transient.solve(bc, initial_conditions)

**AC Small-Signal Analysis:**

.. code-block:: python

   from semidgfem.ac import ACAnalysis
   
   # Create AC analysis
   ac = ACAnalysis(solver)
   
   # Set frequency range
   frequencies = np.logspace(6, 12, 100)  # 1 MHz to 1 THz
   
   # Small-signal perturbation
   ac.set_perturbation_amplitude(0.01)  # V
   
   # Perform AC analysis
   impedance_data = ac.solve(frequencies, dc_solution)
   
   # Extract capacitance and conductance
   capacitance = -impedance_data.imag / (2 * np.pi * frequencies)
   conductance = impedance_data.real

Noise Analysis
~~~~~~~~~~~~~

**Thermal Noise:**

.. code-block:: python

   from semidgfem.noise import NoiseAnalysis
   
   noise = NoiseAnalysis(solver)
   
   # Enable thermal noise
   noise.enable_thermal_noise(temperature=300)
   
   # Enable shot noise
   noise.enable_shot_noise()
   
   # Enable flicker noise
   noise.enable_flicker_noise(alpha=1.0, kf=1e-12)
   
   # Calculate noise spectral density
   frequencies = np.logspace(3, 9, 100)  # 1 kHz to 1 GHz
   noise_psd = noise.calculate_noise_psd(frequencies, dc_solution)

Performance Optimization
-----------------------

GPU Acceleration
~~~~~~~~~~~~~~~

**Enable GPU Computing:**

.. code-block:: python

   from semidgfem.gpu import GPUAcceleration
   
   # Check GPU availability
   if GPUAcceleration.is_available():
       print("GPU acceleration available")
       
       # Enable GPU for solver
       solver.enable_gpu_acceleration()
       
       # Set GPU memory fraction
       solver.set_gpu_memory_fraction(0.8)
       
       # Choose GPU device
       solver.set_gpu_device(0)
   else:
       print("GPU not available, using CPU")

**GPU Performance Tuning:**

.. code-block:: python

   # Optimize GPU kernels
   gpu_options = {
       'block_size': 256,
       'grid_size': 'auto',
       'shared_memory_size': 48000,  # bytes
       'use_texture_memory': True,
       'use_constant_memory': True
   }
   
   solver.set_gpu_options(gpu_options)

SIMD Optimization
~~~~~~~~~~~~~~~~

**Enable Vectorization:**

.. code-block:: python

   from semidgfem.simd import SIMDOptimization
   
   # Check SIMD capabilities
   simd = SIMDOptimization()
   print(f"AVX2 support: {simd.has_avx2()}")
   print(f"FMA support: {simd.has_fma()}")
   
   # Enable SIMD optimization
   solver.enable_simd_optimization()
   
   # Set vector width
   solver.set_simd_vector_width(4)  # AVX2: 4 doubles

Parallel Computing
~~~~~~~~~~~~~~~~~

**OpenMP Threading:**

.. code-block:: python

   import os
   
   # Set number of threads
   os.environ['OMP_NUM_THREADS'] = '8'
   
   # Enable parallel assembly
   solver.enable_parallel_assembly(num_threads=8)

**MPI Parallelization:**

.. code-block:: python

   from mpi4py import MPI
   from semidgfem.parallel import MPISolver
   
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()
   
   # Create MPI solver
   mpi_solver = MPISolver(solver, comm)
   
   # Partition mesh
   mpi_solver.partition_mesh(method='metis')
   
   # Solve in parallel
   results = mpi_solver.solve(boundary_conditions)

Results Analysis
---------------

Data Extraction
~~~~~~~~~~~~~~

**Basic Quantities:**

.. code-block:: python

   # Extract solution fields
   potential = results.potential
   electron_density = results.electron_density
   hole_density = results.hole_density
   electric_field = results.electric_field
   
   # Current densities
   Jn = results.electron_current_density
   Jp = results.hole_current_density
   J_total = Jn + Jp

**Derived Quantities:**

.. code-block:: python

   # Calculate recombination rate
   R = results.calculate_recombination_rate()
   
   # Calculate generation rate
   G = results.calculate_generation_rate()
   
   # Calculate energy densities (for energy transport)
   if hasattr(results, 'electron_temperature'):
       Wn = 1.5 * k_B * results.electron_temperature * results.electron_density
       Wp = 1.5 * k_B * results.hole_temperature * results.hole_density

Device Parameter Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**I-V Characteristics:**

.. code-block:: python

   from semidgfem.analysis import IVAnalysis
   
   iv = IVAnalysis(solver)
   
   # Voltage sweep
   voltages = np.linspace(0, 3.0, 31)
   currents = []
   
   for V in voltages:
       bc.set_voltage('drain', V)
       results = solver.solve(bc)
       current = iv.calculate_terminal_current('drain', results)
       currents.append(current)
   
   # Plot I-V curve
   plt.figure()
   plt.plot(voltages, np.array(currents) * 1e6, 'b-o')
   plt.xlabel('Voltage (V)')
   plt.ylabel('Current (μA)')
   plt.title('I-V Characteristics')
   plt.grid(True)
   plt.show()

**Parameter Extraction:**

.. code-block:: python

   from semidgfem.analysis import ParameterExtraction
   
   extractor = ParameterExtraction()
   
   # Extract threshold voltage
   Vth = extractor.extract_threshold_voltage(voltages, currents)
   
   # Extract transconductance
   gm = extractor.extract_transconductance(voltages, currents)
   
   # Extract output conductance
   gds = extractor.extract_output_conductance(vds_array, ids_matrix)
   
   # Extract mobility
   mobility = extractor.extract_mobility(device_geometry, Vth, gm)
   
   print(f"Threshold voltage: {Vth:.3f} V")
   print(f"Transconductance: {gm*1e6:.1f} μS")
   print(f"Output conductance: {gds*1e6:.1f} μS")
   print(f"Mobility: {mobility*1e4:.0f} cm²/V·s")

Visualization
~~~~~~~~~~~~

**2D Contour Plots:**

.. code-block:: python

   from semidgfem.visualization import ContourPlot
   
   plotter = ContourPlot(results)
   
   # Create multi-panel plot
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   # Potential
   plotter.plot_potential(ax=axes[0,0], levels=20)
   axes[0,0].set_title('Electrostatic Potential')
   
   # Electron density (log scale)
   plotter.plot_electron_density(ax=axes[0,1], log_scale=True)
   axes[0,1].set_title('Electron Density')
   
   # Electric field magnitude
   plotter.plot_electric_field_magnitude(ax=axes[1,0])
   axes[1,0].set_title('Electric Field Magnitude')
   
   # Current density streamlines
   plotter.plot_current_streamlines(ax=axes[1,1])
   axes[1,1].set_title('Current Flow')
   
   plt.tight_layout()
   plt.show()

**3D Surface Plots:**

.. code-block:: python

   from semidgfem.visualization import SurfacePlot
   
   surface = SurfacePlot(results)
   
   # 3D potential surface
   fig = plt.figure(figsize=(10, 8))
   ax = fig.add_subplot(111, projection='3d')
   
   surface.plot_potential_surface(ax=ax, colormap='viridis')
   ax.set_title('3D Potential Distribution')
   plt.show()

**Animation:**

.. code-block:: python

   from semidgfem.visualization import Animation
   
   # For transient results
   animator = Animation(time_points, solutions)
   
   # Create animation
   anim = animator.animate_potential(interval=100)  # ms
   
   # Save animation
   anim.save('potential_evolution.mp4', writer='ffmpeg')

This comprehensive usage guide provides detailed instructions for effectively using all features of the SemiDGFEM framework for advanced semiconductor device simulation.
