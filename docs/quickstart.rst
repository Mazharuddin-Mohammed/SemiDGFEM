Quick Start Guide
=================

This guide will get you up and running with SemiDGFEM in minutes.

.. contents:: Table of Contents
   :local:
   :depth: 2

Installation
------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.8 or higher
- CMake 3.16 or higher
- C++ compiler with C++17 support
- Git

Quick Installation
~~~~~~~~~~~~~~~~~

**Option 1: Using pip (Recommended)**

.. code-block:: bash

   pip install semidgfem

**Option 2: From Source**

.. code-block:: bash

   git clone https://github.com/your-repo/SemiDGFEM.git
   cd SemiDGFEM
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   cd ../python
   pip install -e .

Verify Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import simulator
   print("SemiDGFEM installed successfully!")

Your First Simulation
---------------------

Basic PN Junction
~~~~~~~~~~~~~~~~

Let's simulate a simple PN junction diode:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from semidgfem import *

   # Create 1D device
   device = Device1D(length=2e-6)  # 2 μm
   
   # Define doping profile
   x = np.linspace(0, 2e-6, 200)
   nd = np.where(x >= 1e-6, 1e17, 0)  # n-type right half
   na = np.where(x < 1e-6, 1e17, 0)   # p-type left half
   
   device.set_doping(nd, na)
   
   # Solve Poisson equation
   solver = PoissonSolver(device)
   potential = solver.solve()
   
   # Plot results
   plt.figure(figsize=(10, 6))
   
   plt.subplot(121)
   plt.plot(x*1e6, potential, 'b-', linewidth=2)
   plt.xlabel('Position (μm)')
   plt.ylabel('Potential (V)')
   plt.title('Built-in Potential')
   plt.grid(True)
   
   plt.subplot(122)
   plt.semilogy(x*1e6, nd, 'r-', label='n-type')
   plt.semilogy(x*1e6, na, 'b-', label='p-type')
   plt.xlabel('Position (μm)')
   plt.ylabel('Doping (cm⁻³)')
   plt.title('Doping Profile')
   plt.legend()
   plt.grid(True)
   
   plt.tight_layout()
   plt.show()

2D Device Simulation
~~~~~~~~~~~~~~~~~~~

Now let's try a 2D device with drift-diffusion transport:

.. code-block:: python

   from advanced_transport import create_drift_diffusion_solver
   
   # Create 2D device
   solver = create_drift_diffusion_solver(2e-6, 1e-6)  # 2μm × 1μm
   
   # Set doping profile
   size = 100
   nd = np.full(size, 1e17)  # n-type doping
   na = np.full(size, 1e16)  # p-type background
   solver.set_doping(nd, na)
   
   # Set boundary conditions
   boundary_conditions = [0, 1, 0, 0]  # [left, right, top, bottom] voltages
   
   # Solve transport equations
   results = solver.solve_transport(boundary_conditions, Vg=0.5)
   
   # Display results
   print(f"Solution fields: {list(results.keys())}")
   print(f"Max electron density: {np.max(results['electron_density']):.2e} cm⁻³")
   print(f"Max current density: {np.max(results['current_density_n']):.2e} A/cm²")

MOSFET Simulation
~~~~~~~~~~~~~~~~

Let's simulate a complete MOSFET device:

.. code-block:: python

   from mosfet_simulation import MOSFETDevice, MOSFETType, DeviceGeometry, DopingProfile
   
   # Define device geometry
   geometry = DeviceGeometry(
       length=0.5,          # Gate length (μm)
       width=10.0,          # Gate width (μm)
       tox=5.0,            # Oxide thickness (nm)
       xj=0.2,             # Junction depth (μm)
       channel_length=0.5,  # Channel length (μm)
       source_length=0.5,   # Source length (μm)
       drain_length=0.5     # Drain length (μm)
   )
   
   # Define doping profile
   doping = DopingProfile(
       substrate_doping=1e17,      # p-substrate (cm⁻³)
       source_drain_doping=1e20,   # n+ source/drain (cm⁻³)
       channel_doping=1e17,        # Channel doping (cm⁻³)
       gate_doping=1e20,           # n+ gate (cm⁻³)
       profile_type="uniform"
   )
   
   # Create NMOS device
   nmos = MOSFETDevice(MOSFETType.NMOS, geometry, doping)
   
   print(f"Threshold voltage: {nmos.vth:.3f} V")
   print(f"Oxide capacitance: {nmos.cox*1e4:.2f} μF/cm²")
   
   # Create mesh and solve single bias point
   nmos.create_device_mesh(nx=50, ny=25)
   current = nmos.calculate_drain_current(vgs=1.5, vds=1.0)
   print(f"Drain current: {current*1e6:.2f} μA")

Heterostructure Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulate a GaAs/AlGaAs heterostructure:

.. code-block:: python

   from heterostructure_simulation import (HeterostructureDevice, LayerStructure, 
                                         SemiconductorMaterial)
   
   # Define layer structure
   layers = [
       LayerStructure(
           material=SemiconductorMaterial.GAAS,
           thickness=100.0,  # nm
           composition=0.0,
           doping_type="intrinsic",
           doping_concentration=1e15,
           position=0.0
       ),
       LayerStructure(
           material=SemiconductorMaterial.ALGAS,
           thickness=50.0,   # nm
           composition=0.3,  # Al₀.₃Ga₀.₇As
           doping_type="n",
           doping_concentration=1e18,
           position=100.0
       )
   ]
   
   # Create heterostructure
   hetero = HeterostructureDevice(layers, temperature=300.0)
   
   print(f"Total thickness: {hetero.total_thickness:.1f} nm")
   
   # Calculate band structure
   hetero.create_mesh(nz=200)
   band_structure = hetero.calculate_band_structure()
   
   # Analyze quantum wells
   quantum_wells = hetero.calculate_quantum_wells()
   print(f"Quantum wells detected: {len(quantum_wells)}")

Performance Optimization
------------------------

Enable GPU Acceleration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gpu_acceleration import GPUAcceleratedSolver
   
   # Check GPU availability
   gpu_solver = GPUAcceleratedSolver()
   
   if gpu_solver.is_gpu_available():
       print("GPU acceleration available!")
       
       # Use GPU for large simulations
       potential = np.linspace(0, 1.0, 10000)
       doping_nd = np.full(10000, 1e17)
       doping_na = np.full(10000, 1e16)
       
       results = gpu_solver.solve_transport_gpu(potential, doping_nd, doping_na)
       print("GPU simulation completed!")
   else:
       print("GPU not available, using CPU")

Enable SIMD Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from performance_bindings import PerformanceOptimizer
   
   # Create optimizer
   optimizer = PerformanceOptimizer()
   
   # Check capabilities
   optimizer.print_performance_info()
   
   # Use optimized operations
   a = np.random.random(100000)
   b = np.random.random(100000)
   result = optimizer.optimize_computation('vector_add', a, b)

Next Steps
----------

Now that you've completed the quick start, here are some suggested next steps:

1. **Read the User Guide**: :doc:`usage` - Comprehensive usage instructions
2. **Try Tutorials**: :doc:`tutorials` - Step-by-step tutorials for complex devices
3. **Explore Examples**: :doc:`examples` - Working examples for various applications
4. **API Reference**: :doc:`python_api` - Complete function and class documentation
5. **Performance Optimization**: :doc:`performance_optimization` - Advanced optimization techniques

Common Issues
-------------

Import Errors
~~~~~~~~~~~~~

If you get import errors:

.. code-block:: bash

   # Set library path
   export LD_LIBRARY_PATH=/path/to/SemiDGFEM/build:$LD_LIBRARY_PATH
   export PYTHONPATH=/path/to/SemiDGFEM/python:$PYTHONPATH

GPU Not Detected
~~~~~~~~~~~~~~~~

If GPU acceleration is not working:

.. code-block:: bash

   # Check GPU status
   nvidia-smi  # For NVIDIA GPUs
   clinfo      # For OpenCL devices

Performance Issues
~~~~~~~~~~~~~~~~~

For slow simulations:

1. Enable GPU acceleration if available
2. Use SIMD optimization
3. Reduce mesh resolution for initial testing
4. Check system resources (CPU, memory)

Getting Help
------------

- **Documentation**: Complete guides and API reference
- **Examples**: Working code examples for all features
- **Issues**: Report bugs on GitHub
- **Discussions**: Ask questions in the community forum

You're now ready to start using SemiDGFEM for advanced semiconductor device simulation!
