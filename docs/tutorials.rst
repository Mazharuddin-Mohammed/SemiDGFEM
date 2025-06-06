Tutorials
=========

This section provides step-by-step tutorials for using SemiDGFEM to simulate various semiconductor devices.

.. toctree::
   :maxdepth: 2

   tutorials/getting_started
   tutorials/pn_junction
   tutorials/mosfet_simulation
   tutorials/heterostructure_devices
   tutorials/advanced_features

Getting Started
---------------

Your First Simulation
~~~~~~~~~~~~~~~~~~~~~

Let's start with a simple P-N junction diode simulation to get familiar with SemiDGFEM.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from simulator import Simulator, Device, Method, MeshType

    # Step 1: Create a device
    device = Device(Lx=2e-6, Ly=1e-6)  # 2μm × 1μm device
    
    # Step 2: Initialize simulator
    sim = Simulator(device, Method.DG, MeshType.Structured, order=3)
    
    # Step 3: Set up doping profile
    n_points = sim.get_dof_count()
    Nd = np.zeros(n_points)
    Na = np.zeros(n_points)
    
    # Create P-N junction
    Na[:n_points//2] = 1e16 * 1e6  # P-region (left half)
    Nd[n_points//2:] = 1e16 * 1e6  # N-region (right half)
    
    sim.set_doping(Nd, Na)
    
    # Step 4: Run simulation
    results = sim.solve_drift_diffusion(
        bc=[0.0, 0.7, 0.0, 0.0],  # 0.7V forward bias
        use_amr=True,
        max_steps=100
    )
    
    # Step 5: Analyze results
    print(f"Total current: {np.sum(results['Jn'] + results['Jp']):.2e} A")
    print(f"Peak electron density: {np.max(results['n']):.2e} m⁻³")

Understanding the Results
~~~~~~~~~~~~~~~~~~~~~~~~

The simulation returns a dictionary with the following keys:

- ``V``: Electrostatic potential (V)
- ``n``: Electron concentration (m⁻³)
- ``p``: Hole concentration (m⁻³)
- ``Jn``: Electron current density (A/m²)
- ``Jp``: Hole current density (A/m²)

Visualization
~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Reshape results for 2D plotting
    nx, ny = 100, 50  # Grid dimensions
    V_2d = results['V'].reshape(ny, nx)
    n_2d = results['n'].reshape(ny, nx)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Potential plot
    im1 = ax1.imshow(V_2d, extent=[0, 2e-6, 0, 1e-6], aspect='auto')
    ax1.set_title('Electrostatic Potential (V)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    plt.colorbar(im1, ax=ax1)
    
    # Electron density plot
    im2 = ax2.imshow(np.log10(n_2d), extent=[0, 2e-6, 0, 1e-6], aspect='auto')
    ax2.set_title('Log₁₀(Electron Density) (m⁻³)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

P-N Junction Tutorial
--------------------

Detailed P-N Junction Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial covers comprehensive P-N junction simulation including:

- Forward and reverse bias characteristics
- I-V curve generation
- Capacitance-voltage analysis
- Temperature effects

.. code-block:: python

    import numpy as np
    from simulator import Simulator, Device, Method, MeshType
    
    def simulate_pn_junction_iv():
        """Generate I-V characteristics for P-N junction"""
        
        # Device setup
        device = Device(Lx=2e-6, Ly=1e-6)
        sim = Simulator(device, Method.DG, MeshType.Structured, order=3)
        
        # Doping profile
        n_points = sim.get_dof_count()
        Nd = np.zeros(n_points)
        Na = np.zeros(n_points)
        
        # Abrupt junction at center
        junction_pos = n_points // 2
        Na[:junction_pos] = 1e17 * 1e6  # P-side: 1e17 cm⁻³
        Nd[junction_pos:] = 1e17 * 1e6  # N-side: 1e17 cm⁻³
        
        sim.set_doping(Nd, Na)
        
        # Voltage sweep
        voltages = np.linspace(-1.0, 1.0, 21)
        currents = []
        
        for V in voltages:
            print(f"Simulating V = {V:.2f} V")
            
            # Boundary conditions: [left, right, bottom, top]
            bc = [0.0, V, 0.0, 0.0]
            
            results = sim.solve_drift_diffusion(
                bc=bc,
                max_steps=50,
                use_amr=True
            )
            
            # Calculate total current
            I_total = np.sum(results['Jn'] + results['Jp'])
            currents.append(I_total)
        
        return voltages, currents

MOSFET Simulation Tutorial
--------------------------

Complete MOSFET Device Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial demonstrates how to simulate a MOSFET device with proper:

- Device structure setup
- Gate-oxide coupling
- Source/drain contacts
- I-V characteristics

.. code-block:: python

    from simulator import Simulator, Device, Method, MeshType
    import numpy as np
    
    def setup_mosfet_device():
        """Setup MOSFET device structure"""
        
        # Create device (1μm × 0.5μm)
        device = Device(Lx=1e-6, Ly=0.5e-6)
        sim = Simulator(device, Method.DG, MeshType.Structured, order=3)
        
        # Get grid dimensions
        n_points = sim.get_dof_count()
        nx, ny = 100, 50  # Assuming structured grid
        
        # Initialize doping arrays
        Nd = np.zeros(n_points)
        Na = np.zeros(n_points)
        
        # MOSFET structure parameters
        source_end = nx // 4      # Source region end
        drain_start = 3 * nx // 4 # Drain region start
        surface_depth = 4 * ny // 5  # Surface region depth
        
        # Set up doping profile
        for i in range(n_points):
            x_idx = i % nx
            y_idx = i // nx
            
            # Default: P-type substrate
            Na[i] = 1e17 * 1e6  # 1e17 cm⁻³
            
            # N+ source/drain regions at surface
            if y_idx >= surface_depth:
                if x_idx < source_end:
                    # N+ source
                    Nd[i] = 1e20 * 1e6
                    Na[i] = 0
                elif x_idx >= drain_start:
                    # N+ drain
                    Nd[i] = 1e20 * 1e6
                    Na[i] = 0
        
        sim.set_doping(Nd, Na)
        return sim
    
    def simulate_mosfet_transfer():
        """Simulate MOSFET transfer characteristics"""
        
        sim = setup_mosfet_device()
        
        # Gate voltage sweep
        Vg_values = np.linspace(0, 2.0, 21)
        Vd = 0.1  # Small drain voltage for linear region
        
        drain_currents = []
        
        for Vg in Vg_values:
            print(f"Simulating Vg = {Vg:.2f} V")
            
            # Boundary conditions: [source, drain, substrate, gate]
            bc = [0.0, Vd, 0.0, Vg]
            
            results = sim.solve_drift_diffusion(
                bc=bc,
                max_steps=100,
                use_amr=True
            )
            
            # Calculate drain current (simplified)
            # In practice, this would integrate current density
            # across the drain contact
            Id = calculate_drain_current(results, sim)
            drain_currents.append(Id)
        
        return Vg_values, drain_currents

Advanced Features
-----------------

Adaptive Mesh Refinement
~~~~~~~~~~~~~~~~~~~~~~~~

SemiDGFEM supports automatic mesh refinement to improve accuracy in regions with high gradients:

.. code-block:: python

    # Enable AMR with custom parameters
    results = sim.solve_drift_diffusion(
        bc=[0.0, 0.7, 0.0, 0.0],
        use_amr=True,
        amr_max_levels=3,
        amr_threshold=0.1
    )

GPU Acceleration
~~~~~~~~~~~~~~~

For large problems, enable GPU acceleration:

.. code-block:: python

    # Enable GPU acceleration
    sim.enable_gpu(True)
    
    # Check GPU status
    if sim.is_gpu_enabled():
        print("GPU acceleration active")
    else:
        print("Using CPU solver")

Custom Physics Models
~~~~~~~~~~~~~~~~~~~~

Implement custom mobility models:

.. code-block:: python

    def custom_mobility_model(N_total, carrier_type, temperature=300):
        """Custom mobility model with temperature dependence"""
        
        if carrier_type == 'electron':
            mu_min = 88.0
            mu_max = 1417.0
            N_ref = 9.68e16
            alpha = 0.711
        else:  # hole
            mu_min = 54.3
            mu_max = 470.5
            N_ref = 2.23e17
            alpha = 0.719
        
        # Temperature scaling
        T_ratio = temperature / 300.0
        mu_max *= T_ratio**(-2.3)
        mu_min *= T_ratio**(-0.57)
        
        # Caughey-Thomas model
        mu = mu_min + (mu_max - mu_min) / (1 + (N_total / N_ref)**alpha)
        
        return mu
    
    # Use custom model
    sim.set_mobility_model(custom_mobility_model)

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Convergence Problems**

If the solver fails to converge:

.. code-block:: python

    # Increase tolerance and iterations
    results = sim.solve_drift_diffusion(
        bc=[0.0, 0.7, 0.0, 0.0],
        max_steps=200,
        poisson_tol=1e-8,
        poisson_max_iter=100
    )

**Memory Issues**

For large problems:

.. code-block:: python

    # Use structured mesh for better memory efficiency
    sim = Simulator(device, Method.DG, MeshType.Structured, order=2)
    
    # Enable GPU for large problems
    sim.enable_gpu(True)

**Boundary Condition Issues**

Ensure proper boundary condition setup:

.. code-block:: python

    # Boundary conditions: [left, right, bottom, top]
    # For MOSFET: [source, drain, substrate, gate]
    bc = [0.0, 1.0, 0.0, 0.8]  # Vs, Vd, Vsub, Vg
    
    # Validate boundary conditions
    assert len(bc) == 4, "Need exactly 4 boundary conditions"
    assert all(np.isfinite(v) for v in bc), "All BCs must be finite"

Heterostructure Device Tutorial
------------------------------

GaAs/AlGaAs HEMT Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial demonstrates simulation of a High Electron Mobility Transistor (HEMT) using GaAs/AlGaAs heterostructure:

.. code-block:: python

    from heterostructure_simulation import HeterostructureDevice, LayerStructure, SemiconductorMaterial
    import numpy as np
    import matplotlib.pyplot as plt

    def simulate_gaas_algaas_hemt():
        """Simulate GaAs/AlGaAs HEMT structure"""

        # Define heterostructure layers
        layers = [
            # GaAs buffer layer
            LayerStructure(
                material=SemiconductorMaterial.GAAS,
                thickness=500.0,  # nm
                composition=0.0,
                doping_type="intrinsic",
                doping_concentration=1e14,
                position=0.0
            ),

            # GaAs quantum well
            LayerStructure(
                material=SemiconductorMaterial.GAAS,
                thickness=20.0,   # nm
                composition=0.0,
                doping_type="intrinsic",
                doping_concentration=1e14,
                position=500.0
            ),

            # AlGaAs spacer
            LayerStructure(
                material=SemiconductorMaterial.ALGAS,
                thickness=5.0,    # nm
                composition=0.3,  # Al₀.₃Ga₀.₇As
                doping_type="intrinsic",
                doping_concentration=1e14,
                position=520.0
            ),

            # AlGaAs barrier (doped)
            LayerStructure(
                material=SemiconductorMaterial.ALGAS,
                thickness=30.0,   # nm
                composition=0.3,
                doping_type="n",
                doping_concentration=2e18,
                position=525.0
            ),

            # GaAs cap layer
            LayerStructure(
                material=SemiconductorMaterial.GAAS,
                thickness=10.0,   # nm
                composition=0.0,
                doping_type="n",
                doping_concentration=1e19,
                position=555.0
            )
        ]

        # Create heterostructure device
        hetero = HeterostructureDevice(layers, temperature=300.0)

        print(f"Total thickness: {hetero.total_thickness:.1f} nm")
        print(f"Number of layers: {len(hetero.layers)}")

        # Create mesh and calculate band structure
        hetero.create_mesh(nz=1000)
        band_structure = hetero.calculate_band_structure()
        carrier_densities = hetero.calculate_carrier_densities()

        # Analyze quantum wells
        quantum_wells = hetero.calculate_quantum_wells()

        print(f"Quantum wells detected: {len(quantum_wells)}")
        for i, qw in enumerate(quantum_wells):
            print(f"  Well {i+1}: {qw['width']:.1f} nm, depth: {qw['depth']:.3f} eV")

        # Calculate transport properties
        transport = hetero.calculate_transport_properties()

        # Plot results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        z = band_structure['z'] * 1e9  # Convert to nm

        # Band structure
        ax1.plot(z, band_structure['conduction_band'], 'b-', linewidth=2, label='Conduction Band')
        ax1.plot(z, band_structure['valence_band'], 'r-', linewidth=2, label='Valence Band')
        ax1.set_xlabel('Position (nm)')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title('Band Structure')
        ax1.legend()
        ax1.grid(True)

        # Carrier densities
        ax2.semilogy(z, carrier_densities['electron_density'], 'b-', linewidth=2, label='Electrons')
        ax2.semilogy(z, carrier_densities['hole_density'], 'r-', linewidth=2, label='Holes')
        ax2.set_xlabel('Position (nm)')
        ax2.set_ylabel('Carrier Density (cm⁻³)')
        ax2.set_title('Carrier Densities')
        ax2.legend()
        ax2.grid(True)

        # Electric field
        electric_field = -np.gradient(band_structure['potential'], z*1e-9)
        ax3.plot(z, electric_field*1e-5, 'purple', linewidth=2)
        ax3.set_xlabel('Position (nm)')
        ax3.set_ylabel('Electric Field (V/cm)')
        ax3.set_title('Electric Field')
        ax3.grid(True)

        # Mobility
        mu_e = transport['electron_mobility'] * 1e4  # Convert to cm²/V·s
        ax4.plot(z, mu_e, 'green', linewidth=2)
        ax4.set_xlabel('Position (nm)')
        ax4.set_ylabel('Electron Mobility (cm²/V·s)')
        ax4.set_title('Electron Mobility')
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

        return hetero, band_structure, carrier_densities, transport

Performance Optimization Tutorial
---------------------------------

GPU Acceleration Setup
~~~~~~~~~~~~~~~~~~~~~~

Learn how to maximize performance using GPU acceleration:

.. code-block:: python

    from gpu_acceleration import GPUAcceleratedSolver
    import numpy as np

    def setup_gpu_acceleration():
        """Setup and benchmark GPU acceleration"""

        # Create GPU solver
        gpu_solver = GPUAcceleratedSolver()

        # Check GPU availability
        if gpu_solver.is_gpu_available():
            print("GPU acceleration available!")

            # Get performance info
            perf_info = gpu_solver.get_performance_info()
            print(f"GPU backend: {perf_info['backend']}")
            print(f"Device name: {perf_info['device_name']}")
            print(f"Memory: {perf_info['memory_gb']:.1f} GB")

            # Benchmark performance
            benchmark = gpu_solver.benchmark_performance(size=50000)
            print(f"GPU speedup: {benchmark['speedup']:.1f}x")

            return gpu_solver
        else:
            print("GPU not available, using CPU")
            return None

    def gpu_accelerated_simulation():
        """Run simulation with GPU acceleration"""

        gpu_solver = setup_gpu_acceleration()

        if gpu_solver:
            # Large-scale simulation data
            potential = np.linspace(0, 1.0, 100000)
            doping_nd = np.full(100000, 1e17)
            doping_na = np.full(100000, 1e16)

            # GPU-accelerated transport solution
            results = gpu_solver.solve_transport_gpu(potential, doping_nd, doping_na)

            print(f"GPU solution fields: {list(results.keys())}")
            print(f"Max electron density: {np.max(results['electron_density']):.2e} cm⁻³")

SIMD Optimization Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize performance using SIMD vectorization:

.. code-block:: python

    from performance_bindings import PerformanceOptimizer
    import numpy as np
    import time

    def simd_optimization_demo():
        """Demonstrate SIMD optimization benefits"""

        # Create performance optimizer
        optimizer = PerformanceOptimizer()

        # Print system capabilities
        optimizer.print_performance_info()

        # Large vector operations
        size = 1000000
        a = np.random.random(size)
        b = np.random.random(size)

        # Benchmark standard NumPy
        start_time = time.time()
        result_numpy = a + b
        numpy_time = time.time() - start_time

        # Benchmark SIMD-optimized
        start_time = time.time()
        result_simd = optimizer.optimize_computation('vector_add', a, b)
        simd_time = time.time() - start_time

        # Compare results
        speedup = numpy_time / simd_time
        print(f"NumPy time: {numpy_time:.4f}s")
        print(f"SIMD time: {simd_time:.4f}s")
        print(f"Speedup: {speedup:.1f}x")

        # Verify correctness
        max_error = np.max(np.abs(result_numpy - result_simd))
        print(f"Maximum error: {max_error:.2e}")

Advanced Analysis Tutorial
--------------------------

Parameter Extraction
~~~~~~~~~~~~~~~~~~~~

Extract device parameters from simulation results:

.. code-block:: python

    from mosfet_simulation import MOSFETDevice, MOSFETType, DeviceGeometry, DopingProfile
    import numpy as np

    def extract_mosfet_parameters():
        """Extract MOSFET device parameters"""

        # Create MOSFET device
        geometry = DeviceGeometry(
            length=0.18, width=10.0, tox=4.0, xj=0.15,
            channel_length=0.18, source_length=0.5, drain_length=0.5
        )

        doping = DopingProfile(
            substrate_doping=1e17, source_drain_doping=1e20,
            channel_doping=5e16, gate_doping=1e20, profile_type="uniform"
        )

        nmos = MOSFETDevice(MOSFETType.NMOS, geometry, doping)

        # Calculate I-V characteristics
        vgs_range = np.linspace(0, 3.0, 16)
        vds_range = np.linspace(0, 3.0, 21)

        print("Calculating I-V characteristics...")
        iv_data = nmos.calculate_iv_characteristics(vgs_range, vds_range)

        # Extract device parameters
        parameters = nmos.extract_device_parameters(iv_data)

        print(f"\nExtracted Parameters:")
        print(f"  Threshold Voltage: {parameters['threshold_voltage']:.3f} V")
        print(f"  Transconductance: {parameters['transconductance']*1e6:.1f} μS")
        print(f"  Output Conductance: {parameters['output_conductance']*1e6:.1f} μS")
        print(f"  Intrinsic Gain: {parameters['intrinsic_gain']:.1f}")
        print(f"  Mobility: {parameters['mobility']*1e4:.0f} cm²/V·s")
        print(f"  Subthreshold Slope: {parameters['subthreshold_slope']:.1f} mV/decade")

        # Plot device characteristics
        nmos.plot_device_characteristics(iv_data, 'mosfet_iv.png')

        return parameters

Visualization Tutorial
~~~~~~~~~~~~~~~~~~~~~

Create professional plots and animations:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation

    def create_professional_plots(results):
        """Create publication-quality plots"""

        # Set up matplotlib for publication
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.linewidth': 1.5,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'figure.dpi': 300
        })

        # Create multi-panel figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Potential distribution
        im1 = axes[0,0].contourf(results.x*1e6, results.y*1e6, results.potential,
                                levels=20, cmap='viridis')
        axes[0,0].set_title('Electrostatic Potential (V)')
        axes[0,0].set_xlabel('x (μm)')
        axes[0,0].set_ylabel('y (μm)')
        plt.colorbar(im1, ax=axes[0,0])

        # Electron density (log scale)
        im2 = axes[0,1].contourf(results.x*1e6, results.y*1e6,
                                np.log10(results.electron_density),
                                levels=20, cmap='plasma')
        axes[0,1].set_title('log₁₀(Electron Density) [cm⁻³]')
        axes[0,1].set_xlabel('x (μm)')
        axes[0,1].set_ylabel('y (μm)')
        plt.colorbar(im2, ax=axes[0,1])

        # Electric field magnitude
        E_mag = np.sqrt(results.electric_field_x**2 + results.electric_field_y**2)
        im3 = axes[0,2].contourf(results.x*1e6, results.y*1e6, E_mag*1e-5,
                                levels=20, cmap='hot')
        axes[0,2].set_title('Electric Field Magnitude (V/cm)')
        axes[0,2].set_xlabel('x (μm)')
        axes[0,2].set_ylabel('y (μm)')
        plt.colorbar(im3, ax=axes[0,2])

        # Current density streamlines
        axes[1,0].streamplot(results.x*1e6, results.y*1e6,
                           results.current_density_x, results.current_density_y,
                           density=2, color='blue', linewidth=1)
        axes[1,0].set_title('Current Flow Lines')
        axes[1,0].set_xlabel('x (μm)')
        axes[1,0].set_ylabel('y (μm)')

        # 1D cuts along channel
        y_center = len(results.y) // 2
        axes[1,1].plot(results.x*1e6, results.potential[y_center, :], 'b-',
                      label='Potential')
        axes[1,1].set_xlabel('x (μm)')
        axes[1,1].set_ylabel('Potential (V)')
        axes[1,1].set_title('Potential Along Channel')
        axes[1,1].legend()

        # Carrier densities along channel
        axes[1,2].semilogy(results.x*1e6, results.electron_density[y_center, :],
                          'b-', label='Electrons')
        axes[1,2].semilogy(results.x*1e6, results.hole_density[y_center, :],
                          'r-', label='Holes')
        axes[1,2].set_xlabel('x (μm)')
        axes[1,2].set_ylabel('Carrier Density (cm⁻³)')
        axes[1,2].set_title('Carrier Densities Along Channel')
        axes[1,2].legend()

        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

This comprehensive tutorial collection provides hands-on experience with all major features of the SemiDGFEM framework, from basic device physics to advanced optimization techniques.
