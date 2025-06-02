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
