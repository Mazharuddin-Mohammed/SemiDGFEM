Python API Reference
====================

This section provides detailed documentation for the SemiDGFEM Python API.

Core Classes
------------

Device
~~~~~~

.. autoclass:: simulator.Device
   :members:
   :undoc-members:
   :show-inheritance:

The Device class represents a semiconductor device with material properties and geometry.

**Constructor**

.. code-block:: python

    Device(Lx: float, Ly: float, regions: List[Dict] = None)

**Parameters:**

- ``Lx`` (float): Device width in meters
- ``Ly`` (float): Device height in meters  
- ``regions`` (List[Dict], optional): Material regions with properties

**Example:**

.. code-block:: python

    # Simple silicon device
    device = Device(Lx=1e-6, Ly=0.5e-6)
    
    # Multi-region device
    regions = [
        {"material": "Si", "x_min": 0, "x_max": 1e-6, "y_min": 0, "y_max": 0.5e-6},
        {"material": "SiO2", "x_min": 1e-6, "x_max": 1.2e-6, "y_min": 0, "y_max": 0.5e-6}
    ]
    device = Device(Lx=1.2e-6, Ly=0.5e-6, regions=regions)

**Methods:**

.. py:method:: get_epsilon_at(x: float, y: float) -> float

   Get permittivity at specified coordinates.
   
   :param x: X-coordinate in meters
   :param y: Y-coordinate in meters
   :returns: Relative permittivity at (x,y)
   :raises: ValueError if coordinates are outside device bounds

.. py:method:: get_extents() -> List[float]

   Get device dimensions.
   
   :returns: [width, height] in meters

.. py:method:: is_valid() -> bool

   Check if device is in valid state.
   
   :returns: True if device is valid

Simulator
~~~~~~~~~

.. autoclass:: simulator.Simulator
   :members:
   :undoc-members:
   :show-inheritance:

Main simulation class for semiconductor device analysis.

**Constructor**

.. code-block:: python

    Simulator(device: Device, method: Method, mesh_type: MeshType, order: int = 3)

**Parameters:**

- ``device`` (Device): Device to simulate
- ``method`` (Method): Numerical method (Method.DG for Discontinuous Galerkin)
- ``mesh_type`` (MeshType): Mesh type (MeshType.Structured or MeshType.Unstructured)
- ``order`` (int): Polynomial order (1, 2, or 3)

**Example:**

.. code-block:: python

    from simulator import Simulator, Device, Method, MeshType
    
    device = Device(Lx=2e-6, Ly=1e-6)
    sim = Simulator(device, Method.DG, MeshType.Structured, order=3)

**Methods:**

.. py:method:: set_doping(Nd: np.ndarray, Na: np.ndarray) -> None

   Set doping concentrations.
   
   :param Nd: Donor concentration array (m⁻³)
   :param Na: Acceptor concentration array (m⁻³)
   :raises: ValueError if array sizes don't match DOF count

.. py:method:: solve_drift_diffusion(bc: List[float], Vg: float = 0.0, max_steps: int = 100, use_amr: bool = False, poisson_max_iter: int = 50, poisson_tol: float = 1e-6) -> Dict[str, np.ndarray]

   Solve coupled Poisson and drift-diffusion equations.
   
   :param bc: Boundary conditions [left, right, bottom, top] in volts
   :param Vg: Gate voltage in volts
   :param max_steps: Maximum self-consistent iterations
   :param use_amr: Enable adaptive mesh refinement
   :param poisson_max_iter: Maximum Poisson solver iterations
   :param poisson_tol: Poisson solver tolerance
   :returns: Dictionary with simulation results
   :raises: RuntimeError if solver fails to converge

   **Returns:**
   
   Dictionary containing:
   
   - ``V``: Electrostatic potential (V)
   - ``n``: Electron concentration (m⁻³)
   - ``p``: Hole concentration (m⁻³)
   - ``Jn``: Electron current density (A/m²)
   - ``Jp``: Hole current density (A/m²)

.. py:method:: get_dof_count() -> int

   Get number of degrees of freedom.
   
   :returns: Total DOF count

.. py:method:: enable_gpu(enable: bool) -> None

   Enable or disable GPU acceleration.
   
   :param enable: True to enable GPU acceleration

.. py:method:: is_gpu_enabled() -> bool

   Check if GPU acceleration is enabled.
   
   :returns: True if GPU is enabled

Enumerations
------------

Method
~~~~~~

.. py:class:: Method

   Numerical solution methods.
   
   .. py:attribute:: DG
   
      Discontinuous Galerkin method

MeshType
~~~~~~~~

.. py:class:: MeshType

   Mesh generation types.
   
   .. py:attribute:: Structured
   
      Regular structured mesh
   
   .. py:attribute:: Unstructured
   
      Irregular unstructured mesh

Utility Functions
-----------------

Material Properties
~~~~~~~~~~~~~~~~~~~

.. py:function:: get_material_properties(material: str) -> Dict[str, float]

   Get material properties for common semiconductors.
   
   :param material: Material name ("Si", "GaAs", "SiO2", etc.)
   :returns: Dictionary with material properties
   
   **Example:**
   
   .. code-block:: python
   
       props = get_material_properties("Si")
       print(f"Silicon bandgap: {props['Eg']} eV")
       print(f"Silicon permittivity: {props['epsilon']}")

Physics Utilities
~~~~~~~~~~~~~~~~~

.. py:function:: calculate_built_in_potential(Na: float, Nd: float, ni: float = 1e10, T: float = 300) -> float

   Calculate built-in potential for P-N junction.
   
   :param Na: Acceptor concentration (cm⁻³)
   :param Nd: Donor concentration (cm⁻³)
   :param ni: Intrinsic carrier concentration (cm⁻³)
   :param T: Temperature (K)
   :returns: Built-in potential (V)

.. py:function:: calculate_depletion_width(Na: float, Nd: float, V_applied: float = 0.0, epsilon_r: float = 11.7) -> float

   Calculate depletion width for P-N junction.
   
   :param Na: Acceptor concentration (cm⁻³)
   :param Nd: Donor concentration (cm⁻³)
   :param V_applied: Applied voltage (V)
   :param epsilon_r: Relative permittivity
   :returns: Depletion width (m)

Visualization
~~~~~~~~~~~~~

.. py:function:: plot_2d_results(results: Dict[str, np.ndarray], nx: int, ny: int, device_extents: List[float]) -> None

   Create 2D visualization of simulation results.
   
   :param results: Simulation results dictionary
   :param nx: Number of grid points in x-direction
   :param ny: Number of grid points in y-direction
   :param device_extents: [width, height] in meters

.. py:function:: plot_1d_profile(results: Dict[str, np.ndarray], direction: str = 'x', position: float = 0.5) -> None

   Create 1D profile plot along specified direction.
   
   :param results: Simulation results dictionary
   :param direction: 'x' or 'y' direction
   :param position: Relative position (0-1) along other axis

Error Handling
--------------

SemiDGFEM defines several custom exceptions:

.. py:exception:: SimulationError

   Base exception for simulation errors.

.. py:exception:: ConvergenceError

   Raised when solver fails to converge.
   
   **Example:**
   
   .. code-block:: python
   
       try:
           results = sim.solve_drift_diffusion(bc=[0, 1, 0, 0])
       except ConvergenceError as e:
           print(f"Solver failed to converge: {e}")
           # Try with relaxed tolerance
           results = sim.solve_drift_diffusion(bc=[0, 1, 0, 0], poisson_tol=1e-5)

.. py:exception:: MeshError

   Raised for mesh generation problems.

.. py:exception:: DeviceError

   Raised for invalid device configurations.

Examples
--------

Complete P-N Junction Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from simulator import Simulator, Device, Method, MeshType
    
    def simulate_pn_junction():
        # Create device
        device = Device(Lx=2e-6, Ly=1e-6)
        sim = Simulator(device, Method.DG, MeshType.Structured, order=3)
        
        # Set doping
        n_points = sim.get_dof_count()
        Nd = np.zeros(n_points)
        Na = np.zeros(n_points)
        
        # P-N junction
        Na[:n_points//2] = 1e17 * 1e6  # P-side
        Nd[n_points//2:] = 1e17 * 1e6  # N-side
        
        sim.set_doping(Nd, Na)
        
        # Simulate I-V characteristics
        voltages = np.linspace(-1, 1, 21)
        currents = []
        
        for V in voltages:
            results = sim.solve_drift_diffusion(bc=[0, V, 0, 0])
            I = np.sum(results['Jn'] + results['Jp'])
            currents.append(I)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.semilogy(voltages, np.abs(currents))
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.title('P-N Junction I-V Characteristics')
        plt.grid(True)
        plt.show()
        
        return voltages, currents

MOSFET Transfer Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def simulate_mosfet_transfer():
        # Setup MOSFET device
        device = Device(Lx=1e-6, Ly=0.5e-6)
        sim = Simulator(device, Method.DG, MeshType.Structured, order=3)
        
        # MOSFET doping profile
        n_points = sim.get_dof_count()
        nx, ny = 100, 50
        
        Nd = np.zeros(n_points)
        Na = np.full(n_points, 1e17 * 1e6)  # P-substrate
        
        # N+ source/drain
        for i in range(n_points):
            x_idx = i % nx
            y_idx = i // nx
            
            if y_idx >= 40:  # Surface region
                if x_idx < 25 or x_idx >= 75:  # Source/drain
                    Nd[i] = 1e20 * 1e6
                    Na[i] = 0
        
        sim.set_doping(Nd, Na)
        
        # Transfer characteristics
        Vg_values = np.linspace(0, 2, 21)
        Id_values = []
        
        for Vg in Vg_values:
            results = sim.solve_drift_diffusion(bc=[0, 0.1, 0, Vg])
            # Calculate drain current (simplified)
            Id = calculate_drain_current(results, nx, ny)
            Id_values.append(Id)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.semilogy(Vg_values, np.abs(Id_values))
        plt.xlabel('Gate Voltage (V)')
        plt.ylabel('Drain Current (A)')
        plt.title('MOSFET Transfer Characteristics')
        plt.grid(True)
        plt.show()
        
        return Vg_values, Id_values
