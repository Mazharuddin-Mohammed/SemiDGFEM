# SemiDGFEM API Reference

## Table of Contents
1. [Core Classes](#core-classes)
2. [Device Module](#device-module)
3. [Simulator Module](#simulator-module)
4. [GPU Module](#gpu-module)
5. [Mesh Module](#mesh-module)
6. [Physics Module](#physics-module)
7. [Visualization Module](#visualization-module)
8. [Utilities](#utilities)

## Core Classes

### Device

Represents a semiconductor device with material properties and geometry.

```python
class Device:
    def __init__(self, Lx: float, Ly: float, regions: List[Dict] = None)
```

**Parameters:**
- `Lx` (float): Device length in meters
- `Ly` (float): Device width in meters  
- `regions` (List[Dict], optional): Material regions specification

**Methods:**

#### `get_epsilon_at(x: float, y: float) -> float`
Get permittivity at spatial coordinates.

**Parameters:**
- `x` (float): X coordinate in meters
- `y` (float): Y coordinate in meters

**Returns:**
- `float`: Relative permittivity at (x,y)

#### `get_extents() -> List[float]`
Get device dimensions.

**Returns:**
- `List[float]`: [Lx, Ly] in meters

#### `validate() -> bool`
Validate device parameters.

**Returns:**
- `bool`: True if device is valid

**Example:**
```python
# Simple device
device = Device(Lx=1e-6, Ly=0.5e-6)

# Multi-region device
regions = [
    {"material": "Si", "x_min": 0, "x_max": 1e-6, "y_min": 0, "y_max": 0.5e-6},
    {"material": "SiO2", "x_min": 1e-6, "x_max": 1.2e-6, "y_min": 0, "y_max": 0.5e-6}
]
device = Device(Lx=1.2e-6, Ly=0.5e-6, regions=regions)

# Check permittivity
eps = device.get_epsilon_at(0.5e-6, 0.25e-6)
print(f"Permittivity: {eps}")
```

### Simulator

Main simulation class for semiconductor device analysis.

```python
class Simulator:
    def __init__(self, device: Device, method: Method, mesh_type: MeshType, order: int = 3)
```

**Parameters:**
- `device` (Device): Device to simulate
- `method` (Method): Numerical method (FDM, FEM, DG, etc.)
- `mesh_type` (MeshType): Mesh type (Structured, Unstructured)
- `order` (int): Polynomial order for DG methods (1, 2, or 3)

**Methods:**

#### `set_doping(Nd: np.ndarray, Na: np.ndarray) -> None`
Set doping concentrations.

**Parameters:**
- `Nd` (np.ndarray): Donor concentration in cm⁻³
- `Na` (np.ndarray): Acceptor concentration in cm⁻³

#### `set_trap_level(Et: np.ndarray) -> None`
Set trap energy levels.

**Parameters:**
- `Et` (np.ndarray): Trap energy levels in eV

#### `solve_drift_diffusion(bc: List[float], Vg: float = 0.0, max_steps: int = 100, use_amr: bool = False, poisson_max_iter: int = 50, poisson_tol: float = 1e-6) -> Dict[str, np.ndarray]`
Solve coupled Poisson-drift-diffusion equations.

**Parameters:**
- `bc` (List[float]): Boundary conditions [left, right, bottom, top] in V
- `Vg` (float): Gate voltage in V
- `max_steps` (int): Maximum iteration steps
- `use_amr` (bool): Enable adaptive mesh refinement
- `poisson_max_iter` (int): Maximum Poisson iterations
- `poisson_tol` (float): Poisson solver tolerance

**Returns:**
- `Dict[str, np.ndarray]`: Results dictionary with keys:
  - `'potential'`: Electrostatic potential (V)
  - `'n'`: Electron density (cm⁻³)
  - `'p'`: Hole density (cm⁻³)
  - `'Jn'`: Electron current density (A/m²)
  - `'Jp'`: Hole current density (A/m²)

#### `enable_gpu(enable: bool) -> None`
Enable/disable GPU acceleration.

**Parameters:**
- `enable` (bool): Whether to use GPU

#### `set_num_threads(num_threads: int) -> None`
Set number of OpenMP threads.

**Parameters:**
- `num_threads` (int): Number of threads (-1 for auto)

#### `get_dof_count() -> int`
Get number of degrees of freedom.

**Returns:**
- `int`: Total DOFs in current mesh

#### `get_convergence_residual() -> float`
Get convergence residual from last solve.

**Returns:**
- `float`: L2 norm of residual

**Example:**
```python
import numpy as np
from simulator import Simulator, Device, Method, MeshType

# Create device and simulator
device = Device(1e-6, 0.5e-6)
sim = Simulator(device, Method.DG, MeshType.Structured, order=3)

# Set doping
n_points = sim.get_dof_count()
Nd = np.full(n_points, 1e17)
Na = np.zeros(n_points)
sim.set_doping(Nd, Na)

# Enable optimizations
sim.enable_gpu(True)
sim.set_num_threads(-1)

# Solve
bc = [0.0, 1.0, 0.0, 0.0]  # 1V bias
results = sim.solve_drift_diffusion(bc, use_amr=True)

print(f"DOFs: {sim.get_dof_count()}")
print(f"Residual: {sim.get_convergence_residual():.2e}")
```

## Device Module

### Enumerations

#### Method
```python
class Method(Enum):
    FDM = 0    # Finite Difference Method
    FEM = 1    # Finite Element Method  
    FVM = 2    # Finite Volume Method
    SEM = 3    # Spectral Element Method
    MC = 4     # Monte Carlo Method
    DG = 5     # Discontinuous Galerkin Method
```

#### MeshType
```python
class MeshType(Enum):
    Structured = 0      # Regular grid
    Unstructured = 1    # Irregular triangulation
```

### Material Properties

#### `get_material_property(material: str, property: str, temperature: float = 300.0) -> float`
Get material property value.

**Parameters:**
- `material` (str): Material name ("Si", "GaAs", "SiO2", etc.)
- `property` (str): Property name ("epsilon", "bandgap", "mobility_n", etc.)
- `temperature` (float): Temperature in Kelvin

**Returns:**
- `float`: Property value in SI units

**Example:**
```python
from simulator.device import get_material_property

# Silicon properties at 300K
eps_si = get_material_property("Si", "epsilon", 300.0)
bandgap = get_material_property("Si", "bandgap", 300.0)
mu_n = get_material_property("Si", "mobility_n", 300.0)

print(f"Si permittivity: {eps_si}")
print(f"Si bandgap: {bandgap} eV")
print(f"Si electron mobility: {mu_n} m²/V·s")
```

## Simulator Module

### Performance Control

#### `enable_simd(enable: bool) -> None`
Enable SIMD vectorization.

#### `enable_profiling(enable: bool) -> None`
Enable performance profiling.

#### `get_profile_data() -> List[Dict]`
Get profiling results.

**Returns:**
- `List[Dict]`: Profile data with timing information

#### `set_memory_layout(layout: str) -> None`
Set memory layout optimization.

**Parameters:**
- `layout` (str): Layout type ("aos", "soa", "block", "interleaved")

### Advanced Configuration

#### `set_amr_parameters(params: Dict) -> None`
Configure adaptive mesh refinement.

**Parameters:**
- `params` (Dict): AMR configuration with keys:
  - `'error_estimator'` (str): "kelly", "gradient", "residual", "zz"
  - `'refine_fraction'` (float): Fraction of elements to refine
  - `'coarsen_fraction'` (float): Fraction of elements to coarsen
  - `'max_levels'` (int): Maximum refinement levels
  - `'error_tolerance'` (float): Global error tolerance
  - `'anisotropic'` (bool): Enable anisotropic refinement

#### `set_solver_parameters(params: Dict) -> None`
Configure linear solver.

**Parameters:**
- `params` (Dict): Solver configuration with keys:
  - `'solver_type'` (str): "cg", "gmres", "bicgstab"
  - `'preconditioner'` (str): "jacobi", "ilu", "amg"
  - `'tolerance'` (float): Convergence tolerance
  - `'max_iterations'` (int): Maximum iterations

**Example:**
```python
# Configure AMR
amr_params = {
    'error_estimator': 'kelly',
    'refine_fraction': 0.3,
    'coarsen_fraction': 0.1,
    'max_levels': 5,
    'error_tolerance': 1e-4,
    'anisotropic': True
}
sim.set_amr_parameters(amr_params)

# Configure solver
solver_params = {
    'solver_type': 'gmres',
    'preconditioner': 'ilu',
    'tolerance': 1e-8,
    'max_iterations': 1000
}
sim.set_solver_parameters(solver_params)
```

## GPU Module

### GPUContext

GPU context management for CUDA/OpenCL.

```python
class GPUContext:
    @staticmethod
    def instance() -> GPUContext
```

**Methods:**

#### `initialize(backend: GPUBackend = GPUBackend.AUTO) -> bool`
Initialize GPU context.

**Parameters:**
- `backend` (GPUBackend): Preferred backend (AUTO, CUDA, OPENCL)

**Returns:**
- `bool`: True if initialization successful

#### `get_device_info() -> GPUDeviceInfo`
Get GPU device information.

**Returns:**
- `GPUDeviceInfo`: Device specifications

#### `is_initialized() -> bool`
Check if GPU is initialized.

**Returns:**
- `bool`: True if GPU ready

### GPUDeviceInfo

GPU device information structure.

**Attributes:**
- `name` (str): Device name
- `global_memory` (int): Global memory in bytes
- `compute_capability_major` (int): Compute capability major version
- `compute_capability_minor` (int): Compute capability minor version
- `multiprocessor_count` (int): Number of multiprocessors
- `memory_bandwidth` (float): Memory bandwidth in GB/s
- `supports_double_precision` (bool): Double precision support

### GPUMemory

GPU memory management template.

```python
class GPUMemory[T]:
    def __init__(self, size: int, backend: GPUBackend = GPUBackend.AUTO)
```

**Methods:**

#### `copy_to_device(host_data: np.ndarray) -> None`
Copy data from host to device.

#### `copy_to_host() -> np.ndarray`
Copy data from device to host.

#### `size() -> int`
Get memory size.

**Example:**
```python
from simulator.gpu import GPUContext, GPUMemory, GPUBackend

# Initialize GPU
ctx = GPUContext.instance()
if ctx.initialize(GPUBackend.CUDA):
    info = ctx.get_device_info()
    print(f"GPU: {info.name}")
    print(f"Memory: {info.global_memory / 1e9:.1f} GB")
    
    # Allocate GPU memory
    gpu_mem = GPUMemory[float](1000000)  # 1M floats
    
    # Copy data
    host_data = np.random.random(1000000)
    gpu_mem.copy_to_device(host_data)
    
    # ... GPU computation ...
    
    result = gpu_mem.copy_to_host()
```

## Mesh Module

### Mesh

Mesh generation and management.

```python
class Mesh:
    def __init__(self, device: Device, mesh_type: MeshType)
```

**Methods:**

#### `generate_gmsh_mesh(filename: str) -> None`
Generate mesh using GMSH.

#### `get_grid_points_x() -> List[float]`
Get X coordinates of grid points.

#### `get_grid_points_y() -> List[float]`
Get Y coordinates of grid points.

#### `get_elements() -> List[List[int]]`
Get element connectivity.

#### `refine(refine_flags: List[bool]) -> None`
Refine mesh elements.

#### `get_num_nodes() -> int`
Get number of mesh nodes.

#### `get_num_elements() -> int`
Get number of mesh elements.

#### `compute_quality_metrics() -> Dict[str, float]`
Compute mesh quality metrics.

**Returns:**
- `Dict[str, float]`: Quality metrics (min_angle, max_angle, aspect_ratio, etc.)

**Example:**
```python
from simulator import Mesh, Device, MeshType

device = Device(1e-6, 0.5e-6)
mesh = Mesh(device, MeshType.Unstructured)

# Generate mesh
mesh.generate_gmsh_mesh("device.msh")

# Get mesh data
x_coords = mesh.get_grid_points_x()
y_coords = mesh.get_grid_points_y()
elements = mesh.get_elements()

print(f"Nodes: {mesh.get_num_nodes()}")
print(f"Elements: {mesh.get_num_elements()}")

# Check quality
quality = mesh.compute_quality_metrics()
print(f"Min angle: {quality['min_angle']:.1f}°")
print(f"Max aspect ratio: {quality['max_aspect_ratio']:.2f}")
```

## Physics Module

### Physical Constants

```python
# Fundamental constants
Q_ELECTRON = 1.602e-19      # Elementary charge (C)
K_BOLTZMANN = 1.381e-23     # Boltzmann constant (J/K)
EPSILON_0 = 8.854e-12       # Vacuum permittivity (F/m)
H_PLANCK = 6.626e-34        # Planck constant (J·s)

# Semiconductor constants
NI_SI_300K = 1.0e10 * 1e6   # Si intrinsic concentration at 300K (m⁻³)
BANDGAP_SI = 1.12           # Si bandgap at 300K (eV)
MOBILITY_N_SI = 1350e-4     # Si electron mobility at 300K (m²/V·s)
MOBILITY_P_SI = 480e-4      # Si hole mobility at 300K (m²/V·s)
```

### Physics Functions

#### `compute_intrinsic_concentration(material: str, temperature: float) -> float`
Compute intrinsic carrier concentration.

#### `compute_mobility(material: str, carrier: str, temperature: float, doping: float = 0.0) -> float`
Compute carrier mobility with doping and temperature dependence.

#### `compute_diffusion_coefficient(mobility: float, temperature: float) -> float`
Compute diffusion coefficient using Einstein relation.

#### `compute_recombination_srh(n: float, p: float, ni: float, tau_n: float, tau_p: float) -> float`
Compute Shockley-Read-Hall recombination rate.

**Example:**
```python
from simulator.physics import *

# Silicon at 300K
ni = compute_intrinsic_concentration("Si", 300.0)
mu_n = compute_mobility("Si", "electron", 300.0, 1e17)
mu_p = compute_mobility("Si", "hole", 300.0, 1e17)

print(f"Intrinsic concentration: {ni:.2e} m⁻³")
print(f"Electron mobility: {mu_n:.2e} m²/V·s")
print(f"Hole mobility: {mu_p:.2e} m²/V·s")

# Diffusion coefficients
Dn = compute_diffusion_coefficient(mu_n, 300.0)
Dp = compute_diffusion_coefficient(mu_p, 300.0)

print(f"Electron diffusion: {Dn:.2e} m²/s")
print(f"Hole diffusion: {Dp:.2e} m²/s")
```

## Visualization Module

### Plotting Functions

#### `plot_potential(results: Dict, mesh_data: Dict, **kwargs) -> Figure`
Plot electrostatic potential distribution.

#### `plot_carrier_density(results: Dict, mesh_data: Dict, carrier: str = 'n', **kwargs) -> Figure`
Plot carrier density distribution.

#### `plot_current_density(results: Dict, mesh_data: Dict, **kwargs) -> Figure`
Plot current density vectors.

#### `plot_mesh(mesh_data: Dict, **kwargs) -> Figure`
Plot mesh structure.

#### `create_animation(results_list: List[Dict], mesh_data: Dict, quantity: str, **kwargs) -> Animation`
Create animated visualization.

**Example:**
```python
from simulator.visualization import *
import matplotlib.pyplot as plt

# Plot results
fig1 = plot_potential(results, mesh_data, colormap='viridis')
fig2 = plot_carrier_density(results, mesh_data, carrier='n', log_scale=True)
fig3 = plot_current_density(results, mesh_data, scale=1e-6)

plt.show()

# Create animation
results_list = []  # List of results at different times
animation = create_animation(results_list, mesh_data, 'potential', 
                           interval=100, save_path='potential.gif')
```

## Utilities

### File I/O

#### `save_results(results: Dict, filename: str, format: str = 'hdf5') -> None`
Save simulation results.

#### `load_results(filename: str) -> Dict`
Load simulation results.

#### `export_vtk(results: Dict, mesh_data: Dict, filename: str) -> None`
Export results in VTK format for ParaView.

### Data Processing

#### `interpolate_to_grid(results: Dict, x_grid: np.ndarray, y_grid: np.ndarray) -> Dict`
Interpolate results to regular grid.

#### `compute_iv_curve(device: Device, voltages: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]`
Compute current-voltage characteristics.

#### `extract_line_profile(results: Dict, start_point: Tuple[float, float], end_point: Tuple[float, float], num_points: int = 100) -> Dict`
Extract 1D profile along line.

**Example:**
```python
from simulator.utils import *

# Save results
save_results(results, 'simulation_results.h5', format='hdf5')

# Load results
loaded_results = load_results('simulation_results.h5')

# Export for ParaView
export_vtk(results, mesh_data, 'results.vtk')

# Extract line profile
profile = extract_line_profile(results, (0, 0.25e-6), (1e-6, 0.25e-6))
plt.plot(profile['x'], profile['potential'])
plt.xlabel('Position (m)')
plt.ylabel('Potential (V)')
plt.show()
```
