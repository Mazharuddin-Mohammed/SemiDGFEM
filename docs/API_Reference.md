# SemiDGFEM API Reference Documentation

## Overview

The SemiDGFEM (Semiconductor Discontinuous Galerkin Finite Element Method) framework provides comprehensive semiconductor device simulation capabilities with advanced physics models, GPU acceleration, and complex device support.

## Table of Contents

1. [Core Simulation Framework](#core-simulation-framework)
2. [Advanced Transport Models](#advanced-transport-models)
3. [MOSFET Simulation](#mosfet-simulation)
4. [Heterostructure Simulation](#heterostructure-simulation)
5. [Performance Optimization](#performance-optimization)
6. [GPU Acceleration](#gpu-acceleration)
7. [Visualization and Analysis](#visualization-and-analysis)

---

## Core Simulation Framework

### `simulator.Device`

The core device class for semiconductor device simulation.

```python
class Device:
    def __init__(self, length: float, width: float)
```

**Parameters:**
- `length` (float): Device length in meters
- `width` (float): Device width in meters

**Methods:**

#### `get_dimensions() -> Tuple[float, float]`
Returns the device dimensions.

**Returns:**
- `Tuple[float, float]`: (length, width) in meters

**Example:**
```python
import simulator
device = simulator.Device(2e-6, 1e-6)  # 2μm × 1μm device
length, width = device.get_dimensions()
```

### `simulator.create_device(length: float, width: float) -> Device`

Factory function to create a device instance.

**Parameters:**
- `length` (float): Device length in meters
- `width` (float): Device width in meters

**Returns:**
- `Device`: Configured device instance

---

## Advanced Transport Models

### Drift-Diffusion Transport

#### `advanced_transport.create_drift_diffusion_solver(length: float, width: float) -> DriftDiffusionSolver`

Creates a drift-diffusion transport solver.

**Parameters:**
- `length` (float): Device length in meters
- `width` (float): Device width in meters

**Returns:**
- `DriftDiffusionSolver`: Configured solver instance

**Example:**
```python
from advanced_transport import create_drift_diffusion_solver
import numpy as np

# Create solver
solver = create_drift_diffusion_solver(2e-6, 1e-6)

# Set doping profile
solver.set_doping(
    nd=np.full(100, 1e17),  # n-type doping (cm⁻³)
    na=np.full(100, 1e16)   # p-type doping (cm⁻³)
)

# Solve transport equations
results = solver.solve_transport(
    boundary_conditions=[0, 1, 0, 0],  # [left, right, top, bottom] voltages
    Vg=0.5  # Gate voltage
)
```

#### `DriftDiffusionSolver.solve_transport(boundary_conditions: List[float], **kwargs) -> Dict[str, np.ndarray]`

Solves the drift-diffusion transport equations.

**Parameters:**
- `boundary_conditions` (List[float]): Boundary voltages [left, right, top, bottom]
- `**kwargs`: Additional parameters (Vg, temperature, etc.)

**Returns:**
- `Dict[str, np.ndarray]`: Solution fields including:
  - `potential`: Electrostatic potential (V)
  - `electron_density`: Electron concentration (cm⁻³)
  - `hole_density`: Hole concentration (cm⁻³)
  - `current_density_n`: Electron current density (A/cm²)
  - `current_density_p`: Hole current density (A/cm²)

### Energy Transport

#### `advanced_transport.create_energy_transport_solver(length: float, width: float) -> EnergyTransportSolver`

Creates an energy transport solver for hot carrier effects.

**Additional Fields in Results:**
- `energy_n`: Electron energy density (J/cm³)
- `energy_p`: Hole energy density (J/cm³)
- `temperature_n`: Electron temperature (K)
- `temperature_p`: Hole temperature (K)

### Hydrodynamic Transport

#### `advanced_transport.create_hydrodynamic_solver(length: float, width: float) -> HydrodynamicSolver`

Creates a hydrodynamic transport solver with momentum conservation.

**Additional Fields in Results:**
- `velocity_n`: Electron velocity (m/s)
- `velocity_p`: Hole velocity (m/s)
- `momentum_n`: Electron momentum density (kg⋅m⁻²⋅s⁻¹)
- `momentum_p`: Hole momentum density (kg⋅m⁻²⋅s⁻¹)

### Non-Equilibrium Statistics

#### `advanced_transport.create_non_equilibrium_solver(length: float, width: float) -> NonEquilibriumSolver`

Creates a non-equilibrium statistics solver with quasi-Fermi levels.

**Additional Fields in Results:**
- `quasi_fermi_n`: Electron quasi-Fermi level (eV)
- `quasi_fermi_p`: Hole quasi-Fermi level (eV)
- `generation_rate`: Generation rate (cm⁻³⋅s⁻¹)
- `recombination_rate`: Recombination rate (cm⁻³⋅s⁻¹)

---

## MOSFET Simulation

### `mosfet_simulation.MOSFETDevice`

Advanced MOSFET device simulation with complete I-V characterization.

```python
class MOSFETDevice:
    def __init__(self, mosfet_type: MOSFETType, geometry: DeviceGeometry, 
                 doping: DopingProfile, temperature: float = 300.0)
```

**Parameters:**
- `mosfet_type` (MOSFETType): NMOS or PMOS
- `geometry` (DeviceGeometry): Device geometry parameters
- `doping` (DopingProfile): Doping profile specification
- `temperature` (float): Operating temperature in Kelvin

### Device Geometry

```python
@dataclass
class DeviceGeometry:
    length: float          # Gate length (μm)
    width: float           # Gate width (μm)
    tox: float            # Oxide thickness (nm)
    xj: float             # Junction depth (μm)
    channel_length: float  # Effective channel length (μm)
    source_length: float   # Source region length (μm)
    drain_length: float    # Drain region length (μm)
```

### Doping Profile

```python
@dataclass
class DopingProfile:
    substrate_doping: float      # cm⁻³
    source_drain_doping: float   # cm⁻³
    channel_doping: float        # cm⁻³
    gate_doping: float          # cm⁻³
    profile_type: str           # "uniform", "gaussian", "exponential"
```

### Key Methods

#### `calculate_iv_characteristics(vgs_range: np.ndarray, vds_range: np.ndarray) -> Dict[str, np.ndarray]`

Calculates complete I-V characteristics.

**Parameters:**
- `vgs_range` (np.ndarray): Gate-source voltage range
- `vds_range` (np.ndarray): Drain-source voltage range

**Returns:**
- `Dict[str, np.ndarray]`: I-V data including:
  - `vgs_range`: Gate voltage array
  - `vds_range`: Drain voltage array
  - `ids_matrix`: Drain current matrix (A)
  - `vth`: Threshold voltage (V)

#### `extract_device_parameters(iv_data: Dict[str, np.ndarray]) -> Dict[str, float]`

Extracts key device parameters from I-V data.

**Returns:**
- `Dict[str, float]`: Device parameters including:
  - `threshold_voltage`: Threshold voltage (V)
  - `transconductance`: Transconductance (S)
  - `output_conductance`: Output conductance (S)
  - `mobility`: Carrier mobility (m²/V⋅s)
  - `subthreshold_slope`: Subthreshold slope (mV/decade)
  - `intrinsic_gain`: gm/gds ratio

**Example:**
```python
from mosfet_simulation import MOSFETDevice, MOSFETType, DeviceGeometry, DopingProfile
import numpy as np

# Define device geometry
geometry = DeviceGeometry(
    length=0.18, width=10.0, tox=4.0, xj=0.15,
    channel_length=0.18, source_length=0.5, drain_length=0.5
)

# Define doping profile
doping = DopingProfile(
    substrate_doping=1e17, source_drain_doping=1e20,
    channel_doping=5e16, gate_doping=1e20, profile_type="uniform"
)

# Create NMOS device
nmos = MOSFETDevice(MOSFETType.NMOS, geometry, doping)

# Calculate I-V characteristics
vgs_range = np.linspace(0, 3.0, 16)
vds_range = np.linspace(0, 3.0, 21)
iv_data = nmos.calculate_iv_characteristics(vgs_range, vds_range)

# Extract parameters
parameters = nmos.extract_device_parameters(iv_data)
print(f"Threshold voltage: {parameters['threshold_voltage']:.3f} V")
print(f"Transconductance: {parameters['transconductance']*1e6:.1f} μS")
```

---

## Heterostructure Simulation

### `heterostructure_simulation.HeterostructureDevice`

Advanced heterostructure simulation with quantum effects.

```python
class HeterostructureDevice:
    def __init__(self, layers: List[LayerStructure], temperature: float = 300.0)
```

**Parameters:**
- `layers` (List[LayerStructure]): Layer structure specification
- `temperature` (float): Operating temperature in Kelvin

### Layer Structure

```python
@dataclass
class LayerStructure:
    material: SemiconductorMaterial  # Material type
    thickness: float                 # Layer thickness (nm)
    composition: float              # Alloy composition (0-1)
    doping_type: str               # "n", "p", or "intrinsic"
    doping_concentration: float     # Doping level (cm⁻³)
    position: float                # Starting position (nm)
```

### Supported Materials

```python
class SemiconductorMaterial(Enum):
    SI = "Silicon"
    GE = "Germanium"
    GAAS = "GaAs"
    ALGAS = "AlGaAs"
    INAS = "InAs"
    INGAAS = "InGaAs"
    SIC = "SiC"
    GAN = "GaN"
    ALGAN = "AlGaN"
    INGAN = "InGaN"
```

### Key Methods

#### `calculate_band_structure() -> Dict[str, np.ndarray]`

Calculates the band structure including band bending.

**Returns:**
- `Dict[str, np.ndarray]`: Band structure data including:
  - `z`: Position array (m)
  - `conduction_band`: Conduction band edge (eV)
  - `valence_band`: Valence band edge (eV)
  - `potential`: Electrostatic potential (V)
  - `doping_profile`: Doping concentration (cm⁻³)

#### `calculate_quantum_wells() -> List[Dict[str, Any]]`

Identifies and analyzes quantum wells in the structure.

**Returns:**
- `List[Dict[str, Any]]`: Quantum well information including:
  - `type`: "electron" or "hole"
  - `position`: Well center position (nm)
  - `width`: Well width (nm)
  - `depth`: Confinement depth (eV)
  - `energy_levels`: Confined energy levels (eV)
  - `material`: Well material

**Example:**
```python
from heterostructure_simulation import (HeterostructureDevice, LayerStructure, 
                                      SemiconductorMaterial)

# Define GaAs/AlGaAs heterostructure
layers = [
    LayerStructure(SemiconductorMaterial.GAAS, 500.0, 0.0, "intrinsic", 1e14, 0.0),
    LayerStructure(SemiconductorMaterial.GAAS, 20.0, 0.0, "intrinsic", 1e14, 500.0),
    LayerStructure(SemiconductorMaterial.ALGAS, 30.0, 0.3, "n", 2e18, 520.0)
]

# Create heterostructure
hetero = HeterostructureDevice(layers, temperature=300.0)

# Calculate band structure
band_structure = hetero.calculate_band_structure()

# Analyze quantum wells
quantum_wells = hetero.calculate_quantum_wells()
for qw in quantum_wells:
    print(f"Quantum well: {qw['width']:.1f} nm, depth: {qw['depth']:.3f} eV")
```

---

## Performance Optimization

### `performance_bindings.PerformanceOptimizer`

Automatic performance optimization with SIMD and GPU acceleration.

```python
class PerformanceOptimizer:
    def __init__(self)
    
    def optimize_computation(self, operation: str, *args, force_backend=None) -> Any
    def get_performance_info(self) -> Dict[str, Any]
    def print_performance_info(self) -> None
```

#### Supported Operations

- `vector_add`: SIMD-optimized vector addition
- `dot_product`: Optimized dot product calculation
- `matrix_multiply`: Accelerated matrix multiplication
- `solve_linear`: Linear system solving

**Example:**
```python
from performance_bindings import PerformanceOptimizer
import numpy as np

optimizer = PerformanceOptimizer()

# Automatic backend selection
a = np.random.random(10000)
b = np.random.random(10000)
result = optimizer.optimize_computation('vector_add', a, b)

# Performance information
optimizer.print_performance_info()
```

### `performance_bindings.PhysicsAcceleration`

Specialized acceleration for semiconductor physics computations.

#### Key Methods

- `compute_carrier_densities()`: Accelerated carrier density calculation
- `compute_current_densities()`: Current density computation
- `compute_recombination()`: Recombination rate calculation
- `compute_energy_densities()`: Energy transport calculations
- `compute_momentum_densities()`: Hydrodynamic momentum calculations

---

## GPU Acceleration

### `gpu_acceleration.GPUAcceleratedSolver`

GPU-accelerated solver with CUDA/OpenCL support.

```python
class GPUAcceleratedSolver:
    def __init__(self, preferred_backend=GPUBackend.AUTO)
    
    def is_gpu_available(self) -> bool
    def solve_transport_gpu(self, potential, doping_nd, doping_na, temperature=300.0) -> Dict[str, np.ndarray]
    def benchmark_performance(self, size=10000) -> Dict[str, Any]
```

**Supported Backends:**
- `GPUBackend.CUDA`: NVIDIA GPU acceleration
- `GPUBackend.OPENCL`: Cross-platform GPU acceleration
- `GPUBackend.AUTO`: Automatic backend selection

**Example:**
```python
from gpu_acceleration import GPUAcceleratedSolver
import numpy as np

# Create GPU solver
gpu_solver = GPUAcceleratedSolver()

if gpu_solver.is_gpu_available():
    # GPU-accelerated transport solution
    potential = np.linspace(0, 1.0, 1000)
    doping_nd = np.full(1000, 1e17)
    doping_na = np.full(1000, 1e16)
    
    results = gpu_solver.solve_transport_gpu(potential, doping_nd, doping_na)
    print(f"GPU acceleration: {list(results.keys())}")
```

---

## Visualization and Analysis

### Plotting Functions

All device classes provide comprehensive plotting capabilities:

#### MOSFET Visualization

```python
# Device characteristics
mosfet.plot_device_characteristics(iv_data, save_path="mosfet_iv.png")

# Device structure
mosfet.plot_device_structure(solution, save_path="mosfet_structure.png")
```

#### Heterostructure Visualization

```python
# Band structure
hetero.plot_band_structure(save_path="band_structure.png")

# Transport properties
hetero.plot_transport_properties(save_path="transport.png")
```

### Report Generation

All devices support comprehensive report generation:

```python
# MOSFET report
report = mosfet.generate_device_report(iv_data)

# Heterostructure report
report = hetero.generate_structure_report()
```

---

## Error Handling

All API functions include comprehensive error handling:

```python
try:
    solver = create_drift_diffusion_solver(2e-6, 1e-6)
    results = solver.solve_transport([0, 1, 0, 0])
except ValueError as e:
    print(f"Invalid parameters: {e}")
except RuntimeError as e:
    print(f"Simulation failed: {e}")
```

---

## Performance Considerations

- Use GPU acceleration for large problems (>10,000 mesh points)
- Enable SIMD optimization for vector operations
- Consider memory usage for very large devices
- Use appropriate mesh resolution for accuracy vs. speed trade-off

---

## Version Information

- **Framework Version**: 1.0.0
- **API Version**: 1.0
- **Python Compatibility**: 3.8+
- **Dependencies**: NumPy, SciPy, Matplotlib, PySide6
