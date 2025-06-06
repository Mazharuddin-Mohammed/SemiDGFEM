# SemiDGFEM User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Device Simulation](#basic-device-simulation)
3. [Advanced Transport Models](#advanced-transport-models)
4. [MOSFET Simulation Tutorial](#mosfet-simulation-tutorial)
5. [Heterostructure Simulation Tutorial](#heterostructure-simulation-tutorial)
6. [Performance Optimization](#performance-optimization)
7. [Visualization and Analysis](#visualization-and-analysis)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-repo/SemiDGFEM.git
cd SemiDGFEM
```

2. **Build the framework:**
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

3. **Install Python dependencies:**
```bash
pip install numpy scipy matplotlib PySide6
```

4. **Set up environment:**
```bash
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/python:$PYTHONPATH
```

### Quick Start Example

```python
import sys
sys.path.append('python')

import simulator
import numpy as np

# Create a simple device
device = simulator.Device(2e-6, 1e-6)  # 2μm × 1μm
print(f"Device created: {device.get_dimensions()}")
```

---

## Basic Device Simulation

### Creating Your First Simulation

```python
from advanced_transport import create_drift_diffusion_solver
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create solver
solver = create_drift_diffusion_solver(2e-6, 1e-6)

# Step 2: Set up doping profile
size = 100
nd = np.full(size, 1e17)  # n-type doping (cm⁻³)
na = np.full(size, 1e16)  # p-type doping (cm⁻³)
solver.set_doping(nd, na)

# Step 3: Define boundary conditions
boundary_conditions = [0, 1, 0, 0]  # [left, right, top, bottom] voltages (V)

# Step 4: Solve transport equations
results = solver.solve_transport(boundary_conditions, Vg=0.5)

# Step 5: Analyze results
print(f"Solution fields: {list(results.keys())}")
print(f"Max electron density: {np.max(results['electron_density']):.2e} cm⁻³")
print(f"Max current density: {np.max(results['current_density_n']):.2e} A/cm²")
```

### Understanding the Results

The `solve_transport()` method returns a dictionary with the following fields:

- **`potential`**: Electrostatic potential distribution (V)
- **`electron_density`**: Electron concentration (cm⁻³)
- **`hole_density`**: Hole concentration (cm⁻³)
- **`current_density_n`**: Electron current density (A/cm²)
- **`current_density_p`**: Hole current density (A/cm²)

### Visualizing Results

```python
# Plot potential distribution
plt.figure(figsize=(10, 6))
plt.plot(results['potential'], label='Potential')
plt.xlabel('Position')
plt.ylabel('Potential (V)')
plt.title('Electrostatic Potential')
plt.legend()
plt.grid(True)
plt.show()

# Plot carrier densities
plt.figure(figsize=(10, 6))
plt.semilogy(results['electron_density'], label='Electrons')
plt.semilogy(results['hole_density'], label='Holes')
plt.xlabel('Position')
plt.ylabel('Carrier Density (cm⁻³)')
plt.title('Carrier Densities')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Advanced Transport Models

### Energy Transport for Hot Carrier Effects

```python
from advanced_transport import create_energy_transport_solver

# Create energy transport solver
energy_solver = create_energy_transport_solver(2e-6, 1e-6)
energy_solver.set_doping(nd, na)

# Solve with energy transport
results = energy_solver.solve_transport([0, 2, 0, 0], Vg=1.0)

# Additional fields available:
print(f"Electron temperature: {np.mean(results['temperature_n']):.1f} K")
print(f"Hole temperature: {np.mean(results['temperature_p']):.1f} K")
print(f"Energy density: {np.max(results['energy_n']):.2e} J/cm³")
```

### Hydrodynamic Transport for High-Field Effects

```python
from advanced_transport import create_hydrodynamic_solver

# Create hydrodynamic solver
hydro_solver = create_hydrodynamic_solver(2e-6, 1e-6)
hydro_solver.set_doping(nd, na)

# Solve with momentum conservation
results = hydro_solver.solve_transport([0, 3, 0, 0], Vg=1.5)

# Analyze velocity and momentum
print(f"Max electron velocity: {np.max(results['velocity_n']):.2e} m/s")
print(f"Max hole velocity: {np.max(results['velocity_p']):.2e} m/s")
```

### Non-Equilibrium Statistics

```python
from advanced_transport import create_non_equilibrium_solver

# Create non-equilibrium solver
neq_solver = create_non_equilibrium_solver(2e-6, 1e-6)
neq_solver.set_doping(nd, na)

# Solve with quasi-Fermi levels
results = neq_solver.solve_transport([0, 1, 0, 0], Vg=0.8)

# Analyze quasi-Fermi levels
print(f"Electron quasi-Fermi level: {np.mean(results['quasi_fermi_n']):.3f} eV")
print(f"Hole quasi-Fermi level: {np.mean(results['quasi_fermi_p']):.3f} eV")
```

---

## MOSFET Simulation Tutorial

### Step 1: Define Device Geometry

```python
from mosfet_simulation import DeviceGeometry, DopingProfile, MOSFETDevice, MOSFETType

# Define 180nm technology MOSFET
geometry = DeviceGeometry(
    length=0.18,          # Gate length (μm)
    width=10.0,           # Gate width (μm)
    tox=4.0,             # Oxide thickness (nm)
    xj=0.15,             # Junction depth (μm)
    channel_length=0.18,  # Effective channel length (μm)
    source_length=0.5,    # Source region length (μm)
    drain_length=0.5      # Drain region length (μm)
)
```

### Step 2: Define Doping Profile

```python
# NMOS doping profile
nmos_doping = DopingProfile(
    substrate_doping=1e17,      # p-substrate (cm⁻³)
    source_drain_doping=1e20,   # n+ source/drain (cm⁻³)
    channel_doping=5e16,        # Channel doping (cm⁻³)
    gate_doping=1e20,           # n+ polysilicon gate (cm⁻³)
    profile_type="uniform"
)
```

### Step 3: Create and Analyze MOSFET

```python
# Create NMOS device
nmos = MOSFETDevice(MOSFETType.NMOS, geometry, nmos_doping, temperature=300.0)

print(f"Threshold voltage: {nmos.vth:.3f} V")
print(f"Oxide capacitance: {nmos.cox*1e4:.2f} μF/cm²")

# Create device mesh
nmos.create_device_mesh(nx=100, ny=50)

# Calculate I-V characteristics
import numpy as np
vgs_range = np.linspace(0, 3.0, 16)
vds_range = np.linspace(0, 3.0, 21)

print("Calculating I-V characteristics...")
iv_data = nmos.calculate_iv_characteristics(vgs_range, vds_range)

# Extract device parameters
parameters = nmos.extract_device_parameters(iv_data)
print(f"\nDevice Parameters:")
print(f"  Threshold Voltage: {parameters['threshold_voltage']:.3f} V")
print(f"  Transconductance: {parameters['transconductance']*1e6:.1f} μS")
print(f"  Output Conductance: {parameters['output_conductance']*1e6:.1f} μS")
print(f"  Intrinsic Gain: {parameters['intrinsic_gain']:.1f}")
print(f"  Mobility: {parameters['mobility']*1e4:.0f} cm²/V·s")
print(f"  Subthreshold Slope: {parameters['subthreshold_slope']:.1f} mV/decade")
```

### Step 4: Visualization

```python
# Plot I-V characteristics
nmos.plot_device_characteristics(iv_data, save_path="nmos_iv.png")

# Plot device structure
solution = nmos.solve_poisson_equation(vgs=2.0, vds=1.5)
nmos.plot_device_structure(solution, save_path="nmos_structure.png")

# Generate comprehensive report
report = nmos.generate_device_report(iv_data)
print(report)
```

### PMOS Simulation

```python
# PMOS doping profile (opposite polarity)
pmos_doping = DopingProfile(
    substrate_doping=1e17,      # n-substrate (cm⁻³)
    source_drain_doping=1e20,   # p+ source/drain (cm⁻³)
    channel_doping=5e16,        # Channel doping (cm⁻³)
    gate_doping=1e20,           # p+ polysilicon gate (cm⁻³)
    profile_type="uniform"
)

# Create PMOS device
pmos = MOSFETDevice(MOSFETType.PMOS, geometry, pmos_doping)

# Calculate I-V with negative voltages
vgs_range = np.linspace(0, -3.0, 16)
vds_range = np.linspace(0, -3.0, 21)
pmos_iv_data = pmos.calculate_iv_characteristics(vgs_range, vds_range)
```

---

## Heterostructure Simulation Tutorial

### Step 1: Define Layer Structure

```python
from heterostructure_simulation import (HeterostructureDevice, LayerStructure, 
                                      SemiconductorMaterial)

# GaAs/AlGaAs HEMT structure
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
```

### Step 2: Create and Analyze Heterostructure

```python
# Create heterostructure device
hetero = HeterostructureDevice(layers, temperature=300.0)

print(f"Total thickness: {hetero.total_thickness:.1f} nm")
print(f"Number of layers: {len(hetero.layers)}")

# Create high-resolution mesh
hetero.create_mesh(nz=1000)

# Calculate band structure
print("Calculating band structure...")
band_structure = hetero.calculate_band_structure()

# Calculate carrier densities
print("Calculating carrier densities...")
carrier_densities = hetero.calculate_carrier_densities()

print(f"Fermi level: {carrier_densities['fermi_level']:.3f} eV")
```

### Step 3: Quantum Well Analysis

```python
# Analyze quantum wells
quantum_wells = hetero.calculate_quantum_wells()

print(f"\nQuantum Wells Detected: {len(quantum_wells)}")
for i, qw in enumerate(quantum_wells):
    print(f"  Well {i+1}:")
    print(f"    Type: {qw['type']}")
    print(f"    Position: {qw['position']:.1f} nm")
    print(f"    Width: {qw['width']:.1f} nm")
    print(f"    Depth: {qw['depth']:.3f} eV")
    print(f"    Energy levels: {len(qw['energy_levels'])}")
    if qw['energy_levels']:
        print(f"    Ground state: {qw['energy_levels'][0]:.3f} eV")
```

### Step 4: Transport Properties

```python
# Calculate transport properties
transport = hetero.calculate_transport_properties()

# Analyze 2DEG properties
if quantum_wells:
    qw = quantum_wells[0]  # First quantum well
    z = band_structure['z'] * 1e9  # Convert to nm
    
    # Find quantum well region
    qw_mask = (z >= qw['boundaries'][0]) & (z <= qw['boundaries'][1])
    
    if np.any(qw_mask):
        # Calculate sheet carrier density
        n = carrier_densities['electron_density']
        dz = (z[1] - z[0]) * 1e-9  # Convert to m
        sheet_density = np.sum(n[qw_mask]) * dz * 1e-4  # cm⁻²
        
        # Calculate 2DEG mobility
        mu_e = transport['electron_mobility'] * 1e4  # cm²/V·s
        avg_mobility = np.mean(mu_e[qw_mask])
        
        print(f"\n2DEG Properties:")
        print(f"  Sheet density: {sheet_density:.2e} cm⁻²")
        print(f"  Mobility: {avg_mobility:.0f} cm²/V·s")
```

### Step 5: Visualization and Reporting

```python
# Plot band structure
hetero.plot_band_structure(save_path="hetero_bands.png")

# Plot transport properties
hetero.plot_transport_properties(save_path="hetero_transport.png")

# Generate comprehensive report
report = hetero.generate_structure_report()
print(report)

# Save report to file
with open("heterostructure_report.txt", "w") as f:
    f.write(report)
```

---

## Performance Optimization

### Enabling GPU Acceleration

```python
from gpu_acceleration import GPUAcceleratedSolver
import numpy as np

# Create GPU solver
gpu_solver = GPUAcceleratedSolver()

if gpu_solver.is_gpu_available():
    print("GPU acceleration available!")
    
    # GPU-accelerated transport solution
    potential = np.linspace(0, 1.0, 10000)
    doping_nd = np.full(10000, 1e17)
    doping_na = np.full(10000, 1e16)
    
    results = gpu_solver.solve_transport_gpu(potential, doping_nd, doping_na)
    print(f"GPU solution fields: {list(results.keys())}")
    
    # Performance benchmark
    benchmark = gpu_solver.benchmark_performance(size=50000)
    print(f"GPU performance: {benchmark}")
else:
    print("GPU not available, using CPU")
```

### SIMD Optimization

```python
from performance_bindings import PerformanceOptimizer
import numpy as np

# Create performance optimizer
optimizer = PerformanceOptimizer()

# Print system capabilities
optimizer.print_performance_info()

# Optimized vector operations
a = np.random.random(100000)
b = np.random.random(100000)

# Automatic backend selection
result = optimizer.optimize_computation('vector_add', a, b)
dot_result = optimizer.optimize_computation('dot_product', a, b)

print(f"Optimized computation completed")
```

---

## Visualization and Analysis

### Custom Plotting

```python
import matplotlib.pyplot as plt

# Custom analysis of results
def plot_custom_analysis(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Potential
    ax1.plot(results['potential'])
    ax1.set_title('Potential Distribution')
    ax1.set_ylabel('Potential (V)')
    
    # Carrier densities
    ax2.semilogy(results['electron_density'], label='Electrons')
    ax2.semilogy(results['hole_density'], label='Holes')
    ax2.set_title('Carrier Densities')
    ax2.set_ylabel('Density (cm⁻³)')
    ax2.legend()
    
    # Current densities
    ax3.plot(results['current_density_n'], label='Electron current')
    ax3.plot(results['current_density_p'], label='Hole current')
    ax3.set_title('Current Densities')
    ax3.set_ylabel('Current Density (A/cm²)')
    ax3.legend()
    
    # Electric field
    electric_field = np.gradient(results['potential'])
    ax4.plot(electric_field)
    ax4.set_title('Electric Field')
    ax4.set_ylabel('E-field (V/m)')
    
    plt.tight_layout()
    plt.savefig('custom_analysis.png', dpi=300)
    plt.show()

# Use with your results
plot_custom_analysis(results)
```

---

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure library path is set
   export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
   export PYTHONPATH=$PWD/python:$PYTHONPATH
   ```

2. **GPU Not Available**
   ```python
   # Check GPU status
   from gpu_acceleration import GPUAcceleratedSolver
   solver = GPUAcceleratedSolver()
   print(f"GPU available: {solver.is_gpu_available()}")
   print(f"Performance info: {solver.get_performance_info()}")
   ```

3. **Convergence Issues**
   - Reduce voltage steps
   - Increase mesh resolution
   - Check doping profile validity
   - Verify boundary conditions

4. **Memory Issues**
   - Reduce mesh size
   - Use appropriate data types
   - Enable GPU acceleration for large problems

### Getting Help

- Check the API reference for detailed function documentation
- Review example scripts in the `python/` directory
- Run test suites to verify installation
- Check system requirements and dependencies

### Performance Tips

- Use GPU acceleration for problems with >10,000 mesh points
- Enable SIMD optimization for vector operations
- Consider mesh resolution vs. accuracy trade-offs
- Monitor memory usage for large simulations
