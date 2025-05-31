# SemiDGFEM User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Performance Optimization](#performance-optimization)
7. [GPU Acceleration](#gpu-acceleration)
8. [Troubleshooting](#troubleshooting)
9. [Examples](#examples)

## Introduction

SemiDGFEM is a high-performance 2D semiconductor device simulator using Discontinuous Galerkin (DG) finite element methods. It provides:

- **Advanced Numerical Methods**: P2/P3 DG elements with adaptive mesh refinement
- **High Performance**: SIMD optimization, OpenMP parallelization, and GPU acceleration
- **Complete Physics**: Poisson and drift-diffusion equations with self-consistent coupling
- **Modern Interface**: Python GUI with real-time visualization
- **Production Ready**: Comprehensive error handling and validation

### Key Features
- ✅ P3 Discontinuous Galerkin methods (10 DOFs per element)
- ✅ Adaptive mesh refinement with Kelly error estimator
- ✅ GPU acceleration (CUDA/OpenCL)
- ✅ SIMD optimization (AVX2/FMA)
- ✅ OpenMP parallelization
- ✅ Real-time visualization with Vulkan rendering
- ✅ Self-consistent Poisson-drift-diffusion coupling

## Installation

### Prerequisites

**Required Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake
sudo apt-get install libpetsc-dev libgmsh-dev
sudo apt-get install libboost-dev libomp-dev
sudo apt-get install python3-dev python3-pip

# Optional: GPU support
sudo apt-get install nvidia-cuda-toolkit  # For CUDA
sudo apt-get install opencl-headers libopencl-dev  # For OpenCL
```

**Python Dependencies:**
```bash
pip install numpy scipy matplotlib
pip install PySide6 vulkan  # For GUI
pip install cython setuptools  # For building
```

### Building from Source

1. **Clone the repository:**
```bash
git clone https://github.com/your-repo/SemiDGFEM.git
cd SemiDGFEM
```

2. **Configure build:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_CUDA=ON \
         -DENABLE_OPENCL=ON \
         -DENABLE_VULKAN=ON
```

3. **Build:**
```bash
make -j$(nproc)
make install
```

4. **Build Python interface:**
```bash
cd ../python
python setup.py build_ext --inplace
pip install -e .
```

### Verification

Test the installation:
```bash
# Test C++ library
cd build && ctest

# Test Python interface
python -c "import simulator; print('Installation successful!')"

# Test GPU support
python -c "from simulator.gpu import GPUContext; print('GPU available:', GPUContext.is_available())"
```

## Quick Start

### Basic Simulation (Python)

```python
import numpy as np
from simulator import Simulator, Device, Method, MeshType

# Create device (1μm × 0.5μm)
device = Device(Lx=1e-6, Ly=0.5e-6)

# Create simulator with P3 DG method
sim = Simulator(device, Method.DG, MeshType.Structured, order=3)

# Set doping profile
x = np.linspace(0, 1e-6, 100)
y = np.linspace(0, 0.5e-6, 50)
Nd = 1e17 * np.ones((100, 50))  # n-type doping (cm⁻³)
Na = np.zeros((100, 50))        # No p-type doping

sim.set_doping(Nd.flatten(), Na.flatten())

# Set boundary conditions [left, right, bottom, top] (V)
boundary_conditions = [0.0, 1.0, 0.0, 0.0]

# Run simulation
results = sim.solve_drift_diffusion(
    bc=boundary_conditions,
    Vg=0.0,           # Gate voltage
    max_steps=100,    # Maximum iterations
    use_amr=True,     # Enable adaptive mesh refinement
    poisson_max_iter=50,
    poisson_tol=1e-6
)

# Extract results
potential = results['potential']
n_density = results['n']
p_density = results['p']
current_n = results['Jn']
current_p = results['Jp']

print(f"Simulation completed with {len(potential)} DOFs")
```

### GUI Application

Launch the graphical interface:
```bash
python python/gui/main_gui_2d.py
```

The GUI provides:
- **Device Parameters**: Set dimensions, doping, boundary conditions
- **Simulation Control**: Run/stop, parameter adjustment
- **Real-time Visualization**: Potential, carrier densities, current flow
- **Results Export**: Save data in multiple formats

## Basic Usage

### Device Definition

```python
from simulator import Device

# Simple rectangular device
device = Device(Lx=2e-6, Ly=1e-6)  # 2μm × 1μm

# Device with material regions
regions = [
    {"material": "Si", "x_min": 0, "x_max": 1e-6, "y_min": 0, "y_max": 1e-6},
    {"material": "SiO2", "x_min": 1e-6, "x_max": 2e-6, "y_min": 0, "y_max": 1e-6}
]
device = Device(Lx=2e-6, Ly=1e-6, regions=regions)
```

### Mesh Configuration

```python
from simulator import MeshType

# Structured mesh (faster, regular geometry)
sim = Simulator(device, Method.DG, MeshType.Structured, order=3)

# Unstructured mesh (flexible, complex geometry)
sim = Simulator(device, Method.DG, MeshType.Unstructured, order=3)
```

### Doping Profiles

```python
# Uniform doping
Nd = np.full(n_nodes, 1e17)  # 1×10¹⁷ cm⁻³
Na = np.zeros(n_nodes)

# Gaussian doping profile
x_center, y_center = 0.5e-6, 0.25e-6
sigma = 0.1e-6
for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    distance = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    Nd[i] = 1e17 * np.exp(-(distance/sigma)**2)

sim.set_doping(Nd, Na)
```

### Boundary Conditions

```python
# Dirichlet boundary conditions
bc = [V_left, V_right, V_bottom, V_top]

# Example: Forward bias
bc = [0.0, 0.7, 0.0, 0.0]  # 0.7V applied to right contact

# Example: Reverse bias
bc = [0.0, -5.0, 0.0, 0.0]  # -5V reverse bias
```

## Advanced Features

### Adaptive Mesh Refinement

```python
# Enable AMR with custom parameters
results = sim.solve_drift_diffusion(
    bc=boundary_conditions,
    use_amr=True,
    amr_params={
        'error_estimator': 'kelly',      # kelly, gradient, residual
        'refine_fraction': 0.3,          # Top 30% refined
        'coarsen_fraction': 0.1,         # Bottom 10% coarsened
        'max_levels': 5,                 # Maximum refinement levels
        'error_tolerance': 1e-4,         # Global error tolerance
        'anisotropic': True              # Enable anisotropic refinement
    }
)
```

### Performance Optimization

```python
# Enable SIMD optimization
sim.enable_simd(True)

# Set OpenMP thread count
sim.set_num_threads(8)

# Memory optimization
sim.set_memory_pool_size(1024*1024*1024)  # 1GB pool
```

### Custom Physics Models

```python
# Set custom mobility models
def mobility_model(n, p, T, E_field):
    # Custom mobility calculation
    mu_n = 1000 / (1 + (E_field/1e5)**2)  # Field-dependent mobility
    mu_p = 400 / (1 + (E_field/1e5)**2)
    return mu_n, mu_p

sim.set_mobility_model(mobility_model)

# Set recombination parameters
sim.set_recombination_params(
    tau_n=1e-6,    # Electron lifetime (s)
    tau_p=1e-6,    # Hole lifetime (s)
    B_rad=1e-10,   # Radiative recombination coefficient
    C_aug=1e-30    # Auger recombination coefficient
)
```

## Performance Optimization

### CPU Optimization

```python
# Check available optimizations
print("SIMD support:", sim.has_simd_support())
print("OpenMP threads:", sim.get_max_threads())

# Enable all CPU optimizations
sim.enable_simd(True)
sim.set_num_threads(-1)  # Use all available cores
sim.enable_cache_optimization(True)
```

### Memory Optimization

```python
# Monitor memory usage
stats = sim.get_memory_stats()
print(f"Current usage: {stats['current_mb']:.1f} MB")
print(f"Peak usage: {stats['peak_mb']:.1f} MB")

# Optimize memory layout
sim.set_memory_layout('block')  # block, interleaved, aos, soa
```

### Profiling

```python
# Enable profiling
sim.enable_profiling(True)

# Run simulation
results = sim.solve_drift_diffusion(bc, use_amr=True)

# Get performance data
profile = sim.get_profile_data()
for item in profile:
    print(f"{item['name']}: {item['time_ms']:.2f} ms ({item['percentage']:.1f}%)")
```

## GPU Acceleration

### Enabling GPU Support

```python
from simulator.gpu import GPUContext, GPUBackend

# Initialize GPU context
gpu_ctx = GPUContext()
success = gpu_ctx.initialize(GPUBackend.AUTO)  # AUTO, CUDA, OPENCL

if success:
    print(f"GPU initialized: {gpu_ctx.get_device_info().name}")
    
    # Enable GPU acceleration
    sim.enable_gpu(True)
    sim.set_gpu_backend(GPUBackend.CUDA)
else:
    print("GPU not available, using CPU")
```

### GPU Performance Tuning

```python
# Check GPU capabilities
info = gpu_ctx.get_device_info()
print(f"Global memory: {info.global_memory / 1e9:.1f} GB")
print(f"Compute capability: {info.compute_capability_major}.{info.compute_capability_minor}")
print(f"Multiprocessors: {info.multiprocessor_count}")

# Optimize for GPU
sim.set_gpu_block_size(256)      # CUDA block size
sim.set_gpu_memory_pool(512e6)   # 512MB GPU memory pool
sim.enable_gpu_profiling(True)   # Enable GPU profiling
```

### Hybrid CPU/GPU Execution

```python
# Automatic CPU/GPU selection based on problem size
sim.enable_hybrid_execution(True)

# Set thresholds for GPU usage
sim.set_gpu_thresholds({
    'matrix_assembly': 10000,    # Use GPU for >10k elements
    'linear_solve': 50000,       # Use GPU for >50k DOFs
    'physics_update': 1000       # Use GPU for >1k points
})
```

## Troubleshooting

### Common Issues

**1. Build Failures:**
```bash
# Missing dependencies
sudo apt-get install libpetsc-dev libgmsh-dev

# CMake configuration issues
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON

# Check compiler support
gcc --version  # Requires GCC 7+ for C++17
```

**2. Runtime Errors:**
```python
# Check library loading
import simulator
print(simulator.__file__)  # Verify correct library path

# Validate input data
sim.validate_inputs()  # Check mesh, doping, boundary conditions

# Enable debug mode
sim.set_debug_level(2)  # 0=none, 1=basic, 2=verbose
```

**3. Performance Issues:**
```python
# Check optimization flags
print("Compiler flags:", sim.get_compiler_flags())

# Monitor resource usage
stats = sim.get_performance_stats()
print(f"CPU usage: {stats['cpu_percent']:.1f}%")
print(f"Memory usage: {stats['memory_mb']:.1f} MB")

# Profile bottlenecks
sim.enable_profiling(True)
# ... run simulation ...
sim.print_profile()
```

**4. GPU Issues:**
```python
# Check GPU availability
from simulator.gpu import GPUContext
ctx = GPUContext()
if not ctx.initialize():
    print("GPU not available")
    print("Available devices:", ctx.get_available_devices())

# CUDA-specific debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous execution
```

### Getting Help

- **Documentation**: [https://semidgfem.readthedocs.io](https://semidgfem.readthedocs.io)
- **Issues**: [https://github.com/your-repo/SemiDGFEM/issues](https://github.com/your-repo/SemiDGFEM/issues)
- **Discussions**: [https://github.com/your-repo/SemiDGFEM/discussions](https://github.com/your-repo/SemiDGFEM/discussions)
- **Email**: support@semidgfem.org

## Examples

See the `examples/` directory for complete simulation examples:

- `examples/pn_junction.py` - Basic p-n junction simulation
- `examples/mosfet.py` - MOSFET device simulation
- `examples/solar_cell.py` - Solar cell optimization
- `examples/led.py` - LED efficiency analysis
- `examples/performance_benchmark.py` - Performance testing

Each example includes detailed comments and parameter explanations.
