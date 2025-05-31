# SemiDGFEM User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Tutorials](#tutorials)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

---

## Introduction

SemiDGFEM is a high-performance 2D semiconductor device simulator that uses Discontinuous Galerkin (DG) finite element methods to solve the Poisson and drift-diffusion equations. It provides both C++ and Python interfaces for maximum flexibility.

### Key Features

- **Discontinuous Galerkin Methods**: High-order accuracy with local conservation
- **Adaptive Mesh Refinement**: Automatic mesh adaptation for optimal accuracy
- **GPU Acceleration**: CUDA kernels for high-performance computing
- **Multiple Numerical Methods**: FDM, FEM, FVM, SEM, MC, and DG
- **Python Integration**: Easy-to-use Python interface with NumPy integration

### Supported Devices

- P-N junctions and diodes
- MOSFETs and FinFETs
- Bipolar junction transistors
- Heterostructure devices
- Solar cells and photodetectors

---

## Installation

### Prerequisites

**Required Dependencies:**
- CMake (≥ 3.10)
- C++17 compatible compiler (GCC ≥ 7, Clang ≥ 5)
- PETSc (≥ 3.12)
- Boost (≥ 1.65)
- OpenMP
- Python 3.8+ (for Python interface)

**Optional Dependencies:**
- CUDA (≥ 10.0) for GPU acceleration
- GMSH for advanced mesh generation
- Matplotlib for visualization

### Build Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/SemiDGFEM.git
cd SemiDGFEM
```

2. **Create build directory:**
```bash
mkdir build && cd build
```

3. **Configure with CMake:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

4. **Build C++ library:**
```bash
make -j$(nproc)
```

5. **Build Python extension:**
```bash
cd ../python
python setup.py build_ext --inplace
```

### Verification

Test the installation:
```bash
# Test C++ library
cd ../tests
./working_backend_test

# Test Python interface (when import issues are resolved)
python -c "import simulator; print('Success!')"
```

---

## Quick Start

### Your First Simulation

Let's simulate a simple P-N junction diode:

```python
import numpy as np
import matplotlib.pyplot as plt
import simulator

# 1. Create device
sim = simulator.Simulator(
    extents=[2e-6, 1e-6],      # 2μm × 1μm device
    num_points_x=100,          # 100 points in x
    num_points_y=50,           # 50 points in y
    method="DG",               # Discontinuous Galerkin
    mesh_type="Structured"     # Regular grid
)

# 2. Define doping profile
x = np.linspace(0, 2e-6, 100)
y = np.linspace(0, 1e-6, 50)
X, Y = np.meshgrid(x, y)

# P-type (left) and N-type (right) regions
junction_pos = 1e-6
Nd = np.where(X > junction_pos, 1e16, 0).flatten()  # N-type doping
Na = np.where(X < junction_pos, 1e16, 0).flatten()  # P-type doping

sim.set_doping(Nd, Na)

# 3. Set boundary conditions
bc = [0.0, 0.7, 0.0, 0.0]  # [left, right, bottom, top] voltages

# 4. Solve equations
result = sim.solve_drift_diffusion(
    bc=bc,
    max_steps=50,
    use_amr=False,
    poisson_tol=1e-6
)

# 5. Visualize results
potential = result['potential'].reshape(50, 100)
n = result['n'].reshape(50, 100)
p = result['p'].reshape(50, 100)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot potential
im1 = axes[0].imshow(potential, extent=[0, 2, 0, 1], aspect='auto')
axes[0].set_title('Electrostatic Potential (V)')
axes[0].set_xlabel('x (μm)')
axes[0].set_ylabel('y (μm)')
plt.colorbar(im1, ax=axes[0])

# Plot electron density
im2 = axes[1].imshow(np.log10(n), extent=[0, 2, 0, 1], aspect='auto')
axes[1].set_title('Electron Density (log₁₀ cm⁻³)')
axes[1].set_xlabel('x (μm)')
plt.colorbar(im2, ax=axes[1])

# Plot hole density
im3 = axes[2].imshow(np.log10(p), extent=[0, 2, 0, 1], aspect='auto')
axes[2].set_title('Hole Density (log₁₀ cm⁻³)')
axes[2].set_xlabel('x (μm)')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()
```

---

## Tutorials

### Tutorial 1: Understanding Mesh Types

**Structured Mesh:**
- Regular rectangular grid
- Fast and memory efficient
- Good for simple geometries

**Unstructured Mesh:**
- Triangular elements
- Flexible for complex geometries
- Supports adaptive refinement

```python
# Structured mesh example
sim_struct = simulator.Simulator(mesh_type="Structured")

# Unstructured mesh example  
sim_unstruct = simulator.Simulator(mesh_type="Unstructured")
```

### Tutorial 2: Numerical Methods Comparison

```python
methods = ["FDM", "FEM", "DG"]
results = {}

for method in methods:
    sim = simulator.Simulator(method=method)
    sim.set_doping(Nd, Na)
    result = sim.solve_drift_diffusion(bc)
    results[method] = result['potential']
    
# Compare accuracy and performance
```

### Tutorial 3: Adaptive Mesh Refinement

```python
# Enable AMR for automatic mesh adaptation
result_amr = sim.solve_drift_diffusion(
    bc=bc,
    use_amr=True,        # Enable AMR
    max_steps=100
)

# AMR automatically refines mesh in high-gradient regions
```

### Tutorial 4: GPU Acceleration

```cpp
#include "gpu_acceleration.hpp"

// C++ example for GPU acceleration
simulator::gpu::CudaPoissonSolver gpu_solver;

std::vector<double> rho(nx * ny, 1e16 * 1.602e-19);
std::vector<double> V(nx * ny, 0.0);

gpu_solver.solve_gpu(rho, V, nx, ny, dx, dy, epsilon);
```

---

## Advanced Features

### Custom Material Properties

```python
# Define custom material regions
regions = [
    {
        "material": "Si",
        "x_min": 0, "x_max": 1e-6,
        "y_min": 0, "y_max": 1e-6,
        "epsilon_r": 11.7,
        "bandgap": 1.12
    },
    {
        "material": "SiO2", 
        "x_min": 1e-6, "x_max": 1.1e-6,
        "y_min": 0, "y_max": 1e-6,
        "epsilon_r": 3.9,
        "bandgap": 9.0
    }
]

sim = simulator.Simulator(regions=regions)
```

### Transient Analysis

```python
# Time-dependent simulation
time_points = np.linspace(0, 1e-9, 100)  # 1 ns simulation
results_time = []

for t in time_points:
    # Update boundary conditions for time t
    bc_t = [0.0, 0.7 * np.sin(2*np.pi*1e9*t), 0.0, 0.0]
    result = sim.solve_drift_diffusion(bc_t)
    results_time.append(result)
```

### Performance Profiling

```cpp
#include "performance_optimization.hpp"

simulator::performance::Profiler profiler;

profiler.start_timer("poisson_solve");
// ... solve Poisson equation
profiler.end_timer("poisson_solve");

profiler.start_timer("drift_diffusion_solve");
// ... solve drift-diffusion
profiler.end_timer("drift_diffusion_solve");

// Print performance report
auto profile_data = profiler.get_profile_data();
for (const auto& data : profile_data) {
    std::cout << data.name << ": " << data.total_time << " ms" << std::endl;
}
```

---

## Troubleshooting

### Common Issues

**1. Import Error (Python)**
```
ImportError: cannot import name 'simulator'
```
**Solution**: Rebuild Python extension:
```bash
cd python
python setup.py build_ext --inplace
```

**2. Convergence Issues**
```
RuntimeError: Solver failed to converge
```
**Solutions**:
- Reduce tolerance: `poisson_tol=1e-8`
- Increase iterations: `poisson_max_iter=100`
- Refine mesh: increase `num_points_x`, `num_points_y`
- Enable AMR: `use_amr=True`

**3. Memory Issues**
```
MemoryError: Unable to allocate memory
```
**Solutions**:
- Reduce mesh resolution
- Use structured mesh instead of unstructured
- Enable GPU acceleration for large problems

**4. GPU Errors**
```
CUDA error: out of memory
```
**Solutions**:
- Reduce problem size
- Use CPU solver for very large problems
- Check GPU memory with `nvidia-smi`

### Performance Optimization

**For Small Problems (< 10³ unknowns):**
- Use FDM method
- Structured mesh
- CPU solver

**For Medium Problems (10³ - 10⁵ unknowns):**
- Use DG method
- Enable AMR
- Consider GPU acceleration

**For Large Problems (> 10⁵ unknowns):**
- Use GPU acceleration
- Unstructured mesh with AMR
- Parallel processing

### Getting Help

1. **Check documentation**: `docs/` directory
2. **Run examples**: `examples/` directory  
3. **Check issues**: GitHub issues page
4. **Contact support**: [email/forum]

---

## Best Practices

1. **Start simple**: Begin with structured mesh and FDM
2. **Validate results**: Compare with analytical solutions when possible
3. **Use appropriate tolerances**: Balance accuracy vs. performance
4. **Profile performance**: Identify bottlenecks
5. **Save intermediate results**: For long simulations
6. **Document parameters**: Keep track of simulation settings

---

## Next Steps

- Explore advanced examples in `examples/`
- Read the API reference in `docs/API_REFERENCE.md`
- Try GPU acceleration for performance
- Experiment with different numerical methods
- Contribute to the project on GitHub
