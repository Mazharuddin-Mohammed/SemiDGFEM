# SemiDGFEM API Reference

## Overview

SemiDGFEM is a comprehensive 2D semiconductor device simulator using Discontinuous Galerkin finite element methods. This document provides detailed API reference for both C++ and Python interfaces.

## Table of Contents

1. [C++ API](#cpp-api)
2. [Python API](#python-api)
3. [Examples](#examples)
4. [Error Handling](#error-handling)
5. [Performance Considerations](#performance-considerations)

---

## C++ API

### Core Classes

#### Device Class

```cpp
namespace simulator {
    class Device {
    public:
        // Constructors
        Device(double width, double height);
        Device(double width, double height, 
               const std::vector<std::map<std::string, double>>& regions);
        
        // Material properties
        double get_epsilon_at(double x, double y) const;
        std::vector<double> get_extents() const;
        bool is_valid() const;
        
        // Validation
        void validate() const;
    };
}
```

**Description**: Represents a 2D semiconductor device with geometry and material properties.

**Parameters**:
- `width`: Device width in meters
- `height`: Device height in meters  
- `regions`: Optional material regions with properties

**Example**:
```cpp
// Create a 1μm × 0.5μm silicon device
Device device(1e-6, 0.5e-6);

// Get permittivity at center
double epsilon = device.get_epsilon_at(0.5e-6, 0.25e-6);
```

#### Mesh Class

```cpp
namespace simulator {
    enum class MeshType { Structured, Unstructured };
    
    class Mesh {
    public:
        Mesh(const Device& device, MeshType type);
        
        // Mesh properties
        bool is_valid() const;
        void validate() const;
        size_t get_num_elements() const;
        size_t get_num_nodes() const;
        
        // Mesh access
        std::vector<std::vector<int>> get_elements() const;
        std::vector<std::array<double, 2>> get_nodes() const;
    };
}
```

**Description**: Manages finite element mesh generation and properties.

#### Poisson Solver

```cpp
namespace simulator {
    enum class Method { FDM, FEM, FVM, SEM, MC, DG };
    
    class Poisson {
    public:
        Poisson(const Device& device, Method method, MeshType mesh_type);
        
        // Solver configuration
        void set_charge_density(const std::vector<double>& rho);
        void set_boundary_conditions(const std::vector<double>& bc);
        
        // Solving
        std::vector<double> solve_2d(const std::vector<double>& bc);
        
        // Advanced features
        void set_tolerance(double tol);
        void set_max_iterations(int max_iter);
    };
}
```

**Description**: Solves the Poisson equation: -∇·(ε∇V) = ρ

**Key Methods**:
- `set_charge_density()`: Set space charge density
- `solve_2d()`: Solve for electrostatic potential
- `set_tolerance()`: Set convergence tolerance

#### DriftDiffusion Solver

```cpp
namespace simulator {
    class DriftDiffusion {
    public:
        DriftDiffusion(const Device& device, Method method, 
                      MeshType mesh_type, int order = 3);
        
        // Doping configuration
        void set_doping(const std::vector<double>& Nd, 
                       const std::vector<double>& Na);
        void set_trap_level(const std::vector<double>& Et);
        
        // Solving
        std::map<std::string, std::vector<double>> solve(
            const std::vector<double>& bc, double Vg = 0.0,
            int max_steps = 100, bool use_amr = false);
        
        // Analysis
        void compute_carrier_densities(const std::vector<double>& V,
                                     std::vector<double>& n,
                                     std::vector<double>& p) const;
        void compute_current_densities(const std::vector<double>& V,
                                     const std::vector<double>& n,
                                     const std::vector<double>& p,
                                     std::vector<double>& Jn,
                                     std::vector<double>& Jp) const;
    };
}
```

**Description**: Solves coupled drift-diffusion equations for carrier transport.

### AMR (Adaptive Mesh Refinement)

```cpp
namespace simulator::amr {
    class AMRController {
    public:
        // Error estimation
        std::vector<double> compute_residual_based_error(
            const std::vector<double>& solution,
            const std::vector<std::vector<int>>& elements,
            const std::vector<std::array<double, 2>>& nodes);
            
        std::vector<double> compute_zz_error(
            const std::vector<double>& solution,
            const std::vector<std::vector<int>>& elements,
            const std::vector<std::array<double, 2>>& nodes);
        
        // Anisotropy detection
        std::array<double, 2> detect_anisotropy_direction(
            int element_id,
            const std::vector<double>& solution,
            const std::vector<std::vector<int>>& elements,
            const std::vector<std::array<double, 2>>& vertices);
        
        // Refinement
        std::unordered_map<int, std::vector<int>> perform_refinement(
            const std::vector<ElementRefinement>& refinement_decisions,
            std::vector<std::vector<int>>& elements,
            std::vector<std::array<double, 2>>& vertices,
            std::vector<double>& solution);
    };
}
```

### GPU Acceleration

```cpp
namespace simulator::gpu {
    class CudaPoissonSolver {
    public:
        void solve_gpu(const std::vector<double>& rho,
                      std::vector<double>& V,
                      int nx, int ny, double dx, double dy,
                      double epsilon, double tolerance = 1e-6,
                      int max_iterations = 1000);
    };
    
    class CudaDriftDiffusionSolver {
    public:
        void solve_gpu(const std::vector<double>& V,
                      std::vector<double>& n, std::vector<double>& p,
                      std::vector<double>& Jn, std::vector<double>& Jp,
                      const std::vector<double>& Nd, const std::vector<double>& Na,
                      int nx, int ny, double dx, double dy,
                      double ni, double Vt, double mu_n, double mu_p, double q);
    };
}
```

---

## Python API

### Simulator Class

```python
class Simulator:
    def __init__(self, dimension="TwoD", extents=[1e-6, 0.5e-6], 
                 num_points_x=50, num_points_y=25,
                 method="DG", mesh_type="Structured", regions=None):
        """
        Initialize semiconductor device simulator.
        
        Parameters:
        -----------
        dimension : str
            Simulation dimension ("TwoD")
        extents : list
            Device dimensions [width, height] in meters
        num_points_x, num_points_y : int
            Mesh resolution
        method : str
            Numerical method ("FDM", "FEM", "FVM", "SEM", "MC", "DG")
        mesh_type : str
            Mesh type ("Structured", "Unstructured")
        regions : list, optional
            Material regions
        """
```

### Key Methods

```python
def set_doping(self, Nd, Na):
    """
    Set doping concentrations.
    
    Parameters:
    -----------
    Nd : numpy.ndarray
        Donor concentration (1/m³)
    Na : numpy.ndarray  
        Acceptor concentration (1/m³)
    """

def solve_poisson(self, bc):
    """
    Solve Poisson equation.
    
    Parameters:
    -----------
    bc : array_like
        Boundary conditions [V_left, V_right, V_bottom, V_top]
        
    Returns:
    --------
    numpy.ndarray
        Electrostatic potential
    """

def solve_drift_diffusion(self, bc, Vg=0.0, max_steps=100, 
                         use_amr=False, poisson_max_iter=50, 
                         poisson_tol=1e-6):
    """
    Solve drift-diffusion equations.
    
    Parameters:
    -----------
    bc : array_like
        Boundary conditions
    Vg : float
        Gate voltage (V)
    max_steps : int
        Maximum iteration steps
    use_amr : bool
        Enable adaptive mesh refinement
    poisson_max_iter : int
        Poisson solver max iterations
    poisson_tol : float
        Poisson solver tolerance
        
    Returns:
    --------
    dict
        Solution dictionary with keys:
        - 'potential': Electrostatic potential
        - 'n': Electron concentration
        - 'p': Hole concentration  
        - 'Jn': Electron current density
        - 'Jp': Hole current density
    """
```

---

## Examples

### Basic P-N Junction Simulation

```python
import numpy as np
import simulator

# Create device
sim = simulator.Simulator(
    extents=[2e-6, 1e-6],  # 2μm × 1μm
    num_points_x=100,
    num_points_y=50,
    method="DG"
)

# Set doping profile
x = np.linspace(0, 2e-6, 100)
y = np.linspace(0, 1e-6, 50)
X, Y = np.meshgrid(x, y)

# P-N junction at x = 1μm
Nd = np.where(X < 1e-6, 0, 1e16 * np.ones_like(X)).flatten()
Na = np.where(X < 1e-6, 1e16 * np.ones_like(X), 0).flatten()

sim.set_doping(Nd, Na)

# Solve
bc = [0.0, 0.7, 0.0, 0.0]  # Forward bias
result = sim.solve_drift_diffusion(bc)

# Extract results
potential = result['potential'].reshape(50, 100)
n = result['n'].reshape(50, 100)
p = result['p'].reshape(50, 100)
```

### Advanced AMR Simulation

```cpp
#include "simulator.hpp"

using namespace simulator;

int main() {
    // Create device and mesh
    Device device(2e-6, 1e-6);
    DriftDiffusion solver(device, Method::DG, MeshType::Unstructured);
    
    // Set up AMR controller
    amr::AMRController amr_controller;
    
    // Initial solve
    std::vector<double> bc = {0.0, 0.7, 0.0, 0.0};
    auto result = solver.solve(bc, 0.0, 100, true);  // Enable AMR
    
    return 0;
}
```

---

## Error Handling

### C++ Exceptions

- `std::invalid_argument`: Invalid input parameters
- `std::runtime_error`: Solver failures, memory allocation errors
- `std::out_of_range`: Array access errors

### Python Exceptions

- `ValueError`: Invalid input parameters
- `RuntimeError`: Solver failures
- `MemoryError`: Insufficient memory

---

## Performance Considerations

### Memory Usage

- Structured mesh: O(nx × ny) memory
- Unstructured mesh: O(n_elements × dofs_per_element)
- GPU acceleration: Additional GPU memory required

### Computational Complexity

- Poisson solver: O(n log n) with multigrid
- Drift-diffusion: O(n × iterations)
- AMR: Additional O(n) per refinement cycle

### Optimization Tips

1. **Use GPU acceleration** for large problems (>10⁴ unknowns)
2. **Enable AMR** for problems with sharp gradients
3. **Choose appropriate method**: DG for accuracy, FDM for speed
4. **Optimize mesh resolution** based on device features

---

## Version Information

- **Current Version**: 1.0.0
- **API Stability**: Stable for core classes, experimental for GPU features
- **Dependencies**: PETSc, Boost, OpenMP, CUDA (optional)

For more examples and tutorials, see the `examples/` directory.
