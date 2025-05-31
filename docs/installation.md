# SemiDGFEM Installation Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation](#detailed-installation)
4. [GPU Support](#gpu-support)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Docker Installation](#docker-installation)
8. [Development Setup](#development-setup)

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+, CentOS 7+), macOS 10.14+, Windows 10 (WSL2)
- **CPU**: x86_64 with SSE4.2 support
- **Memory**: 4 GB RAM
- **Storage**: 2 GB free space
- **Compiler**: GCC 7+, Clang 6+, or MSVC 2019+

### Recommended Requirements
- **CPU**: x86_64 with AVX2 support (Intel Haswell+, AMD Excavator+)
- **Memory**: 16 GB RAM
- **Storage**: 10 GB free space (for examples and documentation)
- **GPU**: NVIDIA GPU with Compute Capability 3.5+ (for CUDA acceleration)

### Software Dependencies

**Core Dependencies:**
- CMake 3.16+
- PETSc 3.14+
- GMSH 4.8+
- Boost 1.70+
- OpenMP 4.0+

**Python Dependencies:**
- Python 3.8+
- NumPy 1.19+
- SciPy 1.5+
- Matplotlib 3.3+
- Cython 0.29+

**Optional Dependencies:**
- CUDA Toolkit 11.0+ (for GPU acceleration)
- OpenCL 2.0+ (for cross-platform GPU support)
- Vulkan SDK 1.2+ (for advanced visualization)
- PySide6 6.0+ (for GUI)

## Quick Installation

### Using Conda (Recommended)

```bash
# Create conda environment
conda create -n semidgfem python=3.9
conda activate semidgfem

# Install from conda-forge
conda install -c conda-forge semidgfem

# Verify installation
python -c "import simulator; print('Installation successful!')"
```

### Using pip

```bash
# Install from PyPI
pip install semidgfem

# Install with GPU support
pip install semidgfem[gpu]

# Install with full features
pip install semidgfem[full]
```

## Detailed Installation

### Ubuntu/Debian

#### 1. Install System Dependencies

```bash
# Update package list
sudo apt-get update

# Install build tools
sudo apt-get install -y build-essential cmake git

# Install core dependencies
sudo apt-get install -y libpetsc-dev libpetsc3.14-dev-examples
sudo apt-get install -y libgmsh-dev gmsh
sudo apt-get install -y libboost-all-dev
sudo apt-get install -y libomp-dev

# Install Python development
sudo apt-get install -y python3-dev python3-pip python3-venv

# Optional: Install GPU support
sudo apt-get install -y nvidia-cuda-toolkit  # For CUDA
sudo apt-get install -y opencl-headers libopencl-dev  # For OpenCL

# Optional: Install Vulkan
sudo apt-get install -y vulkan-tools libvulkan-dev
```

#### 2. Create Python Environment

```bash
# Create virtual environment
python3 -m venv semidgfem-env
source semidgfem-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install numpy scipy matplotlib cython
pip install PySide6  # For GUI support
```

#### 3. Build from Source

```bash
# Clone repository
git clone https://github.com/your-repo/SemiDGFEM.git
cd SemiDGFEM

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=ON \
    -DENABLE_OPENCL=ON \
    -DENABLE_VULKAN=ON \
    -DENABLE_OPENMP=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# Build (use all available cores)
make -j$(nproc)

# Install
sudo make install

# Update library path
echo '/usr/local/lib' | sudo tee /etc/ld.so.conf.d/semidgfem.conf
sudo ldconfig
```

#### 4. Install Python Interface

```bash
# Go back to source directory
cd ..

# Build Python extension
cd python
python setup.py build_ext --inplace

# Install in development mode
pip install -e .
```

### CentOS/RHEL/Fedora

#### 1. Install Dependencies

```bash
# CentOS/RHEL 8
sudo dnf groupinstall "Development Tools"
sudo dnf install cmake git
sudo dnf install petsc-devel gmsh-devel boost-devel
sudo dnf install python3-devel python3-pip

# Enable EPEL for additional packages
sudo dnf install epel-release
sudo dnf install libomp-devel

# Fedora
sudo dnf install gcc-c++ cmake git
sudo dnf install petsc-devel gmsh-devel boost-devel
sudo dnf install python3-devel python3-pip libomp-devel
```

#### 2. Build and Install

Follow the same steps as Ubuntu, but use `dnf` instead of `apt-get`.

### macOS

#### 1. Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Install Dependencies

```bash
# Install build tools
brew install cmake git

# Install core dependencies
brew install petsc gmsh boost
brew install libomp

# Install Python
brew install python@3.9
```

#### 3. Set Environment Variables

```bash
# Add to ~/.zshrc or ~/.bash_profile
export LDFLAGS="-L/opt/homebrew/lib"
export CPPFLAGS="-I/opt/homebrew/include"
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"

# For OpenMP
export CC=/opt/homebrew/bin/gcc-11
export CXX=/opt/homebrew/bin/g++-11
```

#### 4. Build and Install

Follow the same build steps as Linux.

### Windows (WSL2)

#### 1. Install WSL2

```powershell
# Run in PowerShell as Administrator
wsl --install -d Ubuntu-20.04
```

#### 2. Setup Ubuntu in WSL2

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade

# Follow Ubuntu installation steps above
```

#### 3. GPU Support in WSL2

```bash
# Install CUDA in WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

## GPU Support

### CUDA Installation

#### Ubuntu/Debian

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### Environment Setup

```bash
# Add to ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# Reload environment
source ~/.bashrc
```

#### Verification

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Test CUDA with SemiDGFEM
python -c "from simulator.gpu import GPUContext; ctx = GPUContext(); print('CUDA available:', ctx.initialize())"
```

### OpenCL Installation

#### Intel OpenCL

```bash
# Ubuntu/Debian
sudo apt-get install intel-opencl-icd

# CentOS/RHEL
sudo dnf install intel-opencl
```

#### AMD OpenCL

```bash
# Download and install AMD APP SDK
wget https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/download/1.0/OCL_SDK_Light_AMD64.tar.xz
tar -xf OCL_SDK_Light_AMD64.tar.xz
sudo ./install.sh
```

#### Verification

```bash
# List OpenCL platforms
clinfo

# Test OpenCL with SemiDGFEM
python -c "from simulator.gpu import GPUContext, GPUBackend; ctx = GPUContext(); print('OpenCL available:', ctx.initialize(GPUBackend.OPENCL))"
```

## Verification

### Basic Functionality Test

```bash
# Run C++ tests
cd build
ctest -V

# Run Python tests
cd ../python
python -m pytest tests/ -v
```

### Performance Test

```python
import numpy as np
from simulator import Simulator, Device, Method, MeshType
import time

# Create test case
device = Device(1e-6, 0.5e-6)
sim = Simulator(device, Method.DG, MeshType.Structured, order=3)

# Set simple doping
n_points = sim.get_dof_count()
Nd = np.full(n_points, 1e17)
Na = np.zeros(n_points)
sim.set_doping(Nd, Na)

# Test CPU performance
start_time = time.time()
results_cpu = sim.solve_drift_diffusion([0.0, 1.0, 0.0, 0.0])
cpu_time = time.time() - start_time

print(f"CPU simulation time: {cpu_time:.2f} seconds")
print(f"DOFs: {sim.get_dof_count()}")

# Test GPU performance (if available)
try:
    sim.enable_gpu(True)
    start_time = time.time()
    results_gpu = sim.solve_drift_diffusion([0.0, 1.0, 0.0, 0.0])
    gpu_time = time.time() - start_time
    
    print(f"GPU simulation time: {gpu_time:.2f} seconds")
    print(f"GPU speedup: {cpu_time/gpu_time:.1f}x")
except:
    print("GPU not available")
```

### GUI Test

```bash
# Test GUI (requires display)
python python/gui/main_gui_2d.py
```

## Troubleshooting

### Common Build Issues

#### PETSc Not Found

```bash
# Check PETSc installation
pkg-config --cflags petsc
pkg-config --libs petsc

# If not found, install development package
sudo apt-get install libpetsc-dev

# Or specify PETSc path manually
cmake .. -DPETSC_DIR=/usr/lib/petsc
```

#### GMSH Not Found

```bash
# Install GMSH development files
sudo apt-get install libgmsh-dev

# Or build GMSH from source
git clone https://gitlab.onelab.info/gmsh/gmsh.git
cd gmsh
mkdir build && cd build
cmake .. -DENABLE_BUILD_SHARED=ON
make -j$(nproc)
sudo make install
```

#### Boost Not Found

```bash
# Install all Boost libraries
sudo apt-get install libboost-all-dev

# Or install specific components
sudo apt-get install libboost-system-dev libboost-filesystem-dev
```

### Runtime Issues

#### Library Loading Errors

```bash
# Check library dependencies
ldd /usr/local/lib/libsimulator.so

# Update library cache
sudo ldconfig

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

#### Python Import Errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Install in development mode
cd python
pip install -e .

# Check extension loading
python -c "import simulator._simulator; print('C++ extension loaded')"
```

#### GPU Issues

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check OpenCL
clinfo

# Test GPU context
python -c "
from simulator.gpu import GPUContext
ctx = GPUContext()
print('GPU available:', ctx.initialize())
if ctx.is_initialized():
    info = ctx.get_device_info()
    print(f'Device: {info.name}')
    print(f'Memory: {info.global_memory / 1e9:.1f} GB')
"
```

### Getting Help

If you encounter issues:

1. **Check the FAQ**: [docs/faq.md](docs/faq.md)
2. **Search existing issues**: [GitHub Issues](https://github.com/your-repo/SemiDGFEM/issues)
3. **Create a new issue**: Include system info, error messages, and steps to reproduce
4. **Join discussions**: [GitHub Discussions](https://github.com/your-repo/SemiDGFEM/discussions)

## Docker Installation

### Using Pre-built Image

```bash
# Pull latest image
docker pull semidgfem/semidgfem:latest

# Run with GPU support
docker run --gpus all -it semidgfem/semidgfem:latest

# Run with volume mounting
docker run --gpus all -v $(pwd):/workspace -it semidgfem/semidgfem:latest
```

### Building Custom Image

```dockerfile
# Dockerfile
FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libpetsc-dev libgmsh-dev libboost-all-dev \
    python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build SemiDGFEM
COPY . /src/SemiDGFEM
WORKDIR /src/SemiDGFEM
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

# Install Python interface
RUN cd python && pip install -e .

WORKDIR /workspace
CMD ["python3"]
```

```bash
# Build image
docker build -t semidgfem-custom .

# Run container
docker run -it semidgfem-custom
```

## Development Setup

### Setting up Development Environment

```bash
# Clone with submodules
git clone --recursive https://github.com/your-repo/SemiDGFEM.git
cd SemiDGFEM

# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Build in debug mode
mkdir build-debug && cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTING=ON
make -j$(nproc)
```

### Code Formatting and Linting

```bash
# Format C++ code
clang-format -i src/**/*.cpp include/**/*.hpp

# Format Python code
black python/
isort python/

# Run linting
cppcheck src/
pylint python/simulator/
```

### Running Tests

```bash
# C++ tests
cd build-debug
ctest -V

# Python tests with coverage
cd python
pytest --cov=simulator tests/

# Integration tests
python tests/integration/test_full_simulation.py
```

This completes the comprehensive installation guide with support for multiple platforms, GPU acceleration, and development setup.
