# SemiDGFEM Installation Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation](#detailed-installation)
4. [GPU Acceleration Setup](#gpu-acceleration-setup)
5. [Python Environment Setup](#python-environment-setup)
6. [Verification and Testing](#verification-and-testing)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

---

## System Requirements

### Minimum Requirements

**Operating System**:
- Linux: Ubuntu 18.04+, CentOS 7+, RHEL 7+
- Windows: Windows 10 (64-bit)
- macOS: 10.14+ (Mojave)

**Hardware**:
- CPU: x86_64 with SSE4.2 support
- RAM: 4 GB minimum, 8 GB recommended
- Storage: 2 GB free space
- GPU: Optional (NVIDIA/AMD for acceleration)

**Software Dependencies**:
- CMake 3.16+
- C++ compiler with C++17 support
- Python 3.8+
- Git

### Recommended Requirements

**Hardware**:
- CPU: Modern x86_64 with AVX2 support
- RAM: 16 GB or more
- Storage: SSD with 10 GB free space
- GPU: NVIDIA RTX series or AMD RX series

**Software**:
- CMake 3.20+
- GCC 9+ or Clang 10+
- Python 3.9+
- CUDA 11.0+ (for NVIDIA GPU acceleration)

---

## Quick Installation

### Linux (Ubuntu/Debian)

```bash
# 1. Install dependencies
sudo apt update
sudo apt install -y cmake build-essential python3 python3-pip git

# 2. Clone repository
git clone https://github.com/your-repo/SemiDGFEM.git
cd SemiDGFEM

# 3. Build framework
mkdir build && cd build
cmake ..
make -j$(nproc)

# 4. Install Python dependencies
pip3 install numpy scipy matplotlib PySide6

# 5. Set environment
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/../python:$PYTHONPATH

# 6. Test installation
cd ../python
python3 -c "import simulator; print('Installation successful!')"
```

### Windows (PowerShell)

```powershell
# 1. Install dependencies (using Chocolatey)
choco install cmake git python3 visualstudio2019buildtools

# 2. Clone repository
git clone https://github.com/your-repo/SemiDGFEM.git
cd SemiDGFEM

# 3. Build framework
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release

# 4. Install Python dependencies
pip install numpy scipy matplotlib PySide6

# 5. Set environment
$env:PATH += ";$PWD\Release"
$env:PYTHONPATH += ";$PWD\..\python"

# 6. Test installation
cd ..\python
python -c "import simulator; print('Installation successful!')"
```

### macOS

```bash
# 1. Install dependencies (using Homebrew)
brew install cmake python3 git

# 2. Clone repository
git clone https://github.com/your-repo/SemiDGFEM.git
cd SemiDGFEM

# 3. Build framework
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

# 4. Install Python dependencies
pip3 install numpy scipy matplotlib PySide6

# 5. Set environment
export DYLD_LIBRARY_PATH=$PWD:$DYLD_LIBRARY_PATH
export PYTHONPATH=$PWD/../python:$PYTHONPATH

# 6. Test installation
cd ../python
python3 -c "import simulator; print('Installation successful!')"
```

---

## Detailed Installation

### Step 1: Install System Dependencies

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install build tools
sudo apt install -y \
    cmake \
    build-essential \
    gcc-9 \
    g++-9 \
    python3 \
    python3-dev \
    python3-pip \
    git \
    pkg-config

# Install optional dependencies
sudo apt install -y \
    libeigen3-dev \
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev
```

#### CentOS/RHEL
```bash
# Install EPEL repository
sudo yum install -y epel-release

# Install build tools
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    cmake3 \
    python3 \
    python3-devel \
    python3-pip \
    git

# Create cmake symlink
sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
```

#### Windows
```powershell
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Install CMake
# Download from: https://cmake.org/download/

# Install Python
# Download from: https://python.org/downloads/

# Install Git
# Download from: https://git-scm.com/download/win
```

### Step 2: Clone and Configure

```bash
# Clone repository
git clone https://github.com/your-repo/SemiDGFEM.git
cd SemiDGFEM

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU_ACCELERATION=ON \
    -DENABLE_SIMD_OPTIMIZATION=ON \
    -DENABLE_PYTHON_BINDINGS=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local
```

### Step 3: Build Framework

```bash
# Build with optimal number of cores
make -j$(nproc)

# Optional: Install system-wide
sudo make install
```

### Step 4: Python Environment Setup

```bash
# Create virtual environment (recommended)
python3 -m venv semidgfem-env
source semidgfem-env/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install \
    numpy>=1.20.0 \
    scipy>=1.7.0 \
    matplotlib>=3.4.0 \
    PySide6>=6.2.0 \
    h5py>=3.1.0 \
    pytest>=6.0.0
```

---

## GPU Acceleration Setup

### NVIDIA CUDA Setup

#### Linux
```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-470

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Set environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi
```

#### Windows
```powershell
# Download and install CUDA Toolkit from:
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version
nvidia-smi
```

### AMD OpenCL Setup

#### Linux
```bash
# Install AMD drivers
sudo apt install mesa-opencl-icd

# Install OpenCL headers
sudo apt install opencl-headers

# Verify installation
clinfo
```

### Intel OpenCL Setup

```bash
# Install Intel OpenCL runtime
sudo apt install intel-opencl-icd

# Verify installation
clinfo
```

---

## Python Environment Setup

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv semidgfem-env

# Activate environment
source semidgfem-env/bin/activate  # Linux/macOS
# or
semidgfem-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Conda Environment

```bash
# Create conda environment
conda create -n semidgfem python=3.9
conda activate semidgfem

# Install dependencies
conda install numpy scipy matplotlib
pip install PySide6
```

### System-wide Installation

```bash
# Install dependencies system-wide
sudo pip3 install numpy scipy matplotlib PySide6

# Or using package manager (Ubuntu)
sudo apt install python3-numpy python3-scipy python3-matplotlib
```

---

## Verification and Testing

### Basic Functionality Test

```bash
# Navigate to Python directory
cd python

# Test basic import
python3 -c "
import simulator
print('✅ Core simulator import successful')

import advanced_transport
print('✅ Advanced transport import successful')

from mosfet_simulation import MOSFETDevice
print('✅ MOSFET simulation import successful')

from heterostructure_simulation import HeterostructureDevice
print('✅ Heterostructure simulation import successful')

print('🎉 All imports successful!')
"
```

### Performance Test

```bash
# Run performance benchmark
python3 -c "
from performance_bindings import PerformanceOptimizer
import numpy as np

optimizer = PerformanceOptimizer()
optimizer.print_performance_info()

# Test SIMD performance
a = np.random.random(100000)
b = np.random.random(100000)
result = optimizer.optimize_computation('vector_add', a, b)
print('✅ SIMD optimization working')
"
```

### GPU Test

```bash
# Test GPU acceleration
python3 -c "
from gpu_acceleration import GPUAcceleratedSolver

solver = GPUAcceleratedSolver()
if solver.is_gpu_available():
    print('✅ GPU acceleration available')
    benchmark = solver.benchmark_performance()
    print(f'GPU performance: {benchmark}')
else:
    print('⚠️  GPU not available, using CPU')
"
```

### Complete Test Suite

```bash
# Run comprehensive tests
cd ../tests
python3 -m pytest test_*.py -v

# Run specific test categories
python3 -m pytest test_core.py -v          # Core functionality
python3 -m pytest test_transport.py -v     # Transport models
python3 -m pytest test_mosfet.py -v        # MOSFET simulation
python3 -m pytest test_hetero.py -v        # Heterostructure simulation
python3 -m pytest test_performance.py -v   # Performance optimization
```

---

## Troubleshooting

### Common Issues

#### 1. Library Not Found
```bash
# Error: libsimulator.so not found
# Solution: Set library path
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH

# For permanent solution, add to ~/.bashrc
echo 'export LD_LIBRARY_PATH=/path/to/SemiDGFEM/build:$LD_LIBRARY_PATH' >> ~/.bashrc
```

#### 2. Python Import Errors
```bash
# Error: No module named 'simulator'
# Solution: Set Python path
export PYTHONPATH=$PWD/python:$PYTHONPATH

# Or install in development mode
cd python
pip install -e .
```

#### 3. CMake Configuration Issues
```bash
# Error: CMake version too old
# Solution: Install newer CMake
pip install cmake

# Error: Compiler not found
# Solution: Install build tools
sudo apt install build-essential
```

#### 4. GPU Not Detected
```bash
# Check GPU status
nvidia-smi  # For NVIDIA
clinfo      # For OpenCL

# Verify drivers
lsmod | grep nvidia
lsmod | grep amdgpu
```

### Performance Issues

#### 1. Slow Compilation
```bash
# Use parallel compilation
make -j$(nproc)

# Use ccache for faster rebuilds
sudo apt install ccache
export CC="ccache gcc"
export CXX="ccache g++"
```

#### 2. Runtime Performance
```bash
# Enable optimizations
cmake .. -DCMAKE_BUILD_TYPE=Release

# Use performance governor
sudo cpupower frequency-set -g performance

# Check CPU features
cat /proc/cpuinfo | grep flags
```

---

## Performance Optimization

### Compiler Optimizations

```bash
# Configure with optimizations
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" \
    -DENABLE_LTO=ON
```

### Memory Optimization

```bash
# Set memory limits
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# Use memory pool
export SEMIDGFEM_USE_MEMORY_POOL=1
```

### GPU Optimization

```bash
# Set GPU memory fraction
export CUDA_MEMORY_FRACTION=0.8

# Enable GPU persistence
sudo nvidia-smi -pm 1
```

---

## Environment Configuration

### Permanent Setup Script

Create `setup_environment.sh`:

```bash
#!/bin/bash
# SemiDGFEM Environment Setup

# Set library paths
export LD_LIBRARY_PATH=/path/to/SemiDGFEM/build:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/SemiDGFEM/python:$PYTHONPATH

# Set performance options
export OMP_NUM_THREADS=$(nproc)
export SEMIDGFEM_USE_SIMD=1
export SEMIDGFEM_USE_GPU=1

# CUDA settings (if available)
if command -v nvcc &> /dev/null; then
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

echo "SemiDGFEM environment configured"
```

Make executable and source:
```bash
chmod +x setup_environment.sh
source setup_environment.sh

# Add to ~/.bashrc for permanent setup
echo 'source /path/to/setup_environment.sh' >> ~/.bashrc
```

---

## Docker Installation

### Dockerfile

```dockerfile
FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake build-essential python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Clone and build
WORKDIR /opt
RUN git clone https://github.com/your-repo/SemiDGFEM.git
WORKDIR /opt/SemiDGFEM
RUN mkdir build && cd build && cmake .. && make -j$(nproc)

# Install Python dependencies
RUN pip3 install numpy scipy matplotlib PySide6

# Set environment
ENV LD_LIBRARY_PATH=/opt/SemiDGFEM/build:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/SemiDGFEM/python:$PYTHONPATH

# Test installation
RUN cd python && python3 -c "import simulator; print('Docker installation successful!')"

WORKDIR /opt/SemiDGFEM
CMD ["/bin/bash"]
```

### Build and Run

```bash
# Build Docker image
docker build -t semidgfem .

# Run container
docker run -it --rm semidgfem

# Run with GPU support (NVIDIA)
docker run --gpus all -it --rm semidgfem
```

---

## Verification Checklist

- [ ] CMake configuration successful
- [ ] C++ compilation successful
- [ ] Python bindings working
- [ ] Core simulator import successful
- [ ] Advanced transport models available
- [ ] MOSFET simulation working
- [ ] Heterostructure simulation working
- [ ] Performance optimization enabled
- [ ] GPU acceleration available (if hardware present)
- [ ] Test suite passes
- [ ] Example scripts run successfully

**Installation Complete!** 🎉

Your SemiDGFEM installation is now ready for semiconductor device simulation.
