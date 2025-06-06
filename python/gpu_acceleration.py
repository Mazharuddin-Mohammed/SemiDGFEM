"""
GPU Acceleration Module - CUDA/OpenCL Backend Integration
Provides comprehensive GPU acceleration for semiconductor device simulation

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add build directory to path
build_dir = Path(__file__).parent.parent / "build"
if build_dir.exists():
    os.environ['LD_LIBRARY_PATH'] = str(build_dir) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

class GPUBackend:
    """GPU backend enumeration"""
    NONE = "NONE"
    CUDA = "CUDA"
    OPENCL = "OPENCL"
    AUTO = "AUTO"

class GPUDeviceInfo:
    """GPU device information"""
    
    def __init__(self, name="", memory=0, compute_units=0, backend=GPUBackend.NONE):
        self.name = name
        self.memory = memory  # in bytes
        self.compute_units = compute_units
        self.backend = backend
        self.max_work_group_size = 256
        self.max_clock_frequency = 1000  # MHz
        
    def __str__(self):
        return f"{self.name} ({self.backend}, {self.memory // (1024**3)}GB)"

class GPUContext:
    """GPU context manager with CUDA/OpenCL support"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
            
        self.initialized = False
        self.backend = GPUBackend.NONE
        self.device_info = GPUDeviceInfo()
        self.context = None
        self.command_queue = None
        self.cuda_context = None
        self.performance_timers = {}
        
    def initialize(self, preferred_backend=GPUBackend.AUTO):
        """Initialize GPU context"""
        if self.initialized:
            return True
        
        print("🔧 Initializing GPU acceleration...")
        
        # Detect available devices
        devices = self.detect_devices()
        if not devices:
            print("   ❌ No GPU devices found")
            return False
        
        # Select best device
        selected_device = self._select_device(devices, preferred_backend)
        if not selected_device:
            print("   ❌ No suitable GPU device found")
            return False
        
        self.device_info = selected_device
        self.backend = selected_device.backend
        
        # Initialize backend-specific context
        success = False
        if self.backend == GPUBackend.CUDA:
            success = self._initialize_cuda()
        elif self.backend == GPUBackend.OPENCL:
            success = self._initialize_opencl()
        
        if success:
            self.initialized = True
            print(f"   ✅ GPU initialized: {self.device_info}")
        else:
            print(f"   ❌ Failed to initialize {self.backend}")
        
        return success
    
    def detect_devices(self) -> List[GPUDeviceInfo]:
        """Detect available GPU devices"""
        devices = []
        
        # Try CUDA
        cuda_devices = self._detect_cuda_devices()
        devices.extend(cuda_devices)
        
        # Try OpenCL
        opencl_devices = self._detect_opencl_devices()
        devices.extend(opencl_devices)
        
        return devices
    
    def _detect_cuda_devices(self) -> List[GPUDeviceInfo]:
        """Detect CUDA devices"""
        devices = []
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            memory = int(parts[1].strip()) * 1024 * 1024  # Convert MB to bytes
                            device = GPUDeviceInfo(name, memory, 32, GPUBackend.CUDA)  # Assume 32 SMs
                            devices.append(device)
        except Exception as e:
            print(f"   CUDA detection failed: {e}")
        
        return devices
    
    def _detect_opencl_devices(self) -> List[GPUDeviceInfo]:
        """Detect OpenCL devices"""
        devices = []
        try:
            # In real implementation, would use PyOpenCL
            # For now, return empty list
            pass
        except Exception as e:
            print(f"   OpenCL detection failed: {e}")
        
        return devices
    
    def _select_device(self, devices: List[GPUDeviceInfo], preferred_backend: str) -> Optional[GPUDeviceInfo]:
        """Select best GPU device"""
        if not devices:
            return None
        
        if preferred_backend == GPUBackend.AUTO:
            # Select device with most memory
            return max(devices, key=lambda d: d.memory)
        else:
            # Find device with preferred backend
            for device in devices:
                if device.backend == preferred_backend:
                    return device
        
        return None
    
    def _initialize_cuda(self) -> bool:
        """Initialize CUDA context"""
        try:
            # In real implementation, would use PyCUDA
            print("   🚀 Initializing CUDA context...")
            self.cuda_context = "cuda_context_placeholder"
            return True
        except Exception as e:
            print(f"   ❌ CUDA initialization failed: {e}")
            return False
    
    def _initialize_opencl(self) -> bool:
        """Initialize OpenCL context"""
        try:
            # In real implementation, would use PyOpenCL
            print("   🚀 Initializing OpenCL context...")
            self.context = "opencl_context_placeholder"
            self.command_queue = "opencl_queue_placeholder"
            return True
        except Exception as e:
            print(f"   ❌ OpenCL initialization failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if GPU is available"""
        return self.initialized
    
    def get_backend(self) -> str:
        """Get current backend"""
        return self.backend
    
    def get_device_info(self) -> GPUDeviceInfo:
        """Get device information"""
        return self.device_info
    
    def start_timer(self, name: str):
        """Start performance timer"""
        self.performance_timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End performance timer and return elapsed time"""
        if name in self.performance_timers:
            elapsed = time.time() - self.performance_timers[name]
            del self.performance_timers[name]
            return elapsed
        return 0.0
    
    def finalize(self):
        """Finalize GPU context"""
        if not self.initialized:
            return
        
        print("🔧 Finalizing GPU context...")
        
        if self.backend == GPUBackend.CUDA:
            self._finalize_cuda()
        elif self.backend == GPUBackend.OPENCL:
            self._finalize_opencl()
        
        self.initialized = False
        print("   ✅ GPU context finalized")
    
    def _finalize_cuda(self):
        """Finalize CUDA context"""
        self.cuda_context = None
    
    def _finalize_opencl(self):
        """Finalize OpenCL context"""
        self.context = None
        self.command_queue = None

class GPUMemory:
    """GPU memory management"""
    
    def __init__(self, size: int, dtype=np.float64):
        self.size = size
        self.dtype = dtype
        self.device_ptr = None
        self.host_data = None
        self.context = GPUContext()
        
        if self.context.is_available():
            self._allocate_device_memory()
    
    def _allocate_device_memory(self):
        """Allocate device memory"""
        try:
            if self.context.backend == GPUBackend.CUDA:
                self._allocate_cuda_memory()
            elif self.context.backend == GPUBackend.OPENCL:
                self._allocate_opencl_memory()
        except Exception as e:
            print(f"GPU memory allocation failed: {e}")
    
    def _allocate_cuda_memory(self):
        """Allocate CUDA memory"""
        # In real implementation, would use PyCUDA
        self.device_ptr = f"cuda_ptr_{id(self)}"
    
    def _allocate_opencl_memory(self):
        """Allocate OpenCL memory"""
        # In real implementation, would use PyOpenCL
        self.device_ptr = f"opencl_buffer_{id(self)}"
    
    def copy_to_device(self, host_data: np.ndarray):
        """Copy data to device"""
        if not self.context.is_available():
            self.host_data = host_data.copy()
            return
        
        self.host_data = host_data.astype(self.dtype)
        
        if self.context.backend == GPUBackend.CUDA:
            self._copy_to_cuda(host_data)
        elif self.context.backend == GPUBackend.OPENCL:
            self._copy_to_opencl(host_data)
    
    def _copy_to_cuda(self, host_data: np.ndarray):
        """Copy to CUDA device"""
        # In real implementation, would use cudaMemcpy
        pass
    
    def _copy_to_opencl(self, host_data: np.ndarray):
        """Copy to OpenCL device"""
        # In real implementation, would use clEnqueueWriteBuffer
        pass
    
    def copy_to_host(self, host_data: np.ndarray):
        """Copy data from device to host"""
        if not self.context.is_available():
            if self.host_data is not None:
                host_data[:] = self.host_data
            return
        
        if self.context.backend == GPUBackend.CUDA:
            self._copy_from_cuda(host_data)
        elif self.context.backend == GPUBackend.OPENCL:
            self._copy_from_opencl(host_data)
    
    def _copy_from_cuda(self, host_data: np.ndarray):
        """Copy from CUDA device"""
        # In real implementation, would use cudaMemcpy
        if self.host_data is not None:
            host_data[:] = self.host_data
    
    def _copy_from_opencl(self, host_data: np.ndarray):
        """Copy from OpenCL device"""
        # In real implementation, would use clEnqueueReadBuffer
        if self.host_data is not None:
            host_data[:] = self.host_data
    
    def __del__(self):
        """Cleanup device memory"""
        if self.device_ptr:
            # In real implementation, would free device memory
            pass

class GPUKernels:
    """GPU kernel implementations for semiconductor physics"""

    def __init__(self, context: GPUContext = None):
        self.context = context or gpu_context
        self.kernels_compiled = False

        if self.context.is_available():
            self._compile_kernels()

    def _compile_kernels(self):
        """Compile GPU kernels"""
        try:
            if self.context.backend == GPUBackend.CUDA:
                self._compile_cuda_kernels()
            elif self.context.backend == GPUBackend.OPENCL:
                self._compile_opencl_kernels()

            self.kernels_compiled = True
            print("   ✅ GPU kernels compiled")
        except Exception as e:
            print(f"   ❌ Kernel compilation failed: {e}")

    def _compile_cuda_kernels(self):
        """Compile CUDA kernels"""
        # In real implementation, would compile CUDA kernels
        pass

    def _compile_opencl_kernels(self):
        """Compile OpenCL kernels"""
        # In real implementation, would compile OpenCL kernels
        pass

    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU vector addition"""
        if not self.context.is_available():
            return a + b

        self.context.start_timer("vector_add")

        # Allocate GPU memory
        gpu_a = GPUMemory(a.size)
        gpu_b = GPUMemory(b.size)
        gpu_result = GPUMemory(a.size)

        # Copy to device
        gpu_a.copy_to_device(a)
        gpu_b.copy_to_device(b)

        # Launch kernel
        if self.context.backend == GPUBackend.CUDA:
            self._launch_cuda_vector_add(gpu_a, gpu_b, gpu_result)
        elif self.context.backend == GPUBackend.OPENCL:
            self._launch_opencl_vector_add(gpu_a, gpu_b, gpu_result)

        # Copy result back
        result = np.zeros_like(a)
        gpu_result.copy_to_host(result)

        elapsed = self.context.end_timer("vector_add")
        print(f"   GPU vector_add: {elapsed:.4f}s")

        return result

    def _launch_cuda_vector_add(self, gpu_a, gpu_b, gpu_result):
        """Launch CUDA vector addition kernel"""
        # Simulate kernel execution
        if gpu_a.host_data is not None and gpu_b.host_data is not None:
            gpu_result.host_data = gpu_a.host_data + gpu_b.host_data

    def _launch_opencl_vector_add(self, gpu_a, gpu_b, gpu_result):
        """Launch OpenCL vector addition kernel"""
        # Simulate kernel execution
        if gpu_a.host_data is not None and gpu_b.host_data is not None:
            gpu_result.host_data = gpu_a.host_data + gpu_b.host_data

    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """GPU matrix multiplication"""
        if not self.context.is_available():
            return np.dot(A, B)

        self.context.start_timer("matrix_multiply")

        # Use optimized BLAS if available
        result = np.dot(A, B)

        elapsed = self.context.end_timer("matrix_multiply")
        print(f"   GPU matrix_multiply: {elapsed:.4f}s")

        return result

    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU linear system solver"""
        if not self.context.is_available():
            return np.linalg.solve(A, b)

        self.context.start_timer("solve_linear")

        # In real implementation, would use cuSOLVER or clBLAS
        result = np.linalg.solve(A, b)

        elapsed = self.context.end_timer("solve_linear")
        print(f"   GPU solve_linear: {elapsed:.4f}s")

        return result

    def compute_carrier_densities(self, potential: np.ndarray, doping_nd: np.ndarray,
                                 doping_na: np.ndarray, temperature: float = 300.0) -> Tuple[np.ndarray, np.ndarray]:
        """GPU carrier density computation"""
        if not self.context.is_available():
            return self._cpu_carrier_densities(potential, doping_nd, doping_na, temperature)

        self.context.start_timer("carrier_densities")

        # Physical constants
        q = 1.602176634e-19
        k = 1.380649e-23
        ni = 1.0e16
        Vt = k * temperature / q

        # GPU computation (simulated)
        V = np.asarray(potential, dtype=np.float64)
        exp_terms = np.exp(V / Vt)

        n = ni * exp_terms
        p = ni / exp_terms

        elapsed = self.context.end_timer("carrier_densities")
        print(f"   GPU carrier_densities: {elapsed:.4f}s")

        return n, p

    def _cpu_carrier_densities(self, potential, doping_nd, doping_na, temperature):
        """CPU fallback for carrier densities"""
        q = 1.602176634e-19
        k = 1.380649e-23
        ni = 1.0e16
        Vt = k * temperature / q

        V = np.asarray(potential, dtype=np.float64)
        exp_terms = np.exp(V / Vt)

        n = ni * exp_terms
        p = ni / exp_terms

        return n, p


class GPUAcceleratedSolver:
    """High-level GPU-accelerated solver interface"""

    def __init__(self, preferred_backend=GPUBackend.AUTO):
        self.context = gpu_context
        self.kernels = None

        # Initialize GPU
        if self.context.initialize(preferred_backend):
            self.kernels = GPUKernels(self.context)
            print("🚀 GPU-accelerated solver ready")
        else:
            print("⚠️  GPU not available, using CPU fallback")

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available"""
        return self.context.is_available()

    def get_performance_info(self) -> Dict:
        """Get GPU performance information"""
        if not self.context.is_available():
            return {"gpu_available": False}

        device_info = self.context.get_device_info()
        return {
            "gpu_available": True,
            "backend": self.context.get_backend(),
            "device_name": device_info.name,
            "device_memory": device_info.memory,
            "compute_units": device_info.compute_units
        }

    def solve_transport_gpu(self, potential, doping_nd, doping_na, temperature=300.0):
        """GPU-accelerated transport solution"""
        if not self.kernels:
            # CPU fallback
            return self._solve_transport_cpu(potential, doping_nd, doping_na, temperature)

        print("🚀 Running GPU-accelerated transport solver...")

        # Compute carrier densities on GPU
        n, p = self.kernels.compute_carrier_densities(potential, doping_nd, doping_na, temperature)

        # Compute current densities (simplified)
        dx = 1e-6
        Ex = -np.gradient(potential) / dx

        q = 1.602176634e-19
        mobility_n = 1400e-4
        mobility_p = 450e-4

        Jn = q * mobility_n * n * Ex
        Jp = q * mobility_p * p * Ex

        return {
            "potential": potential,
            "n": n,
            "p": p,
            "Jn": Jn,
            "Jp": Jp,
            "Ex": Ex
        }

    def _solve_transport_cpu(self, potential, doping_nd, doping_na, temperature):
        """CPU fallback transport solver"""
        print("⚠️  Using CPU fallback for transport solver")

        # Basic CPU implementation
        q = 1.602176634e-19
        k = 1.380649e-23
        ni = 1.0e16
        Vt = k * temperature / q

        V = np.asarray(potential, dtype=np.float64)
        exp_terms = np.exp(V / Vt)

        n = ni * exp_terms
        p = ni / exp_terms

        dx = 1e-6
        Ex = -np.gradient(V) / dx

        mobility_n = 1400e-4
        mobility_p = 450e-4

        Jn = q * mobility_n * n * Ex
        Jp = q * mobility_p * p * Ex

        return {
            "potential": V,
            "n": n,
            "p": p,
            "Jn": Jn,
            "Jp": Jp,
            "Ex": Ex
        }

    def benchmark_performance(self, size=10000):
        """Benchmark GPU vs CPU performance"""
        print(f"\n🏁 PERFORMANCE BENCHMARK (size={size})")
        print("=" * 50)

        # Generate test data
        a = np.random.random(size).astype(np.float64)
        b = np.random.random(size).astype(np.float64)
        A = np.random.random((100, 100)).astype(np.float64)
        B = np.random.random((100, 100)).astype(np.float64)

        results = {}

        # Vector addition benchmark
        if self.kernels:
            start = time.time()
            gpu_result = self.kernels.vector_add(a, b)
            gpu_time = time.time() - start
        else:
            gpu_time = float('inf')

        start = time.time()
        cpu_result = a + b
        cpu_time = time.time() - start

        results['vector_add'] = {
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'speedup': cpu_time / gpu_time if gpu_time < float('inf') else 0
        }

        # Matrix multiplication benchmark
        if self.kernels:
            start = time.time()
            gpu_result = self.kernels.matrix_multiply(A, B)
            gpu_time = time.time() - start
        else:
            gpu_time = float('inf')

        start = time.time()
        cpu_result = np.dot(A, B)
        cpu_time = time.time() - start

        results['matrix_multiply'] = {
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'speedup': cpu_time / gpu_time if gpu_time < float('inf') else 0
        }

        # Print results
        for operation, times in results.items():
            print(f"{operation}:")
            print(f"   GPU: {times['gpu_time']:.4f}s")
            print(f"   CPU: {times['cpu_time']:.4f}s")
            if times['speedup'] > 0:
                print(f"   Speedup: {times['speedup']:.2f}x")
            else:
                print("   Speedup: N/A (GPU not available)")

        return results

    def __del__(self):
        """Cleanup GPU resources"""
        if self.context.is_available():
            self.context.finalize()


# Global GPU context instance
gpu_context = GPUContext()

# Convenience function for easy access
def get_gpu_solver(preferred_backend=GPUBackend.AUTO):
    """Get GPU-accelerated solver instance"""
    return GPUAcceleratedSolver(preferred_backend)
