#!/usr/bin/env python3
"""
Complete GPU Validation and Performance Benchmarking Suite
Comprehensive testing of SIMD/GPU performance optimization capabilities

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import platform
import psutil
import threading
from typing import Dict, List, Tuple, Optional, Any

# Add python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

class SystemInfo:
    """System information collector"""
    
    @staticmethod
    def get_cpu_info():
        """Get CPU information"""
        return {
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "frequency_max": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown",
            "cache_info": SystemInfo._get_cpu_cache_info()
        }
    
    @staticmethod
    def _get_cpu_cache_info():
        """Get CPU cache information"""
        try:
            if platform.system() == "Linux":
                cache_info = {}
                for level in [1, 2, 3]:
                    try:
                        with open(f"/sys/devices/system/cpu/cpu0/cache/index{level}/size", 'r') as f:
                            cache_info[f"L{level}"] = f.read().strip()
                    except:
                        pass
                return cache_info
            else:
                return {"info": "Cache info not available on this platform"}
        except:
            return {"info": "Cache info unavailable"}
    
    @staticmethod
    def get_memory_info():
        """Get memory information"""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_percent": mem.percent,
            "swap_total_gb": psutil.swap_memory().total / (1024**3)
        }
    
    @staticmethod
    def get_gpu_info():
        """Get GPU information"""
        gpu_info = {"available": False, "devices": []}
        
        # Try to get NVIDIA GPU info
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_info["devices"].append({
                    "index": i,
                    "name": name,
                    "memory_total_gb": memory_info.total / (1024**3),
                    "memory_free_gb": memory_info.free / (1024**3),
                    "type": "NVIDIA"
                })
            
            gpu_info["available"] = device_count > 0
            
        except ImportError:
            pass
        except Exception as e:
            gpu_info["error"] = str(e)
        
        # Try to detect other GPU types
        if not gpu_info["available"]:
            # Check for integrated graphics or other GPUs
            try:
                # This is a simplified check - in practice you'd use OpenCL or other APIs
                gpu_info["note"] = "GPU detection requires pynvml for NVIDIA GPUs"
            except:
                pass
        
        return gpu_info

class PerformanceBenchmark:
    """Base class for performance benchmarks"""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def start_timer(self):
        """Start timing"""
        self.start_time = time.perf_counter()
    
    def end_timer(self):
        """End timing and return elapsed time"""
        self.end_time = time.perf_counter()
        return self.end_time - self.start_time
    
    def add_result(self, test_name: str, elapsed_time: float, 
                   throughput: float = None, additional_info: Dict = None):
        """Add benchmark result"""
        result = {
            "test_name": test_name,
            "elapsed_time_ms": elapsed_time * 1000,
            "throughput": throughput,
            "timestamp": datetime.now().isoformat(),
            "additional_info": additional_info or {}
        }
        self.results.append(result)
    
    def get_summary(self):
        """Get benchmark summary"""
        if not self.results:
            return {"name": self.name, "status": "No results"}
        
        times = [r["elapsed_time_ms"] for r in self.results]
        return {
            "name": self.name,
            "total_tests": len(self.results),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "avg_time_ms": sum(times) / len(times),
            "total_time_ms": sum(times)
        }

class SIMDBenchmark(PerformanceBenchmark):
    """SIMD performance benchmarking"""
    
    def __init__(self):
        super().__init__("SIMD Performance")
        self.backend_available = self._check_backend()
    
    def _check_backend(self):
        """Check if performance bindings are available"""
        try:
            import performance_bindings
            return True
        except ImportError:
            return False
    
    def benchmark_vector_operations(self, sizes: List[int] = None):
        """Benchmark SIMD vector operations"""
        if sizes is None:
            sizes = [1000, 10000, 100000, 1000000]
        
        print(f"\n=== {self.name} - Vector Operations ===")
        
        for size in sizes:
            print(f"\nTesting size: {size:,} elements")
            
            # Generate test data
            a = np.random.random(size).astype(np.float64)
            b = np.random.random(size).astype(np.float64)
            
            # Test NumPy baseline
            self._benchmark_numpy_operations(a, b, size)
            
            # Test SIMD operations if available
            if self.backend_available:
                self._benchmark_simd_operations(a, b, size)
            else:
                print("  ⚠ SIMD backend not available - using NumPy baseline only")
    
    def _benchmark_numpy_operations(self, a: np.ndarray, b: np.ndarray, size: int):
        """Benchmark NumPy operations as baseline"""
        
        # Vector addition
        self.start_timer()
        for _ in range(10):  # Multiple iterations for accuracy
            result = a + b
        elapsed = self.end_timer() / 10
        throughput = size / elapsed / 1e6  # Million elements per second
        
        print(f"  NumPy Vector Add: {elapsed*1000:.3f} ms ({throughput:.1f} Melem/s)")
        self.add_result(f"NumPy_VectorAdd_{size}", elapsed, throughput)
        
        # Vector multiplication
        self.start_timer()
        for _ in range(10):
            result = a * b
        elapsed = self.end_timer() / 10
        throughput = size / elapsed / 1e6
        
        print(f"  NumPy Vector Mul: {elapsed*1000:.3f} ms ({throughput:.1f} Melem/s)")
        self.add_result(f"NumPy_VectorMul_{size}", elapsed, throughput)
        
        # Dot product
        self.start_timer()
        for _ in range(10):
            result = np.dot(a, b)
        elapsed = self.end_timer() / 10
        throughput = size / elapsed / 1e6
        
        print(f"  NumPy Dot Product: {elapsed*1000:.3f} ms ({throughput:.1f} Melem/s)")
        self.add_result(f"NumPy_DotProduct_{size}", elapsed, throughput)
    
    def _benchmark_simd_operations(self, a: np.ndarray, b: np.ndarray, size: int):
        """Benchmark SIMD-optimized operations"""
        try:
            import performance_bindings
            
            # Vector addition
            self.start_timer()
            for _ in range(10):
                result = performance_bindings.SIMDKernels.vector_add(a, b)
            elapsed = self.end_timer() / 10
            throughput = size / elapsed / 1e6
            
            print(f"  SIMD Vector Add: {elapsed*1000:.3f} ms ({throughput:.1f} Melem/s)")
            self.add_result(f"SIMD_VectorAdd_{size}", elapsed, throughput)
            
            # Vector multiplication
            self.start_timer()
            for _ in range(10):
                result = performance_bindings.SIMDKernels.vector_multiply(a, b)
            elapsed = self.end_timer() / 10
            throughput = size / elapsed / 1e6
            
            print(f"  SIMD Vector Mul: {elapsed*1000:.3f} ms ({throughput:.1f} Melem/s)")
            self.add_result(f"SIMD_VectorMul_{size}", elapsed, throughput)
            
            # Dot product
            self.start_timer()
            for _ in range(10):
                result = performance_bindings.SIMDKernels.dot_product(a, b)
            elapsed = self.end_timer() / 10
            throughput = size / elapsed / 1e6
            
            print(f"  SIMD Dot Product: {elapsed*1000:.3f} ms ({throughput:.1f} Melem/s)")
            self.add_result(f"SIMD_DotProduct_{size}", elapsed, throughput)
            
        except Exception as e:
            print(f"  ✗ SIMD benchmark failed: {e}")
    
    def benchmark_matrix_operations(self, sizes: List[Tuple[int, int]] = None):
        """Benchmark SIMD matrix operations"""
        if sizes is None:
            sizes = [(100, 100), (500, 500), (1000, 1000)]
        
        print(f"\n=== {self.name} - Matrix Operations ===")
        
        for rows, cols in sizes:
            print(f"\nTesting matrix size: {rows}×{cols}")
            
            # Generate test data
            matrix = np.random.random((rows, cols)).astype(np.float64)
            vector = np.random.random(cols).astype(np.float64)
            
            # NumPy baseline
            self.start_timer()
            for _ in range(5):
                result = matrix @ vector
            elapsed = self.end_timer() / 5
            throughput = (rows * cols) / elapsed / 1e6
            
            print(f"  NumPy MatVec: {elapsed*1000:.3f} ms ({throughput:.1f} Melem/s)")
            self.add_result(f"NumPy_MatVec_{rows}x{cols}", elapsed, throughput)
            
            # SIMD matrix-vector multiplication
            if self.backend_available:
                try:
                    import performance_bindings
                    
                    self.start_timer()
                    for _ in range(5):
                        result = performance_bindings.SIMDKernels.matrix_vector_multiply(matrix, vector)
                    elapsed = self.end_timer() / 5
                    throughput = (rows * cols) / elapsed / 1e6
                    
                    print(f"  SIMD MatVec: {elapsed*1000:.3f} ms ({throughput:.1f} Melem/s)")
                    self.add_result(f"SIMD_MatVec_{rows}x{cols}", elapsed, throughput)
                    
                except Exception as e:
                    print(f"  ✗ SIMD matrix benchmark failed: {e}")

class GPUBenchmark(PerformanceBenchmark):
    """GPU performance benchmarking"""
    
    def __init__(self):
        super().__init__("GPU Performance")
        self.backend_available = self._check_backend()
        self.gpu_available = self._check_gpu()
    
    def _check_backend(self):
        """Check if GPU bindings are available"""
        try:
            import performance_bindings
            return hasattr(performance_bindings, 'GPUAcceleration')
        except ImportError:
            return False
    
    def _check_gpu(self):
        """Check if GPU is available"""
        if not self.backend_available:
            return False
        
        try:
            import performance_bindings
            return performance_bindings.GPUAcceleration.is_available()
        except:
            return False
    
    def benchmark_gpu_operations(self, sizes: List[int] = None):
        """Benchmark GPU operations"""
        if sizes is None:
            sizes = [10000, 100000, 1000000, 10000000]
        
        print(f"\n=== {self.name} - Vector Operations ===")
        print(f"GPU Available: {self.gpu_available}")
        
        if not self.gpu_available:
            print("⚠ GPU not available - skipping GPU benchmarks")
            return
        
        try:
            import performance_bindings
            gpu = performance_bindings.GPUAcceleration()
            
            for size in sizes:
                print(f"\nTesting size: {size:,} elements")
                
                # Generate test data
                a = np.random.random(size).astype(np.float64)
                b = np.random.random(size).astype(np.float64)
                
                # CPU baseline
                self.start_timer()
                cpu_result = a + b
                cpu_time = self.end_timer()
                cpu_throughput = size / cpu_time / 1e6
                
                print(f"  CPU Vector Add: {cpu_time*1000:.3f} ms ({cpu_throughput:.1f} Melem/s)")
                self.add_result(f"CPU_VectorAdd_{size}", cpu_time, cpu_throughput)
                
                # GPU computation
                self.start_timer()
                gpu_result = gpu.vector_add(a, b)
                gpu_time = self.end_timer()
                gpu_throughput = size / gpu_time / 1e6
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                
                print(f"  GPU Vector Add: {gpu_time*1000:.3f} ms ({gpu_throughput:.1f} Melem/s, {speedup:.2f}x)")
                self.add_result(f"GPU_VectorAdd_{size}", gpu_time, gpu_throughput, 
                              {"speedup": speedup, "cpu_time": cpu_time})
                
                # Verify correctness
                if np.allclose(cpu_result, gpu_result, rtol=1e-10):
                    print(f"  ✓ Results match (max diff: {np.max(np.abs(cpu_result - gpu_result)):.2e})")
                else:
                    print(f"  ✗ Results don't match (max diff: {np.max(np.abs(cpu_result - gpu_result)):.2e})")
                
        except Exception as e:
            print(f"✗ GPU benchmark failed: {e}")
    
    def benchmark_gpu_matrix_operations(self, sizes: List[Tuple[int, int]] = None):
        """Benchmark GPU matrix operations"""
        if not self.gpu_available:
            print("⚠ GPU not available - skipping GPU matrix benchmarks")
            return
        
        if sizes is None:
            sizes = [(1000, 1000), (2000, 2000), (5000, 5000)]
        
        print(f"\n=== {self.name} - Matrix Operations ===")
        
        try:
            import performance_bindings
            gpu = performance_bindings.GPUAcceleration()
            
            for rows, cols in sizes:
                print(f"\nTesting matrix size: {rows}×{cols}")
                
                # Generate test data
                matrix = np.random.random((rows, cols)).astype(np.float64)
                vector = np.random.random(cols).astype(np.float64)
                
                # CPU baseline
                self.start_timer()
                cpu_result = matrix @ vector
                cpu_time = self.end_timer()
                cpu_throughput = (rows * cols) / cpu_time / 1e6
                
                print(f"  CPU MatVec: {cpu_time*1000:.3f} ms ({cpu_throughput:.1f} Melem/s)")
                self.add_result(f"CPU_MatVec_{rows}x{cols}", cpu_time, cpu_throughput)
                
                # GPU computation
                self.start_timer()
                gpu_result = gpu.matrix_vector_multiply(matrix, vector)
                gpu_time = self.end_timer()
                gpu_throughput = (rows * cols) / gpu_time / 1e6
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                
                print(f"  GPU MatVec: {gpu_time*1000:.3f} ms ({gpu_throughput:.1f} Melem/s, {speedup:.2f}x)")
                self.add_result(f"GPU_MatVec_{rows}x{cols}", gpu_time, gpu_throughput,
                              {"speedup": speedup, "cpu_time": cpu_time})
                
                # Verify correctness
                if np.allclose(cpu_result, gpu_result, rtol=1e-10):
                    print(f"  ✓ Results match (max diff: {np.max(np.abs(cpu_result - gpu_result)):.2e})")
                else:
                    print(f"  ✗ Results don't match (max diff: {np.max(np.abs(cpu_result - gpu_result)):.2e})")
                
        except Exception as e:
            print(f"✗ GPU matrix benchmark failed: {e}")

class DGPerformanceBenchmark(PerformanceBenchmark):
    """DG-specific performance benchmarking"""
    
    def __init__(self):
        super().__init__("DG Performance")
        self.backend_available = self._check_backend()
    
    def _check_backend(self):
        """Check if DG bindings are available"""
        try:
            import complete_dg
            return True
        except ImportError:
            return False
    
    def benchmark_basis_functions(self, orders: List[int] = None, n_evaluations: int = 100000):
        """Benchmark basis function evaluations"""
        if not self.backend_available:
            print("⚠ DG backend not available - skipping DG benchmarks")
            return
        
        if orders is None:
            orders = [1, 2, 3]
        
        print(f"\n=== {self.name} - Basis Function Evaluation ===")
        
        try:
            import complete_dg
            
            for order in orders:
                print(f"\nTesting P{order} basis functions")
                
                # Generate test points
                xi = np.random.random(n_evaluations)
                eta = np.random.random(n_evaluations)
                
                # Make sure points are in reference triangle
                mask = xi + eta <= 1.0
                xi = xi[mask][:n_evaluations//2]
                eta = eta[mask][:n_evaluations//2]
                n_valid = len(xi)
                
                dofs_per_element = complete_dg.DGBasisFunctions.get_dofs_per_element(order)
                
                # Benchmark basis function evaluation
                self.start_timer()
                for j in range(dofs_per_element):
                    for i in range(n_valid):
                        value = complete_dg.DGBasisFunctions.evaluate_basis_function(
                            xi[i], eta[i], j, order)
                elapsed = self.end_timer()
                
                total_evaluations = dofs_per_element * n_valid
                throughput = total_evaluations / elapsed / 1e6
                
                print(f"  P{order} Basis Eval: {elapsed*1000:.3f} ms ({throughput:.1f} Meval/s)")
                print(f"    DOFs per element: {dofs_per_element}")
                print(f"    Total evaluations: {total_evaluations:,}")
                
                self.add_result(f"BasisEval_P{order}", elapsed, throughput,
                              {"dofs_per_element": dofs_per_element, "total_evaluations": total_evaluations})
                
        except Exception as e:
            print(f"✗ DG basis function benchmark failed: {e}")
    
    def benchmark_element_assembly(self, orders: List[int] = None, n_elements: int = 1000):
        """Benchmark element assembly operations"""
        if not self.backend_available:
            return
        
        if orders is None:
            orders = [1, 2, 3]
        
        print(f"\n=== {self.name} - Element Assembly ===")
        
        try:
            import complete_dg
            
            for order in orders:
                print(f"\nTesting P{order} element assembly")
                
                # Create DG assembly object
                dg_assembly = complete_dg.DGAssembly(order)
                
                # Generate test elements
                elements = []
                for _ in range(n_elements):
                    # Random triangle vertices
                    vertices = np.random.random((3, 2)) * 1e-6
                    elements.append(vertices)
                
                # Benchmark mass matrix assembly
                self.start_timer()
                for vertices in elements:
                    mass_matrix = dg_assembly.assemble_element_matrix(vertices, "mass")
                mass_time = self.end_timer()
                
                # Benchmark stiffness matrix assembly
                self.start_timer()
                for vertices in elements:
                    stiffness_matrix = dg_assembly.assemble_element_matrix(vertices, "stiffness")
                stiffness_time = self.end_timer()
                
                dofs_per_element = complete_dg.DGBasisFunctions.get_dofs_per_element(order)
                
                print(f"  P{order} Mass Assembly: {mass_time*1000:.3f} ms ({n_elements/mass_time:.1f} elem/s)")
                print(f"  P{order} Stiffness Assembly: {stiffness_time*1000:.3f} ms ({n_elements/stiffness_time:.1f} elem/s)")
                
                self.add_result(f"MassAssembly_P{order}", mass_time, n_elements/mass_time,
                              {"n_elements": n_elements, "dofs_per_element": dofs_per_element})
                self.add_result(f"StiffnessAssembly_P{order}", stiffness_time, n_elements/stiffness_time,
                              {"n_elements": n_elements, "dofs_per_element": dofs_per_element})
                
        except Exception as e:
            print(f"✗ DG assembly benchmark failed: {e}")

class TransportModelBenchmark(PerformanceBenchmark):
    """Transport model performance benchmarking"""

    def __init__(self):
        super().__init__("Transport Model Performance")
        self.backend_available = self._check_backend()

    def _check_backend(self):
        """Check if transport model bindings are available"""
        try:
            import unstructured_transport
            return True
        except ImportError:
            return False

    def benchmark_transport_models(self, problem_sizes: List[int] = None):
        """Benchmark transport model execution"""
        if not self.backend_available:
            print("⚠ Transport model backend not available - skipping transport benchmarks")
            return

        if problem_sizes is None:
            problem_sizes = [1000, 5000, 10000]

        print(f"\n=== {self.name} - Transport Model Execution ===")

        try:
            import unstructured_transport
            import simulator

            # Create device
            device = simulator.Device(2e-6, 1e-6)

            for size in problem_sizes:
                print(f"\nTesting problem size: {size:,} DOFs")

                # Generate test data
                potential = np.linspace(0, 1.0, size)
                n = np.full(size, 1e22)
                p = np.full(size, 1e21)
                T_n = np.full(size, 300.0)
                T_p = np.full(size, 300.0)
                Jn = np.full(size, 1e6)
                Jp = np.full(size, -8e5)
                Nd = np.full(size, 1e23)
                Na = np.full(size, 1e22)
                dt = 1e-12

                # Test energy transport
                try:
                    energy_solver = unstructured_transport.create_unstructured_energy_transport(device)

                    self.start_timer()
                    energy_results = energy_solver.solve(potential, n, p, Jn, Jp, dt)
                    energy_time = self.end_timer()

                    print(f"  Energy Transport: {energy_time*1000:.3f} ms ({size/energy_time:.1f} DOF/s)")
                    self.add_result(f"EnergyTransport_{size}", energy_time, size/energy_time)

                except Exception as e:
                    print(f"  ✗ Energy transport failed: {e}")

                # Test hydrodynamic transport
                try:
                    hydro_solver = unstructured_transport.create_unstructured_hydrodynamic(device)

                    self.start_timer()
                    hydro_results = hydro_solver.solve(potential, n, p, T_n, T_p, dt)
                    hydro_time = self.end_timer()

                    print(f"  Hydrodynamic: {hydro_time*1000:.3f} ms ({size/hydro_time:.1f} DOF/s)")
                    self.add_result(f"Hydrodynamic_{size}", hydro_time, size/hydro_time)

                except Exception as e:
                    print(f"  ✗ Hydrodynamic transport failed: {e}")

                # Test non-equilibrium DD
                try:
                    non_eq_solver = unstructured_transport.create_unstructured_non_equilibrium_dd(device)

                    self.start_timer()
                    non_eq_results = non_eq_solver.solve(potential, Nd, Na, dt, 300.0)
                    non_eq_time = self.end_timer()

                    print(f"  Non-Equilibrium DD: {non_eq_time*1000:.3f} ms ({size/non_eq_time:.1f} DOF/s)")
                    self.add_result(f"NonEquilibriumDD_{size}", non_eq_time, size/non_eq_time)

                except Exception as e:
                    print(f"  ✗ Non-equilibrium DD failed: {e}")

        except Exception as e:
            print(f"✗ Transport model benchmark failed: {e}")

class ComprehensiveGPUValidation:
    """Comprehensive GPU validation and performance benchmarking suite"""

    def __init__(self):
        self.system_info = None
        self.benchmarks = []
        self.start_time = None
        self.end_time = None

    def run_complete_validation(self):
        """Run complete GPU validation and benchmarking"""
        print("🚀 SemiDGFEM Complete GPU Validation and Performance Benchmarking")
        print("=" * 80)

        self.start_time = datetime.now()

        # Collect system information
        self._collect_system_info()

        # Run all benchmarks
        self._run_simd_benchmarks()
        self._run_gpu_benchmarks()
        self._run_dg_benchmarks()
        self._run_transport_benchmarks()

        self.end_time = datetime.now()

        # Generate comprehensive report
        self._generate_report()

        return self._get_validation_summary()

    def _collect_system_info(self):
        """Collect comprehensive system information"""
        print("\n=== System Information ===")

        self.system_info = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "python_version": platform.python_version()
            },
            "cpu": SystemInfo.get_cpu_info(),
            "memory": SystemInfo.get_memory_info(),
            "gpu": SystemInfo.get_gpu_info()
        }

        # Display system info
        cpu_info = self.system_info["cpu"]
        mem_info = self.system_info["memory"]
        gpu_info = self.system_info["gpu"]

        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"CPU: {cpu_info['processor']}")
        print(f"Cores: {cpu_info['cores_physical']} physical, {cpu_info['cores_logical']} logical")
        print(f"Memory: {mem_info['total_gb']:.1f} GB total, {mem_info['available_gb']:.1f} GB available")

        if gpu_info["available"]:
            for gpu in gpu_info["devices"]:
                print(f"GPU: {gpu['name']} ({gpu['memory_total_gb']:.1f} GB)")
        else:
            print("GPU: Not available or not detected")

    def _run_simd_benchmarks(self):
        """Run SIMD performance benchmarks"""
        print("\n" + "="*50)
        print("SIMD PERFORMANCE BENCHMARKS")
        print("="*50)

        simd_benchmark = SIMDBenchmark()

        # Vector operations
        simd_benchmark.benchmark_vector_operations([1000, 10000, 100000, 1000000])

        # Matrix operations
        simd_benchmark.benchmark_matrix_operations([(100, 100), (500, 500), (1000, 1000)])

        self.benchmarks.append(simd_benchmark)

    def _run_gpu_benchmarks(self):
        """Run GPU performance benchmarks"""
        print("\n" + "="*50)
        print("GPU PERFORMANCE BENCHMARKS")
        print("="*50)

        gpu_benchmark = GPUBenchmark()

        # Vector operations
        gpu_benchmark.benchmark_gpu_operations([10000, 100000, 1000000, 10000000])

        # Matrix operations
        gpu_benchmark.benchmark_gpu_matrix_operations([(1000, 1000), (2000, 2000), (5000, 5000)])

        self.benchmarks.append(gpu_benchmark)

    def _run_dg_benchmarks(self):
        """Run DG-specific benchmarks"""
        print("\n" + "="*50)
        print("DG DISCRETIZATION BENCHMARKS")
        print("="*50)

        dg_benchmark = DGPerformanceBenchmark()

        # Basis function evaluation
        dg_benchmark.benchmark_basis_functions([1, 2, 3], 50000)

        # Element assembly
        dg_benchmark.benchmark_element_assembly([1, 2, 3], 1000)

        self.benchmarks.append(dg_benchmark)

    def _run_transport_benchmarks(self):
        """Run transport model benchmarks"""
        print("\n" + "="*50)
        print("TRANSPORT MODEL BENCHMARKS")
        print("="*50)

        transport_benchmark = TransportModelBenchmark()

        # Transport model execution
        transport_benchmark.benchmark_transport_models([1000, 5000, 10000])

        self.benchmarks.append(transport_benchmark)

    def _generate_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("="*80)

        total_duration = (self.end_time - self.start_time).total_seconds()
        print(f"Total validation time: {total_duration:.2f} seconds")

        # Summary for each benchmark category
        for benchmark in self.benchmarks:
            summary = benchmark.get_summary()
            print(f"\n--- {summary['name']} ---")

            if summary.get('total_tests', 0) > 0:
                print(f"Tests completed: {summary['total_tests']}")
                print(f"Total time: {summary['total_time_ms']:.1f} ms")
                print(f"Average time: {summary['avg_time_ms']:.3f} ms")
                print(f"Min time: {summary['min_time_ms']:.3f} ms")
                print(f"Max time: {summary['max_time_ms']:.3f} ms")
            else:
                print("No tests completed")

        # Performance analysis
        self._analyze_performance()

        # Save detailed report
        self._save_detailed_report()

    def _analyze_performance(self):
        """Analyze performance results and provide insights"""
        print(f"\n--- Performance Analysis ---")

        # Analyze SIMD performance
        simd_results = next((b for b in self.benchmarks if b.name == "SIMD Performance"), None)
        if simd_results and simd_results.results:
            simd_speedups = []
            for result in simd_results.results:
                if "SIMD_" in result["test_name"]:
                    # Find corresponding NumPy result
                    numpy_test = result["test_name"].replace("SIMD_", "NumPy_")
                    numpy_result = next((r for r in simd_results.results if r["test_name"] == numpy_test), None)
                    if numpy_result:
                        speedup = numpy_result["elapsed_time_ms"] / result["elapsed_time_ms"]
                        simd_speedups.append(speedup)

            if simd_speedups:
                avg_simd_speedup = sum(simd_speedups) / len(simd_speedups)
                print(f"Average SIMD speedup: {avg_simd_speedup:.2f}x")
            else:
                print("SIMD speedup analysis: No comparable results")

        # Analyze GPU performance
        gpu_results = next((b for b in self.benchmarks if b.name == "GPU Performance"), None)
        if gpu_results and gpu_results.results:
            gpu_speedups = []
            for result in gpu_results.results:
                if result["additional_info"] and "speedup" in result["additional_info"]:
                    gpu_speedups.append(result["additional_info"]["speedup"])

            if gpu_speedups:
                avg_gpu_speedup = sum(gpu_speedups) / len(gpu_speedups)
                max_gpu_speedup = max(gpu_speedups)
                print(f"Average GPU speedup: {avg_gpu_speedup:.2f}x")
                print(f"Maximum GPU speedup: {max_gpu_speedup:.2f}x")
            else:
                print("GPU speedup analysis: No GPU results available")

        # Analyze DG performance scaling
        dg_results = next((b for b in self.benchmarks if b.name == "DG Performance"), None)
        if dg_results and dg_results.results:
            p1_results = [r for r in dg_results.results if "P1" in r["test_name"]]
            p3_results = [r for r in dg_results.results if "P3" in r["test_name"]]

            if p1_results and p3_results:
                p1_avg = sum(r["elapsed_time_ms"] for r in p1_results) / len(p1_results)
                p3_avg = sum(r["elapsed_time_ms"] for r in p3_results) / len(p3_results)
                scaling_factor = p3_avg / p1_avg
                print(f"DG order scaling (P3/P1): {scaling_factor:.2f}x")

    def _save_detailed_report(self):
        """Save detailed report to JSON file"""
        report_data = {
            "validation_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds()
            },
            "system_info": self.system_info,
            "benchmark_results": {}
        }

        # Add all benchmark results
        for benchmark in self.benchmarks:
            report_data["benchmark_results"][benchmark.name] = {
                "summary": benchmark.get_summary(),
                "detailed_results": benchmark.results
            }

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpu_validation_report_{timestamp}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n✓ Detailed report saved to: {filename}")
        except Exception as e:
            print(f"\n✗ Failed to save report: {e}")

    def _get_validation_summary(self):
        """Get validation summary"""
        summary = {
            "validation_completed": True,
            "total_benchmarks": len(self.benchmarks),
            "system_info": self.system_info,
            "performance_summary": {}
        }

        for benchmark in self.benchmarks:
            summary["performance_summary"][benchmark.name] = benchmark.get_summary()

        return summary

def main():
    """Main validation entry point"""
    try:
        validator = ComprehensiveGPUValidation()
        summary = validator.run_complete_validation()

        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        if summary["validation_completed"]:
            print("✅ GPU validation and performance benchmarking completed successfully!")
            print(f"Total benchmark categories: {summary['total_benchmarks']}")

            # Check for key capabilities
            gpu_info = summary["system_info"]["gpu"]
            if gpu_info["available"]:
                print("✅ GPU acceleration available and tested")
            else:
                print("⚠ GPU acceleration not available")

            print("\nKey findings:")
            for category, results in summary["performance_summary"].items():
                if results.get("total_tests", 0) > 0:
                    print(f"  • {category}: {results['total_tests']} tests completed")
                else:
                    print(f"  • {category}: No tests completed (backend not available)")

            print("\n🎯 SemiDGFEM performance validation complete!")
            print("   Check the detailed JSON report for comprehensive results.")

            return 0
        else:
            print("❌ Validation failed")
            return 1

    except KeyboardInterrupt:
        print("\n\n⏹ Validation cancelled by user")
        return 0
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
