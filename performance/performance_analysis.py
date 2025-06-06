#!/usr/bin/env python3
"""
Performance Analysis and Comparison Tool
Analyzes and compares performance across different computational backends

Author: Dr. Mazharuddin Mohammed
"""

import sys
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import statistics

# Add python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

class PerformanceAnalyzer:
    """Comprehensive performance analysis tool"""
    
    def __init__(self):
        self.backends = self._detect_available_backends()
        self.results = {}
        
    def _detect_available_backends(self):
        """Detect available computational backends"""
        backends = {
            "numpy": True,  # Always available
            "simd": False,
            "gpu": False,
            "complete_dg": False,
            "transport_models": False
        }
        
        # Check SIMD backend
        try:
            import performance_bindings
            backends["simd"] = True
        except ImportError:
            pass
        
        # Check GPU backend
        try:
            import performance_bindings
            if hasattr(performance_bindings, 'GPUAcceleration'):
                backends["gpu"] = performance_bindings.GPUAcceleration.is_available()
        except:
            pass
        
        # Check DG backend
        try:
            import complete_dg
            backends["complete_dg"] = True
        except ImportError:
            pass
        
        # Check transport models
        try:
            import unstructured_transport
            backends["transport_models"] = True
        except ImportError:
            pass
        
        return backends
    
    def benchmark_computational_kernels(self, sizes: List[int] = None):
        """Benchmark core computational kernels across all backends"""
        if sizes is None:
            sizes = [1000, 10000, 100000, 1000000]
        
        print("=== Computational Kernel Performance Analysis ===")
        print(f"Available backends: {[k for k, v in self.backends.items() if v]}")
        
        for size in sizes:
            print(f"\nBenchmarking size: {size:,} elements")
            
            # Generate test data
            a = np.random.random(size).astype(np.float64)
            b = np.random.random(size).astype(np.float64)
            
            size_results = {}
            
            # NumPy baseline
            size_results["numpy"] = self._benchmark_numpy_kernels(a, b)
            
            # SIMD kernels
            if self.backends["simd"]:
                size_results["simd"] = self._benchmark_simd_kernels(a, b)
            
            # GPU kernels
            if self.backends["gpu"]:
                size_results["gpu"] = self._benchmark_gpu_kernels(a, b)
            
            self.results[f"kernels_{size}"] = size_results
            
            # Display results
            self._display_kernel_results(size, size_results)
    
    def _benchmark_numpy_kernels(self, a: np.ndarray, b: np.ndarray):
        """Benchmark NumPy operations"""
        results = {}
        n_iterations = 10
        
        # Vector addition
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            result = a + b
            end = time.perf_counter()
            times.append(end - start)
        results["vector_add"] = {
            "mean_time": statistics.mean(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "throughput": len(a) / statistics.mean(times) / 1e6
        }
        
        # Vector multiplication
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            result = a * b
            end = time.perf_counter()
            times.append(end - start)
        results["vector_mul"] = {
            "mean_time": statistics.mean(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "throughput": len(a) / statistics.mean(times) / 1e6
        }
        
        # Dot product
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            result = np.dot(a, b)
            end = time.perf_counter()
            times.append(end - start)
        results["dot_product"] = {
            "mean_time": statistics.mean(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "throughput": len(a) / statistics.mean(times) / 1e6
        }
        
        return results
    
    def _benchmark_simd_kernels(self, a: np.ndarray, b: np.ndarray):
        """Benchmark SIMD operations"""
        try:
            import performance_bindings
            
            results = {}
            n_iterations = 10
            
            # Vector addition
            times = []
            for _ in range(n_iterations):
                start = time.perf_counter()
                result = performance_bindings.SIMDKernels.vector_add(a, b)
                end = time.perf_counter()
                times.append(end - start)
            results["vector_add"] = {
                "mean_time": statistics.mean(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "throughput": len(a) / statistics.mean(times) / 1e6
            }
            
            # Vector multiplication
            times = []
            for _ in range(n_iterations):
                start = time.perf_counter()
                result = performance_bindings.SIMDKernels.vector_multiply(a, b)
                end = time.perf_counter()
                times.append(end - start)
            results["vector_mul"] = {
                "mean_time": statistics.mean(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "throughput": len(a) / statistics.mean(times) / 1e6
            }
            
            # Dot product
            times = []
            for _ in range(n_iterations):
                start = time.perf_counter()
                result = performance_bindings.SIMDKernels.dot_product(a, b)
                end = time.perf_counter()
                times.append(end - start)
            results["dot_product"] = {
                "mean_time": statistics.mean(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "throughput": len(a) / statistics.mean(times) / 1e6
            }
            
            return results
            
        except Exception as e:
            print(f"  SIMD benchmark failed: {e}")
            return {}
    
    def _benchmark_gpu_kernels(self, a: np.ndarray, b: np.ndarray):
        """Benchmark GPU operations"""
        try:
            import performance_bindings
            gpu = performance_bindings.GPUAcceleration()
            
            results = {}
            n_iterations = 10
            
            # Warm up GPU
            for _ in range(3):
                _ = gpu.vector_add(a[:1000], b[:1000])
            
            # Vector addition
            times = []
            for _ in range(n_iterations):
                start = time.perf_counter()
                result = gpu.vector_add(a, b)
                end = time.perf_counter()
                times.append(end - start)
            results["vector_add"] = {
                "mean_time": statistics.mean(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "throughput": len(a) / statistics.mean(times) / 1e6
            }
            
            return results
            
        except Exception as e:
            print(f"  GPU benchmark failed: {e}")
            return {}
    
    def _display_kernel_results(self, size: int, results: Dict):
        """Display kernel benchmark results"""
        print(f"\n  Results for {size:,} elements:")
        
        operations = ["vector_add", "vector_mul", "dot_product"]
        
        for op in operations:
            if any(op in backend_results for backend_results in results.values()):
                print(f"\n    {op.replace('_', ' ').title()}:")
                
                # Collect results for this operation
                op_results = {}
                for backend, backend_results in results.items():
                    if op in backend_results:
                        op_results[backend] = backend_results[op]
                
                # Find baseline (NumPy) for speedup calculation
                baseline_time = op_results.get("numpy", {}).get("mean_time", 1.0)
                
                for backend, result in op_results.items():
                    mean_time = result["mean_time"]
                    throughput = result["throughput"]
                    speedup = baseline_time / mean_time if mean_time > 0 else 0
                    
                    print(f"      {backend:>12}: {mean_time*1000:6.3f} ms ({throughput:6.1f} Melem/s, {speedup:4.2f}x)")
    
    def analyze_scaling_behavior(self, sizes: List[int] = None):
        """Analyze how performance scales with problem size"""
        if not self.results:
            print("No benchmark results available. Run benchmark_computational_kernels first.")
            return
        
        print("\n=== Scaling Behavior Analysis ===")
        
        # Extract scaling data
        scaling_data = {}
        
        for result_key, result_data in self.results.items():
            if result_key.startswith("kernels_"):
                size = int(result_key.split("_")[1])
                
                for backend, backend_results in result_data.items():
                    if backend not in scaling_data:
                        scaling_data[backend] = {}
                    
                    for operation, op_result in backend_results.items():
                        if operation not in scaling_data[backend]:
                            scaling_data[backend][operation] = {"sizes": [], "times": [], "throughputs": []}
                        
                        scaling_data[backend][operation]["sizes"].append(size)
                        scaling_data[backend][operation]["times"].append(op_result["mean_time"])
                        scaling_data[backend][operation]["throughputs"].append(op_result["throughput"])
        
        # Analyze scaling for each backend and operation
        for backend, backend_data in scaling_data.items():
            print(f"\n{backend.upper()} Scaling Analysis:")
            
            for operation, op_data in backend_data.items():
                sizes = np.array(op_data["sizes"])
                times = np.array(op_data["times"])
                throughputs = np.array(op_data["throughputs"])
                
                # Sort by size
                sort_idx = np.argsort(sizes)
                sizes = sizes[sort_idx]
                times = times[sort_idx]
                throughputs = throughputs[sort_idx]
                
                # Calculate scaling efficiency
                if len(sizes) > 1:
                    # Ideal linear scaling would have constant throughput
                    throughput_variation = np.std(throughputs) / np.mean(throughputs)
                    
                    # Time scaling factor
                    time_scaling = times[-1] / times[0] / (sizes[-1] / sizes[0])
                    
                    print(f"  {operation}:")
                    print(f"    Throughput variation: {throughput_variation:.3f} (lower is better)")
                    print(f"    Time scaling factor: {time_scaling:.3f} (1.0 is ideal linear)")
                    
                    if throughput_variation < 0.2:
                        print(f"    ✓ Good scaling behavior")
                    else:
                        print(f"    ⚠ Poor scaling behavior")
    
    def generate_performance_report(self, filename: str = None):
        """Generate comprehensive performance report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_analysis_{timestamp}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "available_backends": self.backends,
            "benchmark_results": self.results,
            "analysis": self._generate_analysis_summary()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n✓ Performance report saved to: {filename}")
        except Exception as e:
            print(f"\n✗ Failed to save report: {e}")
    
    def _generate_analysis_summary(self):
        """Generate analysis summary"""
        summary = {
            "best_performers": {},
            "speedup_analysis": {},
            "recommendations": []
        }
        
        # Find best performers for each operation
        for result_key, result_data in self.results.items():
            if result_key.startswith("kernels_"):
                size = int(result_key.split("_")[1])
                
                for operation in ["vector_add", "vector_mul", "dot_product"]:
                    best_backend = None
                    best_throughput = 0
                    
                    for backend, backend_results in result_data.items():
                        if operation in backend_results:
                            throughput = backend_results[operation]["throughput"]
                            if throughput > best_throughput:
                                best_throughput = throughput
                                best_backend = backend
                    
                    if best_backend:
                        key = f"{operation}_{size}"
                        summary["best_performers"][key] = {
                            "backend": best_backend,
                            "throughput": best_throughput
                        }
        
        # Generate recommendations
        if self.backends["gpu"]:
            summary["recommendations"].append("GPU acceleration is available and should be used for large problems")
        
        if self.backends["simd"]:
            summary["recommendations"].append("SIMD optimization is available for CPU computations")
        
        if not self.backends["gpu"] and not self.backends["simd"]:
            summary["recommendations"].append("Consider installing GPU/SIMD backends for better performance")
        
        return summary
    
    def run_complete_analysis(self):
        """Run complete performance analysis"""
        print("🔬 SemiDGFEM Performance Analysis and Comparison")
        print("=" * 60)
        
        # Benchmark computational kernels
        self.benchmark_computational_kernels([1000, 10000, 100000, 1000000])
        
        # Analyze scaling behavior
        self.analyze_scaling_behavior()
        
        # Generate report
        self.generate_performance_report()
        
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS COMPLETE")
        print("="*60)
        
        # Summary
        available_backends = [k for k, v in self.backends.items() if v]
        print(f"Tested backends: {', '.join(available_backends)}")
        
        if self.backends["gpu"]:
            print("✅ GPU acceleration available and tested")
        else:
            print("⚠ GPU acceleration not available")
        
        if self.backends["simd"]:
            print("✅ SIMD optimization available and tested")
        else:
            print("⚠ SIMD optimization not available")
        
        print("\n🎯 Check the generated JSON report for detailed analysis")

def main():
    """Main analysis entry point"""
    try:
        analyzer = PerformanceAnalyzer()
        analyzer.run_complete_analysis()
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹ Analysis cancelled by user")
        return 0
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
