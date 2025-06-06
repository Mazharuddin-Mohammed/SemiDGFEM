#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite
Performance analysis and optimization for all critical paths

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from performance_profiler import PerformanceProfiler, profiler

class CriticalPathBenchmarks:
    """Benchmarks for critical computational paths"""
    
    def __init__(self):
        self.profiler = profiler
        self.results = {}
        
    def benchmark_linear_algebra(self, sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark linear algebra operations"""
        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000]
        
        print("🔢 BENCHMARKING LINEAR ALGEBRA OPERATIONS")
        print("=" * 50)
        
        results = {'sizes': sizes, 'operations': {}}
        
        for size in sizes:
            print(f"   Testing size: {size}×{size}")
            
            # Generate test matrices
            A = np.random.random((size, size)).astype(np.float64)
            B = np.random.random((size, size)).astype(np.float64)
            x = np.random.random(size).astype(np.float64)
            
            # Matrix multiplication
            with self.profiler.profile_operation("matrix_multiply", size*size, "CPU"):
                C = np.dot(A, B)
            
            # Matrix-vector multiplication
            with self.profiler.profile_operation("matvec_multiply", size*size, "CPU"):
                y = np.dot(A, x)
            
            # Linear system solving
            with self.profiler.profile_operation("linear_solve", size*size, "CPU"):
                x_solve = np.linalg.solve(A, x)
            
            # Eigenvalue computation
            if size <= 1000:  # Limit for performance
                with self.profiler.profile_operation("eigenvalues", size*size, "CPU"):
                    eigenvals = np.linalg.eigvals(A)
        
        # Get performance summaries
        operations = ["matrix_multiply", "matvec_multiply", "linear_solve", "eigenvalues"]
        for op in operations:
            summary = self.profiler.get_metrics_summary(op)
            if summary:
                results['operations'][op] = summary
        
        self.results['linear_algebra'] = results
        return results
    
    def benchmark_physics_kernels(self, sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark semiconductor physics kernels"""
        if sizes is None:
            sizes = [1000, 5000, 10000, 25000, 50000]
        
        print("\n⚡ BENCHMARKING PHYSICS KERNELS")
        print("=" * 50)
        
        try:
            from performance_bindings import PhysicsAcceleration
            physics = PhysicsAcceleration()
        except ImportError:
            print("   ❌ Physics acceleration not available")
            return {}
        
        results = {'sizes': sizes, 'operations': {}}
        
        for size in sizes:
            print(f"   Testing size: {size} points")
            
            # Generate test data
            potential = np.linspace(0, 1.0, size)
            doping_nd = np.full(size, 1e17)
            doping_na = np.full(size, 1e16)
            temperature = 300.0
            
            # Carrier density computation
            with self.profiler.profile_operation("carrier_densities", size, "SIMD"):
                n, p = physics.compute_carrier_densities(potential, doping_nd, doping_na, temperature)
            
            # Current density computation
            with self.profiler.profile_operation("current_densities", size, "SIMD"):
                Jn, Jp = physics.compute_current_densities(n, p, potential)
            
            # Recombination computation
            with self.profiler.profile_operation("recombination", size, "SIMD"):
                R = physics.compute_recombination(n, p)
            
            # Energy transport
            T_n = np.full(size, 300.0)
            T_p = np.full(size, 300.0)
            
            with self.profiler.profile_operation("energy_transport", size, "SIMD"):
                Wn, Wp = physics.compute_energy_densities(n, p, T_n, T_p)
            
            # Hydrodynamic transport
            v_n = np.full(size, 1e4)
            v_p = np.full(size, 8e3)
            
            with self.profiler.profile_operation("hydrodynamic", size, "SIMD"):
                Pn, Pp = physics.compute_momentum_densities(n, p, v_n, v_p)
        
        # Get performance summaries
        operations = ["carrier_densities", "current_densities", "recombination", 
                     "energy_transport", "hydrodynamic"]
        for op in operations:
            summary = self.profiler.get_metrics_summary(op)
            if summary:
                results['operations'][op] = summary
        
        self.results['physics_kernels'] = results
        return results
    
    def benchmark_gpu_acceleration(self, sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark GPU acceleration performance"""
        if sizes is None:
            sizes = [1000, 5000, 10000, 25000, 50000]
        
        print("\n🚀 BENCHMARKING GPU ACCELERATION")
        print("=" * 50)
        
        try:
            from gpu_acceleration import GPUAcceleratedSolver, GPUBackend
            gpu_solver = GPUAcceleratedSolver(GPUBackend.AUTO)
        except ImportError:
            print("   ❌ GPU acceleration not available")
            return {}
        
        if not gpu_solver.is_gpu_available():
            print("   ❌ GPU not available")
            return {}
        
        results = {'sizes': sizes, 'operations': {}}
        
        for size in sizes:
            print(f"   Testing size: {size} points")
            
            # Generate test data
            potential = np.linspace(0, 1.0, size)
            doping_nd = np.full(size, 1e17)
            doping_na = np.full(size, 1e16)
            
            # GPU transport solver
            with self.profiler.profile_operation("gpu_transport", size, "GPU"):
                results_gpu = gpu_solver.solve_transport_gpu(potential, doping_nd, doping_na)
            
            # GPU matrix operations
            if size <= 10000:  # Limit for memory
                A = np.random.random((min(size//10, 500), min(size//10, 500)))
                B = np.random.random((min(size//10, 500), min(size//10, 500)))
                
                with self.profiler.profile_operation("gpu_matrix_multiply", A.size, "GPU"):
                    if gpu_solver.kernels:
                        C = gpu_solver.kernels.matrix_multiply(A, B)
        
        # Get performance summaries
        operations = ["gpu_transport", "gpu_matrix_multiply"]
        for op in operations:
            summary = self.profiler.get_metrics_summary(op)
            if summary:
                results['operations'][op] = summary
        
        self.results['gpu_acceleration'] = results
        return results
    
    def benchmark_memory_patterns(self, sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark memory access patterns"""
        if sizes is None:
            sizes = [1000, 5000, 10000, 25000, 50000]
        
        print("\n💾 BENCHMARKING MEMORY PATTERNS")
        print("=" * 50)
        
        results = {'sizes': sizes, 'operations': {}}
        
        for size in sizes:
            print(f"   Testing size: {size} elements")
            
            # Sequential access
            data = np.random.random(size)
            with self.profiler.profile_operation("sequential_access", size, "CPU"):
                result = np.sum(data)
            
            # Random access
            indices = np.random.randint(0, size, size//10)
            with self.profiler.profile_operation("random_access", size, "CPU"):
                result = np.sum(data[indices])
            
            # Strided access
            with self.profiler.profile_operation("strided_access", size, "CPU"):
                result = np.sum(data[::10])
            
            # Cache-friendly operations
            matrix = np.random.random((int(np.sqrt(size)), int(np.sqrt(size))))
            with self.profiler.profile_operation("cache_friendly", size, "CPU"):
                result = np.sum(matrix, axis=0)
            
            # Cache-unfriendly operations
            with self.profiler.profile_operation("cache_unfriendly", size, "CPU"):
                result = np.sum(matrix, axis=1)
        
        # Get performance summaries
        operations = ["sequential_access", "random_access", "strided_access", 
                     "cache_friendly", "cache_unfriendly"]
        for op in operations:
            summary = self.profiler.get_metrics_summary(op)
            if summary:
                results['operations'][op] = summary
        
        self.results['memory_patterns'] = results
        return results
    
    def benchmark_transport_models(self, sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark different transport models"""
        if sizes is None:
            sizes = [100, 500, 1000, 2000]
        
        print("\n🔬 BENCHMARKING TRANSPORT MODELS")
        print("=" * 50)
        
        try:
            from advanced_transport import (create_drift_diffusion_solver,
                                          create_energy_transport_solver,
                                          create_hydrodynamic_solver,
                                          create_non_equilibrium_solver)
        except ImportError:
            print("   ❌ Advanced transport not available")
            return {}
        
        results = {'sizes': sizes, 'models': {}}
        
        for size in sizes:
            print(f"   Testing size: {size} points")
            
            # Test each transport model
            models = {
                'drift_diffusion': create_drift_diffusion_solver,
                'energy_transport': create_energy_transport_solver,
                'hydrodynamic': create_hydrodynamic_solver,
                'non_equilibrium': create_non_equilibrium_solver
            }
            
            for model_name, create_solver in models.items():
                try:
                    solver = create_solver(2e-6, 1e-6)
                    solver.set_doping(np.full(size, 1e17), np.full(size, 1e16))
                    
                    with self.profiler.profile_operation(f"transport_{model_name}", size, "CPU"):
                        results_model = solver.solve_transport([0, 1, 0, 0], Vg=0.5)
                    
                except Exception as e:
                    print(f"      ❌ {model_name} failed: {e}")
        
        # Get performance summaries
        model_operations = [f"transport_{model}" for model in models.keys()]
        for op in model_operations:
            summary = self.profiler.get_metrics_summary(op)
            if summary:
                results['models'][op] = summary
        
        self.results['transport_models'] = results
        return results
    
    def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark suites"""
        print("🏁 COMPREHENSIVE PERFORMANCE BENCHMARKS")
        print("=" * 60)
        
        # Print system information
        self.profiler.print_system_info()
        
        # Clear previous metrics
        self.profiler.clear_metrics()
        
        # Run all benchmark suites
        start_time = time.time()
        
        self.benchmark_linear_algebra()
        self.benchmark_physics_kernels()
        self.benchmark_gpu_acceleration()
        self.benchmark_memory_patterns()
        self.benchmark_transport_models()
        
        total_time = time.time() - start_time
        
        print(f"\n📊 BENCHMARK SUMMARY")
        print("=" * 30)
        print(f"Total benchmark time: {total_time:.2f}s")
        print(f"Total operations tested: {len(self.profiler.metrics_history)}")
        
        # Generate performance report
        report = self.profiler.generate_performance_report()
        print(f"\n{report}")
        
        return self.results
    
    def plot_scaling_analysis(self, save_path: str = None):
        """Plot scaling analysis for all benchmarks"""
        if not self.results:
            print("No benchmark results available for plotting")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        
        for benchmark_name, benchmark_data in self.results.items():
            if plot_idx >= len(axes):
                break
            
            ax = axes[plot_idx]
            
            if 'sizes' in benchmark_data and 'operations' in benchmark_data:
                sizes = benchmark_data['sizes']
                
                for op_name, op_data in benchmark_data['operations'].items():
                    if 'execution_time' in op_data:
                        # Get execution times for each size
                        times = []
                        for size in sizes:
                            # Find metrics for this operation and size
                            matching_metrics = [
                                m for m in self.profiler.metrics_history 
                                if m.operation == op_name and m.problem_size == size*size
                            ]
                            if matching_metrics:
                                times.append(np.mean([m.execution_time for m in matching_metrics]))
                            else:
                                times.append(0)
                        
                        if any(t > 0 for t in times):
                            ax.loglog(sizes, times, 'o-', label=op_name, markersize=4)
            
            ax.set_title(f'{benchmark_name.replace("_", " ").title()}')
            ax.set_xlabel('Problem Size')
            ax.set_ylabel('Execution Time (s)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scaling analysis saved to: {save_path}")
        else:
            plt.show()
    
    def export_results(self, filename: str):
        """Export benchmark results"""
        import json
        
        # Export profiler metrics
        self.profiler.export_metrics(filename.replace('.json', '_metrics.json'))
        
        # Export benchmark results
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Benchmark results exported to: {filename}")

def main():
    """Main benchmarking function"""
    print("🚀 STARTING COMPREHENSIVE PERFORMANCE BENCHMARKS")
    print("=" * 70)
    
    # Create benchmark suite
    benchmarks = CriticalPathBenchmarks()
    
    # Run comprehensive benchmarks
    results = benchmarks.run_comprehensive_benchmarks()
    
    # Plot scaling analysis
    benchmarks.plot_scaling_analysis('performance_scaling_analysis.png')
    
    # Export results
    benchmarks.export_results('benchmark_results.json')
    
    print("\n🎉 BENCHMARKING COMPLETED SUCCESSFULLY!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
