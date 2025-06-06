#!/usr/bin/env python3
"""
Scaling Studies Framework
Advanced performance scaling analysis and optimization

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from performance_profiler import PerformanceProfiler, profiler

@dataclass
class ScalingResult:
    """Scaling analysis result"""
    operation: str
    sizes: List[int]
    times: List[float]
    throughputs: List[float]
    memory_usage: List[float]
    scaling_exponent: float
    efficiency: float
    optimal_size: int
    complexity_class: str

class ScalingAnalyzer:
    """Advanced scaling analysis framework"""
    
    def __init__(self):
        self.profiler = profiler
        self.scaling_results: List[ScalingResult] = []
        
    def analyze_computational_complexity(self, operation: str, sizes: List[int], 
                                       times: List[float]) -> Tuple[float, str]:
        """Analyze computational complexity from timing data"""
        
        if len(sizes) < 3 or len(times) < 3:
            return 0.0, "Unknown"
        
        # Filter out zero times
        valid_data = [(s, t) for s, t in zip(sizes, times) if t > 0]
        if len(valid_data) < 3:
            return 0.0, "Unknown"
        
        sizes_valid, times_valid = zip(*valid_data)
        
        # Log-log regression to find scaling exponent
        log_sizes = np.log(sizes_valid)
        log_times = np.log(times_valid)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_times)
        
        # Classify complexity
        if slope < 0.5:
            complexity_class = "Sub-linear"
        elif slope < 1.5:
            complexity_class = "Linear O(n)"
        elif slope < 2.5:
            complexity_class = "Quadratic O(n²)"
        elif slope < 3.5:
            complexity_class = "Cubic O(n³)"
        else:
            complexity_class = "Higher-order"
        
        return slope, complexity_class
    
    def find_optimal_problem_size(self, sizes: List[int], throughputs: List[float]) -> int:
        """Find optimal problem size based on throughput"""
        if not throughputs:
            return 0
        
        max_throughput_idx = np.argmax(throughputs)
        return sizes[max_throughput_idx]
    
    def calculate_parallel_efficiency(self, sizes: List[int], times: List[float]) -> float:
        """Calculate parallel efficiency metric"""
        if len(times) < 2:
            return 0.0
        
        # Efficiency = (smallest_time * smallest_size) / (time * size)
        # This gives efficiency relative to smallest problem
        base_time = times[0]
        base_size = sizes[0]
        
        efficiencies = []
        for size, time in zip(sizes[1:], times[1:]):
            if time > 0:
                expected_time = base_time * (size / base_size)
                efficiency = expected_time / time
                efficiencies.append(efficiency)
        
        return np.mean(efficiencies) if efficiencies else 0.0
    
    def run_scaling_study(self, operation_func, sizes: List[int], 
                         operation_name: str, **kwargs) -> ScalingResult:
        """Run comprehensive scaling study for an operation"""
        
        print(f"📈 Scaling study: {operation_name}")
        print(f"   Sizes: {sizes}")
        
        times = []
        throughputs = []
        memory_usages = []
        
        for size in sizes:
            print(f"   Testing size: {size}")
            
            # Clear metrics before each test
            initial_metrics_count = len(self.profiler.metrics_history)
            
            # Run operation with profiling
            try:
                with self.profiler.profile_operation(operation_name, size, "CPU"):
                    result = operation_func(size, **kwargs)
                
                # Get the latest metric
                if len(self.profiler.metrics_history) > initial_metrics_count:
                    latest_metric = self.profiler.metrics_history[-1]
                    times.append(latest_metric.execution_time)
                    throughputs.append(latest_metric.throughput)
                    memory_usages.append(latest_metric.memory_usage)
                else:
                    times.append(0.0)
                    throughputs.append(0.0)
                    memory_usages.append(0.0)
                    
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                times.append(0.0)
                throughputs.append(0.0)
                memory_usages.append(0.0)
        
        # Analyze scaling
        scaling_exponent, complexity_class = self.analyze_computational_complexity(
            operation_name, sizes, times)
        
        efficiency = self.calculate_parallel_efficiency(sizes, times)
        optimal_size = self.find_optimal_problem_size(sizes, throughputs)
        
        result = ScalingResult(
            operation=operation_name,
            sizes=sizes,
            times=times,
            throughputs=throughputs,
            memory_usage=memory_usages,
            scaling_exponent=scaling_exponent,
            efficiency=efficiency,
            optimal_size=optimal_size,
            complexity_class=complexity_class
        )
        
        self.scaling_results.append(result)
        
        print(f"   Scaling exponent: {scaling_exponent:.2f}")
        print(f"   Complexity: {complexity_class}")
        print(f"   Efficiency: {efficiency:.2f}")
        print(f"   Optimal size: {optimal_size}")
        
        return result
    
    def compare_backends(self, operation_func, sizes: List[int], 
                        backends: List[str], operation_name: str) -> Dict[str, ScalingResult]:
        """Compare scaling across different backends"""
        
        print(f"\n🔄 Backend comparison: {operation_name}")
        
        results = {}
        
        for backend in backends:
            print(f"\n   Backend: {backend}")
            
            backend_times = []
            backend_throughputs = []
            backend_memory = []
            
            for size in sizes:
                try:
                    with self.profiler.profile_operation(f"{operation_name}_{backend}", size, backend):
                        result = operation_func(size, backend=backend)
                    
                    # Get latest metric
                    if self.profiler.metrics_history:
                        latest_metric = self.profiler.metrics_history[-1]
                        backend_times.append(latest_metric.execution_time)
                        backend_throughputs.append(latest_metric.throughput)
                        backend_memory.append(latest_metric.memory_usage)
                    else:
                        backend_times.append(0.0)
                        backend_throughputs.append(0.0)
                        backend_memory.append(0.0)
                        
                except Exception as e:
                    print(f"      ❌ Size {size} failed: {e}")
                    backend_times.append(0.0)
                    backend_throughputs.append(0.0)
                    backend_memory.append(0.0)
            
            # Analyze scaling for this backend
            scaling_exponent, complexity_class = self.analyze_computational_complexity(
                f"{operation_name}_{backend}", sizes, backend_times)
            
            efficiency = self.calculate_parallel_efficiency(sizes, backend_times)
            optimal_size = self.find_optimal_problem_size(sizes, backend_throughputs)
            
            results[backend] = ScalingResult(
                operation=f"{operation_name}_{backend}",
                sizes=sizes,
                times=backend_times,
                throughputs=backend_throughputs,
                memory_usage=backend_memory,
                scaling_exponent=scaling_exponent,
                efficiency=efficiency,
                optimal_size=optimal_size,
                complexity_class=complexity_class
            )
            
            print(f"      Scaling: {scaling_exponent:.2f} ({complexity_class})")
            print(f"      Efficiency: {efficiency:.2f}")
        
        return results
    
    def plot_scaling_comparison(self, results: List[ScalingResult], 
                               title: str = "Scaling Comparison", 
                               save_path: str = None):
        """Plot scaling comparison"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for result, color in zip(results, colors):
            if not any(t > 0 for t in result.times):
                continue
                
            # Execution time scaling
            valid_data = [(s, t) for s, t in zip(result.sizes, result.times) if t > 0]
            if valid_data:
                sizes, times = zip(*valid_data)
                ax1.loglog(sizes, times, 'o-', color=color, label=result.operation, markersize=4)
        
        ax1.set_title('Execution Time Scaling')
        ax1.set_xlabel('Problem Size')
        ax1.set_ylabel('Time (s)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Throughput comparison
        for result, color in zip(results, colors):
            if any(t > 0 for t in result.throughputs):
                ax2.semilogx(result.sizes, result.throughputs, 'o-', color=color, 
                           label=result.operation, markersize=4)
        
        ax2.set_title('Throughput Comparison')
        ax2.set_xlabel('Problem Size')
        ax2.set_ylabel('Throughput (ops/s)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Memory usage
        for result, color in zip(results, colors):
            if any(m != 0 for m in result.memory_usage):
                ax3.semilogx(result.sizes, result.memory_usage, 'o-', color=color, 
                           label=result.operation, markersize=4)
        
        ax3.set_title('Memory Usage')
        ax3.set_xlabel('Problem Size')
        ax3.set_ylabel('Memory (MB)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Efficiency comparison
        efficiencies = [r.efficiency for r in results]
        operations = [r.operation for r in results]
        
        bars = ax4.bar(range(len(operations)), efficiencies, color=colors)
        ax4.set_title('Parallel Efficiency')
        ax4.set_xlabel('Operation')
        ax4.set_ylabel('Efficiency')
        ax4.set_xticks(range(len(operations)))
        ax4.set_xticklabels(operations, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scaling comparison saved to: {save_path}")
        else:
            plt.show()
    
    def generate_scaling_report(self) -> str:
        """Generate comprehensive scaling analysis report"""
        
        report = []
        report.append("📈 SCALING ANALYSIS REPORT")
        report.append("=" * 50)
        
        if not self.scaling_results:
            report.append("No scaling results available")
            return "\n".join(report)
        
        # Overall summary
        report.append(f"\n📊 SUMMARY:")
        report.append(f"   Operations analyzed: {len(self.scaling_results)}")
        
        avg_efficiency = np.mean([r.efficiency for r in self.scaling_results])
        report.append(f"   Average efficiency: {avg_efficiency:.2f}")
        
        # Per-operation analysis
        report.append(f"\n🔍 DETAILED ANALYSIS:")
        
        for result in self.scaling_results:
            report.append(f"\n   {result.operation}:")
            report.append(f"      Complexity: {result.complexity_class}")
            report.append(f"      Scaling exponent: {result.scaling_exponent:.3f}")
            report.append(f"      Efficiency: {result.efficiency:.3f}")
            report.append(f"      Optimal size: {result.optimal_size}")
            
            if result.times:
                min_time = min(t for t in result.times if t > 0)
                max_time = max(result.times)
                report.append(f"      Time range: {min_time:.6f}s - {max_time:.6f}s")
            
            if result.throughputs:
                max_throughput = max(result.throughputs)
                report.append(f"      Peak throughput: {max_throughput:.0f} ops/s")
        
        # Recommendations
        report.append(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        # Find operations with poor scaling
        poor_scaling = [r for r in self.scaling_results if r.scaling_exponent > 2.0]
        if poor_scaling:
            report.append(f"   Operations with poor scaling (>O(n²)):")
            for r in poor_scaling:
                report.append(f"      - {r.operation}: {r.complexity_class}")
        
        # Find operations with low efficiency
        low_efficiency = [r for r in self.scaling_results if r.efficiency < 0.5]
        if low_efficiency:
            report.append(f"   Operations with low efficiency (<0.5):")
            for r in low_efficiency:
                report.append(f"      - {r.operation}: {r.efficiency:.3f}")
        
        return "\n".join(report)
    
    def export_scaling_results(self, filename: str):
        """Export scaling results to JSON"""
        import json
        
        data = []
        for result in self.scaling_results:
            data.append({
                'operation': result.operation,
                'sizes': result.sizes,
                'times': result.times,
                'throughputs': result.throughputs,
                'memory_usage': result.memory_usage,
                'scaling_exponent': result.scaling_exponent,
                'efficiency': result.efficiency,
                'optimal_size': result.optimal_size,
                'complexity_class': result.complexity_class
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Scaling results exported to: {filename}")

class CriticalPathOptimizer:
    """Critical path optimization framework"""

    def __init__(self):
        self.profiler = profiler
        self.scaling_analyzer = ScalingAnalyzer()
        self.optimization_results = {}

    def identify_bottlenecks(self, threshold_time: float = 0.1) -> List[str]:
        """Identify performance bottlenecks"""

        if not self.profiler.metrics_history:
            return []

        # Group metrics by operation
        operation_times = {}
        for metric in self.profiler.metrics_history:
            if metric.operation not in operation_times:
                operation_times[metric.operation] = []
            operation_times[metric.operation].append(metric.execution_time)

        # Find operations with high average execution time
        bottlenecks = []
        for operation, times in operation_times.items():
            avg_time = np.mean(times)
            if avg_time > threshold_time:
                bottlenecks.append(operation)

        # Sort by average execution time
        bottlenecks.sort(key=lambda op: np.mean(operation_times[op]), reverse=True)

        return bottlenecks

    def optimize_linear_algebra(self) -> Dict[str, Any]:
        """Optimize linear algebra operations"""

        print("🔧 OPTIMIZING LINEAR ALGEBRA OPERATIONS")
        print("=" * 50)

        results = {}

        # Test different BLAS libraries and configurations
        sizes = [500, 1000, 2000]

        for size in sizes:
            print(f"   Testing size: {size}×{size}")

            A = np.random.random((size, size)).astype(np.float64)
            B = np.random.random((size, size)).astype(np.float64)

            # Standard NumPy
            with self.profiler.profile_operation("numpy_matmul", size*size, "NumPy"):
                C1 = np.dot(A, B)

            # NumPy with different order
            with self.profiler.profile_operation("numpy_matmul_F", size*size, "NumPy-F"):
                A_F = np.asfortranarray(A)
                B_F = np.asfortranarray(B)
                C2 = np.dot(A_F, B_F)

            # Matrix multiplication with @ operator
            with self.profiler.profile_operation("numpy_at_operator", size*size, "NumPy-@"):
                C3 = A @ B

        # Analyze results
        operations = ["numpy_matmul", "numpy_matmul_F", "numpy_at_operator"]
        for op in operations:
            summary = self.profiler.get_metrics_summary(op)
            if summary:
                results[op] = summary

        return results

    def optimize_memory_access(self) -> Dict[str, Any]:
        """Optimize memory access patterns"""

        print("\n💾 OPTIMIZING MEMORY ACCESS PATTERNS")
        print("=" * 50)

        results = {}
        sizes = [10000, 50000, 100000]

        for size in sizes:
            print(f"   Testing size: {size} elements")

            # Test different data layouts
            data_c = np.random.random(size).astype(np.float64)  # C-contiguous
            data_f = np.asfortranarray(data_c)  # Fortran-contiguous

            # Sequential access - C order
            with self.profiler.profile_operation("sequential_c", size, "C-order"):
                result = np.sum(data_c)

            # Sequential access - Fortran order
            with self.profiler.profile_operation("sequential_f", size, "F-order"):
                result = np.sum(data_f)

            # Vectorized operations
            data2 = np.random.random(size).astype(np.float64)

            with self.profiler.profile_operation("vectorized_add", size, "Vectorized"):
                result = data_c + data2

            # Cache-optimized matrix operations
            if size <= 50000:  # Limit for memory
                matrix_size = int(np.sqrt(size))
                matrix = np.random.random((matrix_size, matrix_size))

                # Row-wise sum (cache-friendly)
                with self.profiler.profile_operation("rowwise_sum", size, "Cache-friendly"):
                    result = np.sum(matrix, axis=1)

                # Column-wise sum (cache-unfriendly)
                with self.profiler.profile_operation("colwise_sum", size, "Cache-unfriendly"):
                    result = np.sum(matrix, axis=0)

        # Analyze results
        operations = ["sequential_c", "sequential_f", "vectorized_add",
                     "rowwise_sum", "colwise_sum"]
        for op in operations:
            summary = self.profiler.get_metrics_summary(op)
            if summary:
                results[op] = summary

        return results

    def optimize_physics_kernels(self) -> Dict[str, Any]:
        """Optimize semiconductor physics kernels"""

        print("\n⚡ OPTIMIZING PHYSICS KERNELS")
        print("=" * 50)

        try:
            from performance_bindings import PhysicsAcceleration, PerformanceOptimizer
            physics = PhysicsAcceleration()
            optimizer = PerformanceOptimizer()
        except ImportError:
            print("   ❌ Physics acceleration not available")
            return {}

        results = {}
        sizes = [5000, 10000, 25000]

        for size in sizes:
            print(f"   Testing size: {size} points")

            # Generate test data
            potential = np.linspace(0, 1.0, size)
            doping_nd = np.full(size, 1e17)
            doping_na = np.full(size, 1e16)

            # Test different optimization strategies

            # Standard computation
            with self.profiler.profile_operation("physics_standard", size, "Standard"):
                n, p = physics.compute_carrier_densities(potential, doping_nd, doping_na)

            # Optimized computation with performance optimizer
            with self.profiler.profile_operation("physics_optimized", size, "Optimized"):
                # Use optimizer for vector operations
                net_doping = optimizer.optimize_computation('vector_add', doping_nd, -doping_na)
                n_opt, p_opt = physics.compute_carrier_densities(potential, doping_nd, doping_na)

            # Batch processing
            batch_size = size // 4
            with self.profiler.profile_operation("physics_batched", size, "Batched"):
                n_batch = np.zeros(size)
                p_batch = np.zeros(size)
                for i in range(0, size, batch_size):
                    end_idx = min(i + batch_size, size)
                    n_b, p_b = physics.compute_carrier_densities(
                        potential[i:end_idx], doping_nd[i:end_idx], doping_na[i:end_idx])
                    n_batch[i:end_idx] = n_b
                    p_batch[i:end_idx] = p_b

        # Analyze results
        operations = ["physics_standard", "physics_optimized", "physics_batched"]
        for op in operations:
            summary = self.profiler.get_metrics_summary(op)
            if summary:
                results[op] = summary

        return results

    def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """Run comprehensive optimization analysis"""

        print("🚀 COMPREHENSIVE CRITICAL PATH OPTIMIZATION")
        print("=" * 60)

        # Clear previous metrics
        self.profiler.clear_metrics()

        # Identify current bottlenecks
        print("\n🔍 IDENTIFYING BOTTLENECKS...")
        # Run a quick benchmark first to get baseline metrics
        from comprehensive_benchmarks import CriticalPathBenchmarks
        quick_benchmarks = CriticalPathBenchmarks()
        quick_benchmarks.benchmark_linear_algebra([500, 1000])
        quick_benchmarks.benchmark_physics_kernels([5000, 10000])

        bottlenecks = self.identify_bottlenecks(threshold_time=0.01)
        print(f"   Identified bottlenecks: {bottlenecks}")

        # Run optimizations
        optimization_results = {}

        optimization_results['linear_algebra'] = self.optimize_linear_algebra()
        optimization_results['memory_access'] = self.optimize_memory_access()
        optimization_results['physics_kernels'] = self.optimize_physics_kernels()

        # Generate optimization report
        report = self.generate_optimization_report(optimization_results)
        print(f"\n{report}")

        self.optimization_results = optimization_results
        return optimization_results

    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate optimization analysis report"""

        report = []
        report.append("🔧 OPTIMIZATION ANALYSIS REPORT")
        report.append("=" * 50)

        for category, category_results in results.items():
            if not category_results:
                continue

            report.append(f"\n📊 {category.upper().replace('_', ' ')}:")

            # Find best performing operation in each category
            best_op = None
            best_time = float('inf')

            for op_name, op_data in category_results.items():
                if 'execution_time' in op_data:
                    avg_time = op_data['execution_time']['mean']
                    report.append(f"   {op_name}: {avg_time:.6f}s avg")

                    if avg_time < best_time:
                        best_time = avg_time
                        best_op = op_name

            if best_op:
                report.append(f"   ✅ Best: {best_op} ({best_time:.6f}s)")

        # Overall recommendations
        report.append(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        report.append(f"   1. Use vectorized operations for large arrays")
        report.append(f"   2. Prefer C-contiguous memory layout")
        report.append(f"   3. Consider batch processing for very large problems")
        report.append(f"   4. Use performance optimizer for automatic backend selection")
        report.append(f"   5. Profile regularly to identify new bottlenecks")

        return "\n".join(report)

# Global instances
scaling_analyzer = ScalingAnalyzer()
critical_path_optimizer = CriticalPathOptimizer()
