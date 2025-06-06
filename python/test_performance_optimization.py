#!/usr/bin/env python3
"""
Performance Optimization Test Suite
Comprehensive testing of performance optimizations and scaling studies

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_performance_profiler():
    """Test performance profiler functionality"""
    print("🔧 TESTING PERFORMANCE PROFILER")
    print("=" * 40)
    
    try:
        from performance_profiler import PerformanceProfiler
        
        profiler = PerformanceProfiler()
        
        # Test system info
        profiler.print_system_info()
        
        # Test profiling context manager
        with profiler.profile_operation("test_operation", 1000, "CPU"):
            time.sleep(0.01)  # Simulate work
            result = np.sum(np.random.random(1000))
        
        # Test metrics retrieval
        summary = profiler.get_metrics_summary("test_operation")
        print(f"   Metrics summary: {summary}")
        
        # Test continuous monitoring
        profiler.start_continuous_monitoring(0.1)
        time.sleep(0.5)
        profiler.stop_continuous_monitoring()
        
        print("   ✅ Performance profiler tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Performance profiler tests failed: {e}")
        return False

def test_comprehensive_benchmarks():
    """Test comprehensive benchmarking suite"""
    print("\n🏁 TESTING COMPREHENSIVE BENCHMARKS")
    print("=" * 40)
    
    try:
        from comprehensive_benchmarks import CriticalPathBenchmarks
        
        benchmarks = CriticalPathBenchmarks()
        
        # Test individual benchmark suites with smaller sizes for speed
        print("   Testing linear algebra benchmarks...")
        la_results = benchmarks.benchmark_linear_algebra([100, 200])
        assert 'operations' in la_results
        
        print("   Testing physics kernels...")
        physics_results = benchmarks.benchmark_physics_kernels([1000, 2000])
        
        print("   Testing memory patterns...")
        memory_results = benchmarks.benchmark_memory_patterns([1000, 2000])
        
        print("   Testing transport models...")
        transport_results = benchmarks.benchmark_transport_models([50, 100])
        
        # Test plotting (without showing)
        benchmarks.plot_scaling_analysis('test_scaling.png')
        
        # Test export
        benchmarks.export_results('test_benchmark_results.json')
        
        print("   ✅ Comprehensive benchmarks tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Comprehensive benchmarks tests failed: {e}")
        return False

def test_scaling_studies():
    """Test scaling studies framework"""
    print("\n📈 TESTING SCALING STUDIES")
    print("=" * 40)
    
    try:
        from scaling_studies import ScalingAnalyzer
        
        analyzer = ScalingAnalyzer()
        
        # Define a simple test operation
        def test_operation(size, **kwargs):
            data = np.random.random(size)
            return np.sum(data)
        
        # Run scaling study
        sizes = [100, 500, 1000]
        result = analyzer.run_scaling_study(test_operation, sizes, "test_scaling")
        
        assert result.operation == "test_scaling"
        assert len(result.sizes) == len(sizes)
        assert result.scaling_exponent is not None
        
        # Test complexity analysis
        times = [0.001, 0.004, 0.016]  # Quadratic scaling
        exponent, complexity = analyzer.analyze_computational_complexity("test", sizes, times)
        print(f"   Detected complexity: {complexity} (exponent: {exponent:.2f})")
        
        # Test backend comparison
        def backend_operation(size, backend="CPU"):
            data = np.random.random(size)
            if backend == "GPU":
                time.sleep(0.001)  # Simulate GPU overhead
            return np.sum(data)
        
        backend_results = analyzer.compare_backends(
            backend_operation, [100, 500], ["CPU", "GPU"], "backend_test")
        
        assert "CPU" in backend_results
        assert "GPU" in backend_results
        
        # Test report generation
        report = analyzer.generate_scaling_report()
        assert "SCALING ANALYSIS REPORT" in report
        
        print("   ✅ Scaling studies tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Scaling studies tests failed: {e}")
        return False

def test_critical_path_optimization():
    """Test critical path optimization"""
    print("\n🔧 TESTING CRITICAL PATH OPTIMIZATION")
    print("=" * 40)
    
    try:
        from scaling_studies import CriticalPathOptimizer
        
        optimizer = CriticalPathOptimizer()
        
        # Test bottleneck identification
        # First generate some metrics
        with optimizer.profiler.profile_operation("slow_operation", 1000, "CPU"):
            time.sleep(0.1)  # Simulate slow operation
        
        with optimizer.profiler.profile_operation("fast_operation", 1000, "CPU"):
            time.sleep(0.01)  # Simulate fast operation
        
        bottlenecks = optimizer.identify_bottlenecks(threshold_time=0.05)
        print(f"   Identified bottlenecks: {bottlenecks}")
        assert "slow_operation" in bottlenecks
        
        # Test optimization methods
        print("   Testing linear algebra optimization...")
        la_opt = optimizer.optimize_linear_algebra()
        
        print("   Testing memory access optimization...")
        mem_opt = optimizer.optimize_memory_access()
        
        # Test comprehensive optimization
        print("   Running comprehensive optimization...")
        comp_opt = optimizer.run_comprehensive_optimization()
        
        assert 'linear_algebra' in comp_opt
        assert 'memory_access' in comp_opt
        
        print("   ✅ Critical path optimization tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Critical path optimization tests failed: {e}")
        return False

def test_integration_with_simulators():
    """Test integration with existing simulators"""
    print("\n🔗 TESTING SIMULATOR INTEGRATION")
    print("=" * 40)
    
    try:
        # Test with main simulator
        import simulator
        from performance_profiler import profiler
        
        # Create device with profiling
        with profiler.profile_operation("device_creation", 1, "CPU"):
            device = simulator.Device(2e-6, 1e-6)
        
        print(f"   Device created: {device.length:.1e}m × {device.width:.1e}m")
        
        # Test with advanced transport
        from advanced_transport import create_energy_transport_solver
        
        with profiler.profile_operation("transport_creation", 1, "CPU"):
            transport_solver = create_energy_transport_solver(2e-6, 1e-6)
        
        # Test simulation with profiling
        size = 100
        transport_solver.set_doping(np.full(size, 1e17), np.full(size, 1e16))
        
        with profiler.profile_operation("transport_solve", size, "CPU"):
            results = transport_solver.solve_transport([0, 1, 0, 0], Vg=0.5)
        
        print(f"   Transport solved: {len(results)} fields")
        
        # Test with GPU acceleration
        try:
            from gpu_acceleration import GPUAcceleratedSolver
            
            with profiler.profile_operation("gpu_solver_creation", 1, "GPU"):
                gpu_solver = GPUAcceleratedSolver()
            
            print(f"   GPU solver available: {gpu_solver.is_gpu_available()}")
            
        except ImportError:
            print("   GPU acceleration not available for testing")
        
        # Generate performance report
        report = profiler.generate_performance_report()
        print("   Performance report generated")
        
        print("   ✅ Simulator integration tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Simulator integration tests failed: {e}")
        return False

def test_performance_visualization():
    """Test performance visualization capabilities"""
    print("\n📊 TESTING PERFORMANCE VISUALIZATION")
    print("=" * 40)
    
    try:
        from performance_profiler import profiler
        from scaling_studies import ScalingAnalyzer
        
        # Generate some test data
        analyzer = ScalingAnalyzer()
        
        def test_viz_operation(size):
            return np.sum(np.random.random(size))
        
        # Run scaling study for visualization
        sizes = [100, 500, 1000, 2000]
        result = analyzer.run_scaling_study(test_viz_operation, sizes, "viz_test")
        
        # Test plotting
        analyzer.plot_scaling_comparison([result], "Test Scaling", "test_scaling_viz.png")
        
        # Test profiler plotting
        profiler.plot_performance_trends("viz_test", "test_performance_trends.png")
        
        print("   ✅ Performance visualization tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Performance visualization tests failed: {e}")
        return False

def run_performance_stress_test():
    """Run stress test with large problem sizes"""
    print("\n💪 PERFORMANCE STRESS TEST")
    print("=" * 40)
    
    try:
        from comprehensive_benchmarks import CriticalPathBenchmarks
        from performance_profiler import profiler
        
        # Clear previous metrics
        profiler.clear_metrics()
        
        benchmarks = CriticalPathBenchmarks()
        
        # Large-scale tests
        print("   Running large-scale linear algebra test...")
        la_results = benchmarks.benchmark_linear_algebra([1000, 2000])
        
        print("   Running large-scale physics test...")
        physics_results = benchmarks.benchmark_physics_kernels([10000, 25000])
        
        # Check performance metrics
        all_metrics = profiler.metrics_history
        if all_metrics:
            avg_time = np.mean([m.execution_time for m in all_metrics])
            max_memory = max([m.memory_usage for m in all_metrics])
            
            print(f"   Average execution time: {avg_time:.4f}s")
            print(f"   Peak memory usage: {max_memory:.1f}MB")
            
            # Performance thresholds
            if avg_time < 1.0:  # Less than 1 second average
                print("   ✅ Performance within acceptable limits")
            else:
                print("   ⚠️  Performance may need optimization")
        
        print("   ✅ Stress test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Stress test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 PERFORMANCE OPTIMIZATION TEST SUITE")
    print("=" * 60)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests_passed = 0
    total_tests = 6
    
    # Run all tests
    if test_performance_profiler():
        tests_passed += 1
    
    if test_comprehensive_benchmarks():
        tests_passed += 1
    
    if test_scaling_studies():
        tests_passed += 1
    
    if test_critical_path_optimization():
        tests_passed += 1
    
    if test_integration_with_simulators():
        tests_passed += 1
    
    if test_performance_visualization():
        tests_passed += 1
    
    # Run stress test
    stress_test_passed = run_performance_stress_test()
    
    # Print summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {tests_passed/total_tests*100:.1f}%")
    print(f"Stress test: {'✅ Passed' if stress_test_passed else '❌ Failed'}")
    
    if tests_passed == total_tests and stress_test_passed:
        print("\n🎉 ALL PERFORMANCE TESTS PASSED!")
        print("   Performance optimization framework is working correctly")
        return 0
    else:
        print(f"\n⚠️  {total_tests - tests_passed} TESTS FAILED")
        print("   Some performance features may not be working optimally")
        return 1

if __name__ == "__main__":
    sys.exit(main())
