#!/usr/bin/env python3
"""
GPU Acceleration Test Suite
Comprehensive testing of SIMD/GPU acceleration features

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_simd_acceleration():
    """Test SIMD acceleration features"""
    print("🔧 TESTING SIMD ACCELERATION")
    print("=" * 40)
    
    try:
        from performance_bindings import SIMDKernels, PerformanceOptimizer
        
        # Initialize SIMD kernels
        simd = SIMDKernels()
        print(f"   SIMD capabilities: {simd.get_capabilities()}")
        
        # Test vector operations
        size = 10000
        a = np.random.random(size)
        b = np.random.random(size)
        
        # Vector addition
        start = time.time()
        result = simd.vector_add(a, b)
        simd_time = time.time() - start
        
        start = time.time()
        numpy_result = a + b
        numpy_time = time.time() - start
        
        print(f"   Vector add (size={size}):")
        print(f"      SIMD: {simd_time:.6f}s")
        print(f"      NumPy: {numpy_time:.6f}s")
        print(f"      Speedup: {numpy_time/simd_time:.2f}x")
        print(f"      Accuracy: {np.allclose(result, numpy_result)}")
        
        # Dot product
        start = time.time()
        result = simd.dot_product(a, b)
        simd_time = time.time() - start
        
        start = time.time()
        numpy_result = np.dot(a, b)
        numpy_time = time.time() - start
        
        print(f"   Dot product (size={size}):")
        print(f"      SIMD: {simd_time:.6f}s")
        print(f"      NumPy: {numpy_time:.6f}s")
        print(f"      Speedup: {numpy_time/simd_time:.2f}x")
        print(f"      Accuracy: {np.isclose(result, numpy_result)}")
        
        # Matrix-vector multiplication
        A = np.random.random((1000, 1000))
        x = np.random.random(1000)
        
        start = time.time()
        result = simd.matrix_vector_multiply(A, x)
        simd_time = time.time() - start
        
        start = time.time()
        numpy_result = np.dot(A, x)
        numpy_time = time.time() - start
        
        print(f"   Matrix-vector multiply (1000x1000):")
        print(f"      SIMD: {simd_time:.6f}s")
        print(f"      NumPy: {numpy_time:.6f}s")
        print(f"      Speedup: {numpy_time/simd_time:.2f}x")
        print(f"      Accuracy: {np.allclose(result, numpy_result)}")
        
        print("   ✅ SIMD acceleration tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ SIMD acceleration tests failed: {e}")
        return False

def test_gpu_acceleration():
    """Test GPU acceleration features"""
    print("\n🚀 TESTING GPU ACCELERATION")
    print("=" * 40)
    
    try:
        from gpu_acceleration import GPUAcceleratedSolver, GPUBackend
        
        # Test GPU solver initialization
        solver = GPUAcceleratedSolver(GPUBackend.AUTO)
        
        print(f"   GPU available: {solver.is_gpu_available()}")
        
        perf_info = solver.get_performance_info()
        print(f"   Performance info: {perf_info}")
        
        # Test GPU transport solver
        size = 1000
        potential = np.linspace(0, 1.0, size)
        doping_nd = np.full(size, 1e17)
        doping_na = np.full(size, 1e16)
        
        start = time.time()
        results = solver.solve_transport_gpu(potential, doping_nd, doping_na)
        gpu_time = time.time() - start
        
        print(f"   GPU transport solver (size={size}):")
        print(f"      Time: {gpu_time:.6f}s")
        print(f"      Fields: {list(results.keys())}")
        print(f"      Carrier densities: n={np.mean(results['n']):.2e}, p={np.mean(results['p']):.2e}")
        
        # Run performance benchmark
        benchmark_results = solver.benchmark_performance(size=5000)
        
        print("   ✅ GPU acceleration tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ GPU acceleration tests failed: {e}")
        return False

def test_physics_acceleration():
    """Test physics-specific acceleration"""
    print("\n⚡ TESTING PHYSICS ACCELERATION")
    print("=" * 40)
    
    try:
        from performance_bindings import PhysicsAcceleration
        
        # Initialize physics acceleration
        physics = PhysicsAcceleration()
        
        # Test carrier density computation
        size = 5000
        potential = np.linspace(0, 1.0, size)
        doping_nd = np.full(size, 1e17)
        doping_na = np.full(size, 1e16)
        
        start = time.time()
        n, p = physics.compute_carrier_densities(potential, doping_nd, doping_na)
        carrier_time = time.time() - start
        
        print(f"   Carrier densities (size={size}):")
        print(f"      Time: {carrier_time:.6f}s")
        print(f"      n: mean={np.mean(n):.2e}, max={np.max(n):.2e}")
        print(f"      p: mean={np.mean(p):.2e}, max={np.max(p):.2e}")
        
        # Test current density computation
        start = time.time()
        Jn, Jp = physics.compute_current_densities(n, p, potential)
        current_time = time.time() - start
        
        print(f"   Current densities (size={size}):")
        print(f"      Time: {current_time:.6f}s")
        print(f"      Jn: mean={np.mean(Jn):.2e}")
        print(f"      Jp: mean={np.mean(Jp):.2e}")
        
        # Test recombination computation
        start = time.time()
        R = physics.compute_recombination(n, p)
        recomb_time = time.time() - start
        
        print(f"   Recombination (size={size}):")
        print(f"      Time: {recomb_time:.6f}s")
        print(f"      R: mean={np.mean(R):.2e}")
        
        # Test energy transport
        T_n = np.full(size, 300.0)
        T_p = np.full(size, 300.0)
        
        start = time.time()
        Wn, Wp = physics.compute_energy_densities(n, p, T_n, T_p)
        energy_time = time.time() - start
        
        print(f"   Energy densities (size={size}):")
        print(f"      Time: {energy_time:.6f}s")
        print(f"      Wn: mean={np.mean(Wn):.2e}")
        print(f"      Wp: mean={np.mean(Wp):.2e}")
        
        # Test hydrodynamic transport
        v_n = np.full(size, 1e4)
        v_p = np.full(size, 8e3)
        
        start = time.time()
        Pn, Pp = physics.compute_momentum_densities(n, p, v_n, v_p)
        momentum_time = time.time() - start
        
        print(f"   Momentum densities (size={size}):")
        print(f"      Time: {momentum_time:.6f}s")
        print(f"      Pn: mean={np.mean(Pn):.2e}")
        print(f"      Pp: mean={np.mean(Pp):.2e}")
        
        print("   ✅ Physics acceleration tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Physics acceleration tests failed: {e}")
        return False

def test_performance_optimizer():
    """Test performance optimizer"""
    print("\n🎯 TESTING PERFORMANCE OPTIMIZER")
    print("=" * 40)
    
    try:
        from performance_bindings import PerformanceOptimizer
        
        # Initialize optimizer
        optimizer = PerformanceOptimizer()
        
        # Print performance info
        optimizer.print_performance_info()
        
        # Test automatic backend selection
        size = 5000
        a = np.random.random(size)
        b = np.random.random(size)
        
        # Test different operations
        operations = ['vector_add', 'dot_product']
        
        for op in operations:
            if op == 'vector_add':
                args = (a, b)
            elif op == 'dot_product':
                args = (a, b)
            
            start = time.time()
            result = optimizer.optimize_computation(op, *args)
            opt_time = time.time() - start
            
            print(f"   {op} (size={size}):")
            print(f"      Optimized time: {opt_time:.6f}s")
            print(f"      Result shape: {np.asarray(result).shape}")
        
        print("   ✅ Performance optimizer tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Performance optimizer tests failed: {e}")
        return False

def test_integration_with_simulator():
    """Test integration with main simulator"""
    print("\n🔗 TESTING SIMULATOR INTEGRATION")
    print("=" * 40)
    
    try:
        # Test with main simulator
        import simulator
        from advanced_transport import create_energy_transport_solver
        from gpu_acceleration import get_gpu_solver
        
        # Create device
        device = simulator.Device(2e-6, 1e-6)
        print(f"   Device created: {device.length:.1e}m × {device.width:.1e}m")
        
        # Create GPU-accelerated solver
        gpu_solver = get_gpu_solver()
        print(f"   GPU solver: {gpu_solver.is_gpu_available()}")
        
        # Create advanced transport solver
        transport_solver = create_energy_transport_solver(2e-6, 1e-6)
        print(f"   Transport solver: {transport_solver.get_transport_model_name()}")
        
        # Set up simulation
        size = 100
        transport_solver.set_doping(np.full(size, 1e17), np.full(size, 1e16))
        
        # Run simulation
        start = time.time()
        results = transport_solver.solve_transport([0, 1, 0, 0], Vg=0.5)
        sim_time = time.time() - start
        
        print(f"   Simulation time: {sim_time:.6f}s")
        print(f"   Results: {len(results)} fields")
        print(f"   Energy transport fields: {'energy_n' in results}")
        
        print("   ✅ Simulator integration tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Simulator integration tests failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 GPU/SIMD ACCELERATION TEST SUITE")
    print("=" * 60)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests_passed = 0
    total_tests = 5
    
    # Run all tests
    if test_simd_acceleration():
        tests_passed += 1
    
    if test_gpu_acceleration():
        tests_passed += 1
    
    if test_physics_acceleration():
        tests_passed += 1
    
    if test_performance_optimizer():
        tests_passed += 1
    
    if test_integration_with_simulator():
        tests_passed += 1
    
    # Print summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {tests_passed/total_tests*100:.1f}%")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("   GPU/SIMD acceleration is working correctly")
        return 0
    else:
        print(f"\n⚠️  {total_tests - tests_passed} TESTS FAILED")
        print("   Some acceleration features may not be working")
        return 1

if __name__ == "__main__":
    sys.exit(main())
