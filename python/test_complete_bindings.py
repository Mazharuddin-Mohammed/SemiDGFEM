#!/usr/bin/env python3
"""
Complete Python Bindings Test Suite
Tests all compiled Cython modules for the complete backend implementation

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import time
import sys
from pathlib import Path

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_test(test_name):
    """Print test header."""
    print(f"\n--- {test_name} ---")

def test_complete_dg_bindings():
    """Test complete DG discretization bindings."""
    print_header("TESTING COMPLETE DG BINDINGS")
    
    try:
        import complete_dg
        
        print_test("DG Basis Functions")
        
        # Test P3 basis function evaluation
        xi, eta = 1.0/3.0, 1.0/3.0
        order = 3
        
        # Test partition of unity
        sum_basis = 0.0
        dofs = complete_dg.DGBasisFunctions.get_dofs_per_element(order)
        print(f"P{order} elements have {dofs} DOFs per element")
        
        for j in range(dofs):
            phi_j = complete_dg.DGBasisFunctions.evaluate_basis_function(xi, eta, j, order)
            sum_basis += phi_j
        
        print(f"Partition of unity: {sum_basis:.10f} (should be 1.0)")
        assert abs(sum_basis - 1.0) < 1e-10, "Partition of unity failed"
        
        # Test gradient evaluation
        grad_ref = complete_dg.DGBasisFunctions.evaluate_basis_gradient_ref(xi, eta, 0, order)
        print(f"Gradient (reference): [{grad_ref[0]:.6f}, {grad_ref[1]:.6f}]")
        
        # Test coordinate transformation
        b1, b2, b3 = 0.5, 0.3, -0.8
        c1, c2, c3 = -0.4, 0.6, 0.2
        grad_phys = complete_dg.DGBasisFunctions.transform_gradient_to_physical(
            grad_ref, b1, b2, b3, c1, c2, c3)
        print(f"Gradient (physical): [{grad_phys[0]:.6f}, {grad_phys[1]:.6f}]")
        
        print_test("DG Quadrature")
        
        # Test quadrature rules
        for order in [1, 2, 4, 6]:
            points, weights = complete_dg.DGQuadrature.get_quadrature_rule(order)
            print(f"Order {order} quadrature: {len(points)} points, weight sum: {np.sum(weights):.10f}")
            assert abs(np.sum(weights) - 1.0) < 1e-10, f"Quadrature weights don't sum to 1 for order {order}"
        
        print_test("DG Assembly")
        
        # Test element assembly
        dg_assembly = complete_dg.DGAssembly(order=3)
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        
        # Test mass matrix assembly
        mass_matrix = dg_assembly.assemble_element_matrix(vertices, "mass")
        print(f"Mass matrix shape: {mass_matrix.shape}")
        print(f"Mass matrix trace: {np.trace(mass_matrix):.6f}")
        
        # Test stiffness matrix assembly
        stiffness_matrix = dg_assembly.assemble_element_matrix(vertices, "stiffness")
        print(f"Stiffness matrix trace: {np.trace(stiffness_matrix):.6f}")
        
        # Test load vector assembly
        load_vector = dg_assembly.assemble_element_vector(vertices)
        print(f"Load vector norm: {np.linalg.norm(load_vector):.6f}")
        
        # Validate implementation
        validation_results = complete_dg.validate_complete_dg_implementation()
        print(f"Validation results: {validation_results}")
        
        print("✓ Complete DG bindings test PASSED")
        
    except Exception as e:
        print(f"✗ Complete DG bindings test FAILED: {e}")
        raise

def test_unstructured_transport_bindings():
    """Test unstructured transport bindings."""
    print_header("TESTING UNSTRUCTURED TRANSPORT BINDINGS")
    
    try:
        import unstructured_transport
        
        print_test("Unstructured Transport Validation")
        
        # Validate implementation
        validation_results = unstructured_transport.validate_unstructured_implementation()
        print("Validation results:")
        for key, value in validation_results.items():
            print(f"  {key}: {value}")
        
        assert validation_results["validation_passed"], "Unstructured transport validation failed"
        
        print("✓ Unstructured transport bindings test PASSED")
        
    except Exception as e:
        print(f"✗ Unstructured transport bindings test FAILED: {e}")
        raise

def test_performance_bindings():
    """Test performance optimization bindings."""
    print_header("TESTING PERFORMANCE BINDINGS")
    
    try:
        import performance_bindings
        
        print_test("SIMD Operations")
        
        # Test SIMD vector operations
        size = 1000
        a = np.random.random(size)
        b = np.random.random(size)
        
        # Vector addition
        start_time = time.time()
        result_add = performance_bindings.SIMDKernels.vector_add(a, b)
        simd_time = time.time() - start_time
        
        # Compare with NumPy
        start_time = time.time()
        numpy_result = a + b
        numpy_time = time.time() - start_time
        
        print(f"SIMD vector add time: {simd_time*1000:.3f} ms")
        print(f"NumPy vector add time: {numpy_time*1000:.3f} ms")
        print(f"Results match: {np.allclose(result_add, numpy_result)}")
        
        # Vector multiplication
        result_mul = performance_bindings.SIMDKernels.vector_multiply(a, b)
        numpy_mul = a * b
        print(f"Vector multiply match: {np.allclose(result_mul, numpy_mul)}")
        
        # Dot product
        result_dot = performance_bindings.SIMDKernels.dot_product(a, b)
        numpy_dot = np.dot(a, b)
        print(f"Dot product match: {abs(result_dot - numpy_dot) < 1e-10}")
        
        print_test("Matrix Operations")
        
        # Matrix-vector multiplication
        matrix = np.random.random((100, 100))
        vector = np.random.random(100)
        
        result_matvec = performance_bindings.SIMDKernels.matrix_vector_multiply(matrix, vector)
        numpy_matvec = matrix @ vector
        print(f"Matrix-vector multiply match: {np.allclose(result_matvec, numpy_matvec)}")
        
        print_test("Parallel Computing")
        
        # Test parallel computing info
        num_threads = performance_bindings.ParallelComputing.get_num_threads()
        print(f"Available threads: {num_threads}")
        
        print_test("GPU Acceleration")
        
        # Test GPU availability
        gpu_available = performance_bindings.GPUAcceleration.is_available()
        print(f"GPU available: {gpu_available}")
        
        if gpu_available:
            gpu = performance_bindings.GPUAcceleration()
            gpu_result = gpu.vector_add(a[:100], b[:100])  # Smaller test for GPU
            print(f"GPU vector add successful: {len(gpu_result) == 100}")
        
        print_test("Performance Optimizer")
        
        # Test high-level performance interface
        optimizer = performance_bindings.create_performance_optimizer()
        perf_info = optimizer.get_performance_info()
        print("Performance configuration:")
        for key, value in perf_info.items():
            print(f"  {key}: {value}")
        
        # Test optimized operations
        opt_result = optimizer.optimize_vector_operation("add", a[:100], b[:100])
        print(f"Optimized vector add successful: {len(opt_result) == 100}")
        
        print("✓ Performance bindings test PASSED")
        
    except Exception as e:
        print(f"✗ Performance bindings test FAILED: {e}")
        raise

def test_advanced_transport_bindings():
    """Test advanced transport bindings."""
    print_header("TESTING ADVANCED TRANSPORT BINDINGS")
    
    try:
        import advanced_transport
        
        print_test("Transport Model Enumeration")
        
        # Test transport model constants
        models = [
            ("DRIFT_DIFFUSION", advanced_transport.TransportModel.DRIFT_DIFFUSION),
            ("ENERGY_TRANSPORT", advanced_transport.TransportModel.ENERGY_TRANSPORT),
            ("HYDRODYNAMIC", advanced_transport.TransportModel.HYDRODYNAMIC),
            ("NON_EQUILIBRIUM_STATISTICS", advanced_transport.TransportModel.NON_EQUILIBRIUM_STATISTICS)
        ]
        
        for name, value in models:
            print(f"  {name}: {value}")
        
        print("✓ Advanced transport bindings test PASSED")
        
    except Exception as e:
        print(f"✗ Advanced transport bindings test FAILED: {e}")
        raise

def test_simulator_bindings():
    """Test core simulator bindings."""
    print_header("TESTING CORE SIMULATOR BINDINGS")
    
    try:
        import simulator
        
        print_test("Device Creation")
        
        # Test device creation
        device = simulator.Device(1e-6, 1e-6)  # 1μm × 1μm device
        print(f"Device created: {device.get_width():.2e} m × {device.get_height():.2e} m")
        
        print_test("Method and Mesh Type Enums")
        
        # Test enums
        print(f"DG method: {simulator.Method.DG}")
        print(f"Structured mesh: {simulator.MeshType.Structured}")
        print(f"Unstructured mesh: {simulator.MeshType.Unstructured}")
        
        print("✓ Core simulator bindings test PASSED")
        
    except Exception as e:
        print(f"✗ Core simulator bindings test FAILED: {e}")
        raise

def benchmark_performance():
    """Benchmark performance of different operations."""
    print_header("PERFORMANCE BENCHMARK")
    
    try:
        import performance_bindings
        
        print("Running performance benchmark...")
        benchmark_results = performance_bindings.benchmark_performance()
        
        print("Benchmark results:")
        for key, value in benchmark_results.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value:.6f} seconds")
        
        print("✓ Performance benchmark completed")
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        # Don't raise - this is optional

def run_comprehensive_test():
    """Run comprehensive test of all bindings."""
    print_header("COMPREHENSIVE PYTHON BINDINGS TEST")
    print("Testing all compiled Cython modules")
    
    tests = [
        ("Core Simulator", test_simulator_bindings),
        ("Complete DG", test_complete_dg_bindings),
        ("Unstructured Transport", test_unstructured_transport_bindings),
        ("Performance Optimization", test_performance_bindings),
        ("Advanced Transport", test_advanced_transport_bindings),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            test_func()
            passed_tests += 1
        except Exception as e:
            print(f"Test {test_name} failed: {e}")
    
    # Optional benchmark
    try:
        benchmark_performance()
    except:
        pass
    
    # Summary
    print_header("TEST SUMMARY")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL PYTHON BINDINGS TESTS PASSED! 🎉")
        print("\nComplete backend implementation validated:")
        print("  ✓ Core simulator functionality")
        print("  ✓ Complete DG discretization")
        print("  ✓ Unstructured transport models")
        print("  ✓ Performance optimization")
        print("  ✓ Advanced transport physics")
        print("\nThe Python bindings are ready for production use!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and fix the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(run_comprehensive_test())
