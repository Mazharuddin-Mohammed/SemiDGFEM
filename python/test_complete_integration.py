#!/usr/bin/env python3
"""
Comprehensive Python-C++ Integration Test for SemiDGFEM

This test validates the complete integration between the Python frontend
and the C++ backend, including all Cython bindings and high-level API.
"""

import sys
import os
import numpy as np
import time
import traceback

def test_basic_import():
    """Test basic module import."""
    print("=== Testing Basic Import ===")
    
    try:
        import simulator
        print("✅ Basic simulator import successful")
        return True
    except Exception as e:
        print(f"❌ Basic import failed: {e}")
        traceback.print_exc()
        return False

def test_cython_classes():
    """Test individual Cython wrapper classes."""
    print("\n=== Testing Cython Classes ===")
    
    try:
        import simulator
        
        # Test Device creation
        device = simulator.Device(1e-6, 0.5e-6)
        print(f"✅ Device created: {device.length:.2e} × {device.width:.2e} m")
        
        # Test device methods
        epsilon = device.get_epsilon_at(0.5e-6, 0.25e-6)
        extents = device.get_extents()
        print(f"✅ Device methods: ε={epsilon:.2e}, extents={extents}")
        
        # Test Mesh creation
        mesh = simulator.Mesh(device, "Structured")
        num_nodes = mesh.get_num_nodes()
        num_elements = mesh.get_num_elements()
        print(f"✅ Mesh created: {num_nodes} nodes, {num_elements} elements")
        
        # Test PoissonSolver creation
        poisson = simulator.PoissonSolver(device, "DG", "Structured")
        print(f"✅ Poisson solver created: valid={poisson.is_valid()}")
        print(f"   DOF count: {poisson.get_dof_count()}")
        
        # Test DriftDiffusionSolver creation
        dd = simulator.DriftDiffusionSolver(device, "DG", "Structured", 3)
        print(f"✅ Drift-diffusion solver created: valid={dd.is_valid()}")
        print(f"   DOF count: {dd.get_dof_count()}, Order: P{dd.get_order()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cython classes test failed: {e}")
        traceback.print_exc()
        return False

def test_high_level_api():
    """Test high-level Python API."""
    print("\n=== Testing High-Level API ===")
    
    try:
        import semidgfem
        
        # Test SemiDGFEM class creation
        sim = semidgfem.SemiDGFEM(1e-6, 0.5e-6, "DG", "Structured", 3)
        print("✅ SemiDGFEM instance created")
        
        # Test solver info
        info = sim.get_solver_info()
        print(f"✅ Solver info: {info['method']}, DOF={info['dof_count']}")
        
        # Test mesh info
        mesh_info = sim.get_mesh_info()
        print(f"✅ Mesh info: {mesh_info['num_nodes']} nodes, {mesh_info['num_elements']} elements")
        
        # Test doping setup
        sim.set_uniform_doping(1e16, 0.0)
        print("✅ Uniform doping set")
        
        # Test MOSFET doping
        sim.create_mosfet_doping(1e16, 1e20, 0.4)
        print("✅ MOSFET doping profile created")
        
        return True
        
    except Exception as e:
        print(f"❌ High-level API test failed: {e}")
        traceback.print_exc()
        return False

def test_simulation_workflow():
    """Test complete simulation workflow."""
    print("\n=== Testing Simulation Workflow ===")
    
    try:
        import semidgfem
        
        # Create simulator
        sim = semidgfem.SemiDGFEM(1e-6, 0.5e-6, "DG", "Structured", 2)  # Use P2 for faster testing
        sim.set_uniform_doping(1e16, 0.0)
        
        # Test equilibrium solve
        start_time = time.time()
        eq_results = sim.solve_equilibrium([0.0, 0.0, 0.0, 0.0])
        eq_time = time.time() - start_time
        
        print(f"✅ Equilibrium solved in {eq_time:.3f}s")
        print(f"   Potential range: [{eq_results['potential'].min():.3f}, {eq_results['potential'].max():.3f}] V")
        print(f"   DOF count: {eq_results['dof_count']}")
        
        # Test bias point solve
        start_time = time.time()
        bias_results = sim.solve_bias_point(gate_voltage=1.0, drain_voltage=0.5, max_iterations=10)
        bias_time = time.time() - start_time
        
        print(f"✅ Bias point solved in {bias_time:.3f}s")
        print(f"   Potential range: [{bias_results['potential'].min():.3f}, {bias_results['potential'].max():.3f}] V")
        print(f"   Electron density range: [{bias_results['n'].min():.2e}, {bias_results['n'].max():.2e}] cm⁻³")
        print(f"   Convergence residual: {bias_results['convergence_residual']:.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation workflow test failed: {e}")
        traceback.print_exc()
        return False

def test_convenience_functions():
    """Test convenience functions."""
    print("\n=== Testing Convenience Functions ===")
    
    try:
        import semidgfem
        
        # Test MOSFET simulator creation
        mosfet_sim = semidgfem.create_mosfet_simulator(1e-6, 0.5e-6, "DG", 2)
        print("✅ MOSFET simulator created via convenience function")
        
        # Test quick equilibrium solve
        quick_results = semidgfem.quick_equilibrium_solve(1e-6, 0.5e-6)
        print(f"✅ Quick equilibrium solve: {len(quick_results['potential'])} DOFs")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience functions test failed: {e}")
        traceback.print_exc()
        return False

def test_legacy_compatibility():
    """Test legacy Simulator class compatibility."""
    print("\n=== Testing Legacy Compatibility ===")
    
    try:
        import simulator
        
        # Test legacy Simulator class
        legacy_sim = simulator.Simulator(
            dimension="TwoD", 
            extents=[1e-6, 0.5e-6], 
            num_points_x=20, 
            num_points_y=15,
            method="DG", 
            mesh_type="Structured"
        )
        print("✅ Legacy Simulator created")
        
        # Test legacy methods
        doping_size = legacy_sim.get_drift_diffusion_solver().get_dof_count()
        Nd = np.full(doping_size, 1e16, dtype=np.float64)
        Na = np.zeros(doping_size, dtype=np.float64)
        
        legacy_sim.set_doping(Nd, Na)
        print("✅ Legacy doping set")
        
        # Test legacy solve
        bc = [0.0, 0.0, 0.0, 0.0]
        potential = legacy_sim.solve_poisson(bc)
        print(f"✅ Legacy Poisson solve: {len(potential)} DOFs")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_benchmark():
    """Test performance of Python-C++ integration."""
    print("\n=== Performance Benchmark ===")
    
    try:
        import semidgfem
        
        # Create different sized problems
        test_cases = [
            (0.5e-6, 0.25e-6, "Small"),
            (1e-6, 0.5e-6, "Medium"),
            (2e-6, 1e-6, "Large")
        ]
        
        for length, width, size_name in test_cases:
            sim = semidgfem.SemiDGFEM(length, width, "DG", "Structured", 2)
            sim.set_uniform_doping(1e16, 0.0)
            
            # Time equilibrium solve
            start_time = time.time()
            results = sim.solve_equilibrium()
            solve_time = time.time() - start_time
            
            dof_count = results['dof_count']
            throughput = dof_count / solve_time if solve_time > 0 else 0
            
            print(f"✅ {size_name} problem: {dof_count} DOFs in {solve_time:.3f}s ({throughput:.0f} DOF/s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🔬 SemiDGFEM Python-C++ Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Cython Classes", test_cython_classes),
        ("High-Level API", test_high_level_api),
        ("Simulation Workflow", test_simulation_workflow),
        ("Convenience Functions", test_convenience_functions),
        ("Legacy Compatibility", test_legacy_compatibility),
        ("Performance Benchmark", test_performance_benchmark)
    ]
    
    results = []
    total_start = time.time()
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            success = test_func()
            test_time = time.time() - start_time
            results.append((test_name, success, test_time))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False, 0.0))
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, test_time in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name:<25} ({test_time:.3f}s)")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    print(f"⏱️  Total time: {total_time:.3f}s")
    
    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✨ Python-C++ integration is fully functional!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed - integration needs attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
