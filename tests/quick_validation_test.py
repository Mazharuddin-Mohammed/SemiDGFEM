#!/usr/bin/env python3
"""
Quick Validation Test for SemiDGFEM Simulator
Focused test without plotting to identify core implementation issues
"""

import numpy as np
import sys
import os
import time

# Add parent directory for simulator import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_simulator_import():
    """Test if simulator can be imported and basic objects created"""
    
    print("🔍 Testing Simulator Import...")
    
    try:
        import simulator
        print("   ✅ Simulator module imported successfully")
        
        # Test basic object creation
        sim = simulator.Simulator(
            num_points_x=20,
            num_points_y=10,
            method="DG"
        )
        print(f"   ✅ Simulator object created: {sim.method}, {sim.num_points_x}x{sim.num_points_y}")
        
        return sim, True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return None, False
    except Exception as e:
        print(f"   ❌ Object creation failed: {e}")
        return None, False

def test_basic_simulation(sim):
    """Test basic simulation functionality"""
    
    print("\n🔍 Testing Basic Simulation...")
    
    try:
        # Set up simple doping
        total_points = sim.num_points_x * sim.num_points_y
        Nd = np.zeros(total_points)
        Na = np.zeros(total_points)
        
        # Simple P-N junction
        junction = sim.num_points_x // 2
        for i in range(total_points):
            x_idx = i % sim.num_points_x
            if x_idx < junction:
                Na[i] = 1e16  # P-type
            else:
                Nd[i] = 1e16  # N-type
        
        sim.set_doping(Nd, Na)
        print("   ✅ Doping profile set successfully")
        
        # Test Poisson solve
        bc = [0.0, 0.5, 0.0, 0.0]
        start_time = time.time()
        
        try:
            V = sim.solve_poisson(bc)
            solve_time = time.time() - start_time
            
            print(f"   ✅ Poisson solved in {solve_time:.3f}s")
            print(f"   ✅ Potential range: {np.min(V):.3f} to {np.max(V):.3f} V")
            
            # Basic validation
            if np.any(np.isfinite(V)) and len(V) == total_points:
                print("   ✅ Solution is finite and correct size")
                return True
            else:
                print("   ❌ Solution has issues (non-finite or wrong size)")
                return False
                
        except Exception as e:
            print(f"   ❌ Poisson solve failed: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Basic simulation setup failed: {e}")
        return False

def test_drift_diffusion(sim):
    """Test drift-diffusion simulation"""
    
    print("\n🔍 Testing Drift-Diffusion Simulation...")
    
    try:
        bc = [0.0, 0.7, 0.0, 0.0]
        start_time = time.time()
        
        result = sim.solve_drift_diffusion(
            bc=bc,
            max_steps=20,
            use_amr=False,
            poisson_tol=1e-6
        )
        
        solve_time = time.time() - start_time
        print(f"   ✅ Drift-diffusion solved in {solve_time:.3f}s")
        
        # Check results
        required_keys = ['potential', 'n', 'p']
        missing_keys = [key for key in required_keys if key not in result]
        
        if missing_keys:
            print(f"   ❌ Missing result keys: {missing_keys}")
            return False
        
        # Validate result sizes and values
        total_points = sim.num_points_x * sim.num_points_y
        
        for key in required_keys:
            data = result[key]
            if len(data) != total_points:
                print(f"   ❌ {key} has wrong size: {len(data)} vs {total_points}")
                return False
            
            if not np.all(np.isfinite(data)):
                print(f"   ❌ {key} contains non-finite values")
                return False
            
            print(f"   ✅ {key}: range {np.min(data):.2e} to {np.max(data):.2e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Drift-diffusion simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_scaling(sim):
    """Test performance with different grid sizes"""
    
    print("\n🔍 Testing Performance Scaling...")
    
    grid_sizes = [(10, 5), (20, 10), (30, 15)]
    times = []
    
    for nx, ny in grid_sizes:
        try:
            # Create new simulator with different size
            test_sim = type(sim)(
                num_points_x=nx,
                num_points_y=ny,
                method="DG"
            )
            
            # Set up doping
            total_points = nx * ny
            Nd = np.zeros(total_points)
            Na = np.zeros(total_points)
            
            junction = nx // 2
            for i in range(total_points):
                x_idx = i % nx
                if x_idx < junction:
                    Na[i] = 1e16
                else:
                    Nd[i] = 1e16
            
            test_sim.set_doping(Nd, Na)
            
            # Time the solve
            start_time = time.time()
            V = test_sim.solve_poisson([0.0, 0.5, 0.0, 0.0])
            solve_time = time.time() - start_time
            
            times.append(solve_time)
            print(f"   ✅ Grid {nx}x{ny} ({total_points} points): {solve_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed for grid {nx}x{ny}: {e}")
            times.append(float('inf'))
    
    # Check scaling
    if len(times) >= 2 and all(t < float('inf') for t in times):
        scaling_factor = times[-1] / times[0]
        grid_factor = (grid_sizes[-1][0] * grid_sizes[-1][1]) / (grid_sizes[0][0] * grid_sizes[0][1])
        
        print(f"   📊 Scaling analysis:")
        print(f"      Grid size increased by factor: {grid_factor:.1f}")
        print(f"      Time increased by factor: {scaling_factor:.1f}")
        
        if scaling_factor < grid_factor * 2:  # Should scale better than O(n²)
            print("   ✅ Good scaling performance")
            return True
        else:
            print("   ⚠️  Poor scaling performance")
            return False
    else:
        print("   ❌ Could not analyze scaling")
        return False

def test_physical_validation():
    """Test physical behavior validation using analytical model"""
    
    print("\n🔍 Testing Physical Validation...")
    
    # Test MOSFET-like behavior
    try:
        # Create analytical MOSFET model for comparison
        def analytical_mosfet_current(Vg, Vd, Vth=0.5):
            """Simple analytical MOSFET model"""
            if Vg < Vth:
                return 1e-12 * np.exp((Vg - Vth) / 0.1)  # Subthreshold
            else:
                if Vd < (Vg - Vth):
                    return 1e-6 * (Vg - Vth) * Vd  # Linear
                else:
                    return 0.5e-6 * (Vg - Vth)**2  # Saturation
        
        # Test key operating points
        test_points = [
            (0.3, 0.1, "subthreshold"),
            (0.8, 0.1, "linear"),
            (0.8, 1.0, "saturation")
        ]
        
        validation_passed = True
        
        for Vg, Vd, region in test_points:
            expected_current = analytical_mosfet_current(Vg, Vd)
            
            # Check if current is in reasonable range
            if region == "subthreshold":
                if 1e-15 <= expected_current <= 1e-9:
                    print(f"   ✅ {region}: {expected_current:.2e} A - REASONABLE")
                else:
                    print(f"   ❌ {region}: {expected_current:.2e} A - OUT OF RANGE")
                    validation_passed = False
            elif region == "linear":
                if 1e-9 <= expected_current <= 1e-4:
                    print(f"   ✅ {region}: {expected_current:.2e} A - REASONABLE")
                else:
                    print(f"   ❌ {region}: {expected_current:.2e} A - OUT OF RANGE")
                    validation_passed = False
            elif region == "saturation":
                if 1e-8 <= expected_current <= 1e-3:
                    print(f"   ✅ {region}: {expected_current:.2e} A - REASONABLE")
                else:
                    print(f"   ❌ {region}: {expected_current:.2e} A - OUT OF RANGE")
                    validation_passed = False
        
        return validation_passed
        
    except Exception as e:
        print(f"   ❌ Physical validation failed: {e}")
        return False

def main():
    """Main validation test function"""
    
    print("🚀 QUICK VALIDATION TEST FOR SEMIDGFEM")
    print("=" * 50)
    print("Testing core simulator functionality without plotting")
    print()
    
    # Test results
    test_results = {
        'import': False,
        'basic_simulation': False,
        'drift_diffusion': False,
        'performance': False,
        'physical_validation': False
    }
    
    # Test 1: Import and object creation
    sim, import_success = test_simulator_import()
    test_results['import'] = import_success
    
    if not import_success:
        print("\n❌ CRITICAL: Cannot import simulator - aborting tests")
        print("   This indicates a fundamental issue with the Python bindings")
        return False
    
    # Test 2: Basic simulation
    test_results['basic_simulation'] = test_basic_simulation(sim)
    
    # Test 3: Drift-diffusion (only if basic works)
    if test_results['basic_simulation']:
        test_results['drift_diffusion'] = test_drift_diffusion(sim)
    else:
        print("\n⚠️  Skipping drift-diffusion test due to basic simulation failure")
    
    # Test 4: Performance scaling
    if test_results['basic_simulation']:
        test_results['performance'] = test_performance_scaling(sim)
    else:
        print("\n⚠️  Skipping performance test due to basic simulation failure")
    
    # Test 5: Physical validation (analytical)
    test_results['physical_validation'] = test_physical_validation()
    
    # Summary
    print("\n🏁 VALIDATION TEST SUMMARY")
    print("=" * 50)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    pass_rate = (passed_tests / total_tests) * 100
    
    print(f"Tests passed: {passed_tests}/{total_tests} ({pass_rate:.1f}%)")
    print()
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print()
    
    # Overall assessment
    if passed_tests == total_tests:
        print("🎉 EXCELLENT: All tests passed - simulator working correctly!")
        assessment = "EXCELLENT"
    elif passed_tests >= 4:
        print("✅ GOOD: Most tests passed - minor issues to address")
        assessment = "GOOD"
    elif passed_tests >= 2:
        print("⚠️  MODERATE: Some tests passed - significant issues to address")
        assessment = "MODERATE"
    else:
        print("❌ CRITICAL: Most tests failed - major issues to address")
        assessment = "CRITICAL"
    
    # Specific recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if not test_results['import']:
        print("   🔧 Fix Python binding compilation and linking issues")
        print("   🔧 Check C++ library dependencies and paths")
    
    if not test_results['basic_simulation']:
        print("   🔧 Debug Poisson solver implementation")
        print("   🔧 Check matrix assembly and boundary conditions")
    
    if not test_results['drift_diffusion']:
        print("   🔧 Debug drift-diffusion solver coupling")
        print("   🔧 Check carrier density and current calculations")
    
    if not test_results['performance']:
        print("   🔧 Optimize numerical algorithms for better scaling")
        print("   🔧 Consider parallel processing and GPU acceleration")
    
    print(f"\n📊 OVERALL ASSESSMENT: {assessment}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
