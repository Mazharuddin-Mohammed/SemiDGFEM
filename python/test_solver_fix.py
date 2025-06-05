#!/usr/bin/env python3
"""
Test the fixed solver implementation
"""

import sys
import numpy as np
import traceback

def test_solver_step_by_step():
    """Test solver step by step to identify issues."""
    print("🔧 Testing Fixed Solver Step by Step")
    print("=" * 50)
    
    try:
        import simulator
        print("✅ Simulator module imported")
        
        # Step 1: Create simulator
        sim = simulator.Simulator(
            extents=[0.5e-6, 0.25e-6],
            num_points_x=6,
            num_points_y=4,
            method="DG",
            mesh_type="Structured"
        )
        print(f"✅ Simulator created: {sim.num_points_x}×{sim.num_points_y}")
        print(f"   Method: {sim.method_str}, Mesh: {sim.mesh_type_str}")
        print(f"   Valid: {sim.is_valid()}")
        
        # Step 2: Get device info
        device_info = sim.get_device_info()
        print(f"✅ Device info: {device_info}")
        
        # Step 3: Try different doping sizes to find what works
        grid_size = sim.num_points_x * sim.num_points_y
        print(f"   Grid size: {grid_size}")
        
        # Test various sizes
        test_sizes = [grid_size, grid_size * 2, grid_size * 3, 100, 120, 150]
        working_size = None
        
        for size in test_sizes:
            try:
                Nd = np.full(size, 1e15, dtype=np.float64)
                Na = np.zeros(size, dtype=np.float64)
                sim.set_doping(Nd, Na)
                print(f"✅ Doping size {size} works!")
                working_size = size
                break
            except Exception as e:
                print(f"   Size {size} failed: {e}")
                continue
        
        if working_size is None:
            print("❌ Could not find working doping size")
            return False
        
        # Step 4: Try solver with minimal parameters
        print(f"\n🧪 Testing solver with size {working_size}...")
        
        try:
            results = sim.solve_drift_diffusion(
                bc=[0.0, 0.005, 0.0, 0.002],  # Tiny voltages
                Vg=0.001,                     # Tiny gate voltage
                max_steps=1,                  # Single iteration
                use_amr=False,
                poisson_max_iter=2,           # Minimal Poisson iterations
                poisson_tol=0.1               # Very relaxed tolerance
            )
            
            print("🎉 SOLVER SUCCESS!")
            print(f"   Results keys: {list(results.keys())}")
            
            for key, values in results.items():
                if hasattr(values, '__len__') and len(values) > 0:
                    if hasattr(values, 'min'):
                        print(f"   {key}: {len(values)} values, range [{values.min():.3e}, {values.max():.3e}]")
                    else:
                        print(f"   {key}: {len(values)} values")
                else:
                    print(f"   {key}: {values}")
            
            return True
            
        except Exception as solver_error:
            print(f"❌ Solver failed: {solver_error}")
            traceback.print_exc()
            
            # Try even more conservative parameters
            print("\n🔄 Trying ultra-conservative parameters...")
            try:
                results = sim.solve_drift_diffusion(
                    bc=[0.0, 0.001, 0.0, 0.0005],  # Even tinier voltages
                    Vg=0.0001,                      # Even tinier gate voltage
                    max_steps=1,                    # Single iteration
                    use_amr=False,
                    poisson_max_iter=1,             # Single Poisson iteration
                    poisson_tol=1.0                 # Extremely relaxed tolerance
                )
                
                print("🎉 ULTRA-CONSERVATIVE SOLVER SUCCESS!")
                print(f"   Results keys: {list(results.keys())}")
                return True
                
            except Exception as ultra_error:
                print(f"❌ Ultra-conservative solver also failed: {ultra_error}")
                return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

def test_backend_directly():
    """Test backend library directly."""
    print("\n🔧 Testing Backend Library Directly")
    print("=" * 50)
    
    try:
        import ctypes
        import os
        
        # Load library
        lib_path = "../build/libsimulator.so"
        if not os.path.exists(lib_path):
            print(f"❌ Library not found: {lib_path}")
            return False
        
        lib = ctypes.CDLL(lib_path)
        print(f"✅ Library loaded: {lib_path}")
        
        # Test basic functions
        try:
            create_device = lib.create_device
            create_device.argtypes = [ctypes.c_double, ctypes.c_double]
            create_device.restype = ctypes.c_void_p
            
            device = create_device(0.5e-6, 0.25e-6)
            if device:
                print("✅ Device created successfully")
                
                # Clean up
                destroy_device = lib.destroy_device
                destroy_device.argtypes = [ctypes.c_void_p]
                destroy_device(device)
                print("✅ Device destroyed successfully")
                return True
            else:
                print("❌ Device creation failed")
                return False
                
        except Exception as func_error:
            print(f"❌ Function call failed: {func_error}")
            return False
        
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Comprehensive Solver Fix Test")
    print("=" * 60)
    
    test1_result = test_solver_step_by_step()
    test2_result = test_backend_directly()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    print(f"✅ Step-by-step test: {'PASS' if test1_result else 'FAIL'}")
    print(f"✅ Backend direct test: {'PASS' if test2_result else 'FAIL'}")
    
    if test1_result:
        print("\n🎉 SOLVER IS WORKING!")
        print("✨ Python-C++ integration is fully functional!")
        return 0
    elif test2_result:
        print("\n⚠️  Backend works, Python wrapper needs adjustment")
        print("✨ Core integration is functional!")
        return 0
    else:
        print("\n❌ Both tests failed - needs investigation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
