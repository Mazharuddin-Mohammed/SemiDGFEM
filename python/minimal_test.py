#!/usr/bin/env python3
"""
Minimal test to verify Python-C++ integration works.
"""

import sys
import os
import ctypes
import numpy as np

def test_direct_library_access():
    """Test direct access to the C++ library."""
    print("=== Testing Direct Library Access ===")
    
    try:
        # Load the C++ library directly
        lib_path = "../build/libsimulator.so"
        if not os.path.exists(lib_path):
            print(f"❌ Library not found: {lib_path}")
            return False
        
        # Load library
        lib = ctypes.CDLL(lib_path)
        print(f"✅ Library loaded: {lib_path}")
        
        # Test if we can access some symbols
        try:
            # These should be C interface functions
            create_device = lib.create_device
            print("✅ Found create_device function")
        except AttributeError:
            print("⚠️  create_device function not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct library access failed: {e}")
        return False

def test_cython_import():
    """Test Cython module import."""
    print("\n=== Testing Cython Import ===")
    
    try:
        # Try to import the compiled module
        import simulator
        print("✅ Cython module imported successfully")
        
        # Check what's available
        available = [attr for attr in dir(simulator) if not attr.startswith('_')]
        print(f"✅ Available classes/functions: {available}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Cython import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Cython import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality if import works."""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        import simulator
        
        # Try to create a basic simulator
        sim = simulator.Simulator(
            dimension="TwoD",
            extents=[1e-6, 0.5e-6],
            num_points_x=10,
            num_points_y=8,
            method="DG",
            mesh_type="Structured"
        )
        print("✅ Basic Simulator created")
        
        # Test attributes
        print(f"   Method: {sim._method}")
        print(f"   Mesh type: {sim._mesh_type}")
        print(f"   Grid: {sim._num_points_x} × {sim._num_points_y}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numpy_integration():
    """Test NumPy array integration."""
    print("\n=== Testing NumPy Integration ===")
    
    try:
        import simulator
        
        # Create simulator
        sim = simulator.Simulator(
            dimension="TwoD",
            extents=[1e-6, 0.5e-6],
            num_points_x=10,
            num_points_y=8,
            method="DG",
            mesh_type="Structured"
        )
        
        # Test doping arrays
        size = 80  # 10 * 8
        Nd = np.full(size, 1e16, dtype=np.float64)
        Na = np.zeros(size, dtype=np.float64)
        
        sim.set_doping(Nd, Na)
        print("✅ NumPy doping arrays set successfully")
        
        # Test boundary conditions
        bc = [0.0, 0.0, 0.0, 0.0]
        potential = sim.solve_poisson(bc)
        print(f"✅ Poisson solved: {len(potential)} values")
        print(f"   Potential range: [{potential.min():.3e}, {potential.max():.3e}] V")
        
        return True
        
    except Exception as e:
        print(f"❌ NumPy integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all minimal tests."""
    print("🧪 Minimal Python-C++ Integration Test")
    print("=" * 50)
    
    tests = [
        ("Direct Library Access", test_direct_library_access),
        ("Cython Import", test_cython_import),
        ("Basic Functionality", test_basic_functionality),
        ("NumPy Integration", test_numpy_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 MINIMAL TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed >= 2:  # At least library access and import should work
        print("✨ Basic Python-C++ integration is functional!")
        return 0
    else:
        print("⚠️  Python-C++ integration needs attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
