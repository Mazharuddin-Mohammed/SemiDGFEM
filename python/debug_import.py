#!/usr/bin/env python3
"""
Debug script to isolate Python import hanging issue
"""

import sys
import os
import signal
import traceback
import time

def timeout_handler(signum, frame):
    print("TIMEOUT: Import took too long, likely hanging")
    traceback.print_stack(frame)
    sys.exit(1)

def test_basic_imports():
    """Test basic Python functionality"""
    print("1. Testing basic Python imports...")
    
    try:
        import numpy as np
        print("   ✓ NumPy imported successfully")
    except Exception as e:
        print(f"   ✗ NumPy import failed: {e}")
        return False
    
    try:
        import ctypes
        print("   ✓ ctypes imported successfully")
    except Exception as e:
        print(f"   ✗ ctypes import failed: {e}")
        return False
    
    return True

def test_extension_loading():
    """Test loading the extension without importing"""
    print("2. Testing extension file access...")
    
    ext_file = "simulator.cpython-312-x86_64-linux-gnu.so"
    if os.path.exists(ext_file):
        print(f"   ✓ Extension file exists: {ext_file}")
        
        # Check file permissions
        if os.access(ext_file, os.R_OK):
            print("   ✓ Extension file is readable")
        else:
            print("   ✗ Extension file is not readable")
            return False
            
        # Check file size
        size = os.path.getsize(ext_file)
        print(f"   ✓ Extension file size: {size} bytes")
        
        return True
    else:
        print(f"   ✗ Extension file not found: {ext_file}")
        return False

def test_library_dependencies():
    """Test if the underlying C++ library can be loaded"""
    print("3. Testing C++ library dependencies...")
    
    try:
        import ctypes
        
        # Try to load the C++ library directly
        lib_path = "../build/libsimulator.so"
        if os.path.exists(lib_path):
            print(f"   ✓ C++ library exists: {lib_path}")
            
            try:
                lib = ctypes.CDLL(lib_path)
                print("   ✓ C++ library loaded successfully with ctypes")
                return True
            except Exception as e:
                print(f"   ✗ C++ library loading failed: {e}")
                return False
        else:
            print(f"   ✗ C++ library not found: {lib_path}")
            return False
            
    except Exception as e:
        print(f"   ✗ ctypes test failed: {e}")
        return False

def test_step_by_step_import():
    """Test importing the simulator module step by step"""
    print("4. Testing step-by-step import...")
    
    # Set timeout for import
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 second timeout
    
    try:
        print("   Attempting to import simulator module...")
        sys.stdout.flush()
        
        # Try importing
        import simulator
        
        print("   ✓ Import successful!")
        signal.alarm(0)  # Cancel timeout
        return True
        
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        print(f"   ✗ Import failed with exception: {e}")
        traceback.print_exc()
        return False

def test_alternative_import():
    """Test alternative import methods"""
    print("5. Testing alternative import methods...")
    
    try:
        import importlib.util
        
        # Try loading spec
        spec = importlib.util.spec_from_file_location(
            "simulator", 
            "simulator.cpython-312-x86_64-linux-gnu.so"
        )
        
        if spec is None:
            print("   ✗ Could not create module spec")
            return False
            
        print("   ✓ Module spec created")
        
        # Try creating module
        module = importlib.util.module_from_spec(spec)
        print("   ✓ Module object created")
        
        # This is where it might hang - try executing
        print("   Attempting to execute module...")
        sys.stdout.flush()
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout
        
        spec.loader.exec_module(module)
        
        signal.alarm(0)
        print("   ✓ Module executed successfully!")
        return True
        
    except Exception as e:
        signal.alarm(0)
        print(f"   ✗ Alternative import failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("=== Python Import Debug Session ===")
    print()
    
    # Run all tests
    tests = [
        test_basic_imports,
        test_extension_loading,
        test_library_dependencies,
        test_step_by_step_import,
        test_alternative_import
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"   ✗ Test failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    print("=== Debug Summary ===")
    test_names = [
        "Basic Python imports",
        "Extension file access", 
        "C++ library dependencies",
        "Step-by-step import",
        "Alternative import methods"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed! Import issue may be resolved.")
    else:
        print(f"\n⚠️  {sum(results)}/{len(results)} tests passed. Issues identified.")
        
        # Provide specific guidance
        if not results[0]:
            print("   → Fix basic Python environment issues first")
        elif not results[1]:
            print("   → Rebuild Python extension")
        elif not results[2]:
            print("   → Fix C++ library linking issues")
        elif not results[3] or not results[4]:
            print("   → Import hangs during module initialization")
            print("   → Likely causes: static constructor deadlock, PETSc init issues")

if __name__ == "__main__":
    main()
