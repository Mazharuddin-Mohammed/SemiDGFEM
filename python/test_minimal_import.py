#!/usr/bin/env python3
"""
Minimal test to debug Python import issues
"""

import sys
import os

def test_basic_import():
    """Test basic Python functionality before importing simulator"""
    print("1. Testing basic Python functionality...")
    import numpy as np
    print("   ✓ NumPy import successful")
    
    import logging
    print("   ✓ Logging import successful")
    
    print("2. Testing C extension loading...")
    try:
        # Try to load the extension directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("simulator", "simulator.cpython-312-x86_64-linux-gnu.so")
        if spec is None:
            print("   ✗ Could not create spec for simulator extension")
            return False
        
        print("   ✓ Extension spec created successfully")
        
        # Try to create the module
        module = importlib.util.module_from_spec(spec)
        print("   ✓ Module created from spec")
        
        # This is where it might hang - try to execute the module
        print("   Attempting to execute module...")
        sys.stdout.flush()
        
        spec.loader.exec_module(module)
        print("   ✓ Module executed successfully!")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error loading extension: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_library_dependencies():
    """Test if the underlying C++ library can be loaded"""
    print("3. Testing C++ library dependencies...")
    
    import subprocess
    try:
        # Check if the library exists and its dependencies
        result = subprocess.run(['ldd', '../build/libsimulator.so'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✓ C++ library dependencies resolved")
            missing = [line for line in result.stdout.split('\n') if 'not found' in line]
            if missing:
                print("   ⚠ Missing dependencies:")
                for dep in missing:
                    print(f"     {dep}")
            else:
                print("   ✓ All dependencies found")
        else:
            print(f"   ✗ Error checking dependencies: {result.stderr}")
            
    except Exception as e:
        print(f"   ✗ Error checking library: {e}")

def main():
    print("=== Minimal Import Test ===")
    print()
    
    # Test 1: Basic functionality
    test_basic_import()
    print()
    
    # Test 2: Library dependencies
    test_library_dependencies()
    print()
    
    print("=== Test Complete ===")
    print("If the test hangs, the issue is in module execution.")
    print("If it completes, the issue is in the simulator module initialization.")

if __name__ == "__main__":
    main()
