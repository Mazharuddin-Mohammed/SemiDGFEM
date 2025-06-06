#!/usr/bin/env python3
"""
Complete Cython Compilation Script for SemiDGFEM
Compiles all Python bindings for the complete backend implementation

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step, description):
    """Print formatted step."""
    print(f"\n[{step}] {description}")
    print("-" * 40)

def check_prerequisites():
    """Check if all prerequisites are available."""
    print_step("1", "Checking Prerequisites")
    
    # Check Python version
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required")
    print(f"✓ Python {sys.version}")
    
    # Check required packages
    required_packages = ['numpy', 'cython', 'setuptools']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            raise RuntimeError(f"Required package {package} not found. Install with: pip install {package}")
    
    # Check build directory
    build_dir = Path("../build")
    if not build_dir.exists():
        raise RuntimeError("Build directory not found. Run 'make' in the root directory first.")
    print(f"✓ Build directory: {build_dir.absolute()}")
    
    # Check library
    lib_path = build_dir / "libsimulator.so"
    if not lib_path.exists():
        raise RuntimeError("libsimulator.so not found. Run 'make' in the root directory first.")
    print(f"✓ Library: {lib_path}")
    
    # Check include directory
    include_dir = Path("../include")
    if not include_dir.exists():
        raise RuntimeError("Include directory not found.")
    print(f"✓ Include directory: {include_dir.absolute()}")

def clean_previous_builds():
    """Clean previous build artifacts."""
    print_step("2", "Cleaning Previous Builds")
    
    # Directories to clean
    clean_dirs = ['build', 'dist', '__pycache__']
    for dir_name in clean_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✓ Removed {dir_name}/")
    
    # Files to clean
    clean_patterns = ['*.c', '*.cpp', '*.so', '*.pyd', '*.egg-info']
    for pattern in clean_patterns:
        for file_path in Path('.').glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                print(f"✓ Removed {file_path}")

def compile_cython_modules():
    """Compile all Cython modules."""
    print_step("3", "Compiling Cython Modules")
    
    # List of modules to compile
    modules = [
        "simulator.pyx",
        "advanced_transport.pyx", 
        "complete_dg.pyx",
        "unstructured_transport.pyx",
        "performance_bindings.pyx"
    ]
    
    # Check that all modules exist
    for module in modules:
        if not Path(module).exists():
            raise RuntimeError(f"Module {module} not found")
        print(f"✓ Found {module}")
    
    # Run setup.py build_ext
    print("\nRunning Cython compilation...")
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, "setup.py", "build_ext", "--inplace"
        ], capture_output=True, text=True, check=True)
        
        print("✓ Compilation successful")
        print(f"  Compilation time: {time.time() - start_time:.2f} seconds")
        
        if result.stdout:
            print("\nCompilation output:")
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        print("✗ Compilation failed")
        print(f"Error code: {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        raise

def test_compiled_modules():
    """Test that compiled modules can be imported."""
    print_step("4", "Testing Compiled Modules")
    
    modules_to_test = [
        ("simulator", "Basic simulator functionality"),
        ("advanced_transport", "Advanced transport models"),
        ("complete_dg", "Complete DG discretization"),
        ("unstructured_transport", "Unstructured transport models"),
        ("performance_bindings", "Performance optimization")
    ]
    
    for module_name, description in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"✓ {module_name}: {description}")
            
            # Test basic functionality if available
            if hasattr(module, '__version__'):
                print(f"  Version: {module.__version__}")
            
        except ImportError as e:
            print(f"✗ {module_name}: Import failed - {e}")
            raise

def validate_complete_implementation():
    """Validate the complete implementation."""
    print_step("5", "Validating Complete Implementation")
    
    try:
        # Test DG basis functions
        import complete_dg
        validation_results = complete_dg.validate_complete_dg_implementation()
        print("✓ DG basis functions validation:")
        for key, value in validation_results.items():
            print(f"  {key}: {value}")
        
        # Test unstructured transport
        import unstructured_transport
        unstructured_results = unstructured_transport.validate_unstructured_implementation()
        print("✓ Unstructured transport validation:")
        for key, value in unstructured_results.items():
            print(f"  {key}: {value}")
        
        # Test performance bindings
        import performance_bindings
        perf_info = performance_bindings.create_performance_optimizer().get_performance_info()
        print("✓ Performance optimization validation:")
        for key, value in perf_info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        raise

def create_installation_package():
    """Create installation package."""
    print_step("6", "Creating Installation Package")
    
    try:
        # Build wheel
        result = subprocess.run([
            sys.executable, "setup.py", "bdist_wheel"
        ], capture_output=True, text=True, check=True)
        
        print("✓ Wheel package created")
        
        # List created files
        dist_dir = Path("dist")
        if dist_dir.exists():
            for file_path in dist_dir.glob("*.whl"):
                print(f"  Created: {file_path}")
                
    except subprocess.CalledProcessError as e:
        print("✗ Package creation failed")
        print(f"Error: {e.stderr}")
        # Don't raise - this is optional

def print_summary():
    """Print compilation summary."""
    print_header("COMPILATION SUMMARY")
    
    print("✓ Complete Cython compilation successful!")
    print("\nCompiled modules:")
    print("  • simulator - Core device simulation")
    print("  • advanced_transport - Advanced transport models")
    print("  • complete_dg - Complete DG discretization")
    print("  • unstructured_transport - Unstructured mesh support")
    print("  • performance_bindings - SIMD/GPU optimization")
    
    print("\nBackend implementation status:")
    print("  ✓ Structured DG discretization")
    print("  ✓ Unstructured DG discretization")
    print("  ✓ Complete P1/P2/P3 basis functions")
    print("  ✓ Energy transport models")
    print("  ✓ Hydrodynamic transport models")
    print("  ✓ Non-equilibrium drift-diffusion")
    print("  ✓ Performance optimization")
    
    print("\nUsage:")
    print("  import simulator")
    print("  import advanced_transport")
    print("  import complete_dg")
    print("  import unstructured_transport")
    print("  import performance_bindings")
    
    print("\nNext steps:")
    print("  1. Test with example simulations")
    print("  2. Benchmark performance")
    print("  3. Validate physics accuracy")
    print("  4. Deploy to production")

def main():
    """Main compilation workflow."""
    print_header("COMPLETE CYTHON COMPILATION FOR SEMIDGFEM")
    print("Compiling all Python bindings for the complete backend implementation")
    
    try:
        check_prerequisites()
        clean_previous_builds()
        compile_cython_modules()
        test_compiled_modules()
        validate_complete_implementation()
        create_installation_package()
        print_summary()
        
        print("\n🎉 COMPLETE CYTHON COMPILATION SUCCESSFUL! 🎉")
        return 0
        
    except Exception as e:
        print(f"\n❌ COMPILATION FAILED: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
