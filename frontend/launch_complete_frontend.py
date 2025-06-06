#!/usr/bin/env python3
"""
Launch Script for Complete Frontend Integration
Provides easy access to the complete SemiDGFEM frontend

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Setup environment for frontend launch"""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Add python directory to path
    python_dir = project_root / "python"
    if python_dir.exists():
        sys.path.insert(0, str(python_dir))
        print(f"✓ Added Python bindings path: {python_dir}")
    
    # Add frontend directory to path
    frontend_dir = Path(__file__).parent
    sys.path.insert(0, str(frontend_dir))
    print(f"✓ Added frontend path: {frontend_dir}")
    
    # Set working directory to project root
    os.chdir(project_root)
    print(f"✓ Working directory: {project_root}")
    
    return project_root

def check_dependencies():
    """Check if all dependencies are available"""
    
    print("\n=== Checking Dependencies ===")
    
    # Check PySide6
    try:
        import PySide6
        print("✓ PySide6 available")
    except ImportError:
        print("✗ PySide6 not available. Install with: pip install PySide6")
        return False
    
    # Check matplotlib
    try:
        import matplotlib
        print("✓ Matplotlib available")
    except ImportError:
        print("⚠ Matplotlib not available. Install with: pip install matplotlib")
        print("  (Visualization will be limited)")
    
    # Check numpy
    try:
        import numpy
        print("✓ NumPy available")
    except ImportError:
        print("✗ NumPy not available. Install with: pip install numpy")
        return False
    
    return True

def check_backend():
    """Check backend availability"""
    
    print("\n=== Checking Backend ===")
    
    # Check if build directory exists
    build_dir = Path("build")
    if build_dir.exists():
        print(f"✓ Build directory found: {build_dir}")
        
        # Check for library
        lib_file = build_dir / "libsimulator.so"
        if lib_file.exists():
            print(f"✓ Backend library found: {lib_file}")
        else:
            print(f"⚠ Backend library not found: {lib_file}")
            print("  Run 'make' in project root to build backend")
    else:
        print("⚠ Build directory not found")
        print("  Run 'make' in project root to build backend")
    
    # Check Python bindings
    backend_modules = [
        "simulator",
        "complete_dg", 
        "unstructured_transport",
        "performance_bindings"
    ]
    
    available_modules = 0
    for module in backend_modules:
        try:
            __import__(module)
            print(f"✓ {module} module available")
            available_modules += 1
        except ImportError:
            print(f"⚠ {module} module not available")
    
    if available_modules == 0:
        print("\n⚠ No backend modules available")
        print("  To compile Python bindings:")
        print("  1. cd python")
        print("  2. python3 compile_all.py")
        print("  3. python3 test_complete_bindings.py")
    elif available_modules < len(backend_modules):
        print(f"\n⚠ Partial backend available ({available_modules}/{len(backend_modules)} modules)")
    else:
        print(f"\n✓ Complete backend available ({available_modules}/{len(backend_modules)} modules)")
    
    return available_modules > 0

def launch_frontend():
    """Launch the complete frontend"""
    
    print("\n=== Launching Complete Frontend ===")
    
    try:
        from complete_frontend_integration import main
        print("✓ Frontend module loaded")
        
        print("🚀 Starting SemiDGFEM Complete Frontend Integration...")
        return main()
        
    except ImportError as e:
        print(f"✗ Failed to import frontend: {e}")
        return 1
    except Exception as e:
        print(f"✗ Frontend launch failed: {e}")
        return 1

def show_help():
    """Show help information"""
    
    help_text = """
=== SemiDGFEM Complete Frontend Integration ===

This launcher provides access to the complete SemiDGFEM frontend with:
• Advanced transport model configuration
• Real-time simulation control
• Comprehensive visualization
• Backend validation tools

Usage:
  python3 launch_complete_frontend.py [options]

Options:
  --help, -h     Show this help message
  --check, -c    Check dependencies and backend only
  --verbose, -v  Verbose output

Requirements:
  • PySide6 (GUI framework)
  • NumPy (numerical computing)
  • Matplotlib (visualization, optional)
  • Compiled SemiDGFEM backend (optional)

Setup Instructions:
  1. Install dependencies: pip install PySide6 numpy matplotlib
  2. Build backend: make (in project root)
  3. Compile Python bindings: cd python && python3 compile_all.py
  4. Launch frontend: python3 frontend/launch_complete_frontend.py

Features:
  ✓ Complete DG discretization (P1, P2, P3)
  ✓ Structured and unstructured mesh support
  ✓ Energy transport models
  ✓ Hydrodynamic transport models  
  ✓ Non-equilibrium drift-diffusion
  ✓ SIMD and GPU acceleration
  ✓ Advanced visualization and analysis
  ✓ Configuration management
  ✓ Real-time simulation monitoring

Author: Dr. Mazharuddin Mohammed
Version: 2.0.0
"""
    
    print(help_text)

def main():
    """Main launcher function"""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--help', '-h']:
            show_help()
            return 0
        elif arg in ['--check', '-c']:
            setup_environment()
            deps_ok = check_dependencies()
            backend_ok = check_backend()
            print(f"\n=== Summary ===")
            print(f"Dependencies: {'✓' if deps_ok else '✗'}")
            print(f"Backend: {'✓' if backend_ok else '⚠'}")
            return 0 if deps_ok else 1
        elif arg in ['--verbose', '-v']:
            # Verbose mode - just continue with normal launch
            pass
    
    print("=== SemiDGFEM Complete Frontend Integration Launcher ===")
    print("Advanced Semiconductor Device Simulation")
    print("Author: Dr. Mazharuddin Mohammed\n")
    
    # Setup environment
    project_root = setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing required dependencies. Please install them and try again.")
        return 1
    
    # Check backend (optional)
    backend_available = check_backend()
    if not backend_available:
        print("\n⚠ Backend not available - frontend will run in simulation mode")
        print("  All GUI features will work, but simulations will use synthetic data")
    
    # Launch frontend
    try:
        return launch_frontend()
    except KeyboardInterrupt:
        print("\n\n⏹ Frontend launch cancelled by user")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
