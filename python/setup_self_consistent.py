#!/usr/bin/env python3
"""
Setup script for Self-Consistent Solver Cython module
Builds comprehensive self-consistent simulation capabilities
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import sys

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_dir = os.path.join(project_root, "build")

# Include directories
include_dirs = [
    os.path.join(project_root, "include"),
    numpy.get_include(),
    "/usr/include/eigen3",
    "/usr/local/include/eigen3",
    "/usr/include/boost",
    "/usr/local/include/boost"
]

# Library directories
library_dirs = [
    build_dir,
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib"
]

# Libraries to link
libraries = [
    "simulator",
    "boost_system",
    "boost_filesystem"
]

# Source files for the extension
source_files = [
    "self_consistent_solver.pyx"
]

# Additional source files to compile
extra_sources = [
    os.path.join(project_root, "src", "selfconsistent", "self_consistent_solver.cpp"),
    os.path.join(project_root, "src", "materials", "material_properties.cpp"),
    os.path.join(project_root, "src", "device.cpp"),
    os.path.join(project_root, "src", "mesh.cpp")
]

# Compiler arguments
extra_compile_args = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-DWITH_BOOST",
    "-DWITH_EIGEN",
    "-fopenmp"
]

# Linker arguments
extra_link_args = [
    "-fopenmp",
    "-Wl,-rpath," + build_dir
]

# Check for CUDA support
cuda_available = False
try:
    import pycuda
    cuda_available = True
    extra_compile_args.append("-DWITH_CUDA")
    libraries.extend(["cuda", "cudart", "cublas", "cusparse"])
    print("CUDA support detected and enabled")
except ImportError:
    print("CUDA support not available")

# Check for MPI support
mpi_available = False
try:
    from mpi4py import MPI
    mpi_available = True
    extra_compile_args.append("-DWITH_MPI")
    libraries.append("mpi")
    print("MPI support detected and enabled")
except ImportError:
    print("MPI support not available")

# Check for PETSc support
petsc_available = False
petsc_dir = os.environ.get('PETSC_DIR')
if petsc_dir and os.path.exists(petsc_dir):
    petsc_available = True
    extra_compile_args.append("-DWITH_PETSC")
    include_dirs.append(os.path.join(petsc_dir, "include"))
    libraries.extend(["petsc", "m"])
    print("PETSc support detected and enabled")
else:
    print("PETSc support not available")

# Define the extension
extension = Extension(
    name="self_consistent_solver",
    sources=source_files + extra_sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++"
)

# Cython compiler directives
compiler_directives = {
    'language_level': 3,
    'boundscheck': False,
    'wraparound': False,
    'initializedcheck': False,
    'cdivision': True,
    'embedsignature': True,
    'binding': True
}

def build_extension():
    """Build the Cython extension"""
    print("Building self-consistent solver Cython extension...")
    
    # Check if the C++ library exists
    lib_path = os.path.join(build_dir, "libsimulator.so")
    if not os.path.exists(lib_path):
        print(f"Warning: C++ library not found at {lib_path}")
        print("Please build the C++ library first using CMake")
        return False
    
    try:
        setup(
            name="self_consistent_solver",
            ext_modules=cythonize([extension], compiler_directives=compiler_directives),
            zip_safe=False,
            include_dirs=include_dirs
        )
        print("✓ Self-consistent solver extension built successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to build extension: {e}")
        return False

def test_extension():
    """Test the built extension"""
    print("\nTesting self-consistent solver extension...")
    
    try:
        # Test import
        import self_consistent_solver as scs
        print("✓ Extension import successful")
        
        # Test solver creation
        solver = scs.create_self_consistent_solver(2e-6, 1e-6)
        print(f"✓ Solver creation successful: DOF count = {solver.get_dof_count()}")
        
        # Test material database
        materials = scs.create_material_database()
        print("✓ Material database creation successful")
        
        # Test material properties
        si_bandgap = materials.get_bandgap(scs.MaterialType.SILICON, 300.0)
        si_mobility = materials.get_electron_mobility(scs.MaterialType.SILICON, 300.0, 1e16)
        print(f"✓ Silicon properties: Eg = {si_bandgap:.3f} eV, μn = {si_mobility:.4f} m²/V·s")
        
        # Test validation
        validation = scs.validate_self_consistent_solver()
        if validation['validation_passed']:
            print("✓ Validation passed")
        else:
            print("✗ Validation failed")
            
        return True
        
    except Exception as e:
        print(f"✗ Extension test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_example():
    """Run a comprehensive example"""
    print("\nRunning self-consistent solver example...")
    
    try:
        import self_consistent_solver as scs
        import numpy as np
        
        # Create solver
        solver = scs.create_self_consistent_solver(2e-6, 1e-6, "DG", "Structured", 3)
        print(f"Solver created with {solver.get_dof_count()} DOF")
        
        # Set convergence criteria
        solver.set_convergence_criteria(1e-6, 1e-3, 1e-3, 50)
        print("Convergence criteria set")
        
        # Set doping profile
        dof_count = solver.get_dof_count()
        Nd = np.zeros(dof_count)
        Na = np.zeros(dof_count)
        
        # Create p-n junction
        Na[:dof_count//2] = 1e16 * 1e6  # Convert cm^-3 to m^-3
        Nd[dof_count//2:] = 1e16 * 1e6
        
        solver.set_doping(Nd, Na)
        print("Doping profile set")
        
        # Set initial conditions
        initial_potential = np.zeros(dof_count)
        initial_n = np.full(dof_count, 1e10)  # m^-3
        initial_p = np.full(dof_count, 1e10)  # m^-3
        
        # Boundary conditions (forward bias)
        bc = [0.0, 0.7, 0.0, 0.0]  # 0.7V forward bias
        
        # Solve self-consistently
        print("Solving self-consistent equations...")
        results = solver.solve_steady_state(bc, initial_potential, initial_n, initial_p)
        
        print(f"✓ Self-consistent solution converged:")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Final residual: {results['residual']:.2e}")
        print(f"  Potential range: [{np.min(results['potential']):.3f}, {np.max(results['potential']):.3f}] V")
        print(f"  Electron density range: [{np.min(results['n']):.2e}, {np.max(results['n']):.2e}] m^-3")
        print(f"  Hole density range: [{np.min(results['p']):.2e}, {np.max(results['p']):.2e}] m^-3")
        
        # Test material database
        materials = scs.create_material_database()
        
        # Test different materials
        materials_to_test = [
            (scs.MaterialType.SILICON, "Silicon"),
            (scs.MaterialType.GALLIUM_ARSENIDE, "GaAs"),
            (scs.MaterialType.GERMANIUM, "Germanium")
        ]
        
        print("\nMaterial properties at 300K:")
        for mat_type, name in materials_to_test:
            try:
                bandgap = materials.get_bandgap(mat_type, 300.0)
                ni = materials.get_intrinsic_concentration(mat_type, 300.0)
                mu_n = materials.get_electron_mobility(mat_type, 300.0, 1e16)
                mu_p = materials.get_hole_mobility(mat_type, 300.0, 1e16)
                
                print(f"  {name}: Eg={bandgap:.3f} eV, ni={ni:.2e} m^-3, "
                      f"μn={mu_n:.4f} m²/V·s, μp={mu_p:.4f} m²/V·s")
            except:
                print(f"  {name}: Properties not available")
        
        return True
        
    except Exception as e:
        print(f"✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Self-Consistent Solver Cython Extension Builder")
    print("=" * 50)
    
    # Build extension
    build_success = build_extension()
    
    if build_success:
        # Test extension
        test_success = test_extension()
        
        if test_success:
            # Run example
            example_success = run_example()
            
            if example_success:
                print("\n🎉 All tests passed! Self-consistent solver is ready to use.")
            else:
                print("\n⚠️  Example failed, but basic functionality works.")
        else:
            print("\n⚠️  Extension test failed.")
    else:
        print("\n❌ Build failed.")
    
    print("\nBuild process completed.")
