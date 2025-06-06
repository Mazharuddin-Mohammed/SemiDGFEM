"""
Complete Setup script for SemiDGFEM Python bindings
Includes all advanced transport models with structured and unstructured DG discretization

Author: Dr. Mazharuddin Mohammed
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys

# Get the build directory path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "include")
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

# Check if build directory exists
if not os.path.exists(build_dir):
    print(f"Error: Build directory {build_dir} not found. Please run 'make' first.")
    sys.exit(1)

# Check if library exists
lib_path = os.path.join(build_dir, "libsimulator.so")
if not os.path.exists(lib_path):
    print(f"Error: Library {lib_path} not found. Please run 'make' first.")
    sys.exit(1)

print(f"Using build directory: {build_dir}")
print(f"Using include directory: {include_dir}")
print(f"Using source directory: {src_dir}")

# Common compilation flags
common_compile_args = [
    "-std=c++17",
    "-O3",
    "-DWITH_PETSC",
    "-DWITH_GMSH",
    "-fPIC"
]

common_link_args = [
    "-Wl,-rpath," + build_dir,
    "-lpetsc",
    "-lm"
]

# Complete extension modules
ext_modules = [
    # Core simulator module
    Extension(
        "simulator",
        sources=["simulator.pyx"],
        include_dirs=[include_dir, src_dir, np.get_include()],
        libraries=["simulator"],
        library_dirs=[build_dir],
        language="c++",
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),

    # Advanced transport module
    Extension(
        "advanced_transport",
        sources=["advanced_transport.pyx"],
        include_dirs=[include_dir, src_dir, np.get_include()],
        libraries=["simulator"],
        library_dirs=[build_dir],
        language="c++",
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),

    # Complete DG discretization module
    Extension(
        "complete_dg",
        sources=["complete_dg.pyx"],
        include_dirs=[include_dir, src_dir, np.get_include()],
        libraries=["simulator"],
        library_dirs=[build_dir],
        language="c++",
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),

    # Unstructured mesh module
    Extension(
        "unstructured_transport",
        sources=["unstructured_transport.pyx"],
        include_dirs=[include_dir, src_dir, np.get_include()],
        libraries=["simulator"],
        library_dirs=[build_dir],
        language="c++",
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),

    # Performance optimization module
    Extension(
        "performance_bindings",
        sources=["performance_bindings.pyx"],
        include_dirs=[include_dir, src_dir, np.get_include()],
        libraries=["simulator"],
        library_dirs=[build_dir],
        language="c++",
        extra_compile_args=common_compile_args + ["-fopenmp"],
        extra_link_args=common_link_args + ["-fopenmp"],
    )
]

setup(
    name="semiconductor_simulator",
    version="2.0.0",
    description="Complete semiconductor device simulator with advanced transport models",
    author="Dr. Mazharuddin Mohammed",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'embedsignature': True
        }
    ),
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
        'cython>=0.29.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
    ],
)