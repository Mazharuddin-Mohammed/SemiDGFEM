"""
Setup script for SemiDGFEM Python bindings

Author: Dr. Mazharuddin Mohammed
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Get the build directory path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "include")

ext_modules = [
    Extension(
        "simulator",
        sources=["simulator.pyx"],
        include_dirs=[include_dir, np.get_include()],
        libraries=["simulator"],
        library_dirs=[build_dir],
        language="c++",
        extra_compile_args=["-std=c++17"],
        extra_link_args=["-Wl,-rpath," + build_dir],
    ),
    Extension(
        "advanced_transport",
        sources=["advanced_transport.pyx"],
        include_dirs=[include_dir, np.get_include()],
        libraries=["simulator"],
        library_dirs=[build_dir],
        language="c++",
        extra_compile_args=["-std=c++17"],
        extra_link_args=["-Wl,-rpath," + build_dir],
    )
]

setup(
    name="semiconductor_simulator",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}),
    zip_safe=False,
)