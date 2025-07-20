#!/usr/bin/env python3

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_dir = os.path.join(project_root, "build")
include_dir = os.path.join(project_root, "include")

# Define the extension
extensions = [
    Extension(
        "transient_solver",
        sources=["transient_solver.pyx"],
        include_dirs=[
            include_dir,
            numpy.get_include(),
            "/usr/include/petsc",
            "/usr/include/petsc/petsc",
        ],
        library_dirs=[build_dir],
        libraries=["simulator"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-DHAVE_CONFIG_H"],
        extra_link_args=["-std=c++17"],
    )
]

setup(
    name="transient_solver",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
    zip_safe=False,
)
