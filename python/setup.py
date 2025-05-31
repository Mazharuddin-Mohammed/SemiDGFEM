from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "simulator",
        sources=["python/simulator.pyx"],
        include_dirs=["include", np.get_include()],
        libraries=["simulator", "petsc", "gmsh", "boost_numeric_ublas", "vulkan"],
        library_dirs=["/usr/lib"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    name="semiconductor_simulator",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}),
    zip_safe=False,
)