SemiDGFEM Documentation
=======================

**High Performance TCAD Software using Discontinuous Galerkin FEM**

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://semidgfem.readthedocs.io
   :alt: Documentation

.. image:: https://img.shields.io/badge/GPU-CUDA%2FOpenCL-green.svg
   :target: gpu_acceleration.html
   :alt: GPU Support

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python

**SemiDGFEM** is a state-of-the-art, high-performance Technology Computer-Aided Design (TCAD) software for semiconductor device simulation using advanced Discontinuous Galerkin Finite Element Methods. Built for researchers, engineers, and students working on semiconductor device modeling and analysis.

🚀 Key Features
---------------

🔬 **Advanced Numerical Methods**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **P3 Discontinuous Galerkin Elements** with 10 DOFs per triangle for high-order accuracy
- **Self-Consistent Coupling** of Poisson and drift-diffusion equations
- **Adaptive Mesh Refinement (AMR)** with Kelly error estimator and anisotropic refinement
- **Multiple Error Estimators**: Kelly, gradient-based, residual-based, ZZ superconvergent

⚡ **High-Performance Computing**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **GPU Acceleration**: CUDA and OpenCL support with 10-20x speedup
- **SIMD Optimization**: AVX2/FMA vectorization for 4x performance boost
- **OpenMP Parallelization**: Multi-core CPU utilization with 8x scaling
- **Memory Optimization**: Cache-friendly data structures and memory pools

🎯 **Complete Physics Modeling**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Semiconductor Physics**: Drift-diffusion transport with proper carrier statistics
- **Material Properties**: Temperature-dependent mobility and recombination models
- **Device Structures**: Support for complex geometries and multi-region devices
- **Boundary Conditions**: Flexible Dirichlet/Neumann boundary condition handling

🖥️ **Modern User Interface**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Python API**: Intuitive high-level interface for simulation setup
- **GUI Application**: Real-time visualization with PySide6 and Vulkan rendering
- **Visualization**: Professional plots with matplotlib and GPU-accelerated rendering
- **Data Export**: Multiple formats (HDF5, VTK, CSV) for post-processing

Quick Start
-----------

Installation
~~~~~~~~~~~~

Using Conda (Recommended)::

    conda create -n semidgfem python=3.9
    conda activate semidgfem
    conda install -c conda-forge semidgfem

Using pip::

    pip install semidgfem[full]  # Includes GPU support and GUI

From Source::

    git clone https://github.com/your-repo/SemiDGFEM.git
    cd SemiDGFEM
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
    make -j$(nproc) && make install
    cd ../python && pip install -e .

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from simulator import Simulator, Device, Method, MeshType

    # Create a 2μm × 1μm p-n junction device
    device = Device(Lx=2e-6, Ly=1e-6)
    sim = Simulator(device, Method.DG, MeshType.Structured, order=3)

    # Set doping profile
    n_points = sim.get_dof_count()
    Nd = np.zeros(n_points)
    Na = np.zeros(n_points)

    # p-region (left half): Na = 1e16 cm⁻³
    Na[:n_points//2] = 1e16 * 1e6  # Convert to m⁻³
    # n-region (right half): Nd = 1e16 cm⁻³  
    Nd[n_points//2:] = 1e16 * 1e6

    sim.set_doping(Nd, Na)

    # Enable GPU acceleration (if available)
    sim.enable_gpu(True)

    # Run forward bias simulation
    results = sim.solve_drift_diffusion(
        bc=[0.0, 0.7, 0.0, 0.0],  # 0.7V forward bias
        use_amr=True,             # Adaptive mesh refinement
        max_steps=100
    )

    # Analyze results
    print(f"Total current: {np.sum(results['Jn'] + results['Jp']):.2e} A")
    print(f"Peak electron density: {np.max(results['n']):.2e} m⁻³")

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   python_api

.. toctree::
   :maxdepth: 2
   :caption: Theory & Mathematics

   theory
   mathematical_formulation
   numerical_methods
   validation

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   gpu_acceleration
   performance_optimization
   adaptive_mesh_refinement
   physics_models

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials

   basic_simulations
   mosfet_modeling
   heterostructure_devices
   validation_studies

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   architecture
   testing
   release_notes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contact & Support
=================

- **Documentation**: https://semidgfem.readthedocs.io
- **Issues**: `GitHub Issues <https://github.com/your-repo/SemiDGFEM/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/your-repo/SemiDGFEM/discussions>`_
- **Email**: mazharuddin.mohammed.official@gmail.com

License
=======

This project is licensed under the MIT License - see the `LICENSE <https://github.com/your-repo/SemiDGFEM/blob/main/LICENSE>`_ file for details.

Citation
========

If you use SemiDGFEM in your research, please cite:

.. code-block:: bibtex

    @software{semidgfem2024,
      title={SemiDGFEM: High Performance TCAD Software using Discontinuous Galerkin FEM},
      author={Dr. Mazharuddin Mohammed},
      year={2024},
      url={https://github.com/your-repo/SemiDGFEM},
      version={2.0.0},
      note={Comprehensive Enhancement Release with Modern GUI, Advanced MOSFET Modeling, and Heterostructure Support}
    }
