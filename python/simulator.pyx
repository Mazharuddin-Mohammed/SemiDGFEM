# distutils: language = c++
# distutils: sources = src/device.cpp src/mesh.cpp src/2d/dg/structured/poisson_dg_struct_2d.cpp src/2d/dg/structured/driftdiffusion_dg_struct_2d.cpp src/2d/dg/unstructured/poisson_dg_unstruct_2d.cpp src/2d/dg/unstructured/driftdiffusion_dg_unstruct_2d.cpp
# distutils: include_dirs = include
# distutils: libraries = simulator petsc gmsh boost_numeric_ublas vulkan
# distutils: library_dirs = /usr/lib

cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool
import numpy as np
cimport numpy as np

cdef extern from "device.hpp" namespace "simulator":
    cppclass Device:
        Device(double, double, vector[map[string, double]] &)

cdef extern from "mesh.hpp" namespace "simulator":
    enum MeshType:
        Structured,
        Unstructured
    cppclass Mesh:
        Mesh(Device &, MeshType)
        void generate_gmsh_mesh(string &)
        vector[double] get_grid_points_x()
        vector[double] get_grid_points_y()
        vector[vector[int]] get_elements()
        void refine(vector[bool] &)

cdef extern from "poisson.hpp" namespace "simulator":
    enum Method:
        FDM, FEM, FVM, SEM, MC, DG
    Poisson* create_poisson(Device* device, int method, int mesh_type)
    void poisson_set_charge_density(Poisson* poisson, double* rho, int size)
    void poisson_solve_2d(Poisson* poisson, double* bc, int bc_size, double* V, int V_size)
    void poisson_solve_2d_self_consistent(Poisson* poisson, double* bc, int bc_size, 
                                          double* n, double* p, double* Nd, double* Na, int size, 
                                          int max_iter, double tol, double* V, int V_size)

cdef extern from "driftdiffusion.hpp" namespace "simulator":
    DriftDiffusion* create_drift_diffusion(Device* device, int method, int mesh_type, int order)
    void drift_diffusion_set_doping(DriftDiffusion* dd, double* Nd, double* Na, int size)
    void drift_diffusion_set_trap_level(DriftDiffusion* dd, double* Et, int size)
    void drift_diffusion_solve(DriftDiffusion* dd, double* bc, int bc_size, double Vg, 
                               int max_steps, bool use_amr, int poisson_max_iter, double poisson_tol, 
                               double* results, int result_cols, int result_rows)

cdef class Simulator:
    cdef Device* device
    cdef Mesh* mesh
    cdef DriftDiffusion* dd
    cdef int num_points_x, num_points_y
    cdef str mesh_type, method, order

    def __cinit__(self, dimension="TwoD", extents=[1e-6, 0.5e-6], num_points_x=50, num_points_y=25,
                  method="DG", mesh_type="Structured", regions=None, order="P3"):
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.method = method
        self.mesh_type = mesh_type
        self.order = order

        regions = regions or []
        cdef vector[map[string, double]] c_regions
        for r in regions:
            c_regions.push_back(r)

        self.device = new Device(extents[0], extents[1], c_regions)
        self.mesh = new Mesh(self.device[0], Structured if mesh_type == "Structured" else Unstructured)
        self.dd = create_drift_diffusion(self.device, 
                                         {"FDM": 0, "FEM": 1, "FVM": 2, "SEM": 3, "MC": 4, "DG": 5}[method],
                                         {"Structured": 0, "Unstructured": 1}[mesh_type],
                                         {"P2": 2, "P3": 3}[order])

    def __dealloc__(self):
        del self.dd
        del self.mesh
        del self.device

    def generate_mesh(self, filename):
        cdef string c_filename = filename.encode('utf-8')
        self.mesh.generate_gmsh_mesh(c_filename)

    def get_grid_points(self):
        cdef vector[double] grid_x = self.mesh.get_grid_points_x()
        cdef vector[double] grid_y = self.mesh.get_grid_points_y()
        return {"x": np.array(grid_x), "y": np.array(grid_y)}

    def set_doping(self, np.ndarray[double, ndim=1] Nd, np.ndarray[double, ndim=1] Na):
        drift_diffusion_set_doping(self.dd, &Nd[0], &Na[0], Nd.size)

    def set_trap_level(self, np.ndarray[double, ndim=1] Et):
        drift_diffusion_set_trap_level(self.dd, &Et[0], Et.size)

    def solve_drift_diffusion(self, bc, Vg=0.0, max_steps=100, use_amr=False, 
                              poisson_max_iter=50, poisson_tol=1e-6):
        cdef np.ndarray[double, ndim=1] bc_array = np.array(bc, dtype=np.float64)
        cdef np.ndarray[double, ndim=2] results = np.zeros((5, self.num_points_x * self.num_points_y), dtype=np.float64)
        drift_diffusion_solve(self.dd, &bc_array[0], bc_array.size, Vg, max_steps, use_amr, 
                              poisson_max_iter, poisson_tol, &results[0,0], 5, self.num_points_x * self.num_points_y)
        return {
            "potential": results[0],
            "n": results[1],
            "p": results[2],
            "Jn": results[3],
            "Jp": results[4]
        }