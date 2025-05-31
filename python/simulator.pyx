# distutils: language = c++

cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool
from libcpp.memory cimport unique_ptr, make_unique
from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

cdef extern from "device.hpp" namespace "simulator":
    cppclass Device:
        Device(double, double) except +
        Device(double, double, vector[map[string, double]]) except +
        double get_epsilon_at(double x, double y)
        vector[double] get_extents()
        bool is_valid()

# Use C interface to avoid enum class issues
cdef extern from "driftdiffusion.hpp":
    ctypedef struct simulator_Device "simulator::Device"
    ctypedef struct simulator_DriftDiffusion "simulator::DriftDiffusion"
    ctypedef struct simulator_Poisson "simulator::Poisson"

    # C interface functions
    simulator_DriftDiffusion* create_drift_diffusion(simulator_Device* device, int method, int mesh_type, int order)
    void destroy_drift_diffusion(simulator_DriftDiffusion* dd)
    int drift_diffusion_set_doping(simulator_DriftDiffusion* dd, double* Nd, double* Na, int size)
    int drift_diffusion_set_trap_level(simulator_DriftDiffusion* dd, double* Et, int size)
    int drift_diffusion_solve(simulator_DriftDiffusion* dd, double* bc, int bc_size, double Vg,
                             int max_steps, int use_amr, int poisson_max_iter, double poisson_tol,
                             double* V, double* n, double* p, double* Jn, double* Jp, int size)

cdef extern from "poisson.hpp":
    simulator_Poisson* create_poisson(simulator_Device* device, int method, int mesh_type)
    void destroy_poisson(simulator_Poisson* poisson)
    int poisson_solve_2d(simulator_Poisson* poisson, double* bc, int bc_size, double* V, int V_size)
    int poisson_set_charge_density(simulator_Poisson* poisson, double* rho, int size)

cdef class Simulator:
    cdef Device* device
    cdef simulator_Poisson* poisson
    cdef simulator_DriftDiffusion* dd
    cdef int num_points_x, num_points_y
    cdef str mesh_type, method

    def __cinit__(self, dimension="TwoD", extents=[1e-6, 0.5e-6], num_points_x=50, num_points_y=25,
                  method="DG", mesh_type="Structured", regions=None):
        # Declare variables at the beginning
        cdef vector[map[string, double]] c_regions
        cdef int mesh_enum, method_enum

        # Initialize pointers to NULL for safety
        self.device = NULL
        self.poisson = NULL
        self.dd = NULL

        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.method = method
        self.mesh_type = mesh_type

        regions = regions or []
        for r in regions:
            c_regions.push_back(r)

        try:
            # Create C++ objects with error checking
            if len(regions) > 0:
                self.device = new Device(extents[0], extents[1], c_regions)
            else:
                self.device = new Device(extents[0], extents[1])

            if self.device == NULL:
                raise RuntimeError("Failed to create Device")

            # Use integer constants for enums
            mesh_enum = 0 if mesh_type == "Structured" else 1  # 0=Structured, 1=Unstructured
            method_enum = {"FDM": 0, "FEM": 1, "FVM": 2, "SEM": 3, "MC": 4, "DG": 5}[method]

            # Create using C interface with error checking
            self.poisson = create_poisson(<simulator_Device*>self.device, method_enum, mesh_enum)
            if self.poisson == NULL:
                raise RuntimeError("Failed to create Poisson solver")

            self.dd = create_drift_diffusion(<simulator_Device*>self.device, method_enum, mesh_enum, 3)
            if self.dd == NULL:
                raise RuntimeError("Failed to create DriftDiffusion solver")

        except Exception as e:
            # Clean up on error
            if self.dd:
                destroy_drift_diffusion(self.dd)
                self.dd = NULL
            if self.poisson:
                destroy_poisson(self.poisson)
                self.poisson = NULL
            if self.device:
                del self.device
                self.device = NULL
            raise

    def __dealloc__(self):
        if self.dd:
            destroy_drift_diffusion(self.dd)
        if self.poisson:
            destroy_poisson(self.poisson)
        del self.device

    def set_doping(self, np.ndarray[double, ndim=1] Nd, np.ndarray[double, ndim=1] Na):
        if Nd.size != Na.size:
            raise ValueError("Nd and Na arrays must have the same size")
        cdef int result = drift_diffusion_set_doping(self.dd, &Nd[0], &Na[0], Nd.size)
        if result != 0:
            raise RuntimeError("Failed to set doping")

    def set_trap_level(self, np.ndarray[double, ndim=1] Et):
        cdef int result = drift_diffusion_set_trap_level(self.dd, &Et[0], Et.size)
        if result != 0:
            raise RuntimeError("Failed to set trap level")

    def solve_poisson(self, bc):
        cdef np.ndarray[double, ndim=1] bc_array = np.array(bc, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] V = np.zeros(self.num_points_x * self.num_points_y, dtype=np.float64)
        cdef int result = poisson_solve_2d(self.poisson, &bc_array[0], bc_array.size, &V[0], V.size)
        if result != 0:
            raise RuntimeError("Poisson solve failed")
        return V

    def solve_drift_diffusion(self, bc, Vg=0.0, max_steps=100, use_amr=False,
                              poisson_max_iter=50, poisson_tol=1e-6):
        cdef np.ndarray[double, ndim=1] bc_array = np.array(bc, dtype=np.float64)
        cdef int size = self.num_points_x * self.num_points_y

        # Prepare output arrays
        cdef np.ndarray[double, ndim=1] V = np.zeros(size, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] n = np.zeros(size, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] p = np.zeros(size, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] Jn = np.zeros(size, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] Jp = np.zeros(size, dtype=np.float64)

        cdef int use_amr_int = 1 if use_amr else 0
        cdef int result = drift_diffusion_solve(self.dd, &bc_array[0], bc_array.size, Vg,
                                               max_steps, use_amr_int, poisson_max_iter, poisson_tol,
                                               &V[0], &n[0], &p[0], &Jn[0], &Jp[0], size)

        if result != 0:
            raise RuntimeError("Drift-diffusion solve failed")

        return {
            "potential": V,
            "n": n,
            "p": p,
            "Jn": Jn,
            "Jp": Jp
        }