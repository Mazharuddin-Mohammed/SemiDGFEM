# distutils: language = c++
# cython: language_level=3

"""
Advanced Transport Models Python Interface (Simplified)
Provides access to non-equilibrium statistics, energy transport, and hydrodynamics
"""

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool

# C interface declarations for advanced transport (using existing simulator namespace functions)
cdef extern from "advanced_transport.hpp" namespace "simulator":
    # Forward declaration for the solver type
    cdef cppclass AdvancedTransportSolver "simulator::transport::AdvancedTransportSolver":
        pass

# C interface functions in simulator namespace
cdef extern from "advanced_transport.hpp":
    AdvancedTransportSolver* create_advanced_transport_solver "simulator::create_advanced_transport_solver"(
        void* device, int method, int mesh_type, int transport_model, int order)
    void destroy_advanced_transport_solver "simulator::destroy_advanced_transport_solver"(AdvancedTransportSolver* solver)
    int advanced_transport_solver_is_valid "simulator::advanced_transport_solver_is_valid"(AdvancedTransportSolver* solver)
    int advanced_transport_solver_set_doping "simulator::advanced_transport_solver_set_doping"(
        AdvancedTransportSolver* solver, double* Nd, double* Na, int size)
    int advanced_transport_solver_set_trap_level "simulator::advanced_transport_solver_set_trap_level"(
        AdvancedTransportSolver* solver, double* Et, int size)
    int advanced_transport_solver_solve "simulator::advanced_transport_solver_solve"(
        AdvancedTransportSolver* solver, double* bc, int bc_size, double Vg,
        int max_steps, int use_amr, int poisson_max_iter, double poisson_tol,
        double* V, double* n, double* p, double* Jn, double* Jp,
        double* energy_n, double* energy_p, double* T_n, double* T_p, int size)
    int advanced_transport_solver_get_dof_count "simulator::advanced_transport_solver_get_dof_count"(AdvancedTransportSolver* solver)
    double advanced_transport_solver_get_convergence_residual "simulator::advanced_transport_solver_get_convergence_residual"(AdvancedTransportSolver* solver)
    int advanced_transport_solver_get_order "simulator::advanced_transport_solver_get_order"(AdvancedTransportSolver* solver)
    int advanced_transport_solver_get_transport_model "simulator::advanced_transport_solver_get_transport_model"(AdvancedTransportSolver* solver)

# Device interface
cdef extern from "device.hpp":
    void* create_device "simulator::create_device"(double Lx, double Ly)
    void destroy_device "simulator::destroy_device"(void* device)

# Transport model enumeration
class TransportModel:
    DRIFT_DIFFUSION = 0
    ENERGY_TRANSPORT = 1
    HYDRODYNAMIC = 2
    NON_EQUILIBRIUM_STATISTICS = 3

class Method:
    DG = 0
    FEM = 1

class MeshType:
    Structured = 0
    Unstructured = 1

cdef class AdvancedTransport:
    """
    Advanced transport solver with multiple physics models.
    
    Supports:
    - Classical drift-diffusion with Boltzmann statistics
    - Energy transport with hot carrier effects
    - Hydrodynamic transport with momentum conservation
    - Non-equilibrium transport with Fermi-Dirac statistics
    """
    
    cdef AdvancedTransportSolver* _solver
    cdef void* _device
    cdef double _device_width, _device_length
    cdef int _method, _mesh_type, _transport_model, _order

    def __cinit__(self, double device_width, double device_length, int method=0, int mesh_type=0, int transport_model=0, int order=3):
        """
        Initialize advanced transport solver.

        Parameters:
        -----------
        device_width : float
            Device width (m)
        device_length : float
            Device length (m)
        method : int, optional
            Numerical method (0=DG, 1=FEM, default: 0)
        mesh_type : int, optional
            Mesh type (0=Structured, 1=Unstructured, default: 0)
        transport_model : int, optional
            Transport physics model (0=DD, 1=Energy, 2=Hydro, 3=NonEq, default: 0)
        order : int, optional
            Polynomial order (default: 3)
        """
        self._device_width = device_width
        self._device_length = device_length
        self._method = method
        self._mesh_type = mesh_type
        self._transport_model = transport_model
        self._order = order

        # Create device first
        self._device = create_device(device_width, device_length)
        if self._device == NULL:
            raise RuntimeError("Failed to create device")

        # Create solver
        self._solver = create_advanced_transport_solver(
            self._device, method, mesh_type, transport_model, order)

        if self._solver == NULL:
            destroy_device(self._device)
            raise RuntimeError("Failed to create advanced transport solver")
    
    def __dealloc__(self):
        if self._solver != NULL:
            destroy_advanced_transport_solver(self._solver)
        if self._device != NULL:
            destroy_device(self._device)
    
    def set_doping(self, np.ndarray[double, ndim=1] Nd, np.ndarray[double, ndim=1] Na):
        """
        Set doping concentrations.
        
        Parameters:
        -----------
        Nd : array_like
            Donor concentration (m^-3)
        Na : array_like
            Acceptor concentration (m^-3)
        """
        if Nd.size != Na.size:
            raise ValueError("Nd and Na arrays must have the same size")
        
        cdef int result = advanced_transport_solver_set_doping(
            self._solver, &Nd[0], &Na[0], Nd.size)
        
        if result != 0:
            raise RuntimeError("Failed to set doping concentrations")
    
    def set_trap_level(self, np.ndarray[double, ndim=1] Et):
        """
        Set trap energy levels.
        
        Parameters:
        -----------
        Et : array_like
            Trap energy levels (eV)
        """
        cdef int result = advanced_transport_solver_set_trap_level(
            self._solver, &Et[0], Et.size)
        
        if result != 0:
            raise RuntimeError("Failed to set trap levels")
    
    def solve_transport(self, bc, Vg=0.0, max_steps=100, use_amr=False, 
                       poisson_max_iter=50, poisson_tol=1e-6):
        """
        Solve advanced transport equations.
        
        Parameters:
        -----------
        bc : array_like
            Boundary conditions [left, right, bottom, top]
        Vg : float, optional
            Gate voltage (V)
        max_steps : int, optional
            Maximum iteration steps
        use_amr : bool, optional
            Enable adaptive mesh refinement
        poisson_max_iter : int, optional
            Maximum Poisson iterations
        poisson_tol : float, optional
            Poisson tolerance
            
        Returns:
        --------
        dict
            Solution dictionary with fields depending on transport model
        """
        cdef np.ndarray[double, ndim=1] bc_array = np.array(bc, dtype=np.float64)
        cdef int dof_count = advanced_transport_solver_get_dof_count(self._solver)
        
        if dof_count <= 0:
            raise RuntimeError("Invalid DOF count")
        
        # Prepare output arrays
        cdef np.ndarray[double, ndim=1] V = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] n = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] p = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] Jn = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] Jp = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] energy_n = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] energy_p = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] T_n = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] T_p = np.zeros(dof_count, dtype=np.float64)
        
        cdef int use_amr_int = 1 if use_amr else 0
        cdef int result = advanced_transport_solver_solve(
            self._solver, &bc_array[0], bc_array.size, Vg,
            max_steps, use_amr_int, poisson_max_iter, poisson_tol,
            &V[0], &n[0], &p[0], &Jn[0], &Jp[0], 
            &energy_n[0], &energy_p[0], &T_n[0], &T_p[0],
            dof_count)
        
        if result != 0:
            raise RuntimeError("Advanced transport solve failed")
        
        # Build results dictionary based on transport model
        results = {
            "potential": V,
            "n": n,
            "p": p,
            "Jn": Jn,
            "Jp": Jp
        }
        
        # Add model-specific results
        if self._transport_model == TransportModel.ENERGY_TRANSPORT:
            results["energy_n"] = energy_n
            results["energy_p"] = energy_p
            results["T_n"] = T_n
            results["T_p"] = T_p
        elif self._transport_model == TransportModel.HYDRODYNAMIC:
            results["T_n"] = T_n
            results["T_p"] = T_p
            # Add momentum/velocity calculations here
        elif self._transport_model == TransportModel.NON_EQUILIBRIUM_STATISTICS:
            # Add quasi-Fermi levels here
            pass
        
        return results
    
    def is_valid(self):
        """Check if solver is in valid state."""
        return advanced_transport_solver_is_valid(self._solver) == 1
    
    def get_dof_count(self):
        """Get degrees of freedom count."""
        return advanced_transport_solver_get_dof_count(self._solver)
    
    def get_convergence_residual(self):
        """Get convergence residual."""
        return advanced_transport_solver_get_convergence_residual(self._solver)
    
    def get_order(self):
        """Get polynomial order."""
        return advanced_transport_solver_get_order(self._solver)
    
    def get_transport_model(self):
        """Get current transport model."""
        return advanced_transport_solver_get_transport_model(self._solver)
    
    def get_transport_model_name(self):
        """Get transport model name as string."""
        model = self.get_transport_model()
        if model == TransportModel.DRIFT_DIFFUSION:
            return "DRIFT_DIFFUSION"
        elif model == TransportModel.ENERGY_TRANSPORT:
            return "ENERGY_TRANSPORT"
        elif model == TransportModel.HYDRODYNAMIC:
            return "HYDRODYNAMIC"
        elif model == TransportModel.NON_EQUILIBRIUM_STATISTICS:
            return "NON_EQUILIBRIUM_STATISTICS"
        else:
            return "UNKNOWN"
    
    # Properties
    @property
    def device_width(self):
        return self._device_width
    
    @property
    def device_length(self):
        return self._device_length
    
    @property
    def method(self):
        return self._method
    
    @property
    def mesh_type(self):
        return self._mesh_type
    
    @property
    def transport_model(self):
        return self._transport_model
    
    @property
    def order(self):
        return self._order

# Convenience functions
def create_drift_diffusion_solver(device_width, device_length, method=0, mesh_type=0, order=3):
    """Create drift-diffusion transport solver."""
    return AdvancedTransport(device_width, device_length, method, mesh_type, TransportModel.DRIFT_DIFFUSION, order)

def create_energy_transport_solver(device_width, device_length, method=0, mesh_type=0, order=3):
    """Create energy transport solver."""
    return AdvancedTransport(device_width, device_length, method, mesh_type, TransportModel.ENERGY_TRANSPORT, order)

def create_hydrodynamic_solver(device_width, device_length, method=0, mesh_type=0, order=3):
    """Create hydrodynamic transport solver."""
    return AdvancedTransport(device_width, device_length, method, mesh_type, TransportModel.HYDRODYNAMIC, order)

def create_non_equilibrium_solver(device_width, device_length, method=0, mesh_type=0, order=3):
    """Create non-equilibrium statistics solver."""
    return AdvancedTransport(device_width, device_length, method, mesh_type, TransportModel.NON_EQUILIBRIUM_STATISTICS, order)
