# distutils: language = c++
# cython: language_level=3

"""
Advanced Transport Models Python Interface
Provides access to non-equilibrium statistics, energy transport, and hydrodynamics
"""

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool

# Import base classes
from simulator cimport Device, Method, MeshType

# C++ declarations for advanced transport
cdef extern from "advanced_transport.hpp" namespace "simulator::transport":
    cdef cppclass AdvancedTransportSolver:
        AdvancedTransportSolver(const Device& device, Method method, MeshType mesh_type, 
                               int transport_model, int order) except +
        
        void set_doping(const vector[double]& Nd, const vector[double]& Na) except +
        void set_trap_level(const vector[double]& Et) except +
        
        map[string, vector[double]] solve_transport(
            const vector[double]& bc, double Vg, int max_steps, bool use_amr,
            int poisson_max_iter, double poisson_tol) except +
        
        bool is_valid() const
        void validate() except +
        size_t get_dof_count() const
        double get_convergence_residual() const
        int get_order() const
        int get_transport_model() const

# C interface declarations
cdef extern from "advanced_transport.hpp":
    AdvancedTransportSolver* create_advanced_transport_solver(
        Device* device, int method, int mesh_type, int transport_model, int order)
    void destroy_advanced_transport_solver(AdvancedTransportSolver* solver)
    int advanced_transport_solver_is_valid(AdvancedTransportSolver* solver)
    int advanced_transport_solver_set_doping(AdvancedTransportSolver* solver, 
                                            double* Nd, double* Na, int size)
    int advanced_transport_solver_set_trap_level(AdvancedTransportSolver* solver, 
                                                double* Et, int size)
    size_t advanced_transport_solver_get_dof_count(AdvancedTransportSolver* solver)
    double advanced_transport_solver_get_convergence_residual(AdvancedTransportSolver* solver)
    int advanced_transport_solver_get_order(AdvancedTransportSolver* solver)
    int advanced_transport_solver_get_transport_model(AdvancedTransportSolver* solver)

# Transport model enumeration
class TransportModel:
    DRIFT_DIFFUSION = 0
    ENERGY_TRANSPORT = 1
    HYDRODYNAMIC = 2
    NON_EQUILIBRIUM_STATISTICS = 3

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
    cdef Device* _device
    
    def __cinit__(self, Device device, method, mesh_type, transport_model, order=3):
        """
        Initialize advanced transport solver.
        
        Parameters:
        -----------
        device : Device
            Device geometry and material properties
        method : Method
            Numerical method (DG, FEM, etc.)
        mesh_type : MeshType
            Mesh type (Structured, Unstructured)
        transport_model : TransportModel
            Transport physics model
        order : int, optional
            Polynomial order (default: 3)
        """
        self._device = &device._device
        self._solver = create_advanced_transport_solver(
            self._device, method, mesh_type, transport_model, order)
        
        if self._solver == NULL:
            raise RuntimeError("Failed to create advanced transport solver")
    
    def __dealloc__(self):
        if self._solver != NULL:
            destroy_advanced_transport_solver(self._solver)
    
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
            Solution dictionary with fields depending on transport model:
            - 'potential': Electrostatic potential (V)
            - 'n': Electron concentration (m^-3)
            - 'p': Hole concentration (m^-3)
            - 'Jn': Electron current density (A/m^2)
            - 'Jp': Hole current density (A/m^2)
            - 'energy_n': Electron energy density (J/m^3) [energy transport]
            - 'energy_p': Hole energy density (J/m^3) [energy transport]
            - 'T_n': Electron temperature (K) [energy/hydrodynamic]
            - 'T_p': Hole temperature (K) [energy/hydrodynamic]
            - 'velocity_n': Electron velocity (m/s) [hydrodynamic]
            - 'velocity_p': Hole velocity (m/s) [hydrodynamic]
            - 'momentum_n': Electron momentum density (kg⋅m/s⋅m^3) [hydrodynamic]
            - 'momentum_p': Hole momentum density (kg⋅m/s⋅m^3) [hydrodynamic]
            - 'quasi_fermi_n': Electron quasi-Fermi level (V) [non-equilibrium]
            - 'quasi_fermi_p': Hole quasi-Fermi level (V) [non-equilibrium]
        """
        cdef vector[double] bc_vec
        for val in bc:
            bc_vec.push_back(val)
        
        cdef map[string, vector[double]] cpp_results = self._solver.solve_transport(
            bc_vec, Vg, max_steps, use_amr, poisson_max_iter, poisson_tol)
        
        # Convert C++ results to Python dictionary
        results = {}
        cdef vector[double] values
        
        for key_bytes, values in cpp_results:
            key = key_bytes.decode('utf-8')
            results[key] = np.array(values)
        
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

# Convenience functions
def create_drift_diffusion_solver(device, method, mesh_type, order=3):
    """Create drift-diffusion transport solver."""
    return AdvancedTransport(device, method, mesh_type, TransportModel.DRIFT_DIFFUSION, order)

def create_energy_transport_solver(device, method, mesh_type, order=3):
    """Create energy transport solver."""
    return AdvancedTransport(device, method, mesh_type, TransportModel.ENERGY_TRANSPORT, order)

def create_hydrodynamic_solver(device, method, mesh_type, order=3):
    """Create hydrodynamic transport solver."""
    return AdvancedTransport(device, method, mesh_type, TransportModel.HYDRODYNAMIC, order)

def create_non_equilibrium_solver(device, method, mesh_type, order=3):
    """Create non-equilibrium statistics solver."""
    return AdvancedTransport(device, method, mesh_type, TransportModel.NON_EQUILIBRIUM_STATISTICS, order)
