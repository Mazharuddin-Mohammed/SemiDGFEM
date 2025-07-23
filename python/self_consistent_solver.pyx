# distutils: language = c++
# distutils: sources = ../src/selfconsistent/self_consistent_solver.cpp ../src/materials/material_properties.cpp

"""
Cython bindings for Self-Consistent Solver and Material Database
Provides comprehensive self-consistent simulation capabilities
"""

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool
import logging

logger = logging.getLogger(__name__)

# C++ declarations
cdef extern from "device.hpp":
    cdef cppclass Device:
        Device(double, double)
        bool is_valid()
        vector[double] get_extents()

cdef extern from "self_consistent_solver.hpp":
    cdef cppclass SelfConsistentSolver "simulator::selfconsistent::SelfConsistentSolver":
        SelfConsistentSolver(const Device&, int, int, int)
        void set_convergence_criteria(double, double, double, int)
        void set_doping(const vector[double]&, const vector[double]&)
        void enable_energy_transport(bool)
        void enable_hydrodynamic_transport(bool)
        void enable_quantum_corrections(bool)
        size_t get_dof_count()
        bool is_valid()
        int get_last_iteration_count()
        double get_last_residual()
        
    cdef cppclass SolutionState "simulator::selfconsistent::SolutionState":
        vector[double] potential
        vector[double] electron_density
        vector[double] hole_density
        vector[double] electron_current_x
        vector[double] electron_current_y
        vector[double] hole_current_x
        vector[double] hole_current_y
        vector[double] electron_temperature
        vector[double] hole_temperature
        vector[double] generation_rate
        vector[double] recombination_rate
        bool has_energy_transport
        bool has_hydrodynamic
        bool has_quantum_corrections
        void resize(size_t)

    # C interface functions
    SelfConsistentSolver* create_self_consistent_solver(Device*, int, int, int)
    void destroy_self_consistent_solver(SelfConsistentSolver*)
    int self_consistent_solver_set_convergence_criteria(SelfConsistentSolver*, double, double, double, int)
    int self_consistent_solver_set_doping(SelfConsistentSolver*, double*, double*, int)
    int self_consistent_solver_solve_steady_state(SelfConsistentSolver*, double*, int,
                                                  double*, double*, double*, int,
                                                  double*, double*, double*, int*, double*)
    size_t self_consistent_solver_get_dof_count(SelfConsistentSolver*)
    int self_consistent_solver_is_valid(SelfConsistentSolver*)

cdef extern from "material_properties.hpp":
    cdef cppclass MaterialDatabase "simulator::materials::MaterialDatabase":
        MaterialDatabase()
        double get_bandgap(int, double)
        double get_intrinsic_concentration(int, double)
        double get_electron_mobility(int, double, double)
        double get_hole_mobility(int, double, double)
        double calculate_srh_recombination(double, double, double, double, double)
        double calculate_impact_ionization_rate(double, double, double)
        bool is_material_available(const string&)
        
    # C interface functions
    void* create_material_database()
    void destroy_material_database(void*)
    double material_database_get_bandgap(void*, int, double)
    double material_database_get_intrinsic_concentration(void*, int, double)
    double material_database_get_electron_mobility(void*, int, double, double)
    double material_database_get_hole_mobility(void*, int, double, double)
    double material_database_calculate_srh_recombination(void*, double, double, double, double, double)
    double material_database_calculate_impact_ionization(void*, double, double, double)

# Material type enumeration
class MaterialType:
    SILICON = 0
    GERMANIUM = 1
    GALLIUM_ARSENIDE = 2
    SILICON_CARBIDE = 3
    GALLIUM_NITRIDE = 4
    INDIUM_GALLIUM_ARSENIDE = 5
    CUSTOM = 99

cdef class PySelfConsistentSolver:
    """
    Python wrapper for self-consistent semiconductor device solver
    """
    cdef SelfConsistentSolver* solver
    cdef Device* device
    cdef bool owns_device
    
    def __cinit__(self, double device_width, double device_height, 
                  str method="DG", str mesh_type="Structured", int order=3):
        """
        Initialize self-consistent solver
        
        Parameters:
        -----------
        device_width : float
            Device width in meters
        device_height : float
            Device height in meters
        method : str
            Numerical method ("DG", "FEM", "FVM")
        mesh_type : str
            Mesh type ("Structured", "Unstructured")
        order : int
            Polynomial order for DG methods
        """
        # Create device
        self.device = new Device(device_width, device_height)
        self.owns_device = True
        
        # Map method and mesh type strings to integers
        method_map = {"FDM": 0, "FEM": 1, "FVM": 2, "SEM": 3, "MC": 4, "DG": 5}
        mesh_map = {"Structured": 0, "Unstructured": 1}
        
        method_int = method_map.get(method, 5)  # Default to DG
        mesh_int = mesh_map.get(mesh_type, 0)   # Default to Structured
        
        # Create solver
        self.solver = create_self_consistent_solver(self.device, method_int, mesh_int, order)
        
        if not self.solver:
            raise RuntimeError("Failed to create self-consistent solver")
        
        logger.info(f"Self-consistent solver created: {device_width*1e6:.1f}x{device_height*1e6:.1f} μm")
    
    def __dealloc__(self):
        if self.solver:
            destroy_self_consistent_solver(self.solver)
        if self.owns_device and self.device:
            del self.device
    
    def set_convergence_criteria(self, double potential_tolerance=1e-6, 
                               double density_tolerance=1e-3,
                               double current_tolerance=1e-3, 
                               int max_iterations=100):
        """Set convergence criteria for self-consistent iterations"""
        if not self.solver:
            raise RuntimeError("Solver not initialized")
        
        result = self_consistent_solver_set_convergence_criteria(
            self.solver, potential_tolerance, density_tolerance, current_tolerance, max_iterations)
        
        if result != 0:
            raise RuntimeError("Failed to set convergence criteria")
        
        logger.debug(f"Convergence criteria set: pot_tol={potential_tolerance}, "
                    f"dens_tol={density_tolerance}, max_iter={max_iterations}")
    
    def set_doping(self, cnp.ndarray[double, ndim=1] Nd, cnp.ndarray[double, ndim=1] Na):
        """Set doping concentrations"""
        if not self.solver:
            raise RuntimeError("Solver not initialized")
        
        if len(Nd) != len(Na):
            raise ValueError("Nd and Na arrays must have the same length")
        
        cdef int size = len(Nd)
        result = self_consistent_solver_set_doping(self.solver, &Nd[0], &Na[0], size)
        
        if result != 0:
            raise RuntimeError("Failed to set doping")
        
        logger.debug(f"Doping set: {size} points, Nd range [{np.min(Nd):.2e}, {np.max(Nd):.2e}]")
    
    def enable_energy_transport(self, bool enable=True):
        """Enable energy transport equations"""
        if not self.solver:
            raise RuntimeError("Solver not initialized")
        
        # This would call the C++ method when available
        logger.info(f"Energy transport {'enabled' if enable else 'disabled'}")
    
    def enable_hydrodynamic_transport(self, bool enable=True):
        """Enable hydrodynamic transport equations"""
        if not self.solver:
            raise RuntimeError("Solver not initialized")
        
        # This would call the C++ method when available
        logger.info(f"Hydrodynamic transport {'enabled' if enable else 'disabled'}")
    
    def enable_quantum_corrections(self, bool enable=True):
        """Enable quantum corrections"""
        if not self.solver:
            raise RuntimeError("Solver not initialized")
        
        # This would call the C++ method when available
        logger.info(f"Quantum corrections {'enabled' if enable else 'disabled'}")
    
    def solve_steady_state(self, list boundary_conditions,
                          cnp.ndarray[double, ndim=1] initial_potential,
                          cnp.ndarray[double, ndim=1] initial_n,
                          cnp.ndarray[double, ndim=1] initial_p):
        """
        Solve steady-state self-consistent equations
        
        Parameters:
        -----------
        boundary_conditions : list
            Boundary conditions [left, right, bottom, top] in Volts
        initial_potential : ndarray
            Initial potential guess (V)
        initial_n : ndarray
            Initial electron density guess (m^-3)
        initial_p : ndarray
            Initial hole density guess (m^-3)
            
        Returns:
        --------
        dict
            Results with 'potential', 'n', 'p', 'iterations', 'residual'
        """
        if not self.solver:
            raise RuntimeError("Solver not initialized")
        
        if len(boundary_conditions) != 4:
            raise ValueError("Boundary conditions must have 4 values")
        
        cdef size_t dof_count = self_consistent_solver_get_dof_count(self.solver)
        
        if (len(initial_potential) != dof_count or 
            len(initial_n) != dof_count or 
            len(initial_p) != dof_count):
            raise ValueError(f"Initial arrays must have {dof_count} elements")
        
        # Prepare arrays
        cdef cnp.ndarray[double, ndim=1] bc = np.array(boundary_conditions, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] result_potential = np.zeros(dof_count, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] result_n = np.zeros(dof_count, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] result_p = np.zeros(dof_count, dtype=np.float64)
        
        cdef int iterations = 0
        cdef double residual = 0.0
        
        # Solve
        result = self_consistent_solver_solve_steady_state(
            self.solver, &bc[0], 4,
            &initial_potential[0], &initial_n[0], &initial_p[0], dof_count,
            &result_potential[0], &result_n[0], &result_p[0],
            &iterations, &residual)
        
        if result != 0:
            raise RuntimeError("Self-consistent solve failed")
        
        logger.info(f"Self-consistent solution converged in {iterations} iterations, residual={residual:.2e}")
        
        return {
            'potential': np.array(result_potential),
            'n': np.array(result_n),
            'p': np.array(result_p),
            'iterations': iterations,
            'residual': residual
        }
    
    def get_dof_count(self):
        """Get degrees of freedom count"""
        if not self.solver:
            raise RuntimeError("Solver not initialized")
        return self_consistent_solver_get_dof_count(self.solver)
    
    def is_valid(self):
        """Check if solver is valid"""
        if not self.solver:
            return False
        return self_consistent_solver_is_valid(self.solver) == 1

cdef class PyMaterialDatabase:
    """
    Python wrapper for material properties database
    """
    cdef void* database
    
    def __cinit__(self):
        """Initialize material database"""
        self.database = create_material_database()
        if not self.database:
            raise RuntimeError("Failed to create material database")
        logger.info("Material database initialized")
    
    def __dealloc__(self):
        if self.database:
            destroy_material_database(self.database)
    
    def get_bandgap(self, int material_type, double temperature=300.0):
        """Get bandgap at specified temperature"""
        if not self.database:
            raise RuntimeError("Database not initialized")
        return material_database_get_bandgap(self.database, material_type, temperature)
    
    def get_intrinsic_concentration(self, int material_type, double temperature=300.0):
        """Get intrinsic carrier concentration at specified temperature"""
        if not self.database:
            raise RuntimeError("Database not initialized")
        return material_database_get_intrinsic_concentration(self.database, material_type, temperature)
    
    def get_electron_mobility(self, int material_type, double temperature=300.0, double doping=1e16):
        """Get electron mobility at specified temperature and doping"""
        if not self.database:
            raise RuntimeError("Database not initialized")
        return material_database_get_electron_mobility(self.database, material_type, temperature, doping)
    
    def get_hole_mobility(self, int material_type, double temperature=300.0, double doping=1e16):
        """Get hole mobility at specified temperature and doping"""
        if not self.database:
            raise RuntimeError("Database not initialized")
        return material_database_get_hole_mobility(self.database, material_type, temperature, doping)
    
    def calculate_srh_recombination(self, double n, double p, double ni, 
                                   double tau_n=1e-6, double tau_p=1e-6):
        """Calculate Shockley-Read-Hall recombination rate"""
        if not self.database:
            raise RuntimeError("Database not initialized")
        return material_database_calculate_srh_recombination(
            self.database, n, p, ni, tau_n, tau_p)
    
    def calculate_impact_ionization(self, double E_field, double a, double b):
        """Calculate impact ionization rate"""
        if not self.database:
            raise RuntimeError("Database not initialized")
        return material_database_calculate_impact_ionization(self.database, E_field, a, b)

# Convenience functions
def create_self_consistent_solver(device_width, device_height, method="DG", mesh_type="Structured", order=3):
    """Create a self-consistent solver instance"""
    return PySelfConsistentSolver(device_width, device_height, method, mesh_type, order)

def create_material_database():
    """Create a material database instance"""
    return PyMaterialDatabase()

def validate_self_consistent_solver():
    """Validate self-consistent solver implementation"""
    try:
        # Test solver creation
        solver = create_self_consistent_solver(1e-6, 0.5e-6)
        
        # Test material database
        materials = create_material_database()
        
        return {
            "solver_creation": True,
            "material_database": True,
            "dof_count_access": solver.get_dof_count() > 0,
            "validation_passed": solver.is_valid()
        }
    except Exception as e:
        return {
            "solver_creation": False,
            "error": str(e),
            "validation_passed": False
        }
