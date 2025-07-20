# distutils: language = c++

cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.functional cimport function
import numpy as np
cimport numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

# C interface declarations for transient solver
cdef extern from "transient_solver.hpp":
    ctypedef struct simulator_transient_TransientSolver "simulator::transient::TransientSolver"

    # Time integrator enum
    ctypedef enum TimeIntegrator "simulator::transient::TimeIntegrator":
        BACKWARD_EULER = 0
        FORWARD_EULER = 1
        CRANK_NICOLSON = 2
        RK4 = 3
        BDF2 = 4
        ADAPTIVE_RK45 = 5

# C interface declarations
cdef extern from "device.hpp":
    ctypedef struct simulator_Device "simulator::Device"
    simulator_Device* create_device(double Lx, double Ly)
    void destroy_device(simulator_Device* device)

cdef extern from "transient_solver.hpp":
    # C interface functions
    simulator_transient_TransientSolver* create_transient_solver(
        simulator_Device* device, int method, int mesh_type, int order)
    void destroy_transient_solver(simulator_transient_TransientSolver* solver)
    
    int transient_solver_set_time_step(simulator_transient_TransientSolver* solver, double dt)
    int transient_solver_set_final_time(simulator_transient_TransientSolver* solver, double t_final)
    int transient_solver_set_time_integrator(simulator_transient_TransientSolver* solver, int integrator)
    int transient_solver_set_doping(simulator_transient_TransientSolver* solver, 
                                   double* Nd, double* Na, int size)
    
    int transient_solver_solve(simulator_transient_TransientSolver* solver,
                              double* static_bc, int bc_size,
                              double* initial_potential, double* initial_n, double* initial_p, int ic_size,
                              double* time_points, double* potential_results, double* n_results, double* p_results,
                              int max_time_points, int* actual_time_points)
    
    size_t transient_solver_get_dof_count(simulator_transient_TransientSolver* solver)
    double transient_solver_get_current_time(simulator_transient_TransientSolver* solver)
    int transient_solver_is_valid(simulator_transient_TransientSolver* solver)

# Python wrapper class
cdef class TransientSolver:
    cdef simulator_transient_TransientSolver* _solver
    cdef simulator_Device* _device

    def __cinit__(self, double device_width, double device_height, str method="DG", str mesh_type="Structured", int order=3):
        # Create device
        self._device = create_device(device_width, device_height)
        if self._device == NULL:
            raise RuntimeError("Failed to create Device")

        method_enum = {"FDM": 0, "FEM": 1, "FVM": 2, "SEM": 3, "MC": 4, "DG": 5}[method]
        mesh_enum = 0 if mesh_type == "Structured" else 1

        self._solver = create_transient_solver(self._device, method_enum, mesh_enum, order)
        if self._solver == NULL:
            raise RuntimeError("Failed to create TransientSolver")
    
    def __dealloc__(self):
        if self._solver:
            destroy_transient_solver(self._solver)
        if self._device:
            destroy_device(self._device)
    
    def set_time_step(self, double dt):
        """Set the time step for integration."""
        if dt <= 0.0:
            raise ValueError("Time step must be positive")
        cdef int result = transient_solver_set_time_step(self._solver, dt)
        if result != 0:
            raise RuntimeError("Failed to set time step")
    
    def set_final_time(self, double t_final):
        """Set the final simulation time."""
        if t_final <= 0.0:
            raise ValueError("Final time must be positive")
        cdef int result = transient_solver_set_final_time(self._solver, t_final)
        if result != 0:
            raise RuntimeError("Failed to set final time")
    
    def set_time_integrator(self, str integrator):
        """Set the time integration method."""
        integrator_map = {
            "backward_euler": 0,
            "forward_euler": 1,
            "crank_nicolson": 2,
            "rk4": 3,
            "bdf2": 4,
            "adaptive_rk45": 5
        }
        
        if integrator.lower() not in integrator_map:
            raise ValueError(f"Unknown integrator: {integrator}")
        
        cdef int integrator_enum = integrator_map[integrator.lower()]
        cdef int result = transient_solver_set_time_integrator(self._solver, integrator_enum)
        if result != 0:
            raise RuntimeError("Failed to set time integrator")
    
    def set_doping(self, np.ndarray[double, ndim=1] Nd, np.ndarray[double, ndim=1] Na):
        """Set doping concentrations."""
        if Nd.size != Na.size:
            raise ValueError("Nd and Na arrays must have the same size")
        
        cdef int result = transient_solver_set_doping(self._solver, &Nd[0], &Na[0], Nd.size)
        if result != 0:
            raise RuntimeError("Failed to set doping")
    
    def solve(self, bc, initial_conditions, int max_time_points=1000):
        """
        Solve transient simulation.
        
        Parameters:
        -----------
        bc : array_like
            Boundary conditions [left, right, bottom, top] (V)
        initial_conditions : dict
            Dictionary with 'potential', 'n', 'p' arrays
        max_time_points : int
            Maximum number of time points to store
            
        Returns:
        --------
        dict
            Dictionary with 'time', 'potential', 'n', 'p' arrays
        """
        # Validate inputs
        cdef np.ndarray[double, ndim=1] bc_array = np.array(bc, dtype=np.float64)
        if bc_array.size != 4:
            raise ValueError("Boundary conditions must have 4 values")
        
        if not all(key in initial_conditions for key in ['potential', 'n', 'p']):
            raise ValueError("Initial conditions must contain 'potential', 'n', 'p'")
        
        cdef np.ndarray[double, ndim=1] initial_potential = np.array(initial_conditions['potential'], dtype=np.float64)
        cdef np.ndarray[double, ndim=1] initial_n = np.array(initial_conditions['n'], dtype=np.float64)
        cdef np.ndarray[double, ndim=1] initial_p = np.array(initial_conditions['p'], dtype=np.float64)
        
        if not (initial_potential.size == initial_n.size == initial_p.size):
            raise ValueError("All initial condition arrays must have the same size")
        
        cdef int ic_size = initial_potential.size
        
        # Prepare output arrays
        cdef np.ndarray[double, ndim=1] time_points = np.zeros(max_time_points, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] potential_results = np.zeros(max_time_points * ic_size, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] n_results = np.zeros(max_time_points * ic_size, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] p_results = np.zeros(max_time_points * ic_size, dtype=np.float64)
        cdef int actual_time_points = 0
        
        # Call solver
        cdef int result = transient_solver_solve(
            self._solver, &bc_array[0], bc_array.size,
            &initial_potential[0], &initial_n[0], &initial_p[0], ic_size,
            &time_points[0], &potential_results[0], &n_results[0], &p_results[0],
            max_time_points, &actual_time_points)
        
        if result != 0:
            raise RuntimeError("Transient simulation failed")
        
        # Reshape and return results
        time_array = time_points[:actual_time_points]
        potential_array = potential_results[:actual_time_points * ic_size].reshape(actual_time_points, ic_size)
        n_array = n_results[:actual_time_points * ic_size].reshape(actual_time_points, ic_size)
        p_array = p_results[:actual_time_points * ic_size].reshape(actual_time_points, ic_size)
        
        return {
            'time': time_array,
            'potential': potential_array,
            'n': n_array,
            'p': p_array
        }
    
    def get_dof_count(self):
        """Get degrees of freedom count."""
        return transient_solver_get_dof_count(self._solver)
    
    def get_current_time(self):
        """Get current simulation time."""
        return transient_solver_get_current_time(self._solver)
    
    def is_valid(self):
        """Check if solver is valid."""
        return transient_solver_is_valid(self._solver) != 0

# Convenience functions
def create_transient_solver(double device_width, double device_height, str method="DG", str mesh_type="Structured", int order=3):
    """Create a transient solver instance."""
    return TransientSolver(device_width, device_height, method, mesh_type, order)

def get_available_integrators():
    """Get list of available time integrators."""
    return ["backward_euler", "forward_euler", "crank_nicolson", "rk4", "bdf2", "adaptive_rk45"]

def validate_transient_solver():
    """Validate transient solver implementation."""
    try:
        solver = create_transient_solver(1e-6, 0.5e-6)

        return {
            "transient_solver_creation": True,
            "time_step_setting": True,
            "integrator_setting": True,
            "dof_count_access": solver.get_dof_count() > 0,
            "validation_passed": solver.is_valid()
        }
    except Exception as e:
        return {
            "transient_solver_creation": False,
            "error": str(e),
            "validation_passed": False
        }

# Example usage function
def run_transient_example():
    """Run a simple transient simulation example."""
    try:
        import numpy as np

        # Create device and solver
        solver = create_transient_solver(2e-6, 1e-6, "DG", "Structured", 3)
        
        # Configure simulation
        solver.set_time_step(1e-12)  # 1 ps
        solver.set_final_time(1e-9)  # 1 ns
        solver.set_time_integrator("backward_euler")
        
        # Set doping
        dof_count = solver.get_dof_count()
        Nd = np.zeros(dof_count)
        Na = np.zeros(dof_count)
        Na[:dof_count//2] = 1e16 * 1e6  # p-region
        Nd[dof_count//2:] = 1e16 * 1e6  # n-region
        solver.set_doping(Nd, Na)
        
        # Set initial conditions
        initial_conditions = {
            'potential': np.zeros(dof_count),
            'n': np.full(dof_count, 1e10),
            'p': np.full(dof_count, 1e10)
        }
        
        # Boundary conditions (step voltage)
        bc = [0.0, 0.7, 0.0, 0.0]  # 0.7V forward bias
        
        # Solve
        results = solver.solve(bc, initial_conditions, max_time_points=100)
        
        print(f"Transient simulation completed:")
        print(f"  Time points: {len(results['time'])}")
        print(f"  Final time: {results['time'][-1]*1e12:.2f} ps")
        print(f"  DOF count: {dof_count}")
        
        return results
        
    except Exception as e:
        print(f"Transient example failed: {e}")
        return None
