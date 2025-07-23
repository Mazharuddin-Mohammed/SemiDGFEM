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

# Import self-consistent solver if available
try:
    import self_consistent_solver as scs
    SELF_CONSISTENT_AVAILABLE = True
except ImportError:
    SELF_CONSISTENT_AVAILABLE = False

# Complete C interface declarations
cdef extern from "device.hpp" namespace "simulator":
    ctypedef struct simulator_Device "simulator::Device"

    # Device C interface (inside simulator namespace)
    simulator_Device* create_device(double Lx, double Ly)
    void destroy_device(simulator_Device* device)
    double device_get_epsilon_at(simulator_Device* device, double x, double y)
    void device_get_extents(simulator_Device* device, double* extents)

cdef extern from "driftdiffusion.hpp":
    ctypedef struct simulator_DriftDiffusion "simulator::DriftDiffusion"

    # DriftDiffusion C interface
    simulator_DriftDiffusion* create_drift_diffusion(simulator_Device* device, int method, int mesh_type, int order)
    void destroy_drift_diffusion(simulator_DriftDiffusion* dd)
    int drift_diffusion_is_valid(simulator_DriftDiffusion* dd)
    int drift_diffusion_set_doping(simulator_DriftDiffusion* dd, double* Nd, double* Na, int size)
    int drift_diffusion_set_trap_level(simulator_DriftDiffusion* dd, double* Et, int size)
    int drift_diffusion_solve(simulator_DriftDiffusion* dd, double* bc, int bc_size, double Vg,
                             int max_steps, int use_amr, int poisson_max_iter, double poisson_tol,
                             double* V, double* n, double* p, double* Jn, double* Jp, int size)
    size_t drift_diffusion_get_dof_count(simulator_DriftDiffusion* dd)
    double drift_diffusion_get_convergence_residual(simulator_DriftDiffusion* dd)
    int drift_diffusion_get_order(simulator_DriftDiffusion* dd)

cdef extern from "poisson.hpp":
    ctypedef struct simulator_Poisson "simulator::Poisson"

    # Poisson C interface
    simulator_Poisson* create_poisson(simulator_Device* device, int method, int mesh_type)
    void destroy_poisson(simulator_Poisson* poisson)
    int poisson_is_valid(simulator_Poisson* poisson)
    void poisson_set_charge_density(simulator_Poisson* poisson, double* rho, int size)
    int poisson_solve_2d(simulator_Poisson* poisson, double* bc, int bc_size, double* V, int V_size)
    int poisson_solve_2d_self_consistent(simulator_Poisson* poisson, double* bc, int bc_size,
                                        double* n, double* p, double* Nd, double* Na, int size,
                                        int max_iter, double tol, double* V, int V_size)
    size_t poisson_get_dof_count(simulator_Poisson* poisson)
    double poisson_get_residual_norm(simulator_Poisson* poisson)

cdef extern from "mesh.hpp":
    ctypedef struct simulator_Mesh "simulator::Mesh"

    # Mesh C interface
    simulator_Mesh* create_mesh(simulator_Device* device, int mesh_type)
    void destroy_mesh(simulator_Mesh* mesh)
    void mesh_generate_gmsh(simulator_Mesh* mesh, const char* filename)
    int mesh_get_num_nodes(simulator_Mesh* mesh)
    int mesh_get_num_elements(simulator_Mesh* mesh)
    void mesh_get_grid_points_x(simulator_Mesh* mesh, double* points, int size)
    void mesh_get_grid_points_y(simulator_Mesh* mesh, double* points, int size)
    void mesh_get_elements(simulator_Mesh* mesh, int* elements, int num_elements, int nodes_per_element)

# Device wrapper class
cdef class Device:
    cdef simulator_Device* _device
    cdef double _Lx, _Ly

    def __cinit__(self, double Lx, double Ly):
        self._device = create_device(Lx, Ly)
        if self._device == NULL:
            raise RuntimeError("Failed to create Device")
        self._Lx = Lx
        self._Ly = Ly

    def __dealloc__(self):
        if self._device:
            destroy_device(self._device)

    def get_epsilon_at(self, double x, double y):
        """Get permittivity at given coordinates."""
        return device_get_epsilon_at(self._device, x, y)

    def get_extents(self):
        """Get device extents [Lx, Ly]."""
        cdef double extents[2]
        device_get_extents(self._device, extents)
        return [extents[0], extents[1]]

    @property
    def length(self):
        return self._Lx

    @property
    def width(self):
        return self._Ly

# Mesh wrapper class
cdef class Mesh:
    cdef simulator_Mesh* _mesh
    cdef Device _device

    def __cinit__(self, Device device, str mesh_type="Structured"):
        self._device = device
        mesh_enum = 0 if mesh_type == "Structured" else 1
        self._mesh = create_mesh(device._device, mesh_enum)
        if self._mesh == NULL:
            raise RuntimeError("Failed to create Mesh")

    def __dealloc__(self):
        if self._mesh:
            destroy_mesh(self._mesh)

    def generate_gmsh(self, str filename):
        """Generate GMSH mesh file."""
        mesh_generate_gmsh(self._mesh, filename.encode('utf-8'))

    def get_num_nodes(self):
        """Get number of mesh nodes."""
        return mesh_get_num_nodes(self._mesh)

    def get_num_elements(self):
        """Get number of mesh elements."""
        return mesh_get_num_elements(self._mesh)

    def get_grid_points(self):
        """Get mesh grid points."""
        num_nodes = self.get_num_nodes()
        cdef np.ndarray[double, ndim=1] x_points = np.zeros(num_nodes, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] y_points = np.zeros(num_nodes, dtype=np.float64)

        mesh_get_grid_points_x(self._mesh, &x_points[0], num_nodes)
        mesh_get_grid_points_y(self._mesh, &y_points[0], num_nodes)

        return x_points, y_points

# Poisson solver wrapper class
cdef class PoissonSolver:
    cdef simulator_Poisson* _poisson
    cdef Device _device

    def __cinit__(self, Device device, str method="DG", str mesh_type="Structured"):
        self._device = device
        method_enum = {"FDM": 0, "FEM": 1, "FVM": 2, "SEM": 3, "MC": 4, "DG": 5}[method]
        mesh_enum = 0 if mesh_type == "Structured" else 1

        self._poisson = create_poisson(device._device, method_enum, mesh_enum)
        if self._poisson == NULL:
            raise RuntimeError("Failed to create Poisson solver")

    def __dealloc__(self):
        if self._poisson:
            destroy_poisson(self._poisson)

    def is_valid(self):
        """Check if solver is valid."""
        return poisson_is_valid(self._poisson) != 0

    def set_charge_density(self, np.ndarray[double, ndim=1] rho):
        """Set charge density."""
        poisson_set_charge_density(self._poisson, &rho[0], rho.size)

    def solve(self, bc):
        """Solve Poisson equation."""
        cdef np.ndarray[double, ndim=1] bc_array = np.array(bc, dtype=np.float64)
        cdef int dof_count = <int>poisson_get_dof_count(self._poisson)
        cdef np.ndarray[double, ndim=1] V = np.zeros(dof_count, dtype=np.float64)

        cdef int result = poisson_solve_2d(self._poisson, &bc_array[0], bc_array.size, &V[0], V.size)
        if result != 0:
            raise RuntimeError("Poisson solve failed")
        return V

    def solve_self_consistent(self, bc, np.ndarray[double, ndim=1] n, np.ndarray[double, ndim=1] p,
                             np.ndarray[double, ndim=1] Nd, np.ndarray[double, ndim=1] Na,
                             int max_iter=100, double tol=1e-6):
        """Solve self-consistent Poisson equation."""
        cdef np.ndarray[double, ndim=1] bc_array = np.array(bc, dtype=np.float64)
        cdef int dof_count = <int>poisson_get_dof_count(self._poisson)
        cdef np.ndarray[double, ndim=1] V = np.zeros(dof_count, dtype=np.float64)

        cdef int result = poisson_solve_2d_self_consistent(
            self._poisson, &bc_array[0], bc_array.size,
            &n[0], &p[0], &Nd[0], &Na[0], n.size,
            max_iter, tol, &V[0], V.size)

        if result != 0:
            raise RuntimeError("Self-consistent Poisson solve failed")
        return V

    def get_dof_count(self):
        """Get degrees of freedom count."""
        return poisson_get_dof_count(self._poisson)

    def get_residual_norm(self):
        """Get residual norm."""
        return poisson_get_residual_norm(self._poisson)

# DriftDiffusion solver wrapper class
cdef class DriftDiffusionSolver:
    cdef simulator_DriftDiffusion* _dd
    cdef Device _device

    def __cinit__(self, Device device, str method="DG", str mesh_type="Structured", int order=3):
        self._device = device
        method_enum = {"FDM": 0, "FEM": 1, "FVM": 2, "SEM": 3, "MC": 4, "DG": 5}[method]
        mesh_enum = 0 if mesh_type == "Structured" else 1

        self._dd = create_drift_diffusion(device._device, method_enum, mesh_enum, order)
        if self._dd == NULL:
            raise RuntimeError("Failed to create DriftDiffusion solver")

    def __dealloc__(self):
        if self._dd:
            destroy_drift_diffusion(self._dd)

    def is_valid(self):
        """Check if solver is valid."""
        return drift_diffusion_is_valid(self._dd) != 0

    def set_doping(self, np.ndarray[double, ndim=1] Nd, np.ndarray[double, ndim=1] Na):
        """Set doping concentrations."""
        if Nd.size != Na.size:
            raise ValueError("Nd and Na arrays must have the same size")
        cdef int result = drift_diffusion_set_doping(self._dd, &Nd[0], &Na[0], Nd.size)
        if result != 0:
            raise RuntimeError("Failed to set doping")

    def set_trap_level(self, np.ndarray[double, ndim=1] Et):
        """Set trap energy levels."""
        cdef int result = drift_diffusion_set_trap_level(self._dd, &Et[0], Et.size)
        if result != 0:
            raise RuntimeError("Failed to set trap level")

    def solve(self, bc, double Vg=0.0, int max_steps=100, bool use_amr=False,
              int poisson_max_iter=50, double poisson_tol=1e-6):
        """Solve drift-diffusion equations."""
        cdef np.ndarray[double, ndim=1] bc_array = np.array(bc, dtype=np.float64)
        cdef int dof_count = <int>drift_diffusion_get_dof_count(self._dd)

        # Prepare output arrays
        cdef np.ndarray[double, ndim=1] V = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] n = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] p = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] Jn = np.zeros(dof_count, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] Jp = np.zeros(dof_count, dtype=np.float64)

        cdef int use_amr_int = 1 if use_amr else 0
        cdef int result = drift_diffusion_solve(
            self._dd, &bc_array[0], bc_array.size, Vg,
            max_steps, use_amr_int, poisson_max_iter, poisson_tol,
            &V[0], &n[0], &p[0], &Jn[0], &Jp[0], dof_count)

        if result != 0:
            raise RuntimeError("Drift-diffusion solve failed")

        return {
            "potential": V,
            "n": n,
            "p": p,
            "Jn": Jn,
            "Jp": Jp
        }

    def get_dof_count(self):
        """Get degrees of freedom count."""
        return drift_diffusion_get_dof_count(self._dd)

    def get_convergence_residual(self):
        """Get convergence residual."""
        return drift_diffusion_get_convergence_residual(self._dd)

    def get_order(self):
        """Get polynomial order."""
        return drift_diffusion_get_order(self._dd)

# Legacy Simulator class for backward compatibility
cdef class Simulator:
    cdef Device _device
    cdef PoissonSolver _poisson
    cdef DriftDiffusionSolver _dd
    cdef int num_points_x, num_points_y
    cdef str mesh_type, method

    # Python-accessible attributes
    cdef public str _method
    cdef public str _mesh_type
    cdef public int _num_points_x
    cdef public int _num_points_y

    def __cinit__(self, dimension="TwoD", extents=[1e-6, 0.5e-6], num_points_x=50, num_points_y=25,
                  method="DG", mesh_type="Structured", regions=None):

        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.method = method
        self.mesh_type = mesh_type

        # Set public attributes
        self._num_points_x = num_points_x
        self._num_points_y = num_points_y
        self._method = method
        self._mesh_type = mesh_type

        try:
            # Create device using new wrapper
            self._device = Device(extents[0], extents[1])

            # Create solvers using new wrappers
            self._poisson = PoissonSolver(self._device, method, mesh_type)
            self._dd = DriftDiffusionSolver(self._device, method, mesh_type, 3)

        except Exception as e:
            raise RuntimeError(f"Failed to create Simulator: {e}")

    def __dealloc__(self):
        # Cleanup handled by individual wrapper classes
        pass

    def set_doping(self, np.ndarray[double, ndim=1] Nd, np.ndarray[double, ndim=1] Na):
        """Set doping concentrations."""
        self._dd.set_doping(Nd, Na)

    def set_trap_level(self, np.ndarray[double, ndim=1] Et):
        """Set trap energy levels."""
        self._dd.set_trap_level(Et)

    def solve_poisson(self, bc):
        """Solve Poisson equation."""
        return self._poisson.solve(bc)

    def solve_drift_diffusion(self, bc, Vg=0.0, max_steps=100, use_amr=False,
                              poisson_max_iter=50, poisson_tol=1e-6):
        """Solve drift-diffusion equations."""
        return self._dd.solve(bc, Vg, max_steps, use_amr, poisson_max_iter, poisson_tol)

    # Additional convenience methods
    def get_device(self):
        """Get the device object."""
        return self._device

    def get_poisson_solver(self):
        """Get the Poisson solver object."""
        return self._poisson

    def get_drift_diffusion_solver(self):
        """Get the drift-diffusion solver object."""
        return self._dd

    # Python properties to access C attributes
    @property
    def method(self):
        """Get the numerical method"""
        return self._method

    @property
    def mesh_type(self):
        """Get the mesh type"""
        return self._mesh_type

    @property
    def num_points_x(self):
        """Get number of grid points in x direction"""
        return self._num_points_x

    @property
    def num_points_y(self):
        """Get number of grid points in y direction"""
        return self._num_points_y


class SelfConsistentSolver:
    """
    Self-consistent solver for comprehensive semiconductor device simulation

    Provides coupled solving of Poisson, drift-diffusion, and transport equations
    with proper iterative solving and convergence criteria.
    """
    def __init__(self, device, method="DG", mesh_type="Structured", order=3):
        """
        Initialize self-consistent solver

        Parameters:
        -----------
        device : Device
            Device object
        method : str
            Numerical method ("DG", "FEM", "FVM")
        mesh_type : str
            Mesh type ("Structured", "Unstructured")
        order : int
            Polynomial order for DG methods
        """
        if not SELF_CONSISTENT_AVAILABLE:
            raise ImportError("Self-consistent solver module not available. "
                             "Please build the self_consistent_solver extension first.")

        if isinstance(device, Device):
            self.device = device
            self.solver = scs.create_self_consistent_solver(
                device.width, device.height, method, mesh_type, order)
        else:
            raise TypeError("device must be a Device object")

        logger.info(f"Self-consistent solver created with {method} method")

    def set_convergence_criteria(self, potential_tolerance=1e-6, density_tolerance=1e-3,
                               current_tolerance=1e-3, max_iterations=100):
        """
        Set convergence criteria for self-consistent iterations

        Parameters:
        -----------
        potential_tolerance : float
            Potential convergence tolerance (V)
        density_tolerance : float
            Carrier density relative tolerance
        current_tolerance : float
            Current density relative tolerance
        max_iterations : int
            Maximum self-consistent iterations
        """
        self.solver.set_convergence_criteria(
            potential_tolerance, density_tolerance, current_tolerance, max_iterations)

        logger.debug(f"Convergence criteria set: pot_tol={potential_tolerance}, "
                    f"dens_tol={density_tolerance}, max_iter={max_iterations}")

    def set_doping(self, Nd, Na):
        """
        Set doping concentrations

        Parameters:
        -----------
        Nd : array_like
            Donor concentration (m^-3)
        Na : array_like
            Acceptor concentration (m^-3)
        """
        Nd = np.asarray(Nd, dtype=np.float64)
        Na = np.asarray(Na, dtype=np.float64)

        if Nd.shape != Na.shape:
            raise ValueError("Nd and Na arrays must have the same shape")

        self.solver.set_doping(Nd, Na)

        logger.debug(f"Doping set successfully: {len(Nd)} points")
        logger.debug(f"  Nd range: [{np.min(Nd)/1e6:.0e}, {np.max(Nd)/1e6:.0e}] cm⁻³")
        logger.debug(f"  Na range: [{np.min(Na)/1e6:.0e}, {np.max(Na)/1e6:.0e}] cm⁻³")

    def enable_energy_transport(self, enable=True):
        """Enable energy transport equations"""
        self.solver.enable_energy_transport(enable)
        logger.info(f"Energy transport {'enabled' if enable else 'disabled'}")

    def enable_hydrodynamic_transport(self, enable=True):
        """Enable hydrodynamic transport equations"""
        self.solver.enable_hydrodynamic_transport(enable)
        logger.info(f"Hydrodynamic transport {'enabled' if enable else 'disabled'}")

    def enable_quantum_corrections(self, enable=True):
        """Enable quantum corrections"""
        self.solver.enable_quantum_corrections(enable)
        logger.info(f"Quantum corrections {'enabled' if enable else 'disabled'}")

    def solve(self, boundary_conditions, initial_potential=None, initial_n=None, initial_p=None):
        """
        Solve self-consistent equations

        Parameters:
        -----------
        boundary_conditions : list
            Boundary conditions [left, right, bottom, top] in Volts
        initial_potential : array_like, optional
            Initial potential guess (V)
        initial_n : array_like, optional
            Initial electron density guess (m^-3)
        initial_p : array_like, optional
            Initial hole density guess (m^-3)

        Returns:
        --------
        dict
            Results with 'potential', 'n', 'p', 'iterations', 'residual'
        """
        if len(boundary_conditions) != 4:
            raise ValueError("Boundary conditions must have 4 values")

        dof_count = self.solver.get_dof_count()

        # Create default initial conditions if not provided
        if initial_potential is None:
            initial_potential = np.zeros(dof_count, dtype=np.float64)
        else:
            initial_potential = np.asarray(initial_potential, dtype=np.float64)

        if initial_n is None:
            initial_n = np.full(dof_count, 1e10, dtype=np.float64)
        else:
            initial_n = np.asarray(initial_n, dtype=np.float64)

        if initial_p is None:
            initial_p = np.full(dof_count, 1e10, dtype=np.float64)
        else:
            initial_p = np.asarray(initial_p, dtype=np.float64)

        # Validate array sizes
        if (len(initial_potential) != dof_count or
            len(initial_n) != dof_count or
            len(initial_p) != dof_count):
            raise ValueError(f"Initial arrays must have {dof_count} elements")

        # Solve
        logger.info(f"Solving self-consistent equations with boundary conditions: {boundary_conditions}")
        results = self.solver.solve_steady_state(
            boundary_conditions, initial_potential, initial_n, initial_p)

        logger.info(f"Self-consistent solution converged in {results['iterations']} iterations")
        logger.info(f"Final residual: {results['residual']:.2e}")

        return results

    def get_dof_count(self):
        """Get degrees of freedom count"""
        return self.solver.get_dof_count()

    def is_valid(self):
        """Check if solver is valid"""
        return self.solver.is_valid()


class MaterialDatabase:
    """
    Material properties database for semiconductor simulation
    """
    def __init__(self):
        """Initialize material database"""
        if not SELF_CONSISTENT_AVAILABLE:
            raise ImportError("Material database module not available. "
                             "Please build the self_consistent_solver extension first.")

        self.database = scs.create_material_database()
        logger.info("Material database initialized")

    def get_bandgap(self, material_type, temperature=300.0):
        """Get bandgap at specified temperature"""
        return self.database.get_bandgap(material_type, temperature)

    def get_intrinsic_concentration(self, material_type, temperature=300.0):
        """Get intrinsic carrier concentration at specified temperature"""
        return self.database.get_intrinsic_concentration(material_type, temperature)

    def get_electron_mobility(self, material_type, temperature=300.0, doping=1e16):
        """Get electron mobility at specified temperature and doping"""
        return self.database.get_electron_mobility(material_type, temperature, doping)

    def get_hole_mobility(self, material_type, temperature=300.0, doping=1e16):
        """Get hole mobility at specified temperature and doping"""
        return self.database.get_hole_mobility(material_type, temperature, doping)

    def calculate_srh_recombination(self, n, p, ni, tau_n=1e-6, tau_p=1e-6):
        """Calculate Shockley-Read-Hall recombination rate"""
        return self.database.calculate_srh_recombination(n, p, ni, tau_n, tau_p)

    def calculate_impact_ionization(self, E_field, a, b):
        """Calculate impact ionization rate"""
        return self.database.calculate_impact_ionization(E_field, a, b)


# Material type constants
class MaterialType:
    SILICON = 0
    GERMANIUM = 1
    GALLIUM_ARSENIDE = 2
    SILICON_CARBIDE = 3
    GALLIUM_NITRIDE = 4
    INDIUM_GALLIUM_ARSENIDE = 5
    CUSTOM = 99