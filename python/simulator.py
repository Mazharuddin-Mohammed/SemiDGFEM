import numpy as np
import os
import sys
import logging
from ctypes import cdll, c_int, c_double, c_char_p, POINTER, Structure, c_bool, c_void_p, c_size_t
from typing import Dict, List, Optional, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulatorError(Exception):
    """Custom exception for simulator errors."""
    pass

class Mesh(Structure):
    """Opaque mesh structure for ctypes."""
    _fields_ = []

class Device(Structure):
    """Opaque device structure for ctypes."""
    _fields_ = []

class Simulator:
    """
    High-level interface to the SemiDGFEM semiconductor device simulator.

    This class provides a Python interface to the C++ simulation library,
    handling device creation, mesh generation, and physics solving with
    comprehensive error handling and validation.
    """

    def __init__(self, dimension: str = "TwoD",
                 extents: List[float] = [1e-6, 0.5e-6],
                 num_points_x: int = 50,
                 num_points_y: int = 25,
                 method: str = "DG",
                 mesh_type: str = "Structured",
                 regions: Optional[List] = None,
                 order: str = "P3"):
        """
        Initialize the simulator with comprehensive validation.

        Args:
            dimension: Simulation dimension ("TwoD" only supported)
            extents: Device dimensions [Lx, Ly] in meters
            num_points_x: Number of grid points in x direction
            num_points_y: Number of grid points in y direction
            method: Numerical method ("DG" only supported)
            mesh_type: Mesh type ("Structured" or "Unstructured")
            regions: Optional list of material regions
            order: Element order ("P2" or "P3")

        Raises:
            SimulatorError: If initialization fails
            ValueError: If invalid parameters are provided
        """
        # Validate inputs
        self._validate_inputs(dimension, extents, num_points_x, num_points_y, method, mesh_type, order)

        # Store parameters
        self.dimension = dimension
        self.extents = list(extents)  # Make a copy
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.method_str = method
        self.mesh_type_str = mesh_type
        self.order_str = order
        self.regions = regions or []

        # Map string parameters to integers
        self.method = {"FDM": 0, "FEM": 1, "FVM": 2, "SEM": 3, "MC": 4, "DG": 5}[method]
        self.mesh_type = {"Structured": 0, "Unstructured": 1}[mesh_type]
        self.order = {"P2": 2, "P3": 3}[order]

        # Initialize library and objects
        self.lib = None
        self.device = None
        self.mesh = None
        self.drift_diffusion = None

        try:
            self._load_library()
            self._setup_function_signatures()
            self._create_objects()
            logger.info("Simulator initialized successfully")
        except Exception as e:
            self._cleanup()
            raise SimulatorError(f"Failed to initialize simulator: {e}")

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self._cleanup()

    def _validate_inputs(self, dimension: str, extents: List[float],
                        num_points_x: int, num_points_y: int,
                        method: str, mesh_type: str, order: str) -> None:
        """Validate input parameters with detailed error messages."""
        if not isinstance(dimension, str) or dimension != "TwoD":
            raise ValueError("Only 2D simulations are currently supported")

        if not isinstance(extents, (list, tuple)) or len(extents) != 2:
            raise ValueError("Extents must be a list or tuple of 2 values [Lx, Ly]")

        if not all(isinstance(x, (int, float)) and x > 0 for x in extents):
            raise ValueError("All extents must be positive numbers")

        if not isinstance(num_points_x, int) or num_points_x <= 0:
            raise ValueError("num_points_x must be a positive integer")

        if not isinstance(num_points_y, int) or num_points_y <= 0:
            raise ValueError("num_points_y must be a positive integer")

        if method not in ["DG"]:
            raise ValueError("Only DG method is currently supported")

        if mesh_type not in ["Structured", "Unstructured"]:
            raise ValueError("Mesh type must be 'Structured' or 'Unstructured'")

        if order not in ["P2", "P3"]:
            raise ValueError("Order must be 'P2' or 'P3'")

    def _load_library(self) -> None:
        """Load the shared library with multiple fallback paths."""
        # Try multiple possible library locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "build", "libsimulator.so"),
            os.path.join(os.path.dirname(__file__), "..", "libsimulator.so"),
            "./simulator/libsimulator.so",
            "libsimulator.so",
            "./libsimulator.so"
        ]

        for lib_path in possible_paths:
            if os.path.exists(lib_path):
                try:
                    self.lib = cdll.LoadLibrary(lib_path)
                    logger.info(f"Loaded library from: {lib_path}")
                    return
                except OSError as e:
                    logger.warning(f"Failed to load library from {lib_path}: {e}")
                    continue

        raise SimulatorError(
            f"Could not find or load libsimulator.so. Searched paths: {possible_paths}"
        )

    def _setup_function_signatures(self) -> None:
        """Set up ctypes function signatures for type safety and error checking."""
        try:
            # Device functions
            self.lib.create_device.argtypes = [c_double, c_double]
            self.lib.create_device.restype = c_void_p

            self.lib.destroy_device.argtypes = [c_void_p]
            self.lib.destroy_device.restype = None

            self.lib.device_get_epsilon_at.argtypes = [c_void_p, c_double, c_double]
            self.lib.device_get_epsilon_at.restype = c_double

            self.lib.device_get_extents.argtypes = [c_void_p, POINTER(c_double)]
            self.lib.device_get_extents.restype = None

            # Mesh functions
            self.lib.create_mesh.argtypes = [c_void_p, c_int]
            self.lib.create_mesh.restype = c_void_p

            self.lib.destroy_mesh.argtypes = [c_void_p]
            self.lib.destroy_mesh.restype = None

            self.lib.mesh_get_num_nodes.argtypes = [c_void_p]
            self.lib.mesh_get_num_nodes.restype = c_int

            self.lib.mesh_get_num_elements.argtypes = [c_void_p]
            self.lib.mesh_get_num_elements.restype = c_int

            self.lib.mesh_get_grid_points_x.argtypes = [c_void_p, POINTER(c_double), c_int]
            self.lib.mesh_get_grid_points_x.restype = None

            self.lib.mesh_get_grid_points_y.argtypes = [c_void_p, POINTER(c_double), c_int]
            self.lib.mesh_get_grid_points_y.restype = None

            self.lib.mesh_generate_gmsh.argtypes = [c_void_p, c_char_p]
            self.lib.mesh_generate_gmsh.restype = None

            # Drift-diffusion functions
            self.lib.create_drift_diffusion.argtypes = [c_void_p, c_int, c_int, c_int]
            self.lib.create_drift_diffusion.restype = c_void_p

            self.lib.destroy_drift_diffusion.argtypes = [c_void_p]
            self.lib.destroy_drift_diffusion.restype = None

            self.lib.drift_diffusion_is_valid.argtypes = [c_void_p]
            self.lib.drift_diffusion_is_valid.restype = c_int

            self.lib.drift_diffusion_set_doping.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double), c_int]
            self.lib.drift_diffusion_set_doping.restype = c_int

            self.lib.drift_diffusion_set_trap_level.argtypes = [c_void_p, POINTER(c_double), c_int]
            self.lib.drift_diffusion_set_trap_level.restype = c_int

            self.lib.drift_diffusion_solve.argtypes = [
                c_void_p, POINTER(c_double), c_int, c_double, c_int, c_int, c_int, c_double,
                POINTER(c_double), POINTER(c_double), POINTER(c_double),
                POINTER(c_double), POINTER(c_double), c_int
            ]
            self.lib.drift_diffusion_solve.restype = c_int

        except AttributeError as e:
            raise SimulatorError(f"Missing required function in library: {e}")

    def _create_objects(self) -> None:
        """Create device, mesh, and solver objects with error checking."""
        # Create device
        self.device = self.lib.create_device(self.extents[0], self.extents[1])
        if not self.device:
            raise SimulatorError("Failed to create device")

        # Create mesh
        self.mesh = self.lib.create_mesh(self.device, self.mesh_type)
        if not self.mesh:
            raise SimulatorError("Failed to create mesh")

        # Create drift-diffusion solver
        self.drift_diffusion = self.lib.create_drift_diffusion(
            self.device, self.method, self.mesh_type, self.order
        )
        if not self.drift_diffusion:
            raise SimulatorError("Failed to create drift-diffusion solver")

        # Validate objects
        if not self.lib.drift_diffusion_is_valid(self.drift_diffusion):
            raise SimulatorError("Drift-diffusion solver is in invalid state")

    def _cleanup(self) -> None:
        """Clean up allocated resources with error handling."""
        if self.lib:
            if self.drift_diffusion:
                try:
                    self.lib.destroy_drift_diffusion(self.drift_diffusion)
                except Exception as e:
                    logger.warning(f"Error destroying drift-diffusion solver: {e}")
                self.drift_diffusion = None

            if self.mesh:
                try:
                    self.lib.destroy_mesh(self.mesh)
                except Exception as e:
                    logger.warning(f"Error destroying mesh: {e}")
                self.mesh = None

            if self.device:
                try:
                    self.lib.destroy_device(self.device)
                except Exception as e:
                    logger.warning(f"Error destroying device: {e}")
                self.device = None

    def generate_mesh(self, filename: str) -> None:
        """
        Generate mesh file using GMSH.

        Args:
            filename: Output mesh filename

        Raises:
            SimulatorError: If mesh generation fails
            ValueError: If filename is invalid
        """
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Filename must be a non-empty string")

        if not self.mesh:
            raise SimulatorError("Mesh object not initialized")

        try:
            self.lib.mesh_generate_gmsh(self.mesh, c_char_p(filename.encode('utf-8')))
            logger.info(f"Mesh generated successfully: {filename}")
        except Exception as e:
            raise SimulatorError(f"Failed to generate mesh: {e}")

    def get_grid_points(self) -> Dict[str, List[float]]:
        """
        Get mesh grid points.

        Returns:
            Dictionary with 'x' and 'y' coordinate arrays

        Raises:
            SimulatorError: If grid point retrieval fails
        """
        if not self.mesh:
            raise SimulatorError("Mesh object not initialized")

        try:
            # Get number of nodes
            num_nodes = self.lib.mesh_get_num_nodes(self.mesh)
            if num_nodes <= 0:
                raise SimulatorError("Invalid number of mesh nodes")

            # Allocate arrays
            grid_x = (c_double * num_nodes)()
            grid_y = (c_double * num_nodes)()

            # Get grid points
            self.lib.mesh_get_grid_points_x(self.mesh, grid_x, num_nodes)
            self.lib.mesh_get_grid_points_y(self.mesh, grid_y, num_nodes)

            return {
                "x": [float(x) for x in grid_x],
                "y": [float(y) for y in grid_y]
            }
        except Exception as e:
            raise SimulatorError(f"Failed to get grid points: {e}")

    def set_doping(self, Nd: Union[List[float], np.ndarray],
                   Na: Union[List[float], np.ndarray]) -> None:
        """
        Set doping profiles.

        Args:
            Nd: Donor concentration array (1/m³)
            Na: Acceptor concentration array (1/m³)

        Raises:
            SimulatorError: If doping setup fails
            ValueError: If arrays are invalid
        """
        if not self.drift_diffusion:
            raise SimulatorError("Drift-diffusion solver not initialized")

        # Convert to numpy arrays for validation
        Nd = np.asarray(Nd, dtype=float)
        Na = np.asarray(Na, dtype=float)

        if Nd.size == 0 or Na.size == 0:
            raise ValueError("Doping arrays cannot be empty")

        if Nd.size != Na.size:
            raise ValueError("Nd and Na arrays must have the same size")

        if np.any(Nd < 0) or np.any(Na < 0):
            raise ValueError("Doping concentrations must be non-negative")

        try:
            # Convert to ctypes arrays
            Nd_array = (c_double * len(Nd))(*Nd)
            Na_array = (c_double * len(Na))(*Na)

            # Call library function
            result = self.lib.drift_diffusion_set_doping(
                self.drift_diffusion, Nd_array, Na_array, len(Nd)
            )

            if result != 0:
                raise SimulatorError("Failed to set doping profiles")

            logger.info(f"Doping profiles set successfully ({len(Nd)} points)")
        except Exception as e:
            raise SimulatorError(f"Failed to set doping: {e}")

    def set_trap_level(self, Et: Union[List[float], np.ndarray]) -> None:
        """
        Set trap energy levels.

        Args:
            Et: Trap energy level array (eV)

        Raises:
            SimulatorError: If trap level setup fails
            ValueError: If array is invalid
        """
        if not self.drift_diffusion:
            raise SimulatorError("Drift-diffusion solver not initialized")

        # Convert to numpy array for validation
        Et = np.asarray(Et, dtype=float)

        if Et.size == 0:
            raise ValueError("Trap level array cannot be empty")

        if not np.all(np.isfinite(Et)):
            raise ValueError("All trap levels must be finite")

        try:
            # Convert to ctypes array
            Et_array = (c_double * len(Et))(*Et)

            # Call library function
            result = self.lib.drift_diffusion_set_trap_level(
                self.drift_diffusion, Et_array, len(Et)
            )

            if result != 0:
                raise SimulatorError("Failed to set trap levels")

            logger.info(f"Trap levels set successfully ({len(Et)} points)")
        except Exception as e:
            raise SimulatorError(f"Failed to set trap levels: {e}")

    def solve_drift_diffusion(self, bc: Union[List[float], np.ndarray],
                            Vg: float = 0.0,
                            max_steps: int = 100,
                            use_amr: bool = False,
                            poisson_max_iter: int = 50,
                            poisson_tol: float = 1e-6) -> Dict[str, np.ndarray]:
        """
        Solve drift-diffusion equations.

        Args:
            bc: Boundary conditions [left, right, bottom, top] (V)
            Vg: Gate voltage (V)
            max_steps: Maximum iteration steps
            use_amr: Use adaptive mesh refinement
            poisson_max_iter: Maximum Poisson iterations
            poisson_tol: Poisson convergence tolerance

        Returns:
            Dictionary with solution arrays: potential, n, p, Jn, Jp

        Raises:
            SimulatorError: If solution fails
            ValueError: If parameters are invalid
        """
        if not self.drift_diffusion:
            raise SimulatorError("Drift-diffusion solver not initialized")

        # Validate inputs
        bc = np.asarray(bc, dtype=float)
        if bc.size != 4:
            raise ValueError("Boundary conditions must have exactly 4 values")

        if not np.all(np.isfinite(bc)):
            raise ValueError("All boundary conditions must be finite")

        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        if not isinstance(poisson_max_iter, int) or poisson_max_iter <= 0:
            raise ValueError("poisson_max_iter must be a positive integer")

        if not isinstance(poisson_tol, (int, float)) or poisson_tol <= 0:
            raise ValueError("poisson_tol must be positive")

        try:
            # Get number of nodes for result arrays
            num_nodes = self.lib.mesh_get_num_nodes(self.mesh)
            if num_nodes <= 0:
                raise SimulatorError("Invalid number of mesh nodes")

            # Prepare input arrays
            bc_array = (c_double * 4)(*bc)

            # Prepare output arrays
            V_array = (c_double * num_nodes)()
            n_array = (c_double * num_nodes)()
            p_array = (c_double * num_nodes)()
            Jn_array = (c_double * num_nodes)()
            Jp_array = (c_double * num_nodes)()

            # Call solver
            result = self.lib.drift_diffusion_solve(
                self.drift_diffusion, bc_array, 4, c_double(Vg),
                max_steps, int(use_amr), poisson_max_iter, c_double(poisson_tol),
                V_array, n_array, p_array, Jn_array, Jp_array, num_nodes
            )

            if result != 0:
                raise SimulatorError("Drift-diffusion solution failed")

            # Convert results to numpy arrays
            results = {
                "potential": np.array([float(v) for v in V_array]),
                "n": np.array([float(n) for n in n_array]),
                "p": np.array([float(p) for p in p_array]),
                "Jn": np.array([float(jn) for jn in Jn_array]),
                "Jp": np.array([float(jp) for jp in Jp_array])
            }

            logger.info("Drift-diffusion solution completed successfully")
            return results

        except Exception as e:
            raise SimulatorError(f"Failed to solve drift-diffusion: {e}")

    def get_device_info(self) -> Dict[str, Union[float, List[float]]]:
        """
        Get device information.

        Returns:
            Dictionary with device properties
        """
        if not self.device:
            raise SimulatorError("Device not initialized")

        try:
            extents_array = (c_double * 2)()
            self.lib.device_get_extents(self.device, extents_array)

            return {
                "extents": [float(extents_array[0]), float(extents_array[1])],
                "num_points_x": self.num_points_x,
                "num_points_y": self.num_points_y,
                "method": self.method_str,
                "mesh_type": self.mesh_type_str,
                "order": self.order_str
            }
        except Exception as e:
            raise SimulatorError(f"Failed to get device info: {e}")

    def is_valid(self) -> bool:
        """
        Check if simulator is in valid state.

        Returns:
            True if valid, False otherwise
        """
        try:
            return (self.lib is not None and
                   self.device is not None and
                   self.mesh is not None and
                   self.drift_diffusion is not None and
                   self.lib.drift_diffusion_is_valid(self.drift_diffusion))
        except:
            return False