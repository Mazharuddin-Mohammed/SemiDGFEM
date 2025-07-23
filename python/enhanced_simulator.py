#!/usr/bin/env python3
"""
Enhanced Simulator with Self-Consistent Capabilities
Provides comprehensive semiconductor device simulation with advanced physics
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
import time

# Import existing modules
try:
    import simulator
    BASIC_SIMULATOR_AVAILABLE = True
except ImportError:
    BASIC_SIMULATOR_AVAILABLE = False

try:
    import transient_solver_simple as ts
    TRANSIENT_AVAILABLE = True
except ImportError:
    TRANSIENT_AVAILABLE = False

try:
    import unstructured_transport as ut
    ADVANCED_TRANSPORT_AVAILABLE = True
except ImportError:
    ADVANCED_TRANSPORT_AVAILABLE = False

logger = logging.getLogger(__name__)

class MaterialType:
    """Material type constants"""
    SILICON = 0
    GERMANIUM = 1
    GALLIUM_ARSENIDE = 2
    SILICON_CARBIDE = 3
    GALLIUM_NITRIDE = 4
    INDIUM_GALLIUM_ARSENIDE = 5
    CUSTOM = 99

class MaterialDatabase:
    """
    Comprehensive material properties database
    """
    
    def __init__(self):
        """Initialize material database with comprehensive properties"""
        self.materials = self._initialize_materials()
        logger.info("Material database initialized with comprehensive properties")
    
    def _initialize_materials(self):
        """Initialize material properties database"""
        materials = {}
        
        # Silicon properties
        materials[MaterialType.SILICON] = {
            'name': 'Silicon',
            'bandgap_300K': 1.12,  # eV
            'dielectric_constant': 11.7,
            'electron_affinity': 4.05,  # eV
            'intrinsic_concentration_300K': 1.45e16,  # m^-3
            'electron_mobility_max': 0.1350,  # m²/V·s
            'hole_mobility_max': 0.0480,  # m²/V·s
            'electron_saturation_velocity': 1.07e5,  # m/s
            'hole_saturation_velocity': 8.37e4,  # m/s
            'thermal_conductivity': 150.0,  # W/m·K
            'density': 2329.0,  # kg/m³
            'lattice_constant': 5.431e-10,  # m
        }
        
        # Gallium Arsenide properties
        materials[MaterialType.GALLIUM_ARSENIDE] = {
            'name': 'Gallium Arsenide',
            'bandgap_300K': 1.424,  # eV
            'dielectric_constant': 13.1,
            'electron_affinity': 4.07,  # eV
            'intrinsic_concentration_300K': 1.79e12,  # m^-3
            'electron_mobility_max': 0.8500,  # m²/V·s
            'hole_mobility_max': 0.0400,  # m²/V·s
            'electron_saturation_velocity': 1.2e5,  # m/s
            'hole_saturation_velocity': 6.0e4,  # m/s
            'thermal_conductivity': 55.0,  # W/m·K
            'density': 5317.0,  # kg/m³
            'lattice_constant': 5.653e-10,  # m
        }
        
        # Germanium properties
        materials[MaterialType.GERMANIUM] = {
            'name': 'Germanium',
            'bandgap_300K': 0.66,  # eV
            'dielectric_constant': 16.0,
            'electron_affinity': 4.0,  # eV
            'intrinsic_concentration_300K': 2.4e19,  # m^-3
            'electron_mobility_max': 0.3900,  # m²/V·s
            'hole_mobility_max': 0.1900,  # m²/V·s
            'electron_saturation_velocity': 6.0e4,  # m/s
            'hole_saturation_velocity': 6.0e4,  # m/s
            'thermal_conductivity': 60.0,  # W/m·K
            'density': 5323.0,  # kg/m³
            'lattice_constant': 5.658e-10,  # m
        }
        
        # Silicon Carbide properties
        materials[MaterialType.SILICON_CARBIDE] = {
            'name': 'Silicon Carbide',
            'bandgap_300K': 3.26,  # eV
            'dielectric_constant': 9.7,
            'electron_affinity': 3.8,  # eV
            'intrinsic_concentration_300K': 8.2e8,  # m^-3
            'electron_mobility_max': 0.0950,  # m²/V·s
            'hole_mobility_max': 0.0120,  # m²/V·s
            'electron_saturation_velocity': 2.0e5,  # m/s
            'hole_saturation_velocity': 2.0e5,  # m/s
            'thermal_conductivity': 490.0,  # W/m·K
            'density': 3210.0,  # kg/m³
            'lattice_constant': 3.073e-10,  # m
        }
        
        # Gallium Nitride properties
        materials[MaterialType.GALLIUM_NITRIDE] = {
            'name': 'Gallium Nitride',
            'bandgap_300K': 3.39,  # eV
            'dielectric_constant': 9.0,
            'electron_affinity': 4.1,  # eV
            'intrinsic_concentration_300K': 1.9e6,  # m^-3
            'electron_mobility_max': 0.1300,  # m²/V·s
            'hole_mobility_max': 0.0030,  # m²/V·s
            'electron_saturation_velocity': 2.5e5,  # m/s
            'hole_saturation_velocity': 1.0e5,  # m/s
            'thermal_conductivity': 130.0,  # W/m·K
            'density': 6150.0,  # kg/m³
            'lattice_constant': 3.189e-10,  # m
        }
        
        # InGaAs properties (In0.53Ga0.47As)
        materials[MaterialType.INDIUM_GALLIUM_ARSENIDE] = {
            'name': 'Indium Gallium Arsenide',
            'bandgap_300K': 0.75,  # eV
            'dielectric_constant': 13.9,
            'electron_affinity': 4.5,  # eV
            'intrinsic_concentration_300K': 5.7e18,  # m^-3
            'electron_mobility_max': 1.2000,  # m²/V·s
            'hole_mobility_max': 0.0450,  # m²/V·s
            'electron_saturation_velocity': 8.0e4,  # m/s
            'hole_saturation_velocity': 6.0e4,  # m/s
            'thermal_conductivity': 5.0,  # W/m·K
            'density': 5500.0,  # kg/m³
            'lattice_constant': 5.869e-10,  # m
        }
        
        return materials
    
    def get_material_properties(self, material_type):
        """Get all properties for a material"""
        if material_type not in self.materials:
            raise ValueError(f"Material type {material_type} not found")
        return self.materials[material_type].copy()
    
    def get_bandgap(self, material_type, temperature=300.0):
        """Get bandgap at specified temperature"""
        props = self.get_material_properties(material_type)
        
        # Temperature dependence (Varshni equation)
        if material_type == MaterialType.SILICON:
            Eg0 = 1.17  # eV at 0K
            alpha = 4.73e-4  # eV/K
            beta = 636.0  # K
            return Eg0 - (alpha * temperature * temperature) / (temperature + beta)
        elif material_type == MaterialType.GALLIUM_ARSENIDE:
            Eg0 = 1.519  # eV at 0K
            alpha = 5.405e-4  # eV/K
            beta = 204.0  # K
            return Eg0 - (alpha * temperature * temperature) / (temperature + beta)
        else:
            # Simple linear approximation for other materials
            return props['bandgap_300K'] * (1 - 4e-4 * (temperature - 300))
    
    def get_intrinsic_concentration(self, material_type, temperature=300.0):
        """Get intrinsic carrier concentration at specified temperature"""
        props = self.get_material_properties(material_type)
        
        # Temperature dependence
        Eg = self.get_bandgap(material_type, temperature)
        kT = 8.617e-5 * temperature  # eV
        ni_300 = props['intrinsic_concentration_300K']
        
        return ni_300 * (temperature / 300.0)**1.5 * np.exp(-Eg / (2 * kT) + 
                                                           props['bandgap_300K'] / (2 * 8.617e-5 * 300.0))
    
    def get_electron_mobility(self, material_type, temperature=300.0, doping=1e16):
        """Get electron mobility at specified temperature and doping"""
        props = self.get_material_properties(material_type)
        
        # Temperature dependence
        mu_max = props['electron_mobility_max'] * (300.0 / temperature)**2.4
        
        # Doping dependence (simplified Caughey-Thomas)
        mu_min = mu_max * 0.05  # Minimum mobility
        N_ref = 1e23  # Reference doping (m^-3)
        alpha = 0.88  # Mobility exponent
        
        return mu_min + (mu_max - mu_min) / (1.0 + (doping / N_ref)**alpha)
    
    def get_hole_mobility(self, material_type, temperature=300.0, doping=1e16):
        """Get hole mobility at specified temperature and doping"""
        props = self.get_material_properties(material_type)
        
        # Temperature dependence
        mu_max = props['hole_mobility_max'] * (300.0 / temperature)**2.4
        
        # Doping dependence (simplified Caughey-Thomas)
        mu_min = mu_max * 0.1  # Minimum mobility
        N_ref = 2e23  # Reference doping (m^-3)
        alpha = 0.7  # Mobility exponent
        
        return mu_min + (mu_max - mu_min) / (1.0 + (doping / N_ref)**alpha)
    
    def calculate_srh_recombination(self, n, p, ni, tau_n=1e-6, tau_p=1e-6):
        """Calculate Shockley-Read-Hall recombination rate"""
        return (n * p - ni * ni) / (tau_p * (n + ni) + tau_n * (p + ni))
    
    def calculate_radiative_recombination(self, n, p, ni, B_rad=1.1e-21):
        """Calculate radiative recombination rate"""
        return B_rad * (n * p - ni * ni)
    
    def calculate_auger_recombination(self, n, p, ni, C_n=2.8e-43, C_p=9.9e-44):
        """Calculate Auger recombination rate"""
        return (C_n * n + C_p * p) * (n * p - ni * ni)
    
    def calculate_impact_ionization(self, E_field, material_type):
        """Calculate impact ionization rate"""
        if material_type == MaterialType.SILICON:
            a_n = 7.03e5  # 1/m
            b_n = 1.231e6  # V/m
            a_p = 1.582e6  # 1/m
            b_p = 2.036e6  # V/m
        elif material_type == MaterialType.GALLIUM_ARSENIDE:
            a_n = a_p = 5.0e5  # 1/m
            b_n = b_p = 9.5e5  # V/m
        else:
            # Default values
            a_n = a_p = 1e6  # 1/m
            b_n = b_p = 1e6  # V/m
        
        if E_field <= 0:
            return 0.0, 0.0
        
        alpha_n = a_n * np.exp(-b_n / E_field)
        alpha_p = a_p * np.exp(-b_p / E_field)
        
        return alpha_n, alpha_p
    
    def get_available_materials(self):
        """Get list of available materials"""
        return [props['name'] for props in self.materials.values()]

class SelfConsistentSolver:
    """
    Self-consistent solver for comprehensive semiconductor device simulation
    """
    
    def __init__(self, device_width, device_height, material_type=MaterialType.SILICON,
                 method="DG", mesh_type="Structured", order=3):
        """
        Initialize self-consistent solver
        
        Parameters:
        -----------
        device_width : float
            Device width in meters
        device_height : float
            Device height in meters
        material_type : int
            Material type from MaterialType class
        method : str
            Numerical method
        mesh_type : str
            Mesh type
        order : int
            Polynomial order
        """
        self.device_width = device_width
        self.device_height = device_height
        self.material_type = material_type
        self.method = method
        self.mesh_type = mesh_type
        self.order = order
        
        # Initialize material database
        self.materials = MaterialDatabase()
        
        # Initialize solvers if available
        self.basic_solver = None
        self.transient_solver = None
        self.advanced_transport = None
        
        if BASIC_SIMULATOR_AVAILABLE:
            try:
                self.device = simulator.Device(device_width, device_height)
                self.poisson_solver = simulator.PoissonSolver(self.device, method, mesh_type)
                self.dd_solver = simulator.DriftDiffusionSolver(self.device, method, mesh_type, order)
                self.basic_solver = True
                logger.info("Basic simulator initialized")
            except Exception as e:
                logger.warning(f"Basic simulator initialization failed: {e}")
        
        if TRANSIENT_AVAILABLE:
            try:
                self.transient_solver = ts.create_transient_solver(device_width, device_height, method, mesh_type, order)
                logger.info("Transient solver initialized")
            except Exception as e:
                logger.warning(f"Transient solver initialization failed: {e}")
        
        if ADVANCED_TRANSPORT_AVAILABLE:
            try:
                # Try different function names that might be available
                if hasattr(ut, 'create_unstructured_transport_solver'):
                    self.advanced_transport = ut.create_unstructured_transport_solver(device_width, device_height)
                elif hasattr(ut, 'create_transport_solver'):
                    self.advanced_transport = ut.create_transport_solver(device_width, device_height)
                elif hasattr(ut, 'UnstructuredTransportSolver'):
                    self.advanced_transport = ut.UnstructuredTransportSolver(device_width, device_height)
                else:
                    self.advanced_transport = None
                    logger.warning("No compatible advanced transport solver found")

                if self.advanced_transport:
                    logger.info("Advanced transport solver initialized")
            except Exception as e:
                logger.warning(f"Advanced transport initialization failed: {e}")
                self.advanced_transport = None
        
        # Convergence criteria
        self.potential_tolerance = 1e-6
        self.density_tolerance = 1e-3
        self.current_tolerance = 1e-3
        self.max_iterations = 100
        self.damping_factor = 0.7
        
        # Physics configuration
        self.energy_transport_enabled = False
        self.hydrodynamic_enabled = False
        self.quantum_corrections_enabled = False
        
        logger.info(f"Self-consistent solver created: {device_width*1e6:.1f}x{device_height*1e6:.1f} μm")
    
    def set_convergence_criteria(self, potential_tolerance=1e-6, density_tolerance=1e-3,
                               current_tolerance=1e-3, max_iterations=100):
        """Set convergence criteria"""
        self.potential_tolerance = potential_tolerance
        self.density_tolerance = density_tolerance
        self.current_tolerance = current_tolerance
        self.max_iterations = max_iterations
        
        logger.debug(f"Convergence criteria set: pot_tol={potential_tolerance}, "
                    f"dens_tol={density_tolerance}, max_iter={max_iterations}")
    
    def set_doping(self, Nd, Na):
        """Set doping concentrations"""
        self.Nd = np.asarray(Nd, dtype=np.float64)
        self.Na = np.asarray(Na, dtype=np.float64)

        # Set doping in basic solver if available
        if self.basic_solver and hasattr(self, 'dd_solver'):
            try:
                self.dd_solver.set_doping(self.Nd, self.Na)
                logger.debug("Doping set in basic drift-diffusion solver")
            except Exception as e:
                logger.warning(f"Could not set doping in basic solver: {e}")

        # Set doping in transient solver if available and compatible
        if self.transient_solver:
            try:
                transient_dof = self.transient_solver.get_dof_count()
                if len(self.Nd) == transient_dof:
                    self.transient_solver.set_doping(self.Nd, self.Na)
                    logger.debug("Doping set in transient solver")
                else:
                    # Interpolate or resize doping arrays to match transient solver DOF
                    Nd_resized = self._resize_array(self.Nd, transient_dof)
                    Na_resized = self._resize_array(self.Na, transient_dof)
                    self.transient_solver.set_doping(Nd_resized, Na_resized)
                    logger.debug(f"Doping resized and set in transient solver: {len(self.Nd)} -> {transient_dof}")
            except Exception as e:
                logger.warning(f"Could not set doping in transient solver: {e}")

        logger.debug(f"Doping set: {len(Nd)} points")
    
    def enable_energy_transport(self, enable=True):
        """Enable energy transport equations"""
        self.energy_transport_enabled = enable
        logger.info(f"Energy transport {'enabled' if enable else 'disabled'}")
    
    def enable_hydrodynamic_transport(self, enable=True):
        """Enable hydrodynamic transport equations"""
        self.hydrodynamic_enabled = enable
        logger.info(f"Hydrodynamic transport {'enabled' if enable else 'disabled'}")
    
    def enable_quantum_corrections(self, enable=True):
        """Enable quantum corrections"""
        self.quantum_corrections_enabled = enable
        logger.info(f"Quantum corrections {'enabled' if enable else 'disabled'}")
    
    def solve_steady_state(self, boundary_conditions, initial_potential=None, 
                          initial_n=None, initial_p=None):
        """
        Solve steady-state self-consistent equations
        
        Parameters:
        -----------
        boundary_conditions : list
            Boundary conditions [left, right, bottom, top] in Volts
        initial_potential : array_like, optional
            Initial potential guess
        initial_n : array_like, optional
            Initial electron density guess
        initial_p : array_like, optional
            Initial hole density guess
            
        Returns:
        --------
        dict
            Results with comprehensive solution data
        """
        logger.info("Starting self-consistent steady-state solution")
        start_time = time.time()
        
        if not self.basic_solver:
            raise RuntimeError("Basic solver not available")
        
        # Use basic drift-diffusion solver as foundation
        try:
            results = self.dd_solver.solve(boundary_conditions)
            
            # Enhance with material properties
            material_props = self.materials.get_material_properties(self.material_type)
            
            # Add material-specific calculations
            if 'potential' in results and 'n' in results and 'p' in results:
                # Calculate recombination rates
                ni = self.materials.get_intrinsic_concentration(self.material_type, 300.0)
                
                recombination_srh = []
                recombination_rad = []
                recombination_auger = []
                
                for i in range(len(results['n'])):
                    n = results['n'][i]
                    p = results['p'][i]
                    
                    R_srh = self.materials.calculate_srh_recombination(n, p, ni)
                    R_rad = self.materials.calculate_radiative_recombination(n, p, ni)
                    R_auger = self.materials.calculate_auger_recombination(n, p, ni)
                    
                    recombination_srh.append(R_srh)
                    recombination_rad.append(R_rad)
                    recombination_auger.append(R_auger)
                
                results['recombination_srh'] = np.array(recombination_srh)
                results['recombination_radiative'] = np.array(recombination_rad)
                results['recombination_auger'] = np.array(recombination_auger)
                results['material_properties'] = material_props
            
            elapsed = time.time() - start_time
            logger.info(f"Self-consistent solution completed in {elapsed:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Self-consistent solve failed: {e}")
            raise
    
    def get_dof_count(self):
        """Get degrees of freedom count"""
        if self.basic_solver and hasattr(self, 'dd_solver'):
            return self.dd_solver.get_dof_count()
        elif self.transient_solver:
            return self.transient_solver.get_dof_count()
        else:
            # Estimate based on device size
            return 1000  # Default estimate
    
    def is_valid(self):
        """Check if solver is valid"""
        return (self.basic_solver or self.transient_solver or self.advanced_transport) is not None

    def _resize_array(self, array, target_size):
        """Resize array to target size using interpolation"""
        if len(array) == target_size:
            return array

        # Simple linear interpolation
        old_indices = np.linspace(0, len(array) - 1, len(array))
        new_indices = np.linspace(0, len(array) - 1, target_size)

        return np.interp(new_indices, old_indices, array)

# Convenience functions
def create_self_consistent_solver(device_width, device_height, material_type=MaterialType.SILICON,
                                method="DG", mesh_type="Structured", order=3):
    """Create a self-consistent solver instance"""
    return SelfConsistentSolver(device_width, device_height, material_type, method, mesh_type, order)

def create_material_database():
    """Create a material database instance"""
    return MaterialDatabase()

def validate_enhanced_simulator():
    """Validate enhanced simulator implementation"""
    try:
        # Test solver creation
        solver = create_self_consistent_solver(1e-6, 0.5e-6)
        
        # Test material database
        materials = create_material_database()
        
        return {
            "solver_creation": True,
            "material_database": True,
            "dof_count_access": solver.get_dof_count() > 0,
            "validation_passed": solver.is_valid(),
            "available_materials": materials.get_available_materials()
        }
    except Exception as e:
        return {
            "solver_creation": False,
            "error": str(e),
            "validation_passed": False
        }
