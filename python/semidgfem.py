#!/usr/bin/env python3
"""
SemiDGFEM: High-Level Python API for Semiconductor Device Simulation

This module provides a comprehensive, user-friendly Python interface to the
SemiDGFEM C++ backend for semiconductor device simulation using Discontinuous
Galerkin Finite Element Methods.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import logging

# Import the Cython-compiled simulator module
try:
    from . import simulator
except ImportError:
    import simulator

# Set up logging
logger = logging.getLogger(__name__)

class SemiDGFEM:
    """
    High-level interface for semiconductor device simulation.
    
    This class provides a comprehensive API for setting up and running
    semiconductor device simulations using the SemiDGFEM backend.
    """
    
    def __init__(self, device_length: float = 1e-6, device_width: float = 0.5e-6,
                 method: str = "DG", mesh_type: str = "Structured", 
                 polynomial_order: int = 3):
        """
        Initialize the SemiDGFEM simulator.
        
        Args:
            device_length: Device length in meters (default: 1 μm)
            device_width: Device width in meters (default: 0.5 μm)
            method: Numerical method ("DG", "FEM", "FDM", etc.)
            mesh_type: Mesh type ("Structured" or "Unstructured")
            polynomial_order: Polynomial order for DG method (1, 2, or 3)
        """
        self.device_length = device_length
        self.device_width = device_width
        self.method = method
        self.mesh_type = mesh_type
        self.polynomial_order = polynomial_order
        
        # Create simulator using the working implementation
        self.simulator = simulator.Simulator(
            extents=[device_length, device_width],
            num_points_x=int(device_length * 1e6 * 20),  # Reasonable grid density
            num_points_y=int(device_width * 1e6 * 20),
            method=method,
            mesh_type=mesh_type
        )
        
        # Simulation state
        self.doping_set = False
        self.last_results = None
        
        logger.info(f"SemiDGFEM initialized: {device_length*1e6:.2f}μm × {device_width*1e6:.2f}μm")
        logger.info(f"Method: {method}, Mesh: {mesh_type}, Order: P{polynomial_order}")
    
    def set_uniform_doping(self, nd: float = 1e16, na: float = 0.0):
        """
        Set uniform doping concentrations.

        Args:
            nd: Donor concentration in cm⁻³
            na: Acceptor concentration in cm⁻³
        """
        # Calculate grid size
        grid_size = self.simulator.num_points_x * self.simulator.num_points_y
        Nd = np.full(grid_size, nd, dtype=np.float64)
        Na = np.full(grid_size, na, dtype=np.float64)

        self.simulator.set_doping(Nd, Na)
        self.doping_set = True

        logger.info(f"Set uniform doping: Nd={nd:.2e} cm⁻³, Na={na:.2e} cm⁻³")
    
    def set_spatial_doping(self, nd_profile: np.ndarray, na_profile: np.ndarray):
        """
        Set spatially varying doping profiles.

        Args:
            nd_profile: Donor concentration profile (cm⁻³)
            na_profile: Acceptor concentration profile (cm⁻³)
        """
        if len(nd_profile) != len(na_profile):
            raise ValueError("Nd and Na profiles must have the same length")

        grid_size = self.simulator.num_points_x * self.simulator.num_points_y
        if len(nd_profile) != grid_size:
            raise ValueError(f"Doping profile length ({len(nd_profile)}) must match grid size ({grid_size})")

        Nd = np.array(nd_profile, dtype=np.float64)
        Na = np.array(na_profile, dtype=np.float64)

        self.simulator.set_doping(Nd, Na)
        self.doping_set = True

        logger.info(f"Set spatial doping: Nd range [{Nd.min():.2e}, {Nd.max():.2e}] cm⁻³")
    
    def create_mosfet_doping(self, channel_doping: float = 1e16, 
                            source_drain_doping: float = 1e20,
                            gate_length_fraction: float = 0.4):
        """
        Create a simple MOSFET doping profile.
        
        Args:
            channel_doping: Channel doping concentration (cm⁻³)
            source_drain_doping: Source/drain doping concentration (cm⁻³)
            gate_length_fraction: Fraction of device length under gate
        """
        dof_count = self.drift_diffusion.get_dof_count()
        
        # Simple 1D approximation for demonstration
        # In practice, this would use proper 2D mesh coordinates
        gate_start = (1.0 - gate_length_fraction) / 2.0
        gate_end = gate_start + gate_length_fraction
        
        Nd = np.full(dof_count, channel_doping, dtype=np.float64)
        Na = np.zeros(dof_count, dtype=np.float64)
        
        # Set source/drain regions (simplified)
        source_end = int(dof_count * gate_start)
        drain_start = int(dof_count * gate_end)
        
        Nd[:source_end] = source_drain_doping  # Source
        Nd[drain_start:] = source_drain_doping  # Drain
        
        self.drift_diffusion.set_doping(Nd, Na)
        self.doping_set = True
        
        logger.info(f"Created MOSFET doping: Channel={channel_doping:.2e}, S/D={source_drain_doping:.2e} cm⁻³")
    
    def solve_equilibrium(self, boundary_conditions: List[float] = None) -> Dict[str, np.ndarray]:
        """
        Solve for equilibrium (zero bias) conditions.
        
        Args:
            boundary_conditions: Boundary voltages [left, right, bottom, top] in V
            
        Returns:
            Dictionary with simulation results
        """
        if not self.doping_set:
            logger.warning("Doping not set, using default uniform doping")
            self.set_uniform_doping()
        
        if boundary_conditions is None:
            boundary_conditions = [0.0, 0.0, 0.0, 0.0]  # All grounded
        
        # Solve Poisson equation for equilibrium
        potential = self.poisson.solve(boundary_conditions)
        
        results = {
            'potential': potential,
            'boundary_conditions': np.array(boundary_conditions),
            'method': self.method,
            'mesh_type': self.mesh_type,
            'dof_count': len(potential)
        }
        
        self.last_results = results
        logger.info(f"Equilibrium solved: {len(potential)} DOFs, V_range=[{potential.min():.3f}, {potential.max():.3f}] V")
        
        return results
    
    def solve_bias_point(self, gate_voltage: float = 0.0, drain_voltage: float = 0.0,
                        source_voltage: float = 0.0, bulk_voltage: float = 0.0,
                        max_iterations: int = 100, tolerance: float = 1e-6,
                        use_amr: bool = False) -> Dict[str, np.ndarray]:
        """
        Solve for a specific bias point.
        
        Args:
            gate_voltage: Gate voltage in V
            drain_voltage: Drain voltage in V  
            source_voltage: Source voltage in V
            bulk_voltage: Bulk voltage in V
            max_iterations: Maximum self-consistent iterations
            tolerance: Convergence tolerance
            use_amr: Enable adaptive mesh refinement
            
        Returns:
            Dictionary with simulation results
        """
        if not self.doping_set:
            logger.warning("Doping not set, using default uniform doping")
            self.set_uniform_doping()
        
        # Set boundary conditions based on terminal voltages
        boundary_conditions = [source_voltage, drain_voltage, bulk_voltage, gate_voltage]
        
        # Solve coupled Poisson-drift-diffusion equations
        results = self.drift_diffusion.solve(
            boundary_conditions, gate_voltage, max_iterations, use_amr,
            max_iterations, tolerance)
        
        # Add metadata
        results.update({
            'gate_voltage': gate_voltage,
            'drain_voltage': drain_voltage,
            'source_voltage': source_voltage,
            'bulk_voltage': bulk_voltage,
            'boundary_conditions': np.array(boundary_conditions),
            'method': self.method,
            'mesh_type': self.mesh_type,
            'dof_count': len(results['potential']),
            'convergence_residual': self.drift_diffusion.get_convergence_residual()
        })
        
        self.last_results = results
        logger.info(f"Bias point solved: Vg={gate_voltage:.2f}V, Vd={drain_voltage:.2f}V")
        logger.info(f"Convergence residual: {results['convergence_residual']:.2e}")
        
        return results
    
    def compute_iv_characteristics(self, gate_voltages: np.ndarray, drain_voltages: np.ndarray,
                                  source_voltage: float = 0.0, bulk_voltage: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Compute I-V characteristics over voltage ranges.
        
        Args:
            gate_voltages: Array of gate voltages to sweep
            drain_voltages: Array of drain voltages to sweep
            source_voltage: Source voltage (reference)
            bulk_voltage: Bulk voltage
            
        Returns:
            Dictionary with I-V data
        """
        if not self.doping_set:
            logger.warning("Doping not set, using default uniform doping")
            self.set_uniform_doping()
        
        # Initialize result arrays
        ids_matrix = np.zeros((len(gate_voltages), len(drain_voltages)))
        
        logger.info(f"Computing I-V: {len(gate_voltages)} Vg × {len(drain_voltages)} Vd points")
        
        for i, vg in enumerate(gate_voltages):
            for j, vd in enumerate(drain_voltages):
                try:
                    results = self.solve_bias_point(vg, vd, source_voltage, bulk_voltage)
                    
                    # Extract drain current (simplified calculation)
                    # In practice, this would integrate current density at drain contact
                    current_density = results.get('Jn', np.zeros_like(results['potential']))
                    ids = np.mean(current_density) * self.device_width  # Simplified
                    ids_matrix[i, j] = abs(ids)
                    
                except Exception as e:
                    logger.warning(f"Failed at Vg={vg:.2f}V, Vd={vd:.2f}V: {e}")
                    ids_matrix[i, j] = 0.0
        
        iv_results = {
            'gate_voltages': gate_voltages,
            'drain_voltages': drain_voltages,
            'ids_matrix': ids_matrix,
            'source_voltage': source_voltage,
            'bulk_voltage': bulk_voltage
        }
        
        logger.info("I-V characteristics computed successfully")
        return iv_results
    
    def get_mesh_info(self) -> Dict[str, Union[int, np.ndarray]]:
        """
        Get mesh information.
        
        Returns:
            Dictionary with mesh data
        """
        num_nodes = self.mesh.get_num_nodes()
        num_elements = self.mesh.get_num_elements()
        
        try:
            x_points, y_points = self.mesh.get_grid_points()
        except:
            x_points = np.array([])
            y_points = np.array([])
        
        return {
            'num_nodes': num_nodes,
            'num_elements': num_elements,
            'x_points': x_points,
            'y_points': y_points,
            'dof_count': self.drift_diffusion.get_dof_count()
        }
    
    def get_solver_info(self) -> Dict[str, Union[str, int, bool]]:
        """
        Get solver information.
        
        Returns:
            Dictionary with solver data
        """
        return {
            'method': self.method,
            'mesh_type': self.mesh_type,
            'polynomial_order': self.polynomial_order,
            'device_length': self.device_length,
            'device_width': self.device_width,
            'poisson_valid': self.poisson.is_valid(),
            'drift_diffusion_valid': self.drift_diffusion.is_valid(),
            'doping_set': self.doping_set,
            'dof_count': self.drift_diffusion.get_dof_count()
        }

# Convenience functions for quick simulations
def create_mosfet_simulator(length: float = 1e-6, width: float = 0.5e-6, 
                           method: str = "DG", order: int = 3) -> SemiDGFEM:
    """
    Create a MOSFET simulator with default settings.
    
    Args:
        length: Device length in meters
        width: Device width in meters  
        method: Numerical method
        order: Polynomial order
        
    Returns:
        Configured SemiDGFEM instance
    """
    sim = SemiDGFEM(length, width, method, "Structured", order)
    sim.create_mosfet_doping()
    return sim

def quick_equilibrium_solve(length: float = 1e-6, width: float = 0.5e-6) -> Dict[str, np.ndarray]:
    """
    Quick equilibrium solution with default parameters.
    
    Args:
        length: Device length in meters
        width: Device width in meters
        
    Returns:
        Simulation results
    """
    sim = create_mosfet_simulator(length, width)
    return sim.solve_equilibrium()

# Module-level convenience imports
try:
    # Try to import Cython classes if available
    Device = simulator.Device
    PoissonSolver = simulator.PoissonSolver
    DriftDiffusionSolver = simulator.DriftDiffusionSolver
    Mesh = simulator.Mesh
except AttributeError:
    # Fallback to basic simulator if Cython classes not available
    print("⚠️  Advanced Cython classes not available, using basic simulator")
    Device = None
    PoissonSolver = None
    DriftDiffusionSolver = None
    Mesh = None

Simulator = simulator.Simulator  # Legacy compatibility

__all__ = [
    'SemiDGFEM', 'Device', 'PoissonSolver', 'DriftDiffusionSolver', 'Mesh', 'Simulator',
    'create_mosfet_simulator', 'quick_equilibrium_solve'
]
