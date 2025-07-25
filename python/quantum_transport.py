"""
Quantum Transport Models for SemiDGFEM

This module provides comprehensive quantum transport modeling capabilities including:
- Quantum mechanical transport (ballistic, tunneling, coherent)
- Quantum confinement effects (wells, wires, dots)
- Scattering mechanisms (phonons, impurities, interface roughness)
- Non-equilibrium Green's function methods
- Wigner function approach

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from scipy.optimize import minimize_scalar
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physical constants
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
ME = 9.1093837015e-31   # Electron mass (kg)
Q = 1.602176634e-19     # Elementary charge (C)
KB = 1.380649e-23       # Boltzmann constant (J/K)
EV_TO_J = 1.602176634e-19  # eV to Joules conversion

class QuantumTransportType(Enum):
    """Quantum transport model types"""
    BALLISTIC = "ballistic"
    TUNNELING = "tunneling"
    COHERENT = "coherent"
    WIGNER_FUNCTION = "wigner_function"
    NON_EQUILIBRIUM_GREEN = "non_equilibrium_green"
    DENSITY_MATRIX = "density_matrix"

class ConfinementType(Enum):
    """Quantum confinement types"""
    NONE = "none"
    QUANTUM_WELL = "quantum_well"
    QUANTUM_WIRE = "quantum_wire"
    QUANTUM_DOT = "quantum_dot"

class ScatteringType(Enum):
    """Scattering mechanism types"""
    ACOUSTIC_PHONON = "acoustic_phonon"
    OPTICAL_PHONON = "optical_phonon"
    IONIZED_IMPURITY = "ionized_impurity"
    INTERFACE_ROUGHNESS = "interface_roughness"
    ALLOY_DISORDER = "alloy_disorder"
    ELECTRON_ELECTRON = "electron_electron"

@dataclass
class QuantumState:
    """Quantum state information"""
    n: int = 0                              # Principal quantum number
    l: int = 0                              # Angular momentum quantum number
    m: int = 0                              # Magnetic quantum number
    energy: float = 0.0                     # Energy eigenvalue (eV)
    wavefunction: np.ndarray = field(default_factory=lambda: np.array([]))
    occupation: float = 0.0                 # Occupation probability
    momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def __post_init__(self):
        if self.wavefunction.size == 0:
            self.wavefunction = np.array([])
        if self.momentum.size == 0:
            self.momentum = np.zeros(3)

@dataclass
class QuantumTransportParameters:
    """Quantum transport parameters"""
    transport_type: QuantumTransportType = QuantumTransportType.BALLISTIC
    confinement_type: ConfinementType = ConfinementType.NONE
    
    # Physical parameters
    effective_mass: float = 0.067           # Effective mass (m0 units)
    temperature: float = 300.0              # Temperature (K)
    fermi_level: float = 0.0                # Fermi level (eV)
    barrier_height: float = 1.0             # Barrier height (eV)
    barrier_width: float = 1e-9             # Barrier width (m)
    
    # Confinement parameters
    confinement_length: np.ndarray = field(default_factory=lambda: np.array([1e-8, 1e-8, 1e-8]))
    periodic_boundary: np.ndarray = field(default_factory=lambda: np.array([False, False, False]))
    
    # Numerical parameters
    max_states: int = 100                   # Maximum number of states
    energy_cutoff: float = 2.0              # Energy cutoff (eV)
    convergence_tolerance: float = 1e-8     # Convergence tolerance
    max_iterations: int = 1000              # Maximum iterations
    
    # Scattering parameters
    scattering_mechanisms: List[ScatteringType] = field(default_factory=list)
    acoustic_deformation_potential: float = 7.0    # Acoustic deformation potential (eV)
    optical_deformation_potential: float = 1e11    # Optical deformation potential (eV/m)
    interface_roughness_height: float = 0.14e-9    # Interface roughness height (m)
    interface_roughness_correlation: float = 1.5e-9  # Interface roughness correlation length (m)

@dataclass
class QuantumTransportResults:
    """Quantum transport results"""
    states: List[QuantumState] = field(default_factory=list)
    transmission_coefficients: np.ndarray = field(default_factory=lambda: np.array([]))
    reflection_coefficients: np.ndarray = field(default_factory=lambda: np.array([]))
    current_density: np.ndarray = field(default_factory=lambda: np.array([]))
    charge_density: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Transport properties
    conductance: float = 0.0                # Quantum conductance (S)
    resistance: float = 0.0                 # Quantum resistance (Ω)
    mobility: float = 0.0                   # Quantum mobility (m²/V·s)
    diffusion_coefficient: float = 0.0      # Diffusion coefficient (m²/s)
    
    # Energy-dependent properties
    energy_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    density_of_states: np.ndarray = field(default_factory=lambda: np.array([]))
    transmission_function: np.ndarray = field(default_factory=lambda: np.array([]))
    
    converged: bool = False                 # Convergence flag
    iterations: int = 0                     # Number of iterations
    residual: float = 0.0                   # Final residual

class QuantumTransportSolver:
    """Quantum transport solver"""
    
    def __init__(self):
        self.params = QuantumTransportParameters()
        self.vertices = np.array([])
        self.elements = np.array([])
        self.potential = np.array([])
        self.effective_mass = np.array([])
        self.band_offset = np.array([])
        self.results = QuantumTransportResults()
        
        # Matrices
        self.hamiltonian_matrix = None
        self.overlap_matrix = None
        self.green_function_matrix = None
        
    def set_parameters(self, params: QuantumTransportParameters):
        """Set quantum transport parameters"""
        self.params = params
        logger.info(f"Quantum transport parameters set: {params.transport_type.value}")
        
    def set_mesh(self, vertices: np.ndarray, elements: np.ndarray):
        """Set mesh for quantum transport calculation"""
        self.vertices = np.array(vertices)
        self.elements = np.array(elements)
        
        num_nodes = len(vertices)
        self.hamiltonian_matrix = np.zeros((num_nodes, num_nodes))
        self.overlap_matrix = np.eye(num_nodes)
        self.green_function_matrix = np.zeros((num_nodes, num_nodes), dtype=complex)
        
        logger.info(f"Mesh set: {len(vertices)} vertices, {len(elements)} elements")
        
    def set_potential(self, potential: np.ndarray):
        """Set electrostatic potential"""
        self.potential = np.array(potential)
        logger.info(f"Potential set: range [{np.min(potential):.3f}, {np.max(potential):.3f}] eV")
        
    def set_material_properties(self, effective_mass: np.ndarray, band_offset: np.ndarray):
        """Set material properties"""
        self.effective_mass = np.array(effective_mass)
        self.band_offset = np.array(band_offset)
        logger.info("Material properties set")
        
    def solve_schrodinger_equation(self) -> bool:
        """Solve the Schrödinger equation"""
        if self.vertices.size == 0 or self.elements.size == 0:
            logger.error("Mesh not set")
            return False
            
        try:
            # Build Hamiltonian matrix
            self._build_hamiltonian_matrix()
            
            # Solve eigenvalue problem
            return self._solve_eigenvalue_problem()
            
        except Exception as e:
            logger.error(f"Error solving Schrödinger equation: {e}")
            return False
            
    def calculate_quantum_states(self) -> bool:
        """Calculate quantum states"""
        if not self.solve_schrodinger_equation():
            return False
            
        # Update occupation numbers
        self._update_occupation_numbers()
        
        self.results.converged = True
        logger.info(f"Calculated {len(self.results.states)} quantum states")
        return True
        
    def get_bound_states(self) -> List[QuantumState]:
        """Get bound quantum states"""
        bound_states = []
        for state in self.results.states:
            if state.energy < self.params.energy_cutoff:
                bound_states.append(state)
        return bound_states
        
    def get_scattering_states(self, energy: float) -> List[QuantumState]:
        """Get scattering states at given energy"""
        scattering_states = []
        
        # Calculate scattering states (simplified)
        for i in range(10):
            state = QuantumState()
            state.energy = energy
            state.n = i
            
            # Simple plane wave approximation
            k = np.sqrt(2.0 * self.params.effective_mass * ME * energy * EV_TO_J) / HBAR
            x_coords = self.vertices[:, 0] if self.vertices.ndim > 1 else self.vertices
            state.wavefunction = np.exp(1j * k * x_coords)
            
            scattering_states.append(state)
            
        return scattering_states
        
    def calculate_transmission_coefficients(self) -> bool:
        """Calculate transmission coefficients"""
        try:
            # Energy grid
            num_energies = 100
            energy_min = 0.0
            energy_max = self.params.energy_cutoff
            self.results.energy_grid = np.linspace(energy_min, energy_max, num_energies)
            
            # Calculate transmission function
            self.results.transmission_function = np.zeros(num_energies)
            
            for i, energy in enumerate(self.results.energy_grid):
                transmission = self._calculate_transmission_at_energy(energy)
                self.results.transmission_function[i] = transmission
                
            # Store first 10 for compatibility
            self.results.transmission_coefficients = self.results.transmission_function[:10]
            self.results.reflection_coefficients = 1.0 - self.results.transmission_coefficients
            
            logger.info("Transmission coefficients calculated")
            return True
            
        except Exception as e:
            logger.error(f"Error calculating transmission coefficients: {e}")
            return False
            
    def calculate_current_density(self) -> bool:
        """Calculate quantum current density"""
        if not self.results.states:
            return False
            
        try:
            num_nodes = len(self.vertices)
            self.results.current_density = np.zeros((num_nodes, 2))
            
            for state in self.results.states:
                if state.occupation > 1e-10:
                    # Current density calculation (simplified)
                    psi_real = np.real(state.wavefunction)
                    psi_imag = np.imag(state.wavefunction)
                    
                    self.results.current_density[:, 0] += state.occupation * psi_imag * psi_real
                    self.results.current_density[:, 1] += state.occupation * psi_real * psi_imag
                    
            logger.info("Current density calculated")
            return True
            
        except Exception as e:
            logger.error(f"Error calculating current density: {e}")
            return False
            
    def calculate_charge_density(self) -> bool:
        """Calculate quantum charge density"""
        if not self.results.states:
            return False
            
        try:
            num_nodes = len(self.vertices)
            self.results.charge_density = np.zeros((num_nodes, 1))
            
            for state in self.results.states:
                psi_magnitude_sq = np.abs(state.wavefunction)**2
                self.results.charge_density[:, 0] += state.occupation * psi_magnitude_sq
                
            logger.info("Charge density calculated")
            return True
            
        except Exception as e:
            logger.error(f"Error calculating charge density: {e}")
            return False
            
    def add_scattering_mechanism(self, scattering_type: ScatteringType):
        """Add scattering mechanism"""
        if scattering_type not in self.params.scattering_mechanisms:
            self.params.scattering_mechanisms.append(scattering_type)
            logger.info(f"Added scattering mechanism: {scattering_type.value}")
            
    def calculate_scattering_rates(self) -> bool:
        """Calculate scattering rates"""
        try:
            for mechanism in self.params.scattering_mechanisms:
                if mechanism == ScatteringType.ACOUSTIC_PHONON:
                    self._calculate_acoustic_phonon_scattering()
                elif mechanism == ScatteringType.OPTICAL_PHONON:
                    self._calculate_optical_phonon_scattering()
                elif mechanism == ScatteringType.IONIZED_IMPURITY:
                    self._calculate_impurity_scattering()
                    
            logger.info("Scattering rates calculated")
            return True
            
        except Exception as e:
            logger.error(f"Error calculating scattering rates: {e}")
            return False
            
    def solve_boltzmann_equation(self) -> bool:
        """Solve Boltzmann transport equation"""
        try:
            self.calculate_scattering_rates()
            
            # Update transport properties
            self.results.mobility = self._calculate_quantum_mobility()
            self.results.conductance = self._calculate_quantum_conductance()
            self.results.resistance = 1.0 / self.results.conductance if self.results.conductance > 0 else 1e12
            
            logger.info("Boltzmann equation solved")
            return True
            
        except Exception as e:
            logger.error(f"Error solving Boltzmann equation: {e}")
            return False

    def solve_wigner_function(self) -> bool:
        """Solve using Wigner function approach"""
        try:
            # Simplified Wigner function implementation
            logger.info("Wigner function approach not fully implemented - using simplified model")

            # Use quantum states as approximation
            if not self.results.states:
                self.calculate_quantum_states()

            return True

        except Exception as e:
            logger.error(f"Error solving Wigner function: {e}")
            return False

    def solve_green_function(self) -> bool:
        """Solve using Non-equilibrium Green's function method"""
        try:
            # Simplified NEGF implementation
            logger.info("NEGF method not fully implemented - using simplified model")

            # Build Green's function matrix (simplified)
            if self.hamiltonian_matrix is not None:
                num_nodes = self.hamiltonian_matrix.shape[0]
                energy = self.params.fermi_level

                # G = (E*I - H)^(-1)
                identity = np.eye(num_nodes)
                matrix_to_invert = energy * identity - self.hamiltonian_matrix

                try:
                    self.green_function_matrix = np.linalg.inv(matrix_to_invert)
                except np.linalg.LinAlgError:
                    # Use pseudo-inverse if singular
                    self.green_function_matrix = np.linalg.pinv(matrix_to_invert)

            return True

        except Exception as e:
            logger.error(f"Error solving Green's function: {e}")
            return False

    def calculate_coherent_transport(self) -> bool:
        """Calculate coherent transport properties"""
        try:
            # Coherent transport calculation
            if not self.calculate_transmission_coefficients():
                return False

            # Calculate coherent conductance
            self.results.conductance = self._calculate_quantum_conductance()

            logger.info("Coherent transport calculated")
            return True

        except Exception as e:
            logger.error(f"Error calculating coherent transport: {e}")
            return False

    def get_results(self) -> QuantumTransportResults:
        """Get quantum transport results"""
        return self.results

    def get_wavefunction(self, state_index: int) -> np.ndarray:
        """Get wavefunction for specific state"""
        if 0 <= state_index < len(self.results.states):
            return np.real(self.results.states[state_index].wavefunction)
        return np.array([])

    def get_green_function(self, energy: float) -> np.ndarray:
        """Get Green's function at specific energy"""
        if self.green_function_matrix is not None:
            return self.green_function_matrix
        return np.array([])

    def calculate_quantum_capacitance(self) -> float:
        """Calculate quantum capacitance"""
        try:
            if not self.results.density_of_states.size:
                self._calculate_density_of_states()

            # C_q = q^2 * DOS(E_F)
            fermi_index = np.argmin(np.abs(self.results.energy_grid - self.params.fermi_level))
            dos_at_fermi = self.results.density_of_states[fermi_index]

            return Q * Q * dos_at_fermi

        except Exception as e:
            logger.error(f"Error calculating quantum capacitance: {e}")
            return 0.0

    def calculate_shot_noise(self) -> float:
        """Calculate shot noise"""
        try:
            # Shot noise: S = 2qI * (1 - T) for single channel
            # Simplified calculation
            if self.results.transmission_coefficients.size > 0:
                T_avg = np.mean(self.results.transmission_coefficients)
                current = self.results.conductance * 0.1  # Assume 0.1V bias
                return 2 * Q * current * (1 - T_avg)
            return 0.0

        except Exception as e:
            logger.error(f"Error calculating shot noise: {e}")
            return 0.0

    def calculate_local_density_of_states(self) -> np.ndarray:
        """Calculate local density of states"""
        try:
            if self.green_function_matrix is not None:
                # LDOS = -Im(G)/π
                ldos = -np.imag(np.diag(self.green_function_matrix)) / np.pi
                return np.real(ldos)
            return np.array([])

        except Exception as e:
            logger.error(f"Error calculating LDOS: {e}")
            return np.array([])

    # Private methods
    def _build_hamiltonian_matrix(self):
        """Build Hamiltonian matrix"""
        num_nodes = len(self.vertices)
        self.hamiltonian_matrix = np.zeros((num_nodes, num_nodes))

        # Kinetic energy matrix
        self._assemble_kinetic_energy_matrix()

        # Potential energy matrix
        self._assemble_potential_energy_matrix()

    def _assemble_kinetic_energy_matrix(self):
        """Assemble kinetic energy matrix"""
        hbar_sq_over_2m = HBAR**2 / (2.0 * self.params.effective_mass * ME)

        for element in self.elements:
            if len(element) >= 3:
                n1, n2, n3 = element[0], element[1], element[2]

                # Calculate element area
                if self.vertices.ndim > 1:
                    x1, y1 = self.vertices[n1][0], self.vertices[n1][1]
                    x2, y2 = self.vertices[n2][0], self.vertices[n2][1]
                    x3, y3 = self.vertices[n3][0], self.vertices[n3][1]

                    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

                    if area > 1e-15:
                        kinetic_contrib = hbar_sq_over_2m / area

                        # Add to diagonal
                        self.hamiltonian_matrix[n1, n1] += kinetic_contrib
                        self.hamiltonian_matrix[n2, n2] += kinetic_contrib
                        self.hamiltonian_matrix[n3, n3] += kinetic_contrib

                        # Add off-diagonal terms
                        self.hamiltonian_matrix[n1, n2] -= 0.5 * kinetic_contrib
                        self.hamiltonian_matrix[n2, n1] -= 0.5 * kinetic_contrib
                        self.hamiltonian_matrix[n2, n3] -= 0.5 * kinetic_contrib
                        self.hamiltonian_matrix[n3, n2] -= 0.5 * kinetic_contrib
                        self.hamiltonian_matrix[n3, n1] -= 0.5 * kinetic_contrib
                        self.hamiltonian_matrix[n1, n3] -= 0.5 * kinetic_contrib

    def _assemble_potential_energy_matrix(self):
        """Assemble potential energy matrix"""
        if self.potential.size > 0:
            num_nodes = min(len(self.vertices), len(self.potential))
            for i in range(num_nodes):
                self.hamiltonian_matrix[i, i] += self.potential[i] * EV_TO_J

    def _solve_eigenvalue_problem(self) -> bool:
        """Solve eigenvalue problem"""
        try:
            # Use scipy for eigenvalue solving
            eigenvalues, eigenvectors = la.eigh(self.hamiltonian_matrix, self.overlap_matrix)

            # Sort by energy
            sorted_indices = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            # Create quantum states
            self.results.states = []
            num_states = min(self.params.max_states, len(eigenvalues))

            for n in range(num_states):
                state = QuantumState()
                state.n = n
                state.energy = eigenvalues[n] / EV_TO_J  # Convert to eV
                state.wavefunction = eigenvectors[:, n]

                # Normalize wavefunction
                norm = np.sqrt(np.sum(np.abs(state.wavefunction)**2))
                if norm > 1e-15:
                    state.wavefunction /= norm

                self.results.states.append(state)

            return True

        except Exception as e:
            logger.error(f"Error solving eigenvalue problem: {e}")
            return False

    def _update_occupation_numbers(self):
        """Update occupation numbers using Fermi-Dirac statistics"""
        kT = KB * self.params.temperature / EV_TO_J

        for state in self.results.states:
            fermi_factor = 1.0 / (1.0 + np.exp((state.energy - self.params.fermi_level) / kT))
            state.occupation = fermi_factor

    def _calculate_transmission_at_energy(self, energy: float) -> float:
        """Calculate transmission coefficient at specific energy"""
        if self.potential.size == 0:
            return 1.0  # Perfect transmission

        # Simple WKB approximation
        barrier_height = np.max(self.potential)

        if energy > barrier_height:
            return 1.0  # Above barrier

        # Below barrier - tunneling
        barrier_width = self.params.barrier_width
        kappa = np.sqrt(2.0 * self.params.effective_mass * ME * (barrier_height - energy) * EV_TO_J) / HBAR

        return np.exp(-2.0 * kappa * barrier_width)

    def _calculate_acoustic_phonon_scattering(self):
        """Calculate acoustic phonon scattering"""
        deformation_potential = self.params.acoustic_deformation_potential
        sound_velocity = 5000.0  # m/s
        density = 5320.0  # kg/m³

        for state in self.results.states:
            scattering_rate = (deformation_potential**2 * KB * self.params.temperature) / \
                            (HBAR * density * sound_velocity**2)
            state.occupation *= (1.0 - scattering_rate * 1e-12)

    def _calculate_optical_phonon_scattering(self):
        """Calculate optical phonon scattering"""
        phonon_energy = 0.036  # eV

        for state in self.results.states:
            if state.energy > phonon_energy:
                emission_rate = 1e12  # 1/s
                state.occupation *= (1.0 - emission_rate * 1e-12)

    def _calculate_impurity_scattering(self):
        """Calculate impurity scattering"""
        impurity_density = 1e16  # cm⁻³

        for state in self.results.states:
            scattering_rate = impurity_density * 1e6 * 1e-15
            state.occupation *= (1.0 - scattering_rate * 1e-12)

    def _calculate_quantum_mobility(self) -> float:
        """Calculate quantum mobility"""
        total_scattering_rate = 0.0
        total_occupation = 0.0

        for state in self.results.states:
            if state.occupation > 1e-10:
                total_scattering_rate += state.occupation / 1e12
                total_occupation += state.occupation

        if total_occupation > 0 and total_scattering_rate > 0:
            return Q / (self.params.effective_mass * ME * total_scattering_rate / total_occupation)

        return 0.1  # Default mobility

    def _calculate_quantum_conductance(self) -> float:
        """Calculate quantum conductance"""
        if self.results.transmission_function.size == 0:
            return 0.0

        conductance = 0.0
        kT = KB * self.params.temperature / EV_TO_J

        for i, energy in enumerate(self.results.energy_grid):
            transmission = self.results.transmission_function[i]
            fermi_factor = 1.0 / (1.0 + np.exp((energy - self.params.fermi_level) / kT))
            conductance += transmission * fermi_factor

        # Quantum of conductance
        g0 = 2.0 * Q**2 / (2.0 * np.pi * HBAR)  # Factor of 2 for spin

        return g0 * conductance / len(self.results.transmission_function)

    def _calculate_density_of_states(self):
        """Calculate density of states"""
        if self.results.energy_grid.size == 0:
            return

        self.results.density_of_states = np.zeros_like(self.results.energy_grid)

        # Simple DOS calculation from quantum states
        for state in self.results.states:
            # Find closest energy point
            energy_index = np.argmin(np.abs(self.results.energy_grid - state.energy))
            self.results.density_of_states[energy_index] += 1.0

        # Smooth the DOS
        from scipy.ndimage import gaussian_filter1d
        self.results.density_of_states = gaussian_filter1d(self.results.density_of_states, sigma=2.0)


class QuantumConfinementCalculator:
    """Quantum confinement calculator"""

    def __init__(self):
        self.confinement_type = ConfinementType.NONE
        self.dimensions = np.array([1e-8, 1e-8, 1e-8])
        self.effective_mass = 0.067
        self.band_offset = 0.0

    def set_confinement_type(self, confinement_type: ConfinementType):
        """Set confinement type"""
        self.confinement_type = confinement_type
        logger.info(f"Confinement type set to: {confinement_type.value}")

    def set_dimensions(self, dimensions: np.ndarray):
        """Set confinement dimensions"""
        self.dimensions = np.array(dimensions)
        logger.info(f"Dimensions set: {dimensions}")

    def set_material_parameters(self, effective_mass: float, band_offset: float):
        """Set material parameters"""
        self.effective_mass = effective_mass
        self.band_offset = band_offset
        logger.info(f"Material parameters: m*={effective_mass}, offset={band_offset} eV")

    def calculate_energy_levels(self, max_levels: int = 10) -> np.ndarray:
        """Calculate energy levels"""
        if self.confinement_type == ConfinementType.QUANTUM_WELL:
            return self._solve_infinite_well_1d(max_levels)
        elif self.confinement_type == ConfinementType.QUANTUM_WIRE:
            return self._solve_infinite_well_2d(max_levels)
        elif self.confinement_type == ConfinementType.QUANTUM_DOT:
            return self._solve_infinite_well_3d(max_levels)
        else:
            return np.array([])

    def calculate_wavefunctions(self, max_levels: int = 10) -> List[np.ndarray]:
        """Calculate wavefunctions"""
        wavefunctions = []
        num_points = 100

        for n in range(1, max_levels + 1):
            x = np.linspace(0, 1, num_points)

            if self.confinement_type == ConfinementType.QUANTUM_WELL:
                wavefunction = np.sqrt(2.0) * np.sin(n * np.pi * x)
            elif self.confinement_type in [ConfinementType.QUANTUM_WIRE, ConfinementType.QUANTUM_DOT]:
                wavefunction = np.sqrt(2.0) * np.sin(n * np.pi * x) * np.sin(n * np.pi * x)
            else:
                wavefunction = np.ones(num_points)

            wavefunctions.append(wavefunction)

        return wavefunctions

    def calculate_density_of_states(self, energy: float) -> float:
        """Calculate density of states"""
        if self.confinement_type == ConfinementType.QUANTUM_WELL:
            # 2D DOS (step function)
            dos_2d = self.effective_mass * ME / (np.pi * HBAR**2)
            return dos_2d if energy > self.get_ground_state_energy() else 0.0

        elif self.confinement_type == ConfinementType.QUANTUM_WIRE:
            # 1D DOS (1/sqrt(E))
            ground_energy = self.get_ground_state_energy()
            if energy > ground_energy:
                return np.sqrt(2.0 * self.effective_mass * ME) / (np.pi * HBAR * np.sqrt(energy - ground_energy))
            return 0.0

        elif self.confinement_type == ConfinementType.QUANTUM_DOT:
            # 0D DOS (delta functions)
            energy_levels = self.calculate_energy_levels(10)
            dos = 0.0
            broadening = 0.01  # eV

            for level_energy in energy_levels:
                # Lorentzian broadening
                dos += (broadening / np.pi) / ((energy - level_energy)**2 + broadening**2)
            return dos

        return 0.0

    def get_ground_state_energy(self) -> float:
        """Get ground state energy"""
        energy_levels = self.calculate_energy_levels(1)
        return energy_levels[0] if len(energy_levels) > 0 else 0.0

    def get_level_spacing(self) -> float:
        """Get energy level spacing"""
        energy_levels = self.calculate_energy_levels(2)
        if len(energy_levels) < 2:
            return 0.0
        return energy_levels[1] - energy_levels[0]

    def is_quantum_confined(self) -> bool:
        """Check if quantum confined"""
        thermal_energy = KB * 300.0 / EV_TO_J  # kT at room temperature
        level_spacing = self.get_level_spacing()
        return level_spacing > thermal_energy

    def _solve_infinite_well_1d(self, max_levels: int) -> np.ndarray:
        """Solve 1D infinite well"""
        energy_levels = []
        length = self.dimensions[0]

        for n in range(1, max_levels + 1):
            energy = (n**2 * np.pi**2 * HBAR**2) / (2.0 * self.effective_mass * ME * length**2)
            energy_levels.append(energy / EV_TO_J + self.band_offset)  # Convert to eV

        return np.array(energy_levels)

    def _solve_infinite_well_2d(self, max_levels: int) -> np.ndarray:
        """Solve 2D infinite well"""
        energy_levels = []
        lx, ly = self.dimensions[0], self.dimensions[1]

        for nx in range(1, max_levels + 1):
            for ny in range(1, max_levels + 1):
                if len(energy_levels) >= max_levels:
                    break

                energy = (np.pi**2 * HBAR**2 / (2.0 * self.effective_mass * ME)) * \
                        (nx**2 / lx**2 + ny**2 / ly**2)
                energy_levels.append(energy / EV_TO_J + self.band_offset)

            if len(energy_levels) >= max_levels:
                break

        energy_levels.sort()
        return np.array(energy_levels[:max_levels])

    def _solve_infinite_well_3d(self, max_levels: int) -> np.ndarray:
        """Solve 3D infinite well"""
        energy_levels = []
        lx, ly, lz = self.dimensions[0], self.dimensions[1], self.dimensions[2]

        for nx in range(1, max_levels + 1):
            for ny in range(1, max_levels + 1):
                for nz in range(1, max_levels + 1):
                    if len(energy_levels) >= max_levels:
                        break

                    energy = (np.pi**2 * HBAR**2 / (2.0 * self.effective_mass * ME)) * \
                            (nx**2 / lx**2 + ny**2 / ly**2 + nz**2 / lz**2)
                    energy_levels.append(energy / EV_TO_J + self.band_offset)

                if len(energy_levels) >= max_levels:
                    break
            if len(energy_levels) >= max_levels:
                break

        energy_levels.sort()
        return np.array(energy_levels[:max_levels])


class TunnelingCalculator:
    """Tunneling calculator"""

    def __init__(self):
        self.position = np.array([])
        self.potential = np.array([])
        self.min_energy = 0.0
        self.max_energy = 2.0
        self.num_energy_points = 100

    def set_barrier_profile(self, position: np.ndarray, potential: np.ndarray):
        """Set barrier profile"""
        self.position = np.array(position)
        self.potential = np.array(potential)
        logger.info(f"Barrier profile set: {len(position)} points")

    def set_energy_range(self, min_energy: float, max_energy: float, num_points: int):
        """Set energy range for calculation"""
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.num_energy_points = num_points
        logger.info(f"Energy range: [{min_energy}, {max_energy}] eV, {num_points} points")

    def calculate_transmission_coefficients(self) -> np.ndarray:
        """Calculate transmission coefficients"""
        energy_grid = np.linspace(self.min_energy, self.max_energy, self.num_energy_points)
        transmission_coefficients = np.zeros(self.num_energy_points)

        for i, energy in enumerate(energy_grid):
            transmission_coefficients[i] = self._solve_schrodinger_1d(energy)

        return transmission_coefficients

    def calculate_reflection_coefficients(self) -> np.ndarray:
        """Calculate reflection coefficients"""
        transmission = self.calculate_transmission_coefficients()
        return 1.0 - transmission

    def calculate_tunneling_current(self, voltage: float, temperature: float) -> float:
        """Calculate tunneling current using Landauer formula"""
        if self.position.size == 0 or self.potential.size == 0:
            return 0.0

        energy_grid = np.linspace(self.min_energy, self.max_energy, self.num_energy_points)
        transmission = self.calculate_transmission_coefficients()

        current = 0.0
        energy_step = (self.max_energy - self.min_energy) / (self.num_energy_points - 1)

        for i, energy in enumerate(energy_grid):
            T = transmission[i]

            # Fermi-Dirac distributions
            kT = KB * temperature / EV_TO_J
            f_source = 1.0 / (1.0 + np.exp((energy - voltage/2.0) / kT))
            f_drain = 1.0 / (1.0 + np.exp((energy + voltage/2.0) / kT))

            current += T * (f_source - f_drain) * energy_step

        # Quantum of conductance
        g0 = 2.0 * Q**2 / (2.0 * np.pi * HBAR)
        return g0 * current

    def get_barrier_height(self) -> float:
        """Get barrier height"""
        if self.potential.size == 0:
            return 0.0
        return np.max(self.potential)

    def get_barrier_width(self) -> float:
        """Get barrier width"""
        if self.position.size < 2:
            return 0.0

        # Find barrier region
        max_potential = self.get_barrier_height()
        threshold = 0.5 * max_potential

        barrier_indices = np.where(self.potential > threshold)[0]
        if len(barrier_indices) == 0:
            return 0.0

        start_pos = self.position[barrier_indices[0]]
        end_pos = self.position[barrier_indices[-1]]

        return end_pos - start_pos

    def get_wkb_transmission(self, energy: float) -> float:
        """Get WKB transmission coefficient"""
        if energy >= self.get_barrier_height():
            return 1.0  # Above barrier

        # WKB approximation
        integral = 0.0
        effective_mass = 0.067 * ME  # Default effective mass

        for i in range(1, len(self.position)):
            dx = self.position[i] - self.position[i-1]
            V_avg = 0.5 * (self.potential[i] + self.potential[i-1])

            if V_avg > energy:
                k_local = np.sqrt(2.0 * effective_mass * (V_avg - energy) * EV_TO_J) / HBAR
                integral += k_local * dx

        return np.exp(-2.0 * integral)

    def _solve_schrodinger_1d(self, energy: float) -> float:
        """Solve 1D Schrödinger equation"""
        if self.position.size == 0 or self.potential.size == 0:
            return 1.0

        # Use WKB approximation for simplicity
        return self.get_wkb_transmission(energy)

    def _calculate_transfer_matrix(self, energy: float) -> np.ndarray:
        """Calculate transfer matrix (simplified implementation)"""
        # Simplified transfer matrix calculation
        transfer_matrix = np.eye(2, dtype=complex)
        effective_mass = 0.067 * ME

        for i in range(1, len(self.position)):
            dx = self.position[i] - self.position[i-1]
            V = self.potential[i]

            if energy > V:
                k = np.sqrt(2.0 * effective_mass * (energy - V) * EV_TO_J) / HBAR
                phase = k * dx
                propagation_matrix = np.array([[np.exp(1j * phase), 0],
                                             [0, np.exp(-1j * phase)]], dtype=complex)
            else:
                kappa = np.sqrt(2.0 * effective_mass * (V - energy) * EV_TO_J) / HBAR
                decay = kappa * dx
                propagation_matrix = np.array([[np.exp(decay), 0],
                                             [0, np.exp(-decay)]], dtype=complex)

            transfer_matrix = np.dot(transfer_matrix, propagation_matrix)

        return transfer_matrix


# Utility functions
def create_test_quantum_mesh(nx: int = 50, ny: int = 50,
                           domain_x: Tuple[float, float] = (0.0, 1e-7),
                           domain_y: Tuple[float, float] = (0.0, 1e-7)) -> Tuple[np.ndarray, np.ndarray]:
    """Create test mesh for quantum transport"""
    x = np.linspace(domain_x[0], domain_x[1], nx)
    y = np.linspace(domain_y[0], domain_y[1], ny)

    vertices = []
    elements = []

    # Create structured mesh
    for j in range(ny):
        for i in range(nx):
            vertices.append([x[i], y[j]])

    vertices = np.array(vertices)

    # Create triangular elements
    for j in range(ny - 1):
        for i in range(nx - 1):
            # Lower triangle
            n1 = j * nx + i
            n2 = j * nx + (i + 1)
            n3 = (j + 1) * nx + i
            elements.append([n1, n2, n3])

            # Upper triangle
            n1 = j * nx + (i + 1)
            n2 = (j + 1) * nx + (i + 1)
            n3 = (j + 1) * nx + i
            elements.append([n1, n2, n3])

    return vertices, np.array(elements)


def create_quantum_well_potential(vertices: np.ndarray,
                                well_width: float = 5e-9,
                                well_depth: float = 0.3,
                                barrier_height: float = 1.0) -> np.ndarray:
    """Create quantum well potential profile"""
    x_coords = vertices[:, 0] if vertices.ndim > 1 else vertices
    potential = np.full_like(x_coords, barrier_height)

    # Find well region
    x_center = (np.max(x_coords) + np.min(x_coords)) / 2
    well_start = x_center - well_width / 2
    well_end = x_center + well_width / 2

    well_mask = (x_coords >= well_start) & (x_coords <= well_end)
    potential[well_mask] = well_depth

    return potential


def create_tunneling_barrier(position: np.ndarray,
                           barrier_start: float = 2e-8,
                           barrier_end: float = 3e-8,
                           barrier_height: float = 1.0) -> np.ndarray:
    """Create tunneling barrier potential profile"""
    potential = np.zeros_like(position)

    barrier_mask = (position >= barrier_start) & (position <= barrier_end)
    potential[barrier_mask] = barrier_height

    return potential
