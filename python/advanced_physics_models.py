"""
Advanced Physics Models for Semiconductor Device Simulation

This module provides Python interfaces for advanced physics models including:
- Strain effects and mechanical stress
- Thermal transport and self-heating
- Piezoelectric effects
- Thermoelectric coupling
- Optical properties and photogeneration

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class StrainConfig:
    """Configuration for strain effects model"""
    enable_strain_effects: bool = True
    enable_piezoresistance: bool = True
    enable_band_modification: bool = True
    
    # Deformation potential parameters
    shear_deformation_potential: float = 9.0  # eV
    mass_deformation_factor: float = 0.1      # Dimensionless
    
    # Piezoresistance factors
    electron_piezoresistance_factor: float = -100.0  # Typical for Silicon
    hole_piezoresistance_factor: float = 70.0        # Typical for Silicon
    
    # Numerical parameters
    strain_tolerance: float = 1e-8
    max_strain_iterations: int = 50

@dataclass
class ThermalConfig:
    """Configuration for thermal transport model"""
    enable_thermal_transport: bool = True
    enable_joule_heating: bool = True
    enable_thermal_coupling: bool = True
    enable_temperature_dependent_properties: bool = True
    
    # Thermal parameters
    thermal_diffusivity: float = 8.8e-5  # m²/s for Silicon
    ambient_temperature: float = 300.0    # K
    thermal_time_constant: float = 1e-6   # s
    
    # Boundary conditions
    thermal_boundary_resistance: float = 1e-4  # K·m²/W
    
    # Numerical parameters
    thermal_tolerance: float = 1e-6
    max_thermal_iterations: int = 100
    thermal_time_step: float = 1e-9  # s

@dataclass
class PiezoelectricConfig:
    """Configuration for piezoelectric effects"""
    enable_piezoelectric_effects: bool = True
    enable_spontaneous_polarization: bool = True
    enable_piezoelectric_polarization: bool = True
    
    # Piezoelectric constants (C/m² for wurtzite structure)
    e31: float = -0.49  # C/m²
    e33: float = 0.73   # C/m²
    e15: float = -0.40  # C/m²
    
    # Spontaneous polarization
    spontaneous_polarization: float = -0.029  # C/m² for GaN
    
    # Numerical parameters
    piezo_tolerance: float = 1e-8
    max_piezo_iterations: int = 50

@dataclass
class OpticalConfig:
    """Configuration for optical properties model"""
    enable_optical_generation: bool = True
    enable_photon_recycling: bool = False
    enable_stimulated_emission: bool = False
    
    # Optical parameters
    photon_energy: float = 1.24        # eV (1 μm wavelength)
    optical_power: float = 1e-3        # W
    absorption_coefficient: float = 1e4 # cm⁻¹
    quantum_efficiency: float = 0.8     # Dimensionless
    
    # Numerical parameters
    optical_tolerance: float = 1e-8
    max_optical_iterations: int = 50

@dataclass
class MaterialProperties:
    """Material properties for advanced physics models"""
    name: str = "Silicon"
    
    # Basic properties
    bandgap: float = 1.12           # eV at 300K
    electron_affinity: float = 4.05 # eV
    dielectric_constant: float = 11.7 # Relative permittivity
    
    # Effective masses (in units of m0)
    electron_effective_mass: float = 0.26
    hole_effective_mass_heavy: float = 0.49
    hole_effective_mass_light: float = 0.16
    
    # Mobility parameters
    electron_mobility: float = 1350.0  # cm²/V·s at 300K
    hole_mobility: float = 480.0       # cm²/V·s at 300K
    
    # Thermal properties
    thermal_conductivity: float = 148.0  # W/m·K
    specific_heat: float = 700.0         # J/kg·K
    density: float = 2330.0              # kg/m³
    
    # Mechanical properties
    youngs_modulus: float = 170e9        # Pa
    poisson_ratio: float = 0.28          # Dimensionless
    
    # Optical properties
    refractive_index: float = 3.5        # At 1550 nm
    absorption_coefficient: float = 0.0   # cm⁻¹ (for indirect bandgap)

# ============================================================================
# Physical Constants
# ============================================================================

class PhysicalConstants:
    """Physical constants for advanced physics calculations"""
    k = 1.380649e-23    # Boltzmann constant (J/K)
    q = 1.602176634e-19 # Elementary charge (C)
    h = 6.62607015e-34  # Planck constant (J·s)
    hbar = h / (2.0 * np.pi) # Reduced Planck constant
    m0 = 9.1093837015e-31  # Electron rest mass (kg)
    eps0 = 8.8541878128e-12 # Vacuum permittivity (F/m)
    c = 299792458.0      # Speed of light (m/s)

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class StrainTensor:
    """Strain tensor components"""
    exx: np.ndarray = field(default_factory=lambda: np.array([]))
    eyy: np.ndarray = field(default_factory=lambda: np.array([]))
    ezz: np.ndarray = field(default_factory=lambda: np.array([]))
    exy: np.ndarray = field(default_factory=lambda: np.array([]))
    exz: np.ndarray = field(default_factory=lambda: np.array([]))
    eyz: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class BandStructureModification:
    """Band structure modifications due to strain"""
    conduction_band_shift: np.ndarray = field(default_factory=lambda: np.array([]))
    valence_band_shift_heavy: np.ndarray = field(default_factory=lambda: np.array([]))
    valence_band_shift_light: np.ndarray = field(default_factory=lambda: np.array([]))
    valence_band_shift_split: np.ndarray = field(default_factory=lambda: np.array([]))
    effective_mass_modification: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class MobilityModification:
    """Mobility modifications due to strain"""
    electron_mobility_factor: np.ndarray = field(default_factory=lambda: np.array([]))
    hole_mobility_factor: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class PolarizationField:
    """Polarization fields"""
    Px: np.ndarray = field(default_factory=lambda: np.array([]))
    Py: np.ndarray = field(default_factory=lambda: np.array([]))
    Pz: np.ndarray = field(default_factory=lambda: np.array([]))
    bound_charge_density: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class ThermalCoupling:
    """Thermal coupling effects"""
    bandgap_modification: np.ndarray = field(default_factory=lambda: np.array([]))
    mobility_modification: np.ndarray = field(default_factory=lambda: np.array([]))
    thermal_voltage: np.ndarray = field(default_factory=lambda: np.array([]))
    thermal_diffusion_length: np.ndarray = field(default_factory=lambda: np.array([]))

# ============================================================================
# Strain Effects Model
# ============================================================================

class StrainEffectsModel:
    """
    Strain effects model for semiconductor devices
    
    This class implements strain effects including:
    - Strain tensor calculation from displacement fields
    - Band structure modifications due to strain
    - Mobility modifications (piezoresistance effect)
    - Deformation potential calculations
    """
    
    def __init__(self, config: StrainConfig = None, material: MaterialProperties = None):
        self.config = config or StrainConfig()
        self.material = material or MaterialProperties()
        self._initialize_strain_parameters()
    
    def _initialize_strain_parameters(self):
        """Initialize strain-related parameters for different materials"""
        if self.material.name == "Silicon":
            self.deformation_potentials = {
                'conduction_band': -8.6,  # eV
                'valence_band_heavy': -4.8,  # eV
                'valence_band_light': -4.8,  # eV
                'valence_band_split': -5.1   # eV
            }
            self.elastic_constants = {
                'c11': 165.8e9,  # Pa
                'c12': 63.9e9,   # Pa
                'c44': 79.6e9    # Pa
            }
        elif self.material.name == "GaAs":
            self.deformation_potentials = {
                'conduction_band': -7.17,  # eV
                'valence_band_heavy': -1.16,  # eV
                'valence_band_light': -1.16,  # eV
                'valence_band_split': -1.7    # eV
            }
            self.elastic_constants = {
                'c11': 118.8e9,  # Pa
                'c12': 53.8e9,   # Pa
                'c44': 59.4e9    # Pa
            }
        else:
            # Default to Silicon values
            self.material.name = "Silicon"
            self._initialize_strain_parameters()
    
    def calculate_strain_tensor(self, displacement_x: np.ndarray, displacement_y: np.ndarray, 
                              displacement_z: np.ndarray, mesh_spacing: float = 1e-9) -> StrainTensor:
        """
        Calculate strain tensor from displacement fields
        
        Args:
            displacement_x: Displacement field in x-direction (m)
            displacement_y: Displacement field in y-direction (m)
            displacement_z: Displacement field in z-direction (m)
            mesh_spacing: Mesh spacing for gradient calculation (m)
        
        Returns:
            StrainTensor object with all strain components
        """
        strain = StrainTensor()
        
        # Calculate gradients using finite differences
        dux_dx = np.gradient(displacement_x, mesh_spacing, axis=0)
        duy_dy = np.gradient(displacement_y, mesh_spacing, axis=1 if displacement_y.ndim > 1 else 0)
        duz_dz = np.gradient(displacement_z, mesh_spacing, axis=2 if displacement_z.ndim > 2 else 0)
        
        dux_dy = np.gradient(displacement_x, mesh_spacing, axis=1 if displacement_x.ndim > 1 else 0)
        duy_dx = np.gradient(displacement_y, mesh_spacing, axis=0)
        
        dux_dz = np.gradient(displacement_x, mesh_spacing, axis=2 if displacement_x.ndim > 2 else 0)
        duz_dx = np.gradient(displacement_z, mesh_spacing, axis=0)
        
        duy_dz = np.gradient(displacement_y, mesh_spacing, axis=2 if displacement_y.ndim > 2 else 0)
        duz_dy = np.gradient(displacement_z, mesh_spacing, axis=1 if displacement_z.ndim > 1 else 0)
        
        # Calculate strain tensor components
        strain.exx = dux_dx
        strain.eyy = duy_dy
        strain.ezz = duz_dz
        strain.exy = 0.5 * (dux_dy + duy_dx)
        strain.exz = 0.5 * (dux_dz + duz_dx)
        strain.eyz = 0.5 * (duy_dz + duz_dy)
        
        return strain
    
    def calculate_band_modification(self, strain: StrainTensor) -> BandStructureModification:
        """
        Calculate band structure modifications due to strain
        
        Args:
            strain: StrainTensor object
        
        Returns:
            BandStructureModification object
        """
        modification = BandStructureModification()
        
        # Calculate hydrostatic strain
        hydrostatic_strain = strain.exx + strain.eyy + strain.ezz
        
        # Calculate uniaxial strain components
        uniaxial_strain_xx = strain.exx - hydrostatic_strain / 3.0
        uniaxial_strain_yy = strain.eyy - hydrostatic_strain / 3.0
        uniaxial_strain_zz = strain.ezz - hydrostatic_strain / 3.0
        
        # Conduction band shift (hydrostatic deformation potential)
        modification.conduction_band_shift = (
            self.deformation_potentials['conduction_band'] * hydrostatic_strain
        )
        
        # Calculate shear strain magnitude
        shear_strain = np.sqrt(
            uniaxial_strain_xx**2 + uniaxial_strain_yy**2 + uniaxial_strain_zz**2 +
            6.0 * (strain.exy**2 + strain.exz**2 + strain.eyz**2)
        )
        
        # Valence band shifts (more complex due to degeneracy lifting)
        modification.valence_band_shift_heavy = (
            self.deformation_potentials['valence_band_heavy'] * hydrostatic_strain +
            self.config.shear_deformation_potential * shear_strain
        )
        
        modification.valence_band_shift_light = (
            self.deformation_potentials['valence_band_light'] * hydrostatic_strain -
            self.config.shear_deformation_potential * shear_strain
        )
        
        modification.valence_band_shift_split = (
            self.deformation_potentials['valence_band_split'] * hydrostatic_strain
        )
        
        # Effective mass modification (simplified model)
        modification.effective_mass_modification = (
            1.0 + self.config.mass_deformation_factor * np.abs(hydrostatic_strain)
        )
        
        return modification
    
    def calculate_mobility_modification(self, strain: StrainTensor) -> MobilityModification:
        """
        Calculate mobility modifications due to strain (piezoresistance effect)
        
        Args:
            strain: StrainTensor object
        
        Returns:
            MobilityModification object
        """
        modification = MobilityModification()
        
        # Calculate strain magnitude
        strain_magnitude = np.sqrt(
            strain.exx**2 + strain.eyy**2 + strain.ezz**2 +
            2.0 * (strain.exy**2 + strain.exz**2 + strain.eyz**2)
        )
        
        # Piezoresistance effect on mobility
        modification.electron_mobility_factor = (
            1.0 + self.config.electron_piezoresistance_factor * strain_magnitude
        )
        
        modification.hole_mobility_factor = (
            1.0 + self.config.hole_piezoresistance_factor * strain_magnitude
        )
        
        # Ensure positive mobility factors
        modification.electron_mobility_factor = np.maximum(0.1, modification.electron_mobility_factor)
        modification.hole_mobility_factor = np.maximum(0.1, modification.hole_mobility_factor)
        
        return modification

# ============================================================================
# Thermal Transport Model
# ============================================================================

class ThermalTransportModel:
    """
    Thermal transport model for semiconductor devices

    This class implements thermal transport including:
    - Heat equation solver with Joule heating
    - Thermal coupling to electronic properties
    - Temperature-dependent material properties
    - Thermal boundary conditions
    """

    def __init__(self, config: ThermalConfig = None, material: MaterialProperties = None):
        self.config = config or ThermalConfig()
        self.material = material or MaterialProperties()
        self._initialize_thermal_parameters()

    def _initialize_thermal_parameters(self):
        """Initialize thermal properties for different materials"""
        if self.material.name == "Silicon":
            self.thermal_properties = {
                'thermal_conductivity': 148.0,  # W/m·K at 300K
                'specific_heat': 700.0,         # J/kg·K
                'density': 2330.0,              # kg/m³
                'thermal_expansion': 2.6e-6     # 1/K
            }
        elif self.material.name == "GaAs":
            self.thermal_properties = {
                'thermal_conductivity': 55.0,   # W/m·K at 300K
                'specific_heat': 350.0,         # J/kg·K
                'density': 5320.0,              # kg/m³
                'thermal_expansion': 5.7e-6     # 1/K
            }
        else:
            # Default to Silicon values
            self.thermal_properties = {
                'thermal_conductivity': 148.0,
                'specific_heat': 700.0,
                'density': 2330.0,
                'thermal_expansion': 2.6e-6
            }

    def solve_heat_equation(self, initial_temperature: np.ndarray, heat_generation: np.ndarray,
                          boundary_conditions: Dict[str, float], mesh_spacing: float = 1e-9,
                          time_step: float = None, num_time_steps: int = 100) -> np.ndarray:
        """
        Solve the heat equation with Joule heating

        Args:
            initial_temperature: Initial temperature distribution (K)
            heat_generation: Heat generation rate (W/m³)
            boundary_conditions: Dictionary of boundary conditions
            mesh_spacing: Mesh spacing (m)
            time_step: Time step for temporal discretization (s)
            num_time_steps: Number of time steps

        Returns:
            Temperature distribution after time evolution (K)
        """
        if time_step is None:
            time_step = self.config.thermal_time_step

        temperature = initial_temperature.copy()

        # Thermal diffusivity
        alpha = (self.thermal_properties['thermal_conductivity'] /
                (self.thermal_properties['density'] * self.thermal_properties['specific_heat']))

        # Stability condition for explicit scheme
        max_time_step = 0.25 * mesh_spacing**2 / alpha
        if time_step > max_time_step:
            warnings.warn(f"Time step {time_step} may be unstable. "
                         f"Consider using {max_time_step} or smaller.")

        for step in range(num_time_steps):
            # Calculate Laplacian using finite differences
            laplacian = self._calculate_laplacian(temperature, mesh_spacing)

            # Heat equation: ∂T/∂t = α∇²T + Q/(ρcp)
            heat_capacity = (self.thermal_properties['density'] *
                           self.thermal_properties['specific_heat'])

            dT_dt = alpha * laplacian + heat_generation / heat_capacity

            # Update temperature
            temperature += time_step * dT_dt

            # Apply boundary conditions
            temperature = self._apply_thermal_boundary_conditions(temperature, boundary_conditions)

        return temperature

    def calculate_joule_heating(self, current_density_x: np.ndarray, current_density_y: np.ndarray,
                              electric_field_x: np.ndarray, electric_field_y: np.ndarray) -> np.ndarray:
        """
        Calculate Joule heating from current density and electric field

        Args:
            current_density_x: Current density in x-direction (A/m²)
            current_density_y: Current density in y-direction (A/m²)
            electric_field_x: Electric field in x-direction (V/m)
            electric_field_y: Electric field in y-direction (V/m)

        Returns:
            Joule heating rate (W/m³)
        """
        # Joule heating: Q = J·E
        joule_heating = (current_density_x * electric_field_x +
                        current_density_y * electric_field_y)

        # Ensure non-negative heating
        return np.maximum(0.0, joule_heating)

    def calculate_thermal_coupling(self, temperature: np.ndarray,
                                 carrier_density_n: np.ndarray,
                                 carrier_density_p: np.ndarray) -> ThermalCoupling:
        """
        Calculate thermal coupling effects on electronic properties

        Args:
            temperature: Temperature distribution (K)
            carrier_density_n: Electron density (m⁻³)
            carrier_density_p: Hole density (m⁻³)

        Returns:
            ThermalCoupling object
        """
        coupling = ThermalCoupling()
        T_ref = 300.0  # Reference temperature

        # Bandgap temperature dependence (Varshni model)
        alpha = 4.73e-4  # eV/K for Silicon
        beta = 636.0     # K for Silicon
        coupling.bandgap_modification = -alpha * temperature**2 / (temperature + beta)

        # Mobility temperature dependence
        mobility_temp_exponent = -2.4  # For electrons in Silicon
        coupling.mobility_modification = (temperature / T_ref)**mobility_temp_exponent

        # Thermal voltage
        coupling.thermal_voltage = PhysicalConstants.k * temperature / PhysicalConstants.q

        # Thermal diffusion length (simplified)
        diffusion_coefficient = self.config.thermal_diffusivity
        recombination_time = 1e-6  # Typical value
        coupling.thermal_diffusion_length = np.sqrt(diffusion_coefficient * recombination_time)

        return coupling

    def _calculate_laplacian(self, field: np.ndarray, spacing: float) -> np.ndarray:
        """Calculate Laplacian using finite differences"""
        if field.ndim == 1:
            # 1D case
            laplacian = np.zeros_like(field)
            laplacian[1:-1] = (field[2:] - 2*field[1:-1] + field[:-2]) / spacing**2
        elif field.ndim == 2:
            # 2D case
            laplacian = np.zeros_like(field)
            laplacian[1:-1, 1:-1] = (
                (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / spacing**2 +
                (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / spacing**2
            )
        else:
            raise ValueError("Only 1D and 2D fields are supported")

        return laplacian

    def _apply_thermal_boundary_conditions(self, temperature: np.ndarray,
                                         boundary_conditions: Dict[str, float]) -> np.ndarray:
        """Apply thermal boundary conditions"""
        temp = temperature.copy()

        # Apply Dirichlet boundary conditions
        if 'left' in boundary_conditions:
            if temp.ndim == 1:
                temp[0] = boundary_conditions['left']
            else:
                temp[:, 0] = boundary_conditions['left']

        if 'right' in boundary_conditions:
            if temp.ndim == 1:
                temp[-1] = boundary_conditions['right']
            else:
                temp[:, -1] = boundary_conditions['right']

        if 'bottom' in boundary_conditions and temp.ndim == 2:
            temp[0, :] = boundary_conditions['bottom']

        if 'top' in boundary_conditions and temp.ndim == 2:
            temp[-1, :] = boundary_conditions['top']

        return temp

# ============================================================================
# Piezoelectric Effects Model
# ============================================================================

class PiezoelectricModel:
    """
    Piezoelectric effects model for compound semiconductors

    This class implements piezoelectric effects including:
    - Spontaneous and piezoelectric polarization
    - Bound charge calculation from polarization gradients
    - Interface charge at heterojunctions
    - Wurtzite crystal structure effects
    """

    def __init__(self, config: PiezoelectricConfig = None, material: MaterialProperties = None):
        self.config = config or PiezoelectricConfig()
        self.material = material or MaterialProperties()
        self._initialize_piezoelectric_parameters()

    def _initialize_piezoelectric_parameters(self):
        """Initialize piezoelectric constants for different materials"""
        if self.material.name == "GaN":
            self.piezo_constants = {
                'e31': -0.49,  # C/m²
                'e33': 0.73,   # C/m²
                'e15': -0.40   # C/m²
            }
        elif self.material.name == "AlN":
            self.piezo_constants = {
                'e31': -0.60,  # C/m²
                'e33': 1.46,   # C/m²
                'e15': -0.48   # C/m²
            }
        elif self.material.name == "ZnO":
            self.piezo_constants = {
                'e31': -0.57,  # C/m²
                'e33': 1.22,   # C/m²
                'e15': -0.48   # C/m²
            }
        else:
            # Default to GaN values
            self.piezo_constants = {
                'e31': -0.49,
                'e33': 0.73,
                'e15': -0.40
            }

    def calculate_polarization(self, strain: StrainTensor,
                             electric_field_x: np.ndarray = None,
                             electric_field_y: np.ndarray = None,
                             electric_field_z: np.ndarray = None) -> PolarizationField:
        """
        Calculate polarization field from strain and electric field

        Args:
            strain: StrainTensor object
            electric_field_x: Electric field in x-direction (V/m)
            electric_field_y: Electric field in y-direction (V/m)
            electric_field_z: Electric field in z-direction (V/m)

        Returns:
            PolarizationField object
        """
        polarization = PolarizationField()

        # Initialize electric fields if not provided
        if electric_field_x is None:
            electric_field_x = np.zeros_like(strain.exx)
        if electric_field_y is None:
            electric_field_y = np.zeros_like(strain.eyy)
        if electric_field_z is None:
            electric_field_z = np.zeros_like(strain.ezz)

        # Spontaneous polarization (along z-axis for wurtzite)
        P_sp_z = self.config.spontaneous_polarization

        # Piezoelectric polarization components
        e31 = self.piezo_constants['e31']
        e33 = self.piezo_constants['e33']
        e15 = self.piezo_constants['e15']

        # Piezoelectric polarization
        P_pz_x = 2.0 * e15 * strain.exz
        P_pz_y = 2.0 * e15 * strain.eyz
        P_pz_z = e31 * (strain.exx + strain.eyy) + e33 * strain.ezz

        # Total polarization
        polarization.Px = P_pz_x
        polarization.Py = P_pz_y
        polarization.Pz = P_sp_z + P_pz_z

        return polarization

    def calculate_bound_charge_density(self, polarization: PolarizationField,
                                     mesh_spacing: float = 1e-9) -> np.ndarray:
        """
        Calculate bound charge density from polarization divergence

        Args:
            polarization: PolarizationField object
            mesh_spacing: Mesh spacing for gradient calculation (m)

        Returns:
            Bound charge density (C/m³)
        """
        # Calculate divergence of polarization: ∇·P
        div_Px = np.gradient(polarization.Px, mesh_spacing, axis=0)

        if polarization.Py.ndim > 1:
            div_Py = np.gradient(polarization.Py, mesh_spacing, axis=1)
        else:
            div_Py = np.zeros_like(div_Px)

        if polarization.Pz.ndim > 2:
            div_Pz = np.gradient(polarization.Pz, mesh_spacing, axis=2)
        else:
            div_Pz = np.zeros_like(div_Px)

        div_P = div_Px + div_Py + div_Pz

        # Bound charge density: ρ_bound = -∇·P
        return -div_P

    def calculate_interface_charge(self, polarization_1: PolarizationField,
                                 polarization_2: PolarizationField) -> np.ndarray:
        """
        Calculate interface charge at heterojunctions

        Args:
            polarization_1: Polarization in material 1
            polarization_2: Polarization in material 2

        Returns:
            Interface charge density (C/m²)
        """
        # Interface charge density: σ = (P2 - P1)·n
        # Assuming normal vector is along z-direction for simplicity
        delta_Pz = polarization_2.Pz - polarization_1.Pz
        return delta_Pz

# ============================================================================
# Optical Properties Model
# ============================================================================

class OpticalModel:
    """
    Optical properties model for semiconductor devices

    This class implements optical effects including:
    - Optical generation from photon absorption
    - Radiative recombination
    - Beer-Lambert law for light absorption
    - Quantum efficiency calculations
    """

    def __init__(self, config: OpticalConfig = None, material: MaterialProperties = None):
        self.config = config or OpticalConfig()
        self.material = material or MaterialProperties()

    def calculate_optical_generation(self, photon_flux: np.ndarray,
                                   position_z: np.ndarray) -> np.ndarray:
        """
        Calculate optical generation rate from photon flux

        Args:
            photon_flux: Incident photon flux (photons/m²/s)
            position_z: Position along light propagation direction (m)

        Returns:
            Generation rate (carriers/m³/s)
        """
        # Beer-Lambert law: I(z) = I0 * exp(-α*z)
        absorption = np.exp(-self.config.absorption_coefficient * position_z * 1e-2)  # Convert to cm

        # Generation rate: G = α * Φ * η
        generation_rate = (self.config.absorption_coefficient * 1e2 *  # Convert back to m⁻¹
                          photon_flux * absorption * self.config.quantum_efficiency)

        return generation_rate

    def calculate_radiative_recombination(self, electron_density: np.ndarray,
                                        hole_density: np.ndarray) -> np.ndarray:
        """
        Calculate radiative recombination rate

        Args:
            electron_density: Electron density (m⁻³)
            hole_density: Hole density (m⁻³)

        Returns:
            Recombination rate (carriers/m³/s)
        """
        # Radiative recombination coefficient
        if self.material.name == "GaAs":
            B_rad = 7.2e-16  # cm³/s for direct bandgap
        else:
            B_rad = 1e-15    # cm³/s for Silicon (very low due to indirect bandgap)

        # Radiative recombination: R = B * n * p
        recombination_rate = B_rad * 1e6 * electron_density * hole_density  # Convert to m³/s

        return recombination_rate

    def calculate_absorption_coefficient(self, photon_energy: float,
                                       temperature: float = 300.0) -> float:
        """
        Calculate absorption coefficient as function of photon energy

        Args:
            photon_energy: Photon energy (eV)
            temperature: Temperature (K)

        Returns:
            Absorption coefficient (m⁻¹)
        """
        # Temperature-dependent bandgap (Varshni model)
        alpha = 4.73e-4  # eV/K for Silicon
        beta = 636.0     # K for Silicon
        Eg = self.material.bandgap - alpha * temperature**2 / (temperature + beta)

        if photon_energy < Eg:
            # Below bandgap - exponential tail (Urbach rule)
            E_urbach = 0.02  # eV, Urbach energy
            alpha_0 = 1e4    # cm⁻¹, reference absorption
            absorption = alpha_0 * np.exp((photon_energy - Eg) / E_urbach)
        else:
            # Above bandgap - power law dependence
            if self.material.name == "GaAs":
                # Direct bandgap
                absorption = 1e4 * np.sqrt(photon_energy - Eg)  # cm⁻¹
            else:
                # Indirect bandgap (Silicon)
                absorption = 1e3 * (photon_energy - Eg)**2  # cm⁻¹

        return absorption * 1e2  # Convert to m⁻¹
