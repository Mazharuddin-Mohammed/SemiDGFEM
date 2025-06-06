#!/usr/bin/env python3
"""
Advanced Heterostructure Simulation Framework
Multi-material interfaces with band alignment and quantum effects

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

class SemiconductorMaterial(Enum):
    SI = "Silicon"
    GE = "Germanium"
    GAAS = "GaAs"
    ALGAS = "AlGaAs"
    INAS = "InAs"
    INGAAS = "InGaAs"
    SIC = "SiC"
    GAN = "GaN"
    ALGAN = "AlGaN"
    INGAN = "InGaN"

@dataclass
class BandParameters:
    """Band structure parameters for semiconductors"""
    bandgap: float  # eV
    electron_affinity: float  # eV
    effective_mass_electron: float  # m0
    effective_mass_hole_heavy: float  # m0
    effective_mass_hole_light: float  # m0
    dielectric_constant: float
    lattice_constant: float  # Å
    elastic_constant_c11: float  # GPa
    elastic_constant_c12: float  # GPa

@dataclass
class MobilityParameters:
    """Mobility parameters for different materials"""
    electron_mobility_300k: float  # cm²/V·s
    hole_mobility_300k: float  # cm²/V·s
    temperature_exponent_electron: float
    temperature_exponent_hole: float
    field_saturation_electron: float  # V/cm
    field_saturation_hole: float  # V/cm

@dataclass
class LayerStructure:
    """Individual layer in heterostructure"""
    material: SemiconductorMaterial
    thickness: float  # nm
    composition: float  # For alloys (e.g., Al fraction in AlGaAs)
    doping_type: str  # "n", "p", or "intrinsic"
    doping_concentration: float  # cm⁻³
    position: float  # nm (starting position)

class MaterialDatabase:
    """Comprehensive semiconductor material database"""
    
    @staticmethod
    def get_band_parameters(material: SemiconductorMaterial, composition: float = 0.0, 
                           temperature: float = 300.0) -> BandParameters:
        """Get band parameters for semiconductor materials"""
        
        if material == SemiconductorMaterial.SI:
            # Silicon parameters
            Eg_0 = 1.166  # eV at 0K
            alpha = 4.73e-4  # eV/K
            beta = 636  # K
            bandgap = Eg_0 - (alpha * temperature**2) / (temperature + beta)
            
            return BandParameters(
                bandgap=bandgap,
                electron_affinity=4.05,
                effective_mass_electron=0.26,
                effective_mass_hole_heavy=0.39,
                effective_mass_hole_light=0.16,
                dielectric_constant=11.7,
                lattice_constant=5.431,
                elastic_constant_c11=165.8,
                elastic_constant_c12=63.9
            )
        
        elif material == SemiconductorMaterial.GE:
            # Germanium parameters
            return BandParameters(
                bandgap=0.66,
                electron_affinity=4.0,
                effective_mass_electron=0.12,
                effective_mass_hole_heavy=0.28,
                effective_mass_hole_light=0.044,
                dielectric_constant=16.0,
                lattice_constant=5.658,
                elastic_constant_c11=128.5,
                elastic_constant_c12=48.3
            )
        
        elif material == SemiconductorMaterial.GAAS:
            # GaAs parameters
            return BandParameters(
                bandgap=1.424,
                electron_affinity=4.07,
                effective_mass_electron=0.067,
                effective_mass_hole_heavy=0.45,
                effective_mass_hole_light=0.082,
                dielectric_constant=12.9,
                lattice_constant=5.653,
                elastic_constant_c11=118.8,
                elastic_constant_c12=53.8
            )
        
        elif material == SemiconductorMaterial.ALGAS:
            # AlGaAs parameters (composition-dependent)
            # Linear interpolation between GaAs (x=0) and AlAs (x=1)
            x = composition  # Al fraction
            
            # Bandgap bowing
            Eg_GaAs = 1.424
            Eg_AlAs = 2.168
            bowing_parameter = 0.37
            bandgap = (1-x) * Eg_GaAs + x * Eg_AlAs - x * (1-x) * bowing_parameter
            
            # Electron affinity
            chi_GaAs = 4.07
            chi_AlAs = 3.5
            electron_affinity = (1-x) * chi_GaAs + x * chi_AlAs
            
            return BandParameters(
                bandgap=bandgap,
                electron_affinity=electron_affinity,
                effective_mass_electron=0.067 + 0.083 * x,
                effective_mass_hole_heavy=0.45 + 0.25 * x,
                effective_mass_hole_light=0.082 + 0.068 * x,
                dielectric_constant=12.9 - 2.84 * x,
                lattice_constant=5.653 + 0.0078 * x,
                elastic_constant_c11=118.8 + 6.2 * x,
                elastic_constant_c12=53.8 + 2.9 * x
            )
        
        elif material == SemiconductorMaterial.INAS:
            # InAs parameters
            return BandParameters(
                bandgap=0.354,
                electron_affinity=4.9,
                effective_mass_electron=0.023,
                effective_mass_hole_heavy=0.41,
                effective_mass_hole_light=0.026,
                dielectric_constant=15.15,
                lattice_constant=6.058,
                elastic_constant_c11=83.3,
                elastic_constant_c12=45.3
            )
        
        elif material == SemiconductorMaterial.GAN:
            # GaN parameters
            return BandParameters(
                bandgap=3.39,
                electron_affinity=4.1,
                effective_mass_electron=0.20,
                effective_mass_hole_heavy=1.4,
                effective_mass_hole_light=0.15,
                dielectric_constant=9.5,
                lattice_constant=3.189,
                elastic_constant_c11=390,
                elastic_constant_c12=145
            )

        elif material == SemiconductorMaterial.ALGAN:
            # AlGaN parameters (composition-dependent)
            x = composition  # Al fraction

            # Bandgap bowing (GaN to AlN)
            Eg_GaN = 3.39
            Eg_AlN = 6.2
            bowing_parameter = 1.0
            bandgap = (1-x) * Eg_GaN + x * Eg_AlN - x * (1-x) * bowing_parameter

            # Electron affinity
            chi_GaN = 4.1
            chi_AlN = 1.9
            electron_affinity = (1-x) * chi_GaN + x * chi_AlN

            return BandParameters(
                bandgap=bandgap,
                electron_affinity=electron_affinity,
                effective_mass_electron=0.20 + 0.15 * x,
                effective_mass_hole_heavy=1.4 + 0.6 * x,
                effective_mass_hole_light=0.15 + 0.10 * x,
                dielectric_constant=9.5 - 1.0 * x,
                lattice_constant=3.189 - 0.077 * x,
                elastic_constant_c11=390 + 80 * x,
                elastic_constant_c12=145 + 35 * x
            )

        else:
            raise ValueError(f"Material {material} not implemented")
    
    @staticmethod
    def get_mobility_parameters(material: SemiconductorMaterial, 
                               composition: float = 0.0) -> MobilityParameters:
        """Get mobility parameters for semiconductor materials"""
        
        if material == SemiconductorMaterial.SI:
            return MobilityParameters(
                electron_mobility_300k=1400,
                hole_mobility_300k=450,
                temperature_exponent_electron=-2.2,
                temperature_exponent_hole=-2.2,
                field_saturation_electron=1e4,
                field_saturation_hole=8e3
            )
        
        elif material == SemiconductorMaterial.GAAS:
            return MobilityParameters(
                electron_mobility_300k=8500,
                hole_mobility_300k=400,
                temperature_exponent_electron=-1.0,
                temperature_exponent_hole=-2.1,
                field_saturation_electron=4e3,
                field_saturation_hole=6e3
            )
        
        elif material == SemiconductorMaterial.ALGAS:
            # Composition-dependent mobility
            x = composition
            mu_e_GaAs = 8500
            mu_e_AlAs = 200
            mu_h_GaAs = 400
            mu_h_AlAs = 150
            
            return MobilityParameters(
                electron_mobility_300k=(1-x) * mu_e_GaAs + x * mu_e_AlAs,
                hole_mobility_300k=(1-x) * mu_h_GaAs + x * mu_h_AlAs,
                temperature_exponent_electron=-1.0 - 0.5 * x,
                temperature_exponent_hole=-2.1,
                field_saturation_electron=4e3 * (1 - 0.5 * x),
                field_saturation_hole=6e3
            )
        
        elif material == SemiconductorMaterial.GAN:
            return MobilityParameters(
                electron_mobility_300k=1200,
                hole_mobility_300k=25,
                temperature_exponent_electron=-1.5,
                temperature_exponent_hole=-3.0,
                field_saturation_electron=2e5,
                field_saturation_hole=1e5
            )

        elif material == SemiconductorMaterial.ALGAN:
            # Composition-dependent mobility for AlGaN
            x = composition
            mu_e_GaN = 1200
            mu_e_AlN = 300
            mu_h_GaN = 25
            mu_h_AlN = 10

            return MobilityParameters(
                electron_mobility_300k=(1-x) * mu_e_GaN + x * mu_e_AlN,
                hole_mobility_300k=(1-x) * mu_h_GaN + x * mu_h_AlN,
                temperature_exponent_electron=-1.5 - 0.5 * x,
                temperature_exponent_hole=-3.0,
                field_saturation_electron=2e5 * (1 - 0.3 * x),
                field_saturation_hole=1e5
            )

        else:
            # Default parameters
            return MobilityParameters(
                electron_mobility_300k=1000,
                hole_mobility_300k=100,
                temperature_exponent_electron=-2.0,
                temperature_exponent_hole=-2.0,
                field_saturation_electron=1e4,
                field_saturation_hole=8e3
            )

class HeterostructureDevice:
    """Advanced heterostructure device simulation"""
    
    def __init__(self, layers: List[LayerStructure], temperature: float = 300.0):
        self.layers = layers
        self.temperature = temperature
        
        # Physical constants
        self.q = 1.602176634e-19  # Elementary charge (C)
        self.k = 1.380649e-23     # Boltzmann constant (J/K)
        self.eps0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
        self.h = 6.62607015e-34   # Planck constant (J·s)
        self.hbar = self.h / (2 * np.pi)  # Reduced Planck constant
        self.me = 9.1093837015e-31  # Electron mass (kg)
        
        # Calculate total structure thickness
        self.total_thickness = sum(layer.thickness for layer in layers)
        
        # Initialize mesh and material properties
        self.mesh = None
        self.band_structure = None
        self.carrier_densities = None
        
    def create_mesh(self, nz: int = 1000) -> np.ndarray:
        """Create 1D mesh along growth direction"""
        
        z = np.linspace(0, self.total_thickness * 1e-9, nz)  # Convert nm to m
        self.mesh = {'z': z, 'nz': nz, 'dz': z[1] - z[0]}
        
        return z
    
    def calculate_band_structure(self) -> Dict[str, np.ndarray]:
        """Calculate band structure including band offsets"""
        
        if self.mesh is None:
            self.create_mesh()
        
        z = self.mesh['z']
        nz = self.mesh['nz']
        
        # Initialize arrays
        conduction_band = np.zeros(nz)
        valence_band = np.zeros(nz)
        bandgap = np.zeros(nz)
        electron_affinity = np.zeros(nz)
        dielectric_constant = np.zeros(nz)
        doping_profile = np.zeros(nz)
        
        # Fill arrays based on layer structure
        current_position = 0.0
        
        for layer in self.layers:
            # Find mesh points in this layer
            layer_start = current_position * 1e-9  # Convert nm to m
            layer_end = (current_position + layer.thickness) * 1e-9
            
            layer_mask = (z >= layer_start) & (z < layer_end)
            
            # Get material parameters
            band_params = MaterialDatabase.get_band_parameters(
                layer.material, layer.composition, self.temperature)
            
            # Fill material properties
            bandgap[layer_mask] = band_params.bandgap
            electron_affinity[layer_mask] = band_params.electron_affinity
            dielectric_constant[layer_mask] = band_params.dielectric_constant
            
            # Doping
            if layer.doping_type == "n":
                doping_profile[layer_mask] = layer.doping_concentration
            elif layer.doping_type == "p":
                doping_profile[layer_mask] = -layer.doping_concentration
            # intrinsic layers remain zero
            
            current_position += layer.thickness
        
        # Calculate band edges (referenced to vacuum level)
        conduction_band = -electron_affinity  # eV below vacuum
        valence_band = conduction_band - bandgap
        
        # Calculate built-in potential and band bending
        potential = self._solve_poisson_equation(doping_profile, dielectric_constant)
        
        # Apply potential to bands
        conduction_band_bent = conduction_band - potential
        valence_band_bent = valence_band - potential
        
        self.band_structure = {
            'z': z,
            'conduction_band': conduction_band_bent,
            'valence_band': valence_band_bent,
            'bandgap': bandgap,
            'potential': potential,
            'doping_profile': doping_profile,
            'dielectric_constant': dielectric_constant,
            'electron_affinity': electron_affinity
        }
        
        return self.band_structure
    
    def _solve_poisson_equation(self, doping_profile: np.ndarray, 
                               dielectric_constant: np.ndarray) -> np.ndarray:
        """Solve 1D Poisson equation for built-in potential"""
        
        z = self.mesh['z']
        dz = self.mesh['dz']
        nz = self.mesh['nz']
        
        # Initialize potential
        potential = np.zeros(nz)
        
        # Boundary conditions (grounded contacts)
        potential[0] = 0.0
        potential[-1] = 0.0
        
        # Iterative solution of Poisson equation
        for iteration in range(1000):
            potential_old = potential.copy()
            
            for i in range(1, nz-1):
                # Finite difference approximation
                eps_i = dielectric_constant[i] * self.eps0
                rho_i = self.q * doping_profile[i]  # Charge density
                
                # Second derivative approximation
                d2phi_dz2 = (potential[i+1] - 2*potential[i] + potential[i-1]) / dz**2
                
                # Poisson equation: ∇²φ = -ρ/ε
                potential[i] = 0.5 * (potential[i+1] + potential[i-1] + rho_i * dz**2 / eps_i)
            
            # Check convergence
            if np.max(np.abs(potential - potential_old)) < 1e-6:
                break
        
        return potential
    
    def calculate_carrier_densities(self, fermi_level: float = None) -> Dict[str, np.ndarray]:
        """Calculate carrier densities including quantum effects"""
        
        if self.band_structure is None:
            self.calculate_band_structure()
        
        z = self.band_structure['z']
        Ec = self.band_structure['conduction_band']
        Ev = self.band_structure['valence_band']
        
        # Determine Fermi level if not provided
        if fermi_level is None:
            fermi_level = self._calculate_fermi_level()
        
        # Calculate carrier densities
        kT = self.k * self.temperature / self.q  # Thermal voltage in eV
        
        # Electron density (Boltzmann approximation)
        electron_density = np.zeros_like(z)
        hole_density = np.zeros_like(z)
        
        current_position = 0.0
        for layer in self.layers:
            layer_start = current_position * 1e-9
            layer_end = (current_position + layer.thickness) * 1e-9
            layer_mask = (z >= layer_start) & (z < layer_end)
            
            # Get material parameters
            band_params = MaterialDatabase.get_band_parameters(
                layer.material, layer.composition, self.temperature)
            
            # Effective density of states
            me_eff = band_params.effective_mass_electron * self.me
            mh_eff = band_params.effective_mass_hole_heavy * self.me
            
            Nc = 2 * (2 * np.pi * me_eff * self.k * self.temperature / self.h**2)**(3/2)
            Nv = 2 * (2 * np.pi * mh_eff * self.k * self.temperature / self.h**2)**(3/2)
            
            # Convert to cm⁻³
            Nc *= 1e-6
            Nv *= 1e-6
            
            # Calculate densities in this layer
            n_layer = Nc * np.exp((fermi_level - Ec[layer_mask]) / kT)
            p_layer = Nv * np.exp((Ev[layer_mask] - fermi_level) / kT)
            
            electron_density[layer_mask] = n_layer
            hole_density[layer_mask] = p_layer
            
            current_position += layer.thickness
        
        self.carrier_densities = {
            'electron_density': electron_density,
            'hole_density': hole_density,
            'fermi_level': fermi_level
        }
        
        return self.carrier_densities
    
    def _calculate_fermi_level(self) -> float:
        """Calculate Fermi level for charge neutrality"""
        
        # Simplified calculation - in practice would solve self-consistently
        # For now, use average of conduction and valence bands
        Ec = self.band_structure['conduction_band']
        Ev = self.band_structure['valence_band']
        
        # Weight by layer thickness and doping
        total_weight = 0.0
        weighted_energy = 0.0
        
        current_position = 0.0
        for layer in self.layers:
            layer_start = current_position * 1e-9
            layer_end = (current_position + layer.thickness) * 1e-9
            layer_mask = (self.band_structure['z'] >= layer_start) & (self.band_structure['z'] < layer_end)
            
            if np.any(layer_mask):
                layer_weight = layer.thickness
                
                if layer.doping_type == "n" and layer.doping_concentration > 1e16:
                    # n-type: Fermi level near conduction band
                    layer_energy = np.mean(Ec[layer_mask]) - 0.1
                elif layer.doping_type == "p" and layer.doping_concentration > 1e16:
                    # p-type: Fermi level near valence band
                    layer_energy = np.mean(Ev[layer_mask]) + 0.1
                else:
                    # Intrinsic: Fermi level at midgap
                    layer_energy = 0.5 * (np.mean(Ec[layer_mask]) + np.mean(Ev[layer_mask]))
                
                weighted_energy += layer_weight * layer_energy
                total_weight += layer_weight
            
            current_position += layer.thickness
        
        if total_weight > 0:
            return weighted_energy / total_weight
        else:
            return 0.0

    def calculate_quantum_wells(self) -> List[Dict[str, Any]]:
        """Identify and analyze quantum wells in the structure"""

        if self.band_structure is None:
            self.calculate_band_structure()

        z = self.band_structure['z'] * 1e9  # Convert to nm
        Ec = self.band_structure['conduction_band']
        Ev = self.band_structure['valence_band']

        quantum_wells = []

        # Find potential minima in conduction band (electron wells)
        ec_gradient = np.gradient(Ec)
        ec_minima = []

        for i in range(1, len(Ec)-1):
            if Ec[i] < Ec[i-1] and Ec[i] < Ec[i+1]:
                ec_minima.append(i)

        # Analyze each minimum
        for min_idx in ec_minima:
            # Find well boundaries
            left_boundary = min_idx
            right_boundary = min_idx

            # Find left boundary (where potential starts rising)
            for i in range(min_idx, 0, -1):
                if Ec[i] > Ec[min_idx] + 0.05:  # 50 meV above minimum
                    left_boundary = i
                    break

            # Find right boundary
            for i in range(min_idx, len(Ec)):
                if Ec[i] > Ec[min_idx] + 0.05:  # 50 meV above minimum
                    right_boundary = i
                    break

            well_width = z[right_boundary] - z[left_boundary]
            well_depth = min(Ec[left_boundary], Ec[right_boundary]) - Ec[min_idx]

            if well_width > 1.0 and well_depth > 0.01:  # Minimum 1 nm wide, 10 meV deep
                # Calculate quantum energy levels (infinite square well approximation)
                well_width_m = well_width * 1e-9

                # Find effective mass in the well
                well_center = (left_boundary + right_boundary) // 2
                layer_at_center = self._find_layer_at_position(z[well_center])

                if layer_at_center:
                    band_params = MaterialDatabase.get_band_parameters(
                        layer_at_center.material, layer_at_center.composition, self.temperature)
                    m_eff = band_params.effective_mass_electron * self.me

                    # Energy levels: E_n = n²π²ℏ²/(2m*L²)
                    energy_levels = []
                    for n in range(1, 6):  # First 5 levels
                        E_n = (n**2 * np.pi**2 * self.hbar**2) / (2 * m_eff * well_width_m**2)
                        E_n_eV = E_n / self.q  # Convert to eV

                        if E_n_eV < well_depth:
                            energy_levels.append(Ec[min_idx] + E_n_eV)

                    quantum_wells.append({
                        'type': 'electron',
                        'position': z[min_idx],
                        'width': well_width,
                        'depth': well_depth,
                        'energy_levels': energy_levels,
                        'material': layer_at_center.material.value,
                        'boundaries': (z[left_boundary], z[right_boundary])
                    })

        return quantum_wells

    def _find_layer_at_position(self, position_nm: float) -> Optional[LayerStructure]:
        """Find which layer contains the given position"""

        current_position = 0.0
        for layer in self.layers:
            if current_position <= position_nm < current_position + layer.thickness:
                return layer
            current_position += layer.thickness

        return None

    def calculate_transport_properties(self) -> Dict[str, np.ndarray]:
        """Calculate transport properties including mobility"""

        if self.carrier_densities is None:
            self.calculate_carrier_densities()

        z = self.band_structure['z']
        nz = len(z)

        # Initialize arrays
        electron_mobility = np.zeros(nz)
        hole_mobility = np.zeros(nz)
        conductivity = np.zeros(nz)
        resistivity = np.zeros(nz)

        current_position = 0.0
        for layer in self.layers:
            layer_start = current_position * 1e-9
            layer_end = (current_position + layer.thickness) * 1e-9
            layer_mask = (z >= layer_start) & (z < layer_end)

            # Get mobility parameters
            mobility_params = MaterialDatabase.get_mobility_parameters(
                layer.material, layer.composition)

            # Temperature dependence
            temp_factor = (self.temperature / 300.0) ** mobility_params.temperature_exponent_electron
            mu_e = mobility_params.electron_mobility_300k * temp_factor

            temp_factor = (self.temperature / 300.0) ** mobility_params.temperature_exponent_hole
            mu_h = mobility_params.hole_mobility_300k * temp_factor

            electron_mobility[layer_mask] = mu_e * 1e-4  # Convert cm²/V·s to m²/V·s
            hole_mobility[layer_mask] = mu_h * 1e-4

            current_position += layer.thickness

        # Calculate conductivity
        n = self.carrier_densities['electron_density'] * 1e6  # Convert cm⁻³ to m⁻³
        p = self.carrier_densities['hole_density'] * 1e6

        conductivity = self.q * (n * electron_mobility + p * hole_mobility)
        resistivity = 1.0 / np.maximum(conductivity, 1e-10)  # Avoid division by zero

        return {
            'electron_mobility': electron_mobility,
            'hole_mobility': hole_mobility,
            'conductivity': conductivity,
            'resistivity': resistivity
        }

    def plot_band_structure(self, save_path: str = None) -> None:
        """Plot comprehensive band structure diagram"""

        if self.band_structure is None:
            self.calculate_band_structure()

        if self.carrier_densities is None:
            self.calculate_carrier_densities()

        z_nm = self.band_structure['z'] * 1e9  # Convert to nm

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Band diagram
        ax1.plot(z_nm, self.band_structure['conduction_band'], 'b-', linewidth=2, label='Conduction Band')
        ax1.plot(z_nm, self.band_structure['valence_band'], 'r-', linewidth=2, label='Valence Band')

        # Plot Fermi level
        fermi_level = self.carrier_densities['fermi_level']
        ax1.axhline(y=fermi_level, color='green', linestyle='--', linewidth=2, label=f'Fermi Level ({fermi_level:.3f} eV)')

        # Add quantum well energy levels
        quantum_wells = self.calculate_quantum_wells()
        for qw in quantum_wells:
            for level in qw['energy_levels']:
                ax1.axhline(y=level, color='orange', linestyle=':', alpha=0.7)

        ax1.set_xlabel('Position (nm)')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title('Band Structure')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Carrier densities
        ax2.semilogy(z_nm, self.carrier_densities['electron_density'], 'b-', linewidth=2, label='Electrons')
        ax2.semilogy(z_nm, self.carrier_densities['hole_density'], 'r-', linewidth=2, label='Holes')
        ax2.set_xlabel('Position (nm)')
        ax2.set_ylabel('Carrier Density (cm⁻³)')
        ax2.set_title('Carrier Densities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Doping profile
        doping = self.band_structure['doping_profile']
        positive_doping = np.maximum(doping, 0)
        negative_doping = np.minimum(doping, 0)

        ax3.semilogy(z_nm, np.abs(positive_doping), 'g-', linewidth=2, label='n-type')
        ax3.semilogy(z_nm, np.abs(negative_doping), 'm-', linewidth=2, label='p-type')
        ax3.set_xlabel('Position (nm)')
        ax3.set_ylabel('Doping Concentration (cm⁻³)')
        ax3.set_title('Doping Profile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Electric field
        electric_field = -np.gradient(self.band_structure['potential'], self.band_structure['z'])
        ax4.plot(z_nm, electric_field * 1e-5, 'purple', linewidth=2)  # Convert V/m to V/cm
        ax4.set_xlabel('Position (nm)')
        ax4.set_ylabel('Electric Field (V/cm)')
        ax4.set_title('Electric Field')
        ax4.grid(True, alpha=0.3)

        # Add layer boundaries
        for ax in [ax1, ax2, ax3, ax4]:
            current_pos = 0.0
            for layer in self.layers:
                current_pos += layer.thickness
                if current_pos < self.total_thickness:
                    ax.axvline(x=current_pos, color='black', linestyle=':', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Band structure saved to: {save_path}")
        else:
            plt.show()

    def plot_transport_properties(self, save_path: str = None) -> None:
        """Plot transport properties"""

        transport = self.calculate_transport_properties()
        z_nm = self.band_structure['z'] * 1e9

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Mobility
        ax1.semilogy(z_nm, transport['electron_mobility'] * 1e4, 'b-', linewidth=2, label='Electron')
        ax1.semilogy(z_nm, transport['hole_mobility'] * 1e4, 'r-', linewidth=2, label='Hole')
        ax1.set_xlabel('Position (nm)')
        ax1.set_ylabel('Mobility (cm²/V·s)')
        ax1.set_title('Carrier Mobility')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Conductivity
        ax2.semilogy(z_nm, transport['conductivity'], 'g-', linewidth=2)
        ax2.set_xlabel('Position (nm)')
        ax2.set_ylabel('Conductivity (S/m)')
        ax2.set_title('Electrical Conductivity')
        ax2.grid(True, alpha=0.3)

        # 3. Resistivity
        ax3.semilogy(z_nm, transport['resistivity'], 'orange', linewidth=2)
        ax3.set_xlabel('Position (nm)')
        ax3.set_ylabel('Resistivity (Ω·m)')
        ax3.set_title('Electrical Resistivity')
        ax3.grid(True, alpha=0.3)

        # 4. Current density capability
        # Estimate maximum current density
        max_field = 1e5  # V/m (typical breakdown field)
        max_current_density = transport['conductivity'] * max_field / 1e4  # A/cm²

        ax4.semilogy(z_nm, max_current_density, 'red', linewidth=2)
        ax4.set_xlabel('Position (nm)')
        ax4.set_ylabel('Max Current Density (A/cm²)')
        ax4.set_title('Current Handling Capability')
        ax4.grid(True, alpha=0.3)

        # Add layer boundaries
        for ax in [ax1, ax2, ax3, ax4]:
            current_pos = 0.0
            for layer in self.layers:
                current_pos += layer.thickness
                if current_pos < self.total_thickness:
                    ax.axvline(x=current_pos, color='black', linestyle=':', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Transport properties saved to: {save_path}")
        else:
            plt.show()

    def generate_structure_report(self) -> str:
        """Generate comprehensive heterostructure analysis report"""

        # Ensure all calculations are done
        if self.band_structure is None:
            self.calculate_band_structure()
        if self.carrier_densities is None:
            self.calculate_carrier_densities()

        quantum_wells = self.calculate_quantum_wells()
        transport = self.calculate_transport_properties()

        report = []
        report.append("🔬 HETEROSTRUCTURE ANALYSIS REPORT")
        report.append("=" * 60)

        # Structure overview
        report.append(f"\n📐 STRUCTURE OVERVIEW:")
        report.append(f"   Total Thickness: {self.total_thickness:.1f} nm")
        report.append(f"   Number of Layers: {len(self.layers)}")
        report.append(f"   Temperature: {self.temperature:.0f} K")

        # Layer details
        report.append(f"\n🏗️  LAYER STRUCTURE:")
        current_pos = 0.0
        for i, layer in enumerate(self.layers):
            report.append(f"   Layer {i+1}: {layer.material.value}")
            report.append(f"      Position: {current_pos:.1f} - {current_pos + layer.thickness:.1f} nm")
            report.append(f"      Thickness: {layer.thickness:.1f} nm")
            if layer.composition > 0:
                report.append(f"      Composition: {layer.composition:.3f}")
            report.append(f"      Doping: {layer.doping_type}-type, {layer.doping_concentration:.2e} cm⁻³")
            current_pos += layer.thickness

        # Band structure analysis
        Ec = self.band_structure['conduction_band']
        Ev = self.band_structure['valence_band']

        report.append(f"\n⚡ BAND STRUCTURE:")
        report.append(f"   Conduction Band Range: {np.min(Ec):.3f} to {np.max(Ec):.3f} eV")
        report.append(f"   Valence Band Range: {np.min(Ev):.3f} to {np.max(Ev):.3f} eV")
        report.append(f"   Maximum Band Offset: {np.max(Ec) - np.min(Ec):.3f} eV")
        report.append(f"   Fermi Level: {self.carrier_densities['fermi_level']:.3f} eV")

        # Quantum wells
        if quantum_wells:
            report.append(f"\n🌊 QUANTUM WELLS:")
            for i, qw in enumerate(quantum_wells):
                report.append(f"   Well {i+1} ({qw['type']}):")
                report.append(f"      Position: {qw['position']:.1f} nm")
                report.append(f"      Width: {qw['width']:.1f} nm")
                report.append(f"      Depth: {qw['depth']:.3f} eV")
                report.append(f"      Material: {qw['material']}")
                report.append(f"      Energy Levels: {len(qw['energy_levels'])} confined states")
        else:
            report.append(f"\n🌊 QUANTUM WELLS: None detected")

        # Transport properties
        avg_electron_mobility = np.mean(transport['electron_mobility']) * 1e4  # cm²/V·s
        avg_hole_mobility = np.mean(transport['hole_mobility']) * 1e4
        avg_conductivity = np.mean(transport['conductivity'])

        report.append(f"\n🚀 TRANSPORT PROPERTIES:")
        report.append(f"   Average Electron Mobility: {avg_electron_mobility:.0f} cm²/V·s")
        report.append(f"   Average Hole Mobility: {avg_hole_mobility:.0f} cm²/V·s")
        report.append(f"   Average Conductivity: {avg_conductivity:.2e} S/m")

        # Carrier densities
        max_n = np.max(self.carrier_densities['electron_density'])
        max_p = np.max(self.carrier_densities['hole_density'])

        report.append(f"\n👥 CARRIER DENSITIES:")
        report.append(f"   Maximum Electron Density: {max_n:.2e} cm⁻³")
        report.append(f"   Maximum Hole Density: {max_p:.2e} cm⁻³")

        # Interface analysis
        report.append(f"\n🔗 INTERFACE ANALYSIS:")
        interface_count = len(self.layers) - 1
        report.append(f"   Number of Interfaces: {interface_count}")

        if interface_count > 0:
            max_band_discontinuity = 0.0
            current_pos = 0.0

            for i in range(len(self.layers) - 1):
                layer1 = self.layers[i]
                layer2 = self.layers[i + 1]

                # Get band parameters for both layers
                params1 = MaterialDatabase.get_band_parameters(layer1.material, layer1.composition)
                params2 = MaterialDatabase.get_band_parameters(layer2.material, layer2.composition)

                # Calculate band discontinuity
                delta_Ec = abs(params1.electron_affinity - params2.electron_affinity)
                max_band_discontinuity = max(max_band_discontinuity, delta_Ec)

                current_pos += layer1.thickness
                report.append(f"   Interface {i+1} ({layer1.material.value}/{layer2.material.value}):")
                report.append(f"      Position: {current_pos:.1f} nm")
                report.append(f"      ΔEc: {delta_Ec:.3f} eV")

            report.append(f"   Maximum Band Discontinuity: {max_band_discontinuity:.3f} eV")

        return "\n".join(report)
