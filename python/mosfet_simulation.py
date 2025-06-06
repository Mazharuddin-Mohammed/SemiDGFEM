#!/usr/bin/env python3
"""
Advanced MOSFET Simulation Framework
Comprehensive n-channel and p-channel MOSFET device physics

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

class MOSFETType(Enum):
    NMOS = "NMOS"
    PMOS = "PMOS"

class MaterialType(Enum):
    SILICON = "Si"
    SILICON_DIOXIDE = "SiO2"
    POLYSILICON = "PolySi"
    ALUMINUM = "Al"

@dataclass
class MaterialProperties:
    """Material properties for device simulation"""
    name: str
    bandgap: float  # eV
    electron_affinity: float  # eV
    dielectric_constant: float
    electron_mobility: float  # cm²/V·s
    hole_mobility: float  # cm²/V·s
    effective_mass_electron: float  # m0
    effective_mass_hole: float  # m0
    density: float  # g/cm³
    thermal_conductivity: float  # W/cm·K

@dataclass
class DeviceGeometry:
    """MOSFET device geometry parameters"""
    length: float  # Gate length (μm)
    width: float   # Gate width (μm)
    tox: float     # Oxide thickness (nm)
    xj: float      # Junction depth (μm)
    channel_length: float  # Effective channel length (μm)
    source_length: float   # Source region length (μm)
    drain_length: float    # Drain region length (μm)

@dataclass
class DopingProfile:
    """Doping profile specification"""
    substrate_doping: float  # cm⁻³
    source_drain_doping: float  # cm⁻³
    channel_doping: float  # cm⁻³
    gate_doping: float  # cm⁻³
    profile_type: str  # "uniform", "gaussian", "exponential"

class MaterialDatabase:
    """Comprehensive material properties database"""
    
    @staticmethod
    def get_silicon_properties(temperature: float = 300.0) -> MaterialProperties:
        """Get temperature-dependent silicon properties"""
        
        # Temperature dependence for mobility (Arora model)
        mu_n_300 = 1400  # cm²/V·s at 300K
        mu_p_300 = 450   # cm²/V·s at 300K
        
        # Temperature scaling
        temp_factor = (temperature / 300.0) ** (-2.2)
        mu_n = mu_n_300 * temp_factor
        mu_p = mu_p_300 * temp_factor
        
        # Bandgap temperature dependence (Varshni equation)
        Eg_0 = 1.166  # eV at 0K
        alpha = 4.73e-4  # eV/K
        beta = 636  # K
        bandgap = Eg_0 - (alpha * temperature**2) / (temperature + beta)
        
        return MaterialProperties(
            name="Silicon",
            bandgap=bandgap,
            electron_affinity=4.05,
            dielectric_constant=11.7,
            electron_mobility=mu_n,
            hole_mobility=mu_p,
            effective_mass_electron=0.26,
            effective_mass_hole=0.39,
            density=2.33,
            thermal_conductivity=1.5
        )
    
    @staticmethod
    def get_sio2_properties() -> MaterialProperties:
        """Get silicon dioxide properties"""
        return MaterialProperties(
            name="Silicon Dioxide",
            bandgap=9.0,
            electron_affinity=0.95,
            dielectric_constant=3.9,
            electron_mobility=0.0,  # Insulator
            hole_mobility=0.0,      # Insulator
            effective_mass_electron=0.5,
            effective_mass_hole=0.5,
            density=2.20,
            thermal_conductivity=0.014
        )
    
    @staticmethod
    def get_polysilicon_properties() -> MaterialProperties:
        """Get polysilicon properties"""
        return MaterialProperties(
            name="Polysilicon",
            bandgap=1.12,
            electron_affinity=4.05,
            dielectric_constant=11.7,
            electron_mobility=100,  # Reduced due to grain boundaries
            hole_mobility=50,       # Reduced due to grain boundaries
            effective_mass_electron=0.26,
            effective_mass_hole=0.39,
            density=2.33,
            thermal_conductivity=0.3
        )

class MOSFETDevice:
    """Advanced MOSFET device simulation"""
    
    def __init__(self, mosfet_type: MOSFETType, geometry: DeviceGeometry, 
                 doping: DopingProfile, temperature: float = 300.0):
        self.mosfet_type = mosfet_type
        self.geometry = geometry
        self.doping = doping
        self.temperature = temperature
        
        # Material properties
        self.materials = {
            MaterialType.SILICON: MaterialDatabase.get_silicon_properties(temperature),
            MaterialType.SILICON_DIOXIDE: MaterialDatabase.get_sio2_properties(),
            MaterialType.POLYSILICON: MaterialDatabase.get_polysilicon_properties()
        }
        
        # Physical constants
        self.q = 1.602176634e-19  # Elementary charge (C)
        self.k = 1.380649e-23     # Boltzmann constant (J/K)
        self.eps0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
        self.h = 6.62607015e-34   # Planck constant (J·s)
        self.me = 9.1093837015e-31  # Electron mass (kg)
        
        # Device parameters
        self.ni = self._calculate_intrinsic_density()
        self.vt = self.k * temperature / self.q  # Thermal voltage
        self.cox = self._calculate_oxide_capacitance()
        self.vth = self._calculate_threshold_voltage()
        
        # Mesh and solution arrays
        self.mesh = None
        self.potential = None
        self.electron_density = None
        self.hole_density = None
        self.electric_field = None
        
    def _calculate_intrinsic_density(self) -> float:
        """Calculate intrinsic carrier density"""
        si = self.materials[MaterialType.SILICON]
        
        # Effective density of states
        Nc = 2.8e19 * (self.temperature / 300.0) ** 1.5  # cm⁻³
        Nv = 1.04e19 * (self.temperature / 300.0) ** 1.5  # cm⁻³
        
        # Intrinsic density
        ni = np.sqrt(Nc * Nv) * np.exp(-si.bandgap / (2 * self.vt))
        return ni
    
    def _calculate_oxide_capacitance(self) -> float:
        """Calculate oxide capacitance per unit area"""
        sio2 = self.materials[MaterialType.SILICON_DIOXIDE]
        eps_ox = sio2.dielectric_constant * self.eps0
        tox_m = self.geometry.tox * 1e-9  # Convert nm to m
        
        cox = eps_ox / tox_m  # F/m²
        return cox
    
    def _calculate_threshold_voltage(self) -> float:
        """Calculate threshold voltage"""
        si = self.materials[MaterialType.SILICON]
        
        # Work function difference
        phi_ms = -0.6  # V (typical for n+ poly on p-substrate)
        if self.mosfet_type == MOSFETType.PMOS:
            phi_ms = 0.6  # V (typical for p+ poly on n-substrate)
        
        # Surface potential
        phi_f = self.vt * np.log(self.doping.substrate_doping / self.ni)
        if self.mosfet_type == MOSFETType.PMOS:
            phi_f = -phi_f
        
        # Depletion charge
        eps_si = si.dielectric_constant * self.eps0
        Qd = np.sqrt(2 * self.q * eps_si * self.doping.substrate_doping * abs(2 * phi_f))
        
        # Threshold voltage
        vth = phi_ms + 2 * phi_f + Qd / self.cox
        
        return vth
    
    def create_device_mesh(self, nx: int = 100, ny: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Create 2D device mesh"""
        
        # Device dimensions
        total_length = (self.geometry.source_length + self.geometry.channel_length + 
                       self.geometry.drain_length)
        total_height = 2.0 * self.geometry.xj  # Extend below junction
        
        # Create mesh
        x = np.linspace(0, total_length * 1e-6, nx)  # Convert μm to m
        y = np.linspace(0, total_height * 1e-6, ny)  # Convert μm to m
        
        X, Y = np.meshgrid(x, y)
        
        self.mesh = {'x': x, 'y': y, 'X': X, 'Y': Y, 'nx': nx, 'ny': ny}
        
        return X, Y
    
    def create_doping_profile(self) -> np.ndarray:
        """Create 2D doping profile"""
        
        if self.mesh is None:
            raise ValueError("Mesh must be created before doping profile")
        
        X, Y = self.mesh['X'], self.mesh['Y']
        nx, ny = self.mesh['nx'], self.mesh['ny']
        
        # Initialize with substrate doping
        doping = np.full((ny, nx), self.doping.substrate_doping)
        
        # Source and drain regions
        source_end = self.geometry.source_length * 1e-6
        channel_start = source_end
        channel_end = channel_start + self.geometry.channel_length * 1e-6
        drain_start = channel_end
        
        junction_depth = self.geometry.xj * 1e-6
        
        # Source region
        source_mask = (X <= source_end) & (Y <= junction_depth)
        doping[source_mask] = self.doping.source_drain_doping
        
        # Drain region
        drain_mask = (X >= drain_start) & (Y <= junction_depth)
        doping[drain_mask] = self.doping.source_drain_doping
        
        # Channel region (if different from substrate)
        if self.doping.channel_doping != self.doping.substrate_doping:
            channel_mask = ((X >= channel_start) & (X <= channel_end) & 
                           (Y <= junction_depth))
            doping[channel_mask] = self.doping.channel_doping
        
        # Apply MOSFET type sign convention
        if self.mosfet_type == MOSFETType.NMOS:
            # For NMOS: p-substrate (negative), n+ source/drain (positive)
            substrate_mask = doping == self.doping.substrate_doping
            doping[substrate_mask] *= -1  # p-type
            # Source/drain remain positive (n-type)
        else:
            # For PMOS: n-substrate (positive), p+ source/drain (negative)
            source_drain_mask = ((doping == self.doping.source_drain_doping) | 
                                (doping == self.doping.channel_doping))
            doping[source_drain_mask] *= -1  # p-type
            # Substrate remains positive (n-type)
        
        return doping
    
    def solve_poisson_equation(self, vgs: float, vds: float, vbs: float = 0.0) -> Dict[str, np.ndarray]:
        """Solve 2D Poisson equation for given bias conditions"""
        
        if self.mesh is None:
            self.create_device_mesh()
        
        # Create doping profile
        doping = self.create_doping_profile()
        
        # Initialize potential
        nx, ny = self.mesh['nx'], self.mesh['ny']
        potential = np.zeros((ny, nx))
        
        # Apply boundary conditions
        # Gate voltage (top boundary in channel region)
        channel_start_idx = int(self.geometry.source_length / 
                               (self.geometry.source_length + self.geometry.channel_length + 
                                self.geometry.drain_length) * nx)
        channel_end_idx = int((self.geometry.source_length + self.geometry.channel_length) / 
                             (self.geometry.source_length + self.geometry.channel_length + 
                              self.geometry.drain_length) * nx)
        
        # Source contact (left boundary)
        potential[:, 0] = vbs  # Source at reference + body bias
        
        # Drain contact (right boundary)
        potential[:, -1] = vds + vbs  # Drain voltage + body bias
        
        # Gate voltage (applied through oxide - simplified)
        gate_potential = vgs + vbs
        
        # Simplified Poisson solution using finite differences
        # This is a simplified implementation - in practice would use advanced solvers
        
        si = self.materials[MaterialType.SILICON]
        eps_si = si.dielectric_constant * self.eps0
        
        # Iterative solution
        for iteration in range(100):
            potential_old = potential.copy()
            
            for i in range(1, ny-1):
                for j in range(1, nx-1):
                    # Finite difference Laplacian
                    d2phi_dx2 = (potential[i, j+1] - 2*potential[i, j] + potential[i, j-1])
                    d2phi_dy2 = (potential[i+1, j] - 2*potential[i, j] + potential[i-1, j])
                    
                    # Charge density (simplified)
                    rho = self.q * doping[i, j]
                    
                    # Update potential
                    dx = self.mesh['x'][1] - self.mesh['x'][0]
                    dy = self.mesh['y'][1] - self.mesh['y'][0]
                    
                    potential[i, j] = 0.25 * (potential[i, j+1] + potential[i, j-1] + 
                                            potential[i+1, j] + potential[i-1, j] + 
                                            rho * dx * dy / eps_si)
            
            # Apply gate boundary condition (simplified)
            if iteration % 10 == 0:
                # Apply gate voltage in channel region
                for j in range(channel_start_idx, channel_end_idx):
                    potential[0, j] = gate_potential - self.vth  # Simplified gate coupling
            
            # Check convergence
            if np.max(np.abs(potential - potential_old)) < 1e-6:
                break
        
        # Calculate carrier densities
        electron_density = self.ni * np.exp((potential - self.vt * np.log(self.doping.substrate_doping / self.ni)) / self.vt)
        hole_density = self.ni * np.exp(-(potential - self.vt * np.log(self.doping.substrate_doping / self.ni)) / self.vt)
        
        # Calculate electric field
        ey, ex = np.gradient(-potential)
        electric_field = np.sqrt(ex**2 + ey**2)
        
        # Store results
        self.potential = potential
        self.electron_density = electron_density
        self.hole_density = hole_density
        self.electric_field = electric_field
        
        return {
            'potential': potential,
            'electron_density': electron_density,
            'hole_density': hole_density,
            'electric_field': electric_field,
            'doping': doping
        }
    
    def calculate_drain_current(self, vgs: float, vds: float, vbs: float = 0.0) -> float:
        """Calculate drain current using drift-diffusion model"""
        
        # Solve for potential distribution
        solution = self.solve_poisson_equation(vgs, vds, vbs)
        
        # Extract channel region
        channel_start_idx = int(self.geometry.source_length / 
                               (self.geometry.source_length + self.geometry.channel_length + 
                                self.geometry.drain_length) * self.mesh['nx'])
        channel_end_idx = int((self.geometry.source_length + self.geometry.channel_length) / 
                             (self.geometry.source_length + self.geometry.channel_length + 
                              self.geometry.drain_length) * self.mesh['nx'])
        
        # Calculate current density in channel
        si = self.materials[MaterialType.SILICON]
        
        if self.mosfet_type == MOSFETType.NMOS:
            mobility = si.electron_mobility * 1e-4  # Convert cm²/V·s to m²/V·s
            carrier_density = solution['electron_density']
        else:
            mobility = si.hole_mobility * 1e-4  # Convert cm²/V·s to m²/V·s
            carrier_density = solution['hole_density']
        
        # Current density J = q * μ * n * E
        ex, ey = np.gradient(-solution['potential'])
        current_density_x = self.q * mobility * carrier_density * ex
        
        # Integrate current over channel cross-section
        dy = self.mesh['y'][1] - self.mesh['y'][0]
        channel_current = 0.0
        
        for j in range(channel_start_idx, channel_end_idx):
            # Integrate over channel depth
            channel_depth_idx = int(self.geometry.xj * 1e-6 / dy)
            current_slice = np.sum(current_density_x[:channel_depth_idx, j]) * dy
            channel_current += current_slice
        
        # Multiply by device width
        total_current = channel_current * self.geometry.width * 1e-6  # Convert μm to m
        
        return abs(total_current)  # Return magnitude

    def calculate_iv_characteristics(self, vgs_range: np.ndarray, vds_range: np.ndarray,
                                   vbs: float = 0.0) -> Dict[str, np.ndarray]:
        """Calculate complete I-V characteristics"""

        print(f"🔬 Calculating I-V characteristics for {self.mosfet_type.value}")
        print(f"   VGS range: {vgs_range[0]:.1f}V to {vgs_range[-1]:.1f}V ({len(vgs_range)} points)")
        print(f"   VDS range: {vds_range[0]:.1f}V to {vds_range[-1]:.1f}V ({len(vds_range)} points)")

        # Initialize current arrays
        ids_matrix = np.zeros((len(vgs_range), len(vds_range)))

        # Calculate current for each bias point
        for i, vgs in enumerate(vgs_range):
            for j, vds in enumerate(vds_range):
                try:
                    ids = self.calculate_drain_current(vgs, vds, vbs)
                    ids_matrix[i, j] = ids
                except Exception as e:
                    print(f"   Warning: Failed at VGS={vgs:.1f}V, VDS={vds:.1f}V: {e}")
                    ids_matrix[i, j] = 0.0

            if i % max(1, len(vgs_range)//10) == 0:
                print(f"   Progress: {i/len(vgs_range)*100:.0f}%")

        return {
            'vgs_range': vgs_range,
            'vds_range': vds_range,
            'ids_matrix': ids_matrix,
            'vth': self.vth,
            'cox': self.cox
        }

    def extract_device_parameters(self, iv_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract key device parameters from I-V data"""

        vgs_range = iv_data['vgs_range']
        vds_range = iv_data['vds_range']
        ids_matrix = iv_data['ids_matrix']

        # Extract threshold voltage from transfer characteristics
        # Use linear region (low VDS)
        vds_linear = vds_range[1]  # Use second point to avoid VDS=0
        linear_idx = 1

        ids_linear = ids_matrix[:, linear_idx]

        # Find threshold voltage (extrapolation method)
        # Linear region: IDS ∝ (VGS - VTH)
        valid_points = ids_linear > np.max(ids_linear) * 0.1  # Above 10% of max
        if np.any(valid_points):
            vgs_valid = vgs_range[valid_points]
            ids_valid = ids_linear[valid_points]

            # Linear fit
            coeffs = np.polyfit(vgs_valid, ids_valid, 1)
            vth_extracted = -coeffs[1] / coeffs[0]  # x-intercept
        else:
            vth_extracted = self.vth

        # Extract transconductance (gm = dIDS/dVGS)
        gm_max = 0.0
        for j in range(len(vds_range)):
            if vds_range[j] > 0.1:  # Avoid very low VDS
                gm = np.gradient(ids_matrix[:, j], vgs_range)
                gm_max = max(gm_max, np.max(gm))

        # Extract output conductance (gds = dIDS/dVDS)
        gds_avg = 0.0
        count = 0
        for i in range(len(vgs_range)):
            if vgs_range[i] > vth_extracted + 0.5:  # Well above threshold
                gds = np.gradient(ids_matrix[i, :], vds_range)
                gds_avg += np.mean(gds[vds_range > 0.5])  # Average in saturation
                count += 1

        if count > 0:
            gds_avg /= count

        # Calculate mobility (simplified)
        # μ = gm * L / (W * COX * VDS)
        if gm_max > 0 and len(vds_range) > 1:
            mobility_extracted = (gm_max * self.geometry.channel_length * 1e-6 /
                                (self.geometry.width * 1e-6 * self.cox * vds_range[1]))
        else:
            si = self.materials[MaterialType.SILICON]
            mobility_extracted = (si.electron_mobility if self.mosfet_type == MOSFETType.NMOS
                                else si.hole_mobility) * 1e-4

        # Subthreshold slope
        # S = d(VGS)/d(log10(IDS)) in mV/decade
        subthreshold_slope = 0.0
        if len(vgs_range) > 10:
            # Find subthreshold region
            ids_log = np.log10(np.maximum(ids_linear, 1e-15))  # Avoid log(0)
            subthreshold_mask = (vgs_range < vth_extracted) & (ids_linear > 1e-12)

            if np.sum(subthreshold_mask) > 5:
                vgs_sub = vgs_range[subthreshold_mask]
                ids_log_sub = ids_log[subthreshold_mask]

                # Linear fit in subthreshold region
                coeffs_sub = np.polyfit(ids_log_sub, vgs_sub, 1)
                subthreshold_slope = coeffs_sub[0] * 1000  # Convert to mV/decade

        return {
            'threshold_voltage': vth_extracted,
            'transconductance': gm_max,
            'output_conductance': gds_avg,
            'mobility': mobility_extracted,
            'subthreshold_slope': subthreshold_slope,
            'oxide_capacitance': self.cox,
            'intrinsic_gain': gm_max / gds_avg if gds_avg > 0 else 0.0
        }

    def plot_device_characteristics(self, iv_data: Dict[str, np.ndarray],
                                  save_path: str = None) -> None:
        """Plot comprehensive device characteristics"""

        vgs_range = iv_data['vgs_range']
        vds_range = iv_data['vds_range']
        ids_matrix = iv_data['ids_matrix']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Output characteristics (IDS vs VDS for different VGS)
        colors = plt.cm.viridis(np.linspace(0, 1, len(vgs_range)))
        for i, (vgs, color) in enumerate(zip(vgs_range, colors)):
            if i % max(1, len(vgs_range)//5) == 0:  # Plot every 5th curve
                ax1.plot(vds_range, ids_matrix[i, :] * 1e6, color=color,
                        label=f'VGS = {vgs:.1f}V', linewidth=2)

        ax1.set_xlabel('VDS (V)')
        ax1.set_ylabel('IDS (μA)')
        ax1.set_title(f'{self.mosfet_type.value} Output Characteristics')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Transfer characteristics (IDS vs VGS for different VDS)
        vds_indices = [len(vds_range)//4, len(vds_range)//2, -1]  # Low, medium, high VDS
        for idx in vds_indices:
            vds = vds_range[idx]
            ax2.semilogy(vgs_range, np.maximum(ids_matrix[:, idx] * 1e6, 1e-3),
                        'o-', label=f'VDS = {vds:.1f}V', markersize=4)

        ax2.axvline(x=self.vth, color='red', linestyle='--', label=f'VTH = {self.vth:.2f}V')
        ax2.set_xlabel('VGS (V)')
        ax2.set_ylabel('IDS (μA)')
        ax2.set_title(f'{self.mosfet_type.value} Transfer Characteristics')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Transconductance (gm vs VGS)
        for idx in vds_indices:
            vds = vds_range[idx]
            gm = np.gradient(ids_matrix[:, idx], vgs_range) * 1e6  # Convert to μS
            ax3.plot(vgs_range, gm, 'o-', label=f'VDS = {vds:.1f}V', markersize=4)

        ax3.set_xlabel('VGS (V)')
        ax3.set_ylabel('gm (μS)')
        ax3.set_title('Transconductance')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Output conductance (gds vs VDS)
        vgs_indices = [len(vgs_range)//2, 3*len(vgs_range)//4, -1]  # Medium to high VGS
        for idx in vgs_indices:
            vgs = vgs_range[idx]
            if vgs > self.vth:  # Only plot above threshold
                gds = np.gradient(ids_matrix[idx, :], vds_range) * 1e6  # Convert to μS
                ax4.plot(vds_range, gds, 'o-', label=f'VGS = {vgs:.1f}V', markersize=4)

        ax4.set_xlabel('VDS (V)')
        ax4.set_ylabel('gds (μS)')
        ax4.set_title('Output Conductance')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Device characteristics saved to: {save_path}")
        else:
            plt.show()

    def plot_device_structure(self, solution: Dict[str, np.ndarray] = None,
                            save_path: str = None) -> None:
        """Plot device structure and potential distribution"""

        if solution is None:
            # Generate default solution
            solution = self.solve_poisson_equation(vgs=1.0, vds=1.0)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        X, Y = self.mesh['X'] * 1e6, self.mesh['Y'] * 1e6  # Convert to μm

        # 1. Doping profile
        im1 = ax1.contourf(X, Y, np.log10(np.abs(solution['doping'])), levels=20, cmap='RdBu')
        ax1.set_xlabel('X (μm)')
        ax1.set_ylabel('Y (μm)')
        ax1.set_title('Doping Profile (log10|N|)')
        plt.colorbar(im1, ax=ax1, label='log10(cm⁻³)')

        # 2. Potential distribution
        im2 = ax2.contourf(X, Y, solution['potential'], levels=20, cmap='viridis')
        ax2.set_xlabel('X (μm)')
        ax2.set_ylabel('Y (μm)')
        ax2.set_title('Potential Distribution')
        plt.colorbar(im2, ax=ax2, label='Potential (V)')

        # 3. Electron density
        im3 = ax3.contourf(X, Y, np.log10(solution['electron_density']), levels=20, cmap='plasma')
        ax3.set_xlabel('X (μm)')
        ax3.set_ylabel('Y (μm)')
        ax3.set_title('Electron Density (log10)')
        plt.colorbar(im3, ax=ax3, label='log10(cm⁻³)')

        # 4. Electric field
        im4 = ax4.contourf(X, Y, solution['electric_field'], levels=20, cmap='hot')
        ax4.set_xlabel('X (μm)')
        ax4.set_ylabel('Y (μm)')
        ax4.set_title('Electric Field Magnitude')
        plt.colorbar(im4, ax=ax4, label='E-field (V/m)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Device structure saved to: {save_path}")
        else:
            plt.show()

    def generate_device_report(self, iv_data: Dict[str, np.ndarray] = None) -> str:
        """Generate comprehensive device analysis report"""

        if iv_data is None:
            # Quick I-V calculation for report
            vgs_range = np.linspace(-1, 3, 21)
            vds_range = np.linspace(0, 3, 16)
            iv_data = self.calculate_iv_characteristics(vgs_range, vds_range)

        parameters = self.extract_device_parameters(iv_data)

        report = []
        report.append(f"🔬 {self.mosfet_type.value} DEVICE ANALYSIS REPORT")
        report.append("=" * 60)

        # Device geometry
        report.append(f"\n📐 DEVICE GEOMETRY:")
        report.append(f"   Gate Length: {self.geometry.length:.2f} μm")
        report.append(f"   Gate Width: {self.geometry.width:.2f} μm")
        report.append(f"   Oxide Thickness: {self.geometry.tox:.1f} nm")
        report.append(f"   Junction Depth: {self.geometry.xj:.2f} μm")
        report.append(f"   Channel Length: {self.geometry.channel_length:.2f} μm")

        # Doping profile
        report.append(f"\n🧬 DOPING PROFILE:")
        report.append(f"   Substrate: {self.doping.substrate_doping:.2e} cm⁻³")
        report.append(f"   Source/Drain: {self.doping.source_drain_doping:.2e} cm⁻³")
        report.append(f"   Channel: {self.doping.channel_doping:.2e} cm⁻³")

        # Material properties
        si = self.materials[MaterialType.SILICON]
        report.append(f"\n🔬 MATERIAL PROPERTIES (T={self.temperature:.0f}K):")
        report.append(f"   Bandgap: {si.bandgap:.3f} eV")
        report.append(f"   Electron Mobility: {si.electron_mobility:.0f} cm²/V·s")
        report.append(f"   Hole Mobility: {si.hole_mobility:.0f} cm²/V·s")
        report.append(f"   Intrinsic Density: {self.ni:.2e} cm⁻³")

        # Device parameters
        report.append(f"\n⚡ DEVICE PARAMETERS:")
        report.append(f"   Threshold Voltage: {parameters['threshold_voltage']:.3f} V")
        report.append(f"   Oxide Capacitance: {self.cox*1e4:.2f} μF/cm²")
        report.append(f"   Transconductance: {parameters['transconductance']*1e6:.1f} μS")
        report.append(f"   Output Conductance: {parameters['output_conductance']*1e6:.1f} μS")
        report.append(f"   Intrinsic Gain: {parameters['intrinsic_gain']:.1f}")
        report.append(f"   Mobility (extracted): {parameters['mobility']*1e4:.0f} cm²/V·s")
        report.append(f"   Subthreshold Slope: {parameters['subthreshold_slope']:.1f} mV/decade")

        # Performance metrics
        max_current = np.max(iv_data['ids_matrix']) * 1e6  # μA
        report.append(f"\n📊 PERFORMANCE METRICS:")
        report.append(f"   Maximum Current: {max_current:.1f} μA")
        report.append(f"   Current Density: {max_current/(self.geometry.width*self.geometry.length):.1f} μA/μm²")
        report.append(f"   On/Off Ratio: {self._calculate_on_off_ratio(iv_data):.1e}")

        return "\n".join(report)

    def _calculate_on_off_ratio(self, iv_data: Dict[str, np.ndarray]) -> float:
        """Calculate on/off current ratio"""

        vgs_range = iv_data['vgs_range']
        ids_matrix = iv_data['ids_matrix']

        # Find on current (high VGS)
        high_vgs_idx = -1  # Highest VGS
        on_current = np.max(ids_matrix[high_vgs_idx, :])

        # Find off current (low VGS, below threshold)
        low_vgs_indices = vgs_range < self.vth - 0.5
        if np.any(low_vgs_indices):
            off_current = np.min(ids_matrix[low_vgs_indices, :])
            off_current = max(off_current, 1e-15)  # Avoid division by zero
        else:
            off_current = 1e-12  # Default small current

        return on_current / off_current
