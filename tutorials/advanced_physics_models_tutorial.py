"""
Advanced Physics Models Tutorial

This tutorial demonstrates the usage of advanced physics models in the SemiDGFEM simulator:
- Strain effects and mechanical stress
- Thermal transport and self-heating
- Piezoelectric effects
- Optical properties and photogeneration

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from advanced_physics_models import (
    StrainEffectsModel, ThermalTransportModel, PiezoelectricModel, OpticalModel,
    StrainConfig, ThermalConfig, PiezoelectricConfig, OpticalConfig,
    MaterialProperties, StrainTensor, PhysicalConstants
)

def tutorial_1_strain_effects():
    """Tutorial 1: Strain Effects in Semiconductor Devices"""
    print("\n" + "="*60)
    print("Tutorial 1: Strain Effects in Semiconductor Devices")
    print("="*60)
    
    # Create Silicon material and strain model
    material = MaterialProperties(name="Silicon")
    config = StrainConfig(
        enable_strain_effects=True,
        enable_piezoresistance=True,
        enable_band_modification=True
    )
    strain_model = StrainEffectsModel(config, material)
    
    print("1.1 Creating strain tensor from displacement fields...")
    
    # Simulate displacement fields (e.g., from mechanical stress)
    x = np.linspace(0, 100e-9, 50)  # 100 nm device
    displacement_x = 1e-12 * x**2   # Quadratic displacement (bending)
    displacement_y = 0.5e-12 * x    # Linear displacement
    displacement_z = np.zeros_like(x)
    
    # Calculate strain tensor
    strain = strain_model.calculate_strain_tensor(
        displacement_x, displacement_y, displacement_z, mesh_spacing=2e-9
    )
    
    print(f"   - Strain tensor calculated for {len(strain.exx)} points")
    print(f"   - Maximum exx strain: {np.max(strain.exx):.2e}")
    print(f"   - Maximum exy shear strain: {np.max(strain.exy):.2e}")
    
    print("\n1.2 Calculating band structure modifications...")
    
    # Calculate band structure changes due to strain
    band_mod = strain_model.calculate_band_modification(strain)
    
    print(f"   - Conduction band shift range: {np.min(band_mod.conduction_band_shift):.3f} to {np.max(band_mod.conduction_band_shift):.3f} eV")
    print(f"   - Heavy hole band shift range: {np.min(band_mod.valence_band_shift_heavy):.3f} to {np.max(band_mod.valence_band_shift_heavy):.3f} eV")
    print(f"   - Effective mass modification range: {np.min(band_mod.effective_mass_modification):.3f} to {np.max(band_mod.effective_mass_modification):.3f}")
    
    print("\n1.3 Calculating mobility modifications (piezoresistance)...")
    
    # Calculate mobility changes due to strain
    mobility_mod = strain_model.calculate_mobility_modification(strain)
    
    print(f"   - Electron mobility factor range: {np.min(mobility_mod.electron_mobility_factor):.3f} to {np.max(mobility_mod.electron_mobility_factor):.3f}")
    print(f"   - Hole mobility factor range: {np.min(mobility_mod.hole_mobility_factor):.3f} to {np.max(mobility_mod.hole_mobility_factor):.3f}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x*1e9, strain.exx*1e3, 'b-', label='εxx')
    plt.plot(x*1e9, strain.eyy*1e3, 'r-', label='εyy')
    plt.xlabel('Position (nm)')
    plt.ylabel('Strain (×10⁻³)')
    plt.title('Strain Components')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x*1e9, band_mod.conduction_band_shift*1e3, 'g-', label='CB shift')
    plt.plot(x*1e9, band_mod.valence_band_shift_heavy*1e3, 'm-', label='VB heavy shift')
    plt.xlabel('Position (nm)')
    plt.ylabel('Band Shift (meV)')
    plt.title('Band Structure Modifications')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x*1e9, mobility_mod.electron_mobility_factor, 'b-', label='Electron')
    plt.plot(x*1e9, mobility_mod.hole_mobility_factor, 'r-', label='Hole')
    plt.xlabel('Position (nm)')
    plt.ylabel('Mobility Factor')
    plt.title('Mobility Modifications')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(x*1e9, band_mod.effective_mass_modification, 'k-')
    plt.xlabel('Position (nm)')
    plt.ylabel('Effective Mass Factor')
    plt.title('Effective Mass Modifications')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('strain_effects_tutorial.png', dpi=150, bbox_inches='tight')
    print("\n   - Strain effects visualization saved as 'strain_effects_tutorial.png'")

def tutorial_2_thermal_transport():
    """Tutorial 2: Thermal Transport and Self-Heating"""
    print("\n" + "="*60)
    print("Tutorial 2: Thermal Transport and Self-Heating")
    print("="*60)
    
    # Create Silicon material and thermal model
    material = MaterialProperties(name="Silicon")
    config = ThermalConfig(
        enable_thermal_transport=True,
        enable_joule_heating=True,
        enable_thermal_coupling=True
    )
    thermal_model = ThermalTransportModel(config, material)
    
    print("2.1 Solving heat equation with Joule heating...")
    
    # Setup 1D thermal problem
    x = np.linspace(0, 10e-6, 100)  # 10 μm device
    initial_temp = np.full_like(x, 300.0)  # 300K initial temperature
    
    # Create heat generation profile (higher in the center)
    heat_generation = 1e8 * np.exp(-((x - 5e-6) / 2e-6)**2)  # Gaussian profile
    
    # Boundary conditions
    boundary_conditions = {'left': 300.0, 'right': 300.0}  # Fixed temperature at contacts
    
    # Solve heat equation
    final_temp = thermal_model.solve_heat_equation(
        initial_temp, heat_generation, boundary_conditions,
        mesh_spacing=x[1]-x[0], num_time_steps=50
    )
    
    print(f"   - Temperature solved for {len(final_temp)} points")
    print(f"   - Maximum temperature: {np.max(final_temp):.1f} K")
    print(f"   - Temperature rise: {np.max(final_temp) - 300.0:.1f} K")
    
    print("\n2.2 Calculating Joule heating from current flow...")
    
    # Simulate current density and electric field
    current_density_x = 1e6 * np.ones_like(x)  # 1 MA/m² uniform current
    current_density_y = np.zeros_like(x)
    electric_field_x = 1e3 * np.ones_like(x)   # 1 kV/m uniform field
    electric_field_y = np.zeros_like(x)
    
    joule_heating = thermal_model.calculate_joule_heating(
        current_density_x, current_density_y, electric_field_x, electric_field_y
    )
    
    print(f"   - Joule heating calculated: {np.mean(joule_heating):.2e} W/m³")
    
    print("\n2.3 Calculating thermal coupling effects...")
    
    # Calculate thermal coupling to electronic properties
    electron_density = np.full_like(x, 1e16)  # m⁻³
    hole_density = np.full_like(x, 1e15)      # m⁻³
    
    coupling = thermal_model.calculate_thermal_coupling(final_temp, electron_density, hole_density)
    
    print(f"   - Bandgap modification range: {np.min(coupling.bandgap_modification)*1e3:.2f} to {np.max(coupling.bandgap_modification)*1e3:.2f} meV")
    print(f"   - Mobility modification range: {np.min(coupling.mobility_modification):.3f} to {np.max(coupling.mobility_modification):.3f}")
    print(f"   - Thermal voltage range: {np.min(coupling.thermal_voltage)*1e3:.2f} to {np.max(coupling.thermal_voltage)*1e3:.2f} mV")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x*1e6, heat_generation*1e-8, 'r-', linewidth=2)
    plt.xlabel('Position (μm)')
    plt.ylabel('Heat Generation (×10⁸ W/m³)')
    plt.title('Heat Generation Profile')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x*1e6, final_temp, 'b-', linewidth=2)
    plt.axhline(y=300, color='k', linestyle='--', alpha=0.5, label='Initial temp')
    plt.xlabel('Position (μm)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x*1e6, coupling.bandgap_modification*1e3, 'g-', linewidth=2)
    plt.xlabel('Position (μm)')
    plt.ylabel('Bandgap Shift (meV)')
    plt.title('Thermal Bandgap Modification')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(x*1e6, coupling.mobility_modification, 'm-', linewidth=2)
    plt.xlabel('Position (μm)')
    plt.ylabel('Mobility Factor')
    plt.title('Thermal Mobility Modification')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('thermal_transport_tutorial.png', dpi=150, bbox_inches='tight')
    print("\n   - Thermal transport visualization saved as 'thermal_transport_tutorial.png'")

def tutorial_3_piezoelectric_effects():
    """Tutorial 3: Piezoelectric Effects in Compound Semiconductors"""
    print("\n" + "="*60)
    print("Tutorial 3: Piezoelectric Effects in Compound Semiconductors")
    print("="*60)
    
    # Create GaN material and piezoelectric model
    material = MaterialProperties(name="GaN")
    config = PiezoelectricConfig(
        enable_piezoelectric_effects=True,
        enable_spontaneous_polarization=True
    )
    piezo_model = PiezoelectricModel(config, material)
    
    print("3.1 Creating strain tensor for wurtzite structure...")
    
    # Create strain tensor for GaN heterostructure
    z = np.linspace(0, 20e-9, 40)  # 20 nm structure
    strain = StrainTensor()
    
    # Simulate lattice mismatch strain
    strain.exx = np.full_like(z, 0.02)    # 2% tensile strain
    strain.eyy = np.full_like(z, 0.02)    # 2% tensile strain  
    strain.ezz = np.full_like(z, -0.008)  # Poisson compression
    strain.exy = np.zeros_like(z)
    strain.exz = np.zeros_like(z)
    strain.eyz = np.zeros_like(z)
    
    print(f"   - Strain tensor created for {len(strain.exx)} points")
    print(f"   - In-plane strain: {strain.exx[0]:.3f}")
    print(f"   - Out-of-plane strain: {strain.ezz[0]:.3f}")
    
    print("\n3.2 Calculating polarization fields...")
    
    # Calculate polarization from strain
    polarization = piezo_model.calculate_polarization(strain)
    
    print(f"   - Spontaneous polarization: {config.spontaneous_polarization:.3f} C/m²")
    print(f"   - Piezoelectric Pz range: {np.min(polarization.Pz):.3f} to {np.max(polarization.Pz):.3f} C/m²")
    print(f"   - Piezoelectric Px range: {np.min(polarization.Px):.3f} to {np.max(polarization.Px):.3f} C/m²")
    
    print("\n3.3 Calculating bound charge density...")
    
    # Calculate bound charge from polarization divergence
    bound_charge = piezo_model.calculate_bound_charge_density(polarization, mesh_spacing=z[1]-z[0])
    
    print(f"   - Bound charge density range: {np.min(bound_charge):.2e} to {np.max(bound_charge):.2e} C/m³")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(z*1e9, strain.exx*100, 'b-', label='εxx')
    plt.plot(z*1e9, strain.eyy*100, 'r-', label='εyy')
    plt.plot(z*1e9, strain.ezz*100, 'g-', label='εzz')
    plt.xlabel('Position (nm)')
    plt.ylabel('Strain (%)')
    plt.title('Strain Components')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(z*1e9, polarization.Pz, 'b-', linewidth=2, label='Pz')
    plt.plot(z*1e9, polarization.Px, 'r-', linewidth=2, label='Px')
    plt.xlabel('Position (nm)')
    plt.ylabel('Polarization (C/m²)')
    plt.title('Polarization Fields')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(z*1e9, bound_charge*1e-6, 'g-', linewidth=2)
    plt.xlabel('Position (nm)')
    plt.ylabel('Bound Charge (×10⁶ C/m³)')
    plt.title('Bound Charge Density')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('piezoelectric_effects_tutorial.png', dpi=150, bbox_inches='tight')
    print("\n   - Piezoelectric effects visualization saved as 'piezoelectric_effects_tutorial.png'")

def tutorial_4_optical_properties():
    """Tutorial 4: Optical Properties and Photogeneration"""
    print("\n" + "="*60)
    print("Tutorial 4: Optical Properties and Photogeneration")
    print("="*60)
    
    # Create GaAs material (direct bandgap) and optical model
    material = MaterialProperties(name="GaAs", bandgap=1.42)
    config = OpticalConfig(
        enable_optical_generation=True,
        absorption_coefficient=1e4,  # cm⁻¹
        quantum_efficiency=0.9
    )
    optical_model = OpticalModel(config, material)
    
    print("4.1 Calculating optical generation profile...")
    
    # Setup optical absorption problem
    z = np.linspace(0, 5e-6, 50)  # 5 μm penetration depth
    photon_flux = np.full_like(z, 1e20)  # photons/m²/s
    
    # Calculate optical generation
    generation_rate = optical_model.calculate_optical_generation(photon_flux, z)
    
    print(f"   - Generation rate at surface: {generation_rate[0]:.2e} carriers/m³/s")
    print(f"   - Generation rate at 1 μm: {generation_rate[10]:.2e} carriers/m³/s")
    print(f"   - 1/e penetration depth: {1/(config.absorption_coefficient*1e2)*1e6:.2f} μm")
    
    print("\n4.2 Calculating radiative recombination...")
    
    # Simulate carrier densities
    electron_density = 1e16 * np.ones_like(z)  # m⁻³
    hole_density = 1e16 * np.ones_like(z)      # m⁻³
    
    recombination_rate = optical_model.calculate_radiative_recombination(electron_density, hole_density)
    
    print(f"   - Radiative recombination rate: {recombination_rate[0]:.2e} carriers/m³/s")
    
    print("\n4.3 Calculating absorption coefficient vs photon energy...")
    
    # Calculate absorption spectrum
    photon_energies = np.linspace(1.0, 2.5, 100)  # eV
    absorption_coeffs = [optical_model.calculate_absorption_coefficient(E) for E in photon_energies]
    absorption_coeffs = np.array(absorption_coeffs) * 1e-2  # Convert to cm⁻¹
    
    print(f"   - Absorption at 1.5 eV: {absorption_coeffs[50]:.2e} cm⁻¹")
    print(f"   - Absorption at 2.0 eV: {absorption_coeffs[80]:.2e} cm⁻¹")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.semilogy(z*1e6, generation_rate, 'b-', linewidth=2)
    plt.xlabel('Depth (μm)')
    plt.ylabel('Generation Rate (m⁻³s⁻¹)')
    plt.title('Optical Generation Profile')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(z*1e6, electron_density*1e-16, 'b-', label='Electrons')
    plt.plot(z*1e6, hole_density*1e-16, 'r-', label='Holes')
    plt.xlabel('Depth (μm)')
    plt.ylabel('Carrier Density (×10¹⁶ m⁻³)')
    plt.title('Carrier Densities')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.semilogy(photon_energies, absorption_coeffs, 'g-', linewidth=2)
    plt.axvline(x=material.bandgap, color='r', linestyle='--', label=f'Bandgap ({material.bandgap} eV)')
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel('Absorption Coefficient (cm⁻¹)')
    plt.title('Absorption Spectrum')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.semilogy(z*1e6, recombination_rate, 'r-', linewidth=2)
    plt.xlabel('Depth (μm)')
    plt.ylabel('Recombination Rate (m⁻³s⁻¹)')
    plt.title('Radiative Recombination')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optical_properties_tutorial.png', dpi=150, bbox_inches='tight')
    print("\n   - Optical properties visualization saved as 'optical_properties_tutorial.png'")

def run_all_tutorials():
    """Run all advanced physics models tutorials"""
    print("Advanced Physics Models Tutorial Suite")
    print("SemiDGFEM Semiconductor Device Simulator")
    print("Author: Dr. Mazharuddin Mohammed")
    
    try:
        tutorial_1_strain_effects()
        tutorial_2_thermal_transport()
        tutorial_3_piezoelectric_effects()
        tutorial_4_optical_properties()
        
        print("\n" + "="*60)
        print("✓ All tutorials completed successfully!")
        print("Generated visualization files:")
        print("  - strain_effects_tutorial.png")
        print("  - thermal_transport_tutorial.png")
        print("  - piezoelectric_effects_tutorial.png")
        print("  - optical_properties_tutorial.png")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Tutorial failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tutorials()
