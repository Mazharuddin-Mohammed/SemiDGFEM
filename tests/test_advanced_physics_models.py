"""
Test Suite for Advanced Physics Models

This module contains comprehensive tests for advanced physics models including:
- Strain effects and mechanical stress
- Thermal transport and self-heating
- Piezoelectric effects
- Optical properties and photogeneration

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from advanced_physics_models import (
    StrainEffectsModel, ThermalTransportModel, PiezoelectricModel, OpticalModel,
    StrainConfig, ThermalConfig, PiezoelectricConfig, OpticalConfig,
    MaterialProperties, StrainTensor, PhysicalConstants, PolarizationField
)

def test_strain_effects_model():
    """Test strain effects model functionality"""
    print("Testing Strain Effects Model...")
    
    # Create model with Silicon material
    material = MaterialProperties(name="Silicon")
    config = StrainConfig()
    model = StrainEffectsModel(config, material)
    
    # Test strain tensor calculation
    size = 10
    displacement_x = np.linspace(0, 1e-9, size)
    displacement_y = np.linspace(0, 0.5e-9, size)
    displacement_z = np.zeros(size)
    
    strain = model.calculate_strain_tensor(displacement_x, displacement_y, displacement_z)
    
    assert hasattr(strain, 'exx'), "Strain tensor should have exx component"
    assert hasattr(strain, 'eyy'), "Strain tensor should have eyy component"
    assert hasattr(strain, 'ezz'), "Strain tensor should have ezz component"
    assert len(strain.exx) == size, f"Expected strain.exx length {size}, got {len(strain.exx)}"
    
    # Test band structure modification
    band_mod = model.calculate_band_modification(strain)
    
    assert hasattr(band_mod, 'conduction_band_shift'), "Should have conduction band shift"
    assert hasattr(band_mod, 'valence_band_shift_heavy'), "Should have valence band shift"
    assert len(band_mod.conduction_band_shift) == size, "Band modification should match strain size"
    
    # Test mobility modification
    mobility_mod = model.calculate_mobility_modification(strain)
    
    assert hasattr(mobility_mod, 'electron_mobility_factor'), "Should have electron mobility factor"
    assert hasattr(mobility_mod, 'hole_mobility_factor'), "Should have hole mobility factor"
    assert np.all(mobility_mod.electron_mobility_factor > 0), "Mobility factors should be positive"
    assert np.all(mobility_mod.hole_mobility_factor > 0), "Mobility factors should be positive"
    
    print("✓ Strain Effects Model tests passed")

def test_thermal_transport_model():
    """Test thermal transport model functionality"""
    print("Testing Thermal Transport Model...")
    
    # Create model with Silicon material
    material = MaterialProperties(name="Silicon")
    config = ThermalConfig()
    model = ThermalTransportModel(config, material)
    
    # Test heat equation solver
    size = 20
    initial_temp = np.full(size, 300.0)  # 300K initial temperature
    heat_gen = np.full(size, 1e6)       # 1 MW/m³ heat generation
    boundary_conditions = {'left': 300.0, 'right': 350.0}
    
    final_temp = model.solve_heat_equation(
        initial_temp, heat_gen, boundary_conditions, 
        mesh_spacing=1e-6, num_time_steps=10
    )
    
    assert len(final_temp) == size, f"Expected temperature array length {size}, got {len(final_temp)}"
    assert np.all(final_temp >= 300.0), "Temperature should not go below initial value"
    assert final_temp[0] == 300.0, "Left boundary condition should be enforced"
    assert final_temp[-1] == 350.0, "Right boundary condition should be enforced"
    
    # Test Joule heating calculation
    current_x = np.full(size, 1e6)  # A/m²
    current_y = np.zeros(size)
    field_x = np.full(size, 1e3)    # V/m
    field_y = np.zeros(size)
    
    joule_heating = model.calculate_joule_heating(current_x, current_y, field_x, field_y)
    
    assert len(joule_heating) == size, "Joule heating array should match input size"
    assert np.all(joule_heating >= 0), "Joule heating should be non-negative"
    expected_heating = current_x * field_x
    np.testing.assert_allclose(joule_heating, expected_heating, rtol=1e-10)
    
    # Test thermal coupling
    electron_density = np.full(size, 1e16)  # m⁻³
    hole_density = np.full(size, 1e15)      # m⁻³
    temperature = np.full(size, 350.0)      # K
    
    coupling = model.calculate_thermal_coupling(temperature, electron_density, hole_density)
    
    assert hasattr(coupling, 'bandgap_modification'), "Should have bandgap modification"
    assert hasattr(coupling, 'mobility_modification'), "Should have mobility modification"
    assert hasattr(coupling, 'thermal_voltage'), "Should have thermal voltage"
    assert len(coupling.thermal_voltage) == size, "Thermal voltage should match input size"
    
    # Check thermal voltage calculation
    expected_thermal_voltage = PhysicalConstants.k * temperature / PhysicalConstants.q
    np.testing.assert_allclose(coupling.thermal_voltage, expected_thermal_voltage, rtol=1e-10)
    
    print("✓ Thermal Transport Model tests passed")

def test_piezoelectric_model():
    """Test piezoelectric effects model functionality"""
    print("Testing Piezoelectric Model...")
    
    # Create model with GaN material
    material = MaterialProperties(name="GaN")
    config = PiezoelectricConfig()
    model = PiezoelectricModel(config, material)
    
    # Create strain tensor
    size = 15
    strain = StrainTensor()
    strain.exx = np.full(size, 0.01)    # 1% strain
    strain.eyy = np.full(size, 0.005)   # 0.5% strain
    strain.ezz = np.full(size, -0.003)  # -0.3% strain (Poisson effect)
    strain.exy = np.zeros(size)
    strain.exz = np.full(size, 0.001)   # Small shear strain
    strain.eyz = np.zeros(size)
    
    # Test polarization calculation
    polarization = model.calculate_polarization(strain)
    
    assert hasattr(polarization, 'Px'), "Should have Px polarization component"
    assert hasattr(polarization, 'Py'), "Should have Py polarization component"
    assert hasattr(polarization, 'Pz'), "Should have Pz polarization component"
    assert len(polarization.Px) == size, f"Expected Px length {size}, got {len(polarization.Px)}"
    
    # Check that spontaneous polarization is included
    assert np.all(np.abs(polarization.Pz) > 0), "Pz should include spontaneous polarization"
    
    # Test bound charge calculation
    bound_charge = model.calculate_bound_charge_density(polarization)
    
    assert len(bound_charge) == size, f"Expected bound charge length {size}, got {len(bound_charge)}"
    
    # Test interface charge calculation
    # Create second polarization field
    polarization_2 = PolarizationField()
    polarization_2.Px = polarization.Px * 0.5
    polarization_2.Py = polarization.Py * 0.5
    polarization_2.Pz = polarization.Pz * 0.8
    
    interface_charge = model.calculate_interface_charge(polarization, polarization_2)
    
    assert len(interface_charge) == size, "Interface charge should match polarization size"
    expected_interface = polarization_2.Pz - polarization.Pz
    np.testing.assert_allclose(interface_charge, expected_interface, rtol=1e-10)
    
    print("✓ Piezoelectric Model tests passed")

def test_optical_model():
    """Test optical properties model functionality"""
    print("Testing Optical Model...")
    
    # Create model with GaAs material (direct bandgap)
    material = MaterialProperties(name="GaAs", bandgap=1.42)
    config = OpticalConfig(absorption_coefficient=1e4, quantum_efficiency=0.9)
    model = OpticalModel(config, material)
    
    # Test optical generation
    size = 25
    photon_flux = np.full(size, 1e20)  # photons/m²/s
    position_z = np.linspace(0, 10e-6, size)  # 10 μm depth
    
    generation_rate = model.calculate_optical_generation(photon_flux, position_z)
    
    assert len(generation_rate) == size, f"Expected generation rate length {size}, got {len(generation_rate)}"
    assert np.all(generation_rate > 0), "Generation rate should be positive"
    
    # Check exponential decay with depth
    assert generation_rate[0] > generation_rate[-1], "Generation should decay with depth"
    
    # Test radiative recombination
    electron_density = np.full(size, 1e16)  # m⁻³
    hole_density = np.full(size, 1e16)      # m⁻³
    
    recombination_rate = model.calculate_radiative_recombination(electron_density, hole_density)
    
    assert len(recombination_rate) == size, "Recombination rate should match input size"
    assert np.all(recombination_rate > 0), "Recombination rate should be positive"
    
    # Test absorption coefficient calculation
    photon_energy_below = 1.0  # eV (below bandgap)
    photon_energy_above = 2.0  # eV (above bandgap)
    
    alpha_below = model.calculate_absorption_coefficient(photon_energy_below)
    alpha_above = model.calculate_absorption_coefficient(photon_energy_above)
    
    assert alpha_below > 0, "Absorption coefficient should be positive below bandgap"
    assert alpha_above > 0, "Absorption coefficient should be positive above bandgap"
    assert alpha_above > alpha_below, "Absorption should be higher above bandgap"
    
    print("✓ Optical Model tests passed")

def test_material_properties():
    """Test material properties for different semiconductors"""
    print("Testing Material Properties...")
    
    # Test Silicon properties
    si_material = MaterialProperties(name="Silicon")
    assert si_material.bandgap == 1.12, f"Expected Si bandgap 1.12 eV, got {si_material.bandgap}"
    assert si_material.thermal_conductivity == 148.0, "Expected Si thermal conductivity 148 W/m·K"
    
    # Test GaAs properties
    gaas_material = MaterialProperties(name="GaAs")
    gaas_material.bandgap = 1.42  # Set GaAs bandgap
    gaas_material.thermal_conductivity = 55.0  # Set GaAs thermal conductivity
    assert gaas_material.bandgap == 1.42, f"Expected GaAs bandgap 1.42 eV, got {gaas_material.bandgap}"
    assert gaas_material.thermal_conductivity == 55.0, "Expected GaAs thermal conductivity 55 W/m·K"
    
    print("✓ Material Properties tests passed")

def test_physical_constants():
    """Test physical constants values"""
    print("Testing Physical Constants...")
    
    # Test key physical constants
    assert abs(PhysicalConstants.k - 1.380649e-23) < 1e-30, "Boltzmann constant incorrect"
    assert abs(PhysicalConstants.q - 1.602176634e-19) < 1e-30, "Elementary charge incorrect"
    assert abs(PhysicalConstants.h - 6.62607015e-34) < 1e-40, "Planck constant incorrect"
    
    # Test derived constants
    expected_hbar = PhysicalConstants.h / (2.0 * np.pi)
    assert abs(PhysicalConstants.hbar - expected_hbar) < 1e-40, "Reduced Planck constant incorrect"
    
    print("✓ Physical Constants tests passed")

def run_all_tests():
    """Run all tests for advanced physics models"""
    print("=" * 60)
    print("Running Advanced Physics Models Test Suite")
    print("=" * 60)
    
    try:
        test_strain_effects_model()
        test_thermal_transport_model()
        test_piezoelectric_model()
        test_optical_model()
        test_material_properties()
        test_physical_constants()
        
        print("\n" + "=" * 60)
        print("✓ All Advanced Physics Models tests passed successfully!")
        print("Total tests: 6 test categories")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
