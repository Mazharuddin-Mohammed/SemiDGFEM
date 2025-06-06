#!/usr/bin/env python3
"""
Test Complex Devices: MOSFET and Heterostructure Simulations
Simplified test suite for complex device physics

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_mosfet_basic():
    """Test basic MOSFET functionality"""
    print("🔬 TESTING BASIC MOSFET FUNCTIONALITY")
    print("=" * 50)
    
    try:
        from mosfet_simulation import (MOSFETDevice, MOSFETType, DeviceGeometry, 
                                     DopingProfile)
        
        # Simple device geometry
        geometry = DeviceGeometry(
            length=0.5, width=10.0, tox=5.0, xj=0.2,
            channel_length=0.5, source_length=0.5, drain_length=0.5
        )
        
        # Simple doping profile
        doping = DopingProfile(
            substrate_doping=1e17, source_drain_doping=1e20,
            channel_doping=1e17, gate_doping=1e20, profile_type="uniform"
        )
        
        # Create NMOS device
        print("   Creating NMOS device...")
        nmos = MOSFETDevice(MOSFETType.NMOS, geometry, doping, temperature=300.0)
        
        print(f"   Threshold voltage: {nmos.vth:.3f} V")
        print(f"   Oxide capacitance: {nmos.cox*1e4:.2f} μF/cm²")
        print(f"   Thermal voltage: {nmos.vt:.4f} V")
        
        # Create mesh
        print("   Creating device mesh...")
        X, Y = nmos.create_device_mesh(nx=50, ny=25)
        print(f"   Mesh created: {X.shape}")
        
        # Create doping profile
        print("   Creating doping profile...")
        doping_2d = nmos.create_doping_profile()
        print(f"   Doping profile shape: {doping_2d.shape}")
        print(f"   Doping range: {np.min(doping_2d):.2e} to {np.max(doping_2d):.2e} cm⁻³")
        
        # Test single bias point
        print("   Testing single bias point...")
        solution = nmos.solve_poisson_equation(vgs=1.0, vds=0.5)
        print(f"   Solution fields: {list(solution.keys())}")
        
        # Test current calculation
        print("   Testing current calculation...")
        current = nmos.calculate_drain_current(vgs=1.5, vds=1.0)
        print(f"   Drain current: {current*1e6:.2f} μA")
        
        # Test I-V calculation (small range)
        print("   Testing I-V characteristics...")
        vgs_range = np.linspace(0, 2.0, 6)
        vds_range = np.linspace(0, 2.0, 6)
        
        iv_data = nmos.calculate_iv_characteristics(vgs_range, vds_range)
        print(f"   I-V matrix shape: {iv_data['ids_matrix'].shape}")
        print(f"   Max current: {np.max(iv_data['ids_matrix'])*1e6:.2f} μA")
        
        # Extract parameters
        print("   Extracting device parameters...")
        parameters = nmos.extract_device_parameters(iv_data)
        print(f"   Extracted VTH: {parameters['threshold_voltage']:.3f} V")
        print(f"   Transconductance: {parameters['transconductance']*1e6:.1f} μS")
        
        print("   ✅ MOSFET basic test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ MOSFET basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_heterostructure_basic():
    """Test basic heterostructure functionality"""
    print("\n🔬 TESTING BASIC HETEROSTRUCTURE FUNCTIONALITY")
    print("=" * 50)
    
    try:
        from heterostructure_simulation import (HeterostructureDevice, LayerStructure, 
                                              SemiconductorMaterial)
        
        # Simple two-layer structure
        layers = [
            LayerStructure(
                material=SemiconductorMaterial.GAAS,
                thickness=100.0,  # 100 nm
                composition=0.0,
                doping_type="intrinsic",
                doping_concentration=1e15,
                position=0.0
            ),
            LayerStructure(
                material=SemiconductorMaterial.ALGAS,
                thickness=50.0,   # 50 nm
                composition=0.3,  # Al₀.₃Ga₀.₇As
                doping_type="n",
                doping_concentration=1e18,
                position=100.0
            )
        ]
        
        print("   Creating heterostructure device...")
        hetero = HeterostructureDevice(layers, temperature=300.0)
        
        print(f"   Total thickness: {hetero.total_thickness:.1f} nm")
        print(f"   Number of layers: {len(hetero.layers)}")
        
        # Create mesh
        print("   Creating mesh...")
        z = hetero.create_mesh(nz=200)
        print(f"   Mesh points: {len(z)}")
        print(f"   Mesh range: {z[0]*1e9:.1f} to {z[-1]*1e9:.1f} nm")
        
        # Calculate band structure
        print("   Calculating band structure...")
        band_structure = hetero.calculate_band_structure()
        print(f"   Band structure fields: {list(band_structure.keys())}")
        
        Ec = band_structure['conduction_band']
        Ev = band_structure['valence_band']
        print(f"   Conduction band range: {np.min(Ec):.3f} to {np.max(Ec):.3f} eV")
        print(f"   Valence band range: {np.min(Ev):.3f} to {np.max(Ev):.3f} eV")
        
        # Calculate carrier densities
        print("   Calculating carrier densities...")
        carrier_densities = hetero.calculate_carrier_densities()
        print(f"   Carrier density fields: {list(carrier_densities.keys())}")
        
        n = carrier_densities['electron_density']
        p = carrier_densities['hole_density']
        print(f"   Electron density range: {np.min(n):.2e} to {np.max(n):.2e} cm⁻³")
        print(f"   Hole density range: {np.min(p):.2e} to {np.max(p):.2e} cm⁻³")
        print(f"   Fermi level: {carrier_densities['fermi_level']:.3f} eV")
        
        # Analyze quantum wells
        print("   Analyzing quantum wells...")
        quantum_wells = hetero.calculate_quantum_wells()
        print(f"   Quantum wells detected: {len(quantum_wells)}")
        
        for i, qw in enumerate(quantum_wells):
            print(f"   Well {i+1}: {qw['type']} at {qw['position']:.1f} nm, width {qw['width']:.1f} nm")
        
        # Calculate transport properties
        print("   Calculating transport properties...")
        transport = hetero.calculate_transport_properties()
        print(f"   Transport fields: {list(transport.keys())}")
        
        avg_mobility_e = np.mean(transport['electron_mobility']) * 1e4
        avg_mobility_h = np.mean(transport['hole_mobility']) * 1e4
        print(f"   Average electron mobility: {avg_mobility_e:.0f} cm²/V·s")
        print(f"   Average hole mobility: {avg_mobility_h:.0f} cm²/V·s")
        
        print("   ✅ Heterostructure basic test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Heterostructure basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_material_database():
    """Test material database functionality"""
    print("\n🔬 TESTING MATERIAL DATABASE")
    print("=" * 50)
    
    try:
        from mosfet_simulation import MaterialDatabase as MOSMaterialDB
        from heterostructure_simulation import MaterialDatabase as HeteroMaterialDB
        from heterostructure_simulation import SemiconductorMaterial
        
        # Test MOSFET materials
        print("   Testing MOSFET materials...")
        si_props = MOSMaterialDB.get_silicon_properties(300.0)
        print(f"   Silicon bandgap: {si_props.bandgap:.3f} eV")
        print(f"   Silicon mobility (e): {si_props.electron_mobility:.0f} cm²/V·s")
        
        sio2_props = MOSMaterialDB.get_sio2_properties()
        print(f"   SiO₂ bandgap: {sio2_props.bandgap:.1f} eV")
        print(f"   SiO₂ dielectric: {sio2_props.dielectric_constant:.1f}")
        
        # Test heterostructure materials
        print("   Testing heterostructure materials...")
        gaas_props = HeteroMaterialDB.get_band_parameters(SemiconductorMaterial.GAAS, 0.0, 300.0)
        print(f"   GaAs bandgap: {gaas_props.bandgap:.3f} eV")
        print(f"   GaAs electron affinity: {gaas_props.electron_affinity:.2f} eV")
        
        algaas_props = HeteroMaterialDB.get_band_parameters(SemiconductorMaterial.ALGAS, 0.3, 300.0)
        print(f"   Al₀.₃Ga₀.₇As bandgap: {algaas_props.bandgap:.3f} eV")
        
        gan_props = HeteroMaterialDB.get_band_parameters(SemiconductorMaterial.GAN, 0.0, 300.0)
        print(f"   GaN bandgap: {gan_props.bandgap:.2f} eV")
        
        algan_props = HeteroMaterialDB.get_band_parameters(SemiconductorMaterial.ALGAN, 0.25, 300.0)
        print(f"   Al₀.₂₅Ga₀.₇₅N bandgap: {algan_props.bandgap:.2f} eV")
        
        # Test mobility parameters
        print("   Testing mobility parameters...")
        gaas_mobility = HeteroMaterialDB.get_mobility_parameters(SemiconductorMaterial.GAAS, 0.0)
        print(f"   GaAs electron mobility: {gaas_mobility.electron_mobility_300k:.0f} cm²/V·s")
        
        algan_mobility = HeteroMaterialDB.get_mobility_parameters(SemiconductorMaterial.ALGAN, 0.25)
        print(f"   AlGaN electron mobility: {algan_mobility.electron_mobility_300k:.0f} cm²/V·s")
        
        print("   ✅ Material database test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Material database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_visualization():
    """Test device visualization capabilities"""
    print("\n🔬 TESTING DEVICE VISUALIZATION")
    print("=" * 50)
    
    try:
        # Test MOSFET visualization
        print("   Testing MOSFET visualization...")
        from mosfet_simulation import (MOSFETDevice, MOSFETType, DeviceGeometry, 
                                     DopingProfile)
        
        geometry = DeviceGeometry(
            length=0.5, width=10.0, tox=5.0, xj=0.2,
            channel_length=0.5, source_length=0.5, drain_length=0.5
        )
        
        doping = DopingProfile(
            substrate_doping=1e17, source_drain_doping=1e20,
            channel_doping=1e17, gate_doping=1e20, profile_type="uniform"
        )
        
        nmos = MOSFETDevice(MOSFETType.NMOS, geometry, doping)
        nmos.create_device_mesh(nx=40, ny=20)
        
        # Test structure plotting
        solution = nmos.solve_poisson_equation(vgs=1.0, vds=0.5)
        nmos.plot_device_structure(solution, 'test_nmos_structure.png')
        print("   MOSFET structure plot created")
        
        # Test I-V plotting
        vgs_range = np.linspace(0, 2.0, 5)
        vds_range = np.linspace(0, 2.0, 5)
        iv_data = nmos.calculate_iv_characteristics(vgs_range, vds_range)
        nmos.plot_device_characteristics(iv_data, 'test_nmos_iv.png')
        print("   MOSFET I-V plot created")
        
        # Test heterostructure visualization
        print("   Testing heterostructure visualization...")
        from heterostructure_simulation import (HeterostructureDevice, LayerStructure, 
                                              SemiconductorMaterial)
        
        layers = [
            LayerStructure(SemiconductorMaterial.GAAS, 100.0, 0.0, "intrinsic", 1e15, 0.0),
            LayerStructure(SemiconductorMaterial.ALGAS, 50.0, 0.3, "n", 1e18, 100.0)
        ]
        
        hetero = HeterostructureDevice(layers)
        hetero.create_mesh(nz=100)
        hetero.calculate_band_structure()
        hetero.calculate_carrier_densities()
        
        hetero.plot_band_structure('test_hetero_bands.png')
        print("   Heterostructure band plot created")
        
        hetero.plot_transport_properties('test_hetero_transport.png')
        print("   Heterostructure transport plot created")
        
        print("   ✅ Device visualization test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Device visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 COMPLEX DEVICE SIMULATION TEST SUITE")
    print("=" * 60)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests_passed = 0
    total_tests = 4
    
    # Run all tests
    if test_mosfet_basic():
        tests_passed += 1
    
    if test_heterostructure_basic():
        tests_passed += 1
    
    if test_material_database():
        tests_passed += 1
    
    if test_device_visualization():
        tests_passed += 1
    
    # Print summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {tests_passed/total_tests*100:.1f}%")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL COMPLEX DEVICE TESTS PASSED!")
        print("   MOSFET and heterostructure simulations are working correctly")
        return 0
    else:
        print(f"\n⚠️  {total_tests - tests_passed} TESTS FAILED")
        print("   Some complex device features may not be working")
        return 1

if __name__ == "__main__":
    sys.exit(main())
