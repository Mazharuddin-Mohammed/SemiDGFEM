#!/usr/bin/env python3
"""
Complex Device Demonstration: MOSFET and Heterostructure Success
Comprehensive demonstration of advanced device physics capabilities

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

def demonstrate_heterostructure_capabilities():
    """Demonstrate comprehensive heterostructure simulation capabilities"""
    print("🔬 HETEROSTRUCTURE SIMULATION DEMONSTRATION")
    print("=" * 60)
    
    try:
        from heterostructure_simulation import (HeterostructureDevice, LayerStructure, 
                                              SemiconductorMaterial, MaterialDatabase)
        
        # Create advanced GaAs/AlGaAs heterostructure
        print("   Creating GaAs/AlGaAs HEMT structure...")
        layers = [
            # GaAs buffer
            LayerStructure(
                material=SemiconductorMaterial.GAAS,
                thickness=500.0,  # 500 nm buffer
                composition=0.0,
                doping_type="intrinsic",
                doping_concentration=1e14,
                position=0.0
            ),
            # GaAs quantum well
            LayerStructure(
                material=SemiconductorMaterial.GAAS,
                thickness=15.0,   # 15 nm quantum well
                composition=0.0,
                doping_type="intrinsic",
                doping_concentration=1e14,
                position=500.0
            ),
            # AlGaAs spacer
            LayerStructure(
                material=SemiconductorMaterial.ALGAS,
                thickness=5.0,    # 5 nm spacer
                composition=0.3,  # Al₀.₃Ga₀.₇As
                doping_type="intrinsic",
                doping_concentration=1e14,
                position=515.0
            ),
            # AlGaAs barrier (doped)
            LayerStructure(
                material=SemiconductorMaterial.ALGAS,
                thickness=30.0,   # 30 nm barrier
                composition=0.3,
                doping_type="n",
                doping_concentration=2e18,
                position=520.0
            ),
            # GaAs cap
            LayerStructure(
                material=SemiconductorMaterial.GAAS,
                thickness=10.0,   # 10 nm cap
                composition=0.0,
                doping_type="n",
                doping_concentration=1e19,
                position=550.0
            )
        ]
        
        # Create heterostructure device
        hetero = HeterostructureDevice(layers, temperature=300.0)
        
        print(f"   Structure created:")
        print(f"      Total thickness: {hetero.total_thickness:.1f} nm")
        print(f"      Number of layers: {len(hetero.layers)}")
        print(f"      Temperature: {hetero.temperature:.0f} K")
        
        # Create high-resolution mesh
        print("   Creating high-resolution mesh...")
        z = hetero.create_mesh(nz=1000)
        print(f"      Mesh points: {len(z)}")
        print(f"      Resolution: {(z[1]-z[0])*1e9:.2f} nm")
        
        # Calculate band structure
        print("   Calculating band structure...")
        start_time = time.time()
        band_structure = hetero.calculate_band_structure()
        band_time = time.time() - start_time
        
        print(f"      Calculation time: {band_time:.3f}s")
        print(f"      Band fields: {len(band_structure)} calculated")
        
        # Analyze band edges
        Ec = band_structure['conduction_band']
        Ev = band_structure['valence_band']
        print(f"      Conduction band range: {np.min(Ec):.3f} to {np.max(Ec):.3f} eV")
        print(f"      Valence band range: {np.min(Ev):.3f} to {np.max(Ev):.3f} eV")
        print(f"      Band offset: {np.max(Ec) - np.min(Ec):.3f} eV")
        
        # Calculate carrier densities
        print("   Calculating carrier densities...")
        start_time = time.time()
        carrier_densities = hetero.calculate_carrier_densities()
        carrier_time = time.time() - start_time
        
        print(f"      Calculation time: {carrier_time:.3f}s")
        print(f"      Fermi level: {carrier_densities['fermi_level']:.3f} eV")
        
        n = carrier_densities['electron_density']
        p = carrier_densities['hole_density']
        print(f"      Electron density: {np.min(n):.2e} to {np.max(n):.2e} cm⁻³")
        print(f"      Hole density: {np.min(p):.2e} to {np.max(p):.2e} cm⁻³")
        
        # Analyze quantum wells
        print("   Analyzing quantum confinement...")
        quantum_wells = hetero.calculate_quantum_wells()
        
        print(f"      Quantum wells detected: {len(quantum_wells)}")
        for i, qw in enumerate(quantum_wells):
            print(f"      Well {i+1}: {qw['type']} carrier well")
            print(f"         Position: {qw['position']:.1f} nm")
            print(f"         Width: {qw['width']:.1f} nm")
            print(f"         Depth: {qw['depth']:.3f} eV")
            print(f"         Material: {qw['material']}")
            print(f"         Confined states: {len(qw['energy_levels'])}")
            
            if qw['energy_levels']:
                print(f"         Ground state: {qw['energy_levels'][0]:.3f} eV")
        
        # Calculate transport properties
        print("   Calculating transport properties...")
        start_time = time.time()
        transport = hetero.calculate_transport_properties()
        transport_time = time.time() - start_time
        
        print(f"      Calculation time: {transport_time:.3f}s")
        
        # Analyze mobility
        mu_e = transport['electron_mobility'] * 1e4  # Convert to cm²/V·s
        mu_h = transport['hole_mobility'] * 1e4
        
        print(f"      Electron mobility range: {np.min(mu_e):.0f} to {np.max(mu_e):.0f} cm²/V·s")
        print(f"      Hole mobility range: {np.min(mu_h):.0f} to {np.max(mu_h):.0f} cm²/V·s")
        
        # Find high-mobility regions (quantum wells)
        high_mobility_mask = mu_e > np.mean(mu_e) * 2
        if np.any(high_mobility_mask):
            high_mobility_positions = z[high_mobility_mask] * 1e9
            print(f"      High-mobility regions: {np.min(high_mobility_positions):.1f} to {np.max(high_mobility_positions):.1f} nm")
        
        # Calculate sheet carrier density (2DEG)
        if quantum_wells:
            qw = quantum_wells[0]  # First quantum well
            qw_start = qw['boundaries'][0]
            qw_end = qw['boundaries'][1]
            
            # Find indices corresponding to quantum well
            qw_mask = (z * 1e9 >= qw_start) & (z * 1e9 <= qw_end)
            
            if np.any(qw_mask):
                # Integrate carrier density over quantum well
                dz = z[1] - z[0]  # m
                sheet_density = np.sum(n[qw_mask]) * dz * 1e-4  # Convert to cm⁻²
                
                print(f"      2DEG sheet density: {sheet_density:.2e} cm⁻²")
                
                # Calculate 2DEG mobility
                avg_mobility_2deg = np.mean(mu_e[qw_mask])
                print(f"      2DEG mobility: {avg_mobility_2deg:.0f} cm²/V·s")
        
        # Material analysis
        print("   Material interface analysis...")
        interface_count = len(layers) - 1
        print(f"      Number of interfaces: {interface_count}")
        
        max_discontinuity = 0.0
        for i in range(len(layers) - 1):
            layer1 = layers[i]
            layer2 = layers[i + 1]
            
            # Get band parameters
            params1 = MaterialDatabase.get_band_parameters(layer1.material, layer1.composition)
            params2 = MaterialDatabase.get_band_parameters(layer2.material, layer2.composition)
            
            # Calculate band discontinuity
            delta_Ec = abs(params1.electron_affinity - params2.electron_affinity)
            max_discontinuity = max(max_discontinuity, delta_Ec)
            
            print(f"      Interface {i+1}: {layer1.material.value}/{layer2.material.value}")
            print(f"         Band offset: {delta_Ec:.3f} eV")
        
        print(f"      Maximum band discontinuity: {max_discontinuity:.3f} eV")
        
        # Performance summary
        total_time = band_time + carrier_time + transport_time
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"      Total calculation time: {total_time:.3f}s")
        print(f"      Mesh points processed: {len(z)}")
        print(f"      Processing rate: {len(z)/total_time:.0f} points/second")
        
        # Generate comprehensive report
        print("   Generating structure report...")
        report = hetero.generate_structure_report()
        
        # Save report to file
        with open('heterostructure_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print("      Report saved to: heterostructure_analysis_report.txt")
        
        print("   ✅ Heterostructure demonstration completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Heterostructure demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_material_database():
    """Demonstrate comprehensive material database capabilities"""
    print("\n🔬 MATERIAL DATABASE DEMONSTRATION")
    print("=" * 60)
    
    try:
        from heterostructure_simulation import MaterialDatabase, SemiconductorMaterial
        
        print("   Comprehensive material property analysis...")
        
        # Test all implemented materials
        materials = [
            (SemiconductorMaterial.SI, 0.0, "Silicon"),
            (SemiconductorMaterial.GE, 0.0, "Germanium"),
            (SemiconductorMaterial.GAAS, 0.0, "Gallium Arsenide"),
            (SemiconductorMaterial.ALGAS, 0.3, "Al₀.₃Ga₀.₇As"),
            (SemiconductorMaterial.INAS, 0.0, "Indium Arsenide"),
            (SemiconductorMaterial.GAN, 0.0, "Gallium Nitride"),
            (SemiconductorMaterial.ALGAN, 0.25, "Al₀.₂₅Ga₀.₇₅N")
        ]
        
        print(f"   Testing {len(materials)} materials...")
        
        for material, composition, name in materials:
            try:
                # Get band parameters
                band_params = MaterialDatabase.get_band_parameters(material, composition, 300.0)
                mobility_params = MaterialDatabase.get_mobility_parameters(material, composition)
                
                print(f"\n   {name}:")
                print(f"      Bandgap: {band_params.bandgap:.3f} eV")
                print(f"      Electron affinity: {band_params.electron_affinity:.2f} eV")
                print(f"      Dielectric constant: {band_params.dielectric_constant:.1f}")
                print(f"      Electron mobility: {mobility_params.electron_mobility_300k:.0f} cm²/V·s")
                print(f"      Hole mobility: {mobility_params.hole_mobility_300k:.0f} cm²/V·s")
                print(f"      Lattice constant: {band_params.lattice_constant:.3f} Å")
                
            except Exception as e:
                print(f"      ❌ Failed to get properties for {name}: {e}")
        
        # Test temperature dependence
        print(f"\n   Temperature dependence analysis (Silicon):")
        temperatures = [77, 200, 300, 400, 500]  # K
        
        for T in temperatures:
            si_props = MaterialDatabase.get_band_parameters(SemiconductorMaterial.SI, 0.0, T)
            print(f"      T = {T:3d}K: Eg = {si_props.bandgap:.3f} eV")
        
        # Test composition dependence
        print(f"\n   Composition dependence analysis (AlGaAs):")
        compositions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        for x in compositions:
            algaas_props = MaterialDatabase.get_band_parameters(SemiconductorMaterial.ALGAS, x, 300.0)
            print(f"      x = {x:.1f}: Eg = {algaas_props.bandgap:.3f} eV, χ = {algaas_props.electron_affinity:.2f} eV")
        
        print("   ✅ Material database demonstration completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Material database demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_quantum_effects():
    """Demonstrate quantum confinement effects"""
    print("\n🔬 QUANTUM EFFECTS DEMONSTRATION")
    print("=" * 60)
    
    try:
        from heterostructure_simulation import HeterostructureDevice, LayerStructure, SemiconductorMaterial
        
        print("   Creating quantum well structures with different widths...")
        
        well_widths = [5.0, 10.0, 15.0, 20.0, 30.0]  # nm
        
        for width in well_widths:
            print(f"\n   Analyzing {width:.0f} nm quantum well:")
            
            # Create quantum well structure
            layers = [
                # Barrier
                LayerStructure(SemiconductorMaterial.ALGAS, 50.0, 0.3, "intrinsic", 1e15, 0.0),
                # Quantum well
                LayerStructure(SemiconductorMaterial.GAAS, width, 0.0, "intrinsic", 1e15, 50.0),
                # Barrier
                LayerStructure(SemiconductorMaterial.ALGAS, 50.0, 0.3, "intrinsic", 1e15, 50.0 + width)
            ]
            
            hetero = HeterostructureDevice(layers, temperature=300.0)
            hetero.create_mesh(nz=300)
            hetero.calculate_band_structure()
            hetero.calculate_carrier_densities()
            
            # Analyze quantum wells
            quantum_wells = hetero.calculate_quantum_wells()
            
            if quantum_wells:
                qw = quantum_wells[0]
                print(f"      Well detected: {qw['width']:.1f} nm wide")
                print(f"      Confinement depth: {qw['depth']:.3f} eV")
                print(f"      Confined energy levels: {len(qw['energy_levels'])}")
                
                if qw['energy_levels']:
                    print(f"      Ground state energy: {qw['energy_levels'][0]:.3f} eV")
                    
                    # Calculate confinement energy
                    band_structure = hetero.band_structure
                    Ec = band_structure['conduction_band']
                    well_bottom = np.min(Ec)
                    confinement_energy = qw['energy_levels'][0] - well_bottom
                    print(f"      Confinement energy: {confinement_energy*1000:.1f} meV")
            else:
                print("      No quantum well detected")
        
        print("   ✅ Quantum effects demonstration completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Quantum effects demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demonstration function"""
    print("🚀 COMPLEX DEVICE SIMULATION - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    demonstrations_passed = 0
    total_demonstrations = 3
    
    # Run all demonstrations
    if demonstrate_heterostructure_capabilities():
        demonstrations_passed += 1
    
    if demonstrate_material_database():
        demonstrations_passed += 1
    
    if demonstrate_quantum_effects():
        demonstrations_passed += 1
    
    # Print summary
    print(f"\n📊 DEMONSTRATION SUMMARY")
    print("=" * 40)
    print(f"Demonstrations completed: {demonstrations_passed}/{total_demonstrations}")
    print(f"Success rate: {demonstrations_passed/total_demonstrations*100:.1f}%")
    
    if demonstrations_passed == total_demonstrations:
        print("\n🎉 ALL COMPLEX DEVICE DEMONSTRATIONS SUCCESSFUL!")
        print("   Advanced heterostructure and quantum physics simulations working perfectly")
        print("   Material database with comprehensive semiconductor properties")
        print("   Quantum confinement effects and energy level calculations")
        print("   Transport properties and 2DEG analysis")
        print("   Multi-material interface analysis")
        return 0
    else:
        print(f"\n⚠️  {total_demonstrations - demonstrations_passed} DEMONSTRATIONS INCOMPLETE")
        return 1

if __name__ == "__main__":
    sys.exit(main())
