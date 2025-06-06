#!/usr/bin/env python3
"""
Complex Device Examples: MOSFET and Heterostructure Simulations
Comprehensive examples demonstrating advanced device physics

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

def example_nmos_characterization():
    """Example: Complete NMOS transistor characterization"""
    print("🔬 NMOS TRANSISTOR CHARACTERIZATION EXAMPLE")
    print("=" * 60)
    
    try:
        from mosfet_simulation import (MOSFETDevice, MOSFETType, DeviceGeometry, 
                                     DopingProfile)
        
        # Define device geometry (typical 180nm technology)
        geometry = DeviceGeometry(
            length=0.18,      # 180 nm gate length
            width=10.0,       # 10 μm gate width
            tox=4.0,          # 4 nm oxide thickness
            xj=0.15,          # 150 nm junction depth
            channel_length=0.18,
            source_length=0.5,
            drain_length=0.5
        )
        
        # Define doping profile
        doping = DopingProfile(
            substrate_doping=1e17,      # p-substrate: 1×10¹⁷ cm⁻³
            source_drain_doping=1e20,   # n+ source/drain: 1×10²⁰ cm⁻³
            channel_doping=5e16,        # Channel: 5×10¹⁶ cm⁻³
            gate_doping=1e20,           # n+ polysilicon gate
            profile_type="uniform"
        )
        
        # Create NMOS device
        print("   Creating NMOS device...")
        nmos = MOSFETDevice(MOSFETType.NMOS, geometry, doping, temperature=300.0)
        
        print(f"   Threshold voltage: {nmos.vth:.3f} V")
        print(f"   Oxide capacitance: {nmos.cox*1e4:.2f} μF/cm²")
        print(f"   Intrinsic density: {nmos.ni:.2e} cm⁻³")
        
        # Create device mesh
        print("   Creating device mesh...")
        nmos.create_device_mesh(nx=80, ny=40)
        
        # Calculate I-V characteristics
        print("   Calculating I-V characteristics...")
        vgs_range = np.linspace(0, 3.0, 16)
        vds_range = np.linspace(0, 3.0, 21)
        
        start_time = time.time()
        iv_data = nmos.calculate_iv_characteristics(vgs_range, vds_range)
        calc_time = time.time() - start_time
        
        print(f"   I-V calculation completed in {calc_time:.2f}s")
        
        # Extract device parameters
        parameters = nmos.extract_device_parameters(iv_data)
        
        print(f"\n📊 EXTRACTED PARAMETERS:")
        print(f"   Threshold Voltage: {parameters['threshold_voltage']:.3f} V")
        print(f"   Transconductance: {parameters['transconductance']*1e6:.1f} μS")
        print(f"   Output Conductance: {parameters['output_conductance']*1e6:.1f} μS")
        print(f"   Intrinsic Gain: {parameters['intrinsic_gain']:.1f}")
        print(f"   Mobility: {parameters['mobility']*1e4:.0f} cm²/V·s")
        print(f"   Subthreshold Slope: {parameters['subthreshold_slope']:.1f} mV/decade")
        
        # Plot characteristics
        print("   Generating plots...")
        nmos.plot_device_characteristics(iv_data, 'nmos_characteristics.png')
        
        # Plot device structure
        solution = nmos.solve_poisson_equation(vgs=2.0, vds=1.5)
        nmos.plot_device_structure(solution, 'nmos_structure.png')
        
        # Generate report
        report = nmos.generate_device_report(iv_data)
        print(f"\n{report}")
        
        print("   ✅ NMOS characterization completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ NMOS characterization failed: {e}")
        return False

def example_pmos_characterization():
    """Example: Complete PMOS transistor characterization"""
    print("\n🔬 PMOS TRANSISTOR CHARACTERIZATION EXAMPLE")
    print("=" * 60)
    
    try:
        from mosfet_simulation import (MOSFETDevice, MOSFETType, DeviceGeometry, 
                                     DopingProfile)
        
        # Define device geometry (same as NMOS for comparison)
        geometry = DeviceGeometry(
            length=0.18,      # 180 nm gate length
            width=20.0,       # 20 μm gate width (wider for PMOS)
            tox=4.0,          # 4 nm oxide thickness
            xj=0.15,          # 150 nm junction depth
            channel_length=0.18,
            source_length=0.5,
            drain_length=0.5
        )
        
        # Define doping profile (opposite polarity)
        doping = DopingProfile(
            substrate_doping=1e17,      # n-substrate: 1×10¹⁷ cm⁻³
            source_drain_doping=1e20,   # p+ source/drain: 1×10²⁰ cm⁻³
            channel_doping=5e16,        # Channel: 5×10¹⁶ cm⁻³
            gate_doping=1e20,           # p+ polysilicon gate
            profile_type="uniform"
        )
        
        # Create PMOS device
        print("   Creating PMOS device...")
        pmos = MOSFETDevice(MOSFETType.PMOS, geometry, doping, temperature=300.0)
        
        print(f"   Threshold voltage: {pmos.vth:.3f} V")
        print(f"   Oxide capacitance: {pmos.cox*1e4:.2f} μF/cm²")
        
        # Create device mesh
        print("   Creating device mesh...")
        pmos.create_device_mesh(nx=80, ny=40)
        
        # Calculate I-V characteristics (negative voltages for PMOS)
        print("   Calculating I-V characteristics...")
        vgs_range = np.linspace(0, -3.0, 16)
        vds_range = np.linspace(0, -3.0, 21)
        
        start_time = time.time()
        iv_data = pmos.calculate_iv_characteristics(vgs_range, vds_range)
        calc_time = time.time() - start_time
        
        print(f"   I-V calculation completed in {calc_time:.2f}s")
        
        # Extract device parameters
        parameters = pmos.extract_device_parameters(iv_data)
        
        print(f"\n📊 EXTRACTED PARAMETERS:")
        print(f"   Threshold Voltage: {parameters['threshold_voltage']:.3f} V")
        print(f"   Transconductance: {parameters['transconductance']*1e6:.1f} μS")
        print(f"   Output Conductance: {parameters['output_conductance']*1e6:.1f} μS")
        print(f"   Intrinsic Gain: {parameters['intrinsic_gain']:.1f}")
        print(f"   Mobility: {parameters['mobility']*1e4:.0f} cm²/V·s")
        
        # Plot characteristics
        print("   Generating plots...")
        pmos.plot_device_characteristics(iv_data, 'pmos_characteristics.png')
        
        # Plot device structure
        solution = pmos.solve_poisson_equation(vgs=-2.0, vds=-1.5)
        pmos.plot_device_structure(solution, 'pmos_structure.png')
        
        print("   ✅ PMOS characterization completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ PMOS characterization failed: {e}")
        return False

def example_gaas_algaas_heterostructure():
    """Example: GaAs/AlGaAs heterostructure simulation"""
    print("\n🔬 GaAs/AlGaAs HETEROSTRUCTURE EXAMPLE")
    print("=" * 60)
    
    try:
        from heterostructure_simulation import (HeterostructureDevice, LayerStructure, 
                                              SemiconductorMaterial)
        
        # Define heterostructure layers (HEMT-like structure)
        layers = [
            # Buffer layer
            LayerStructure(
                material=SemiconductorMaterial.GAAS,
                thickness=500.0,  # 500 nm
                composition=0.0,
                doping_type="intrinsic",
                doping_concentration=1e14,
                position=0.0
            ),
            # Channel layer
            LayerStructure(
                material=SemiconductorMaterial.GAAS,
                thickness=20.0,   # 20 nm quantum well
                composition=0.0,
                doping_type="intrinsic",
                doping_concentration=1e14,
                position=500.0
            ),
            # Spacer layer
            LayerStructure(
                material=SemiconductorMaterial.ALGAS,
                thickness=5.0,    # 5 nm undoped spacer
                composition=0.3,  # Al₀.₃Ga₀.₇As
                doping_type="intrinsic",
                doping_concentration=1e14,
                position=520.0
            ),
            # Doped barrier
            LayerStructure(
                material=SemiconductorMaterial.ALGAS,
                thickness=25.0,   # 25 nm doped barrier
                composition=0.3,
                doping_type="n",
                doping_concentration=2e18,
                position=525.0
            ),
            # Cap layer
            LayerStructure(
                material=SemiconductorMaterial.GAAS,
                thickness=10.0,   # 10 nm cap
                composition=0.0,
                doping_type="n",
                doping_concentration=1e19,
                position=550.0
            )
        ]
        
        print("   Creating heterostructure device...")
        hetero = HeterostructureDevice(layers, temperature=300.0)
        
        print(f"   Total thickness: {hetero.total_thickness:.1f} nm")
        print(f"   Number of layers: {len(hetero.layers)}")
        
        # Create mesh and calculate band structure
        print("   Creating mesh and calculating band structure...")
        hetero.create_mesh(nz=1000)
        band_structure = hetero.calculate_band_structure()
        
        print("   Calculating carrier densities...")
        carrier_densities = hetero.calculate_carrier_densities()
        
        print("   Analyzing quantum wells...")
        quantum_wells = hetero.calculate_quantum_wells()
        
        print(f"\n🌊 QUANTUM WELLS DETECTED: {len(quantum_wells)}")
        for i, qw in enumerate(quantum_wells):
            print(f"   Well {i+1}:")
            print(f"      Type: {qw['type']}")
            print(f"      Position: {qw['position']:.1f} nm")
            print(f"      Width: {qw['width']:.1f} nm")
            print(f"      Depth: {qw['depth']:.3f} eV")
            print(f"      Energy levels: {len(qw['energy_levels'])}")
            print(f"      Material: {qw['material']}")
        
        # Calculate transport properties
        print("   Calculating transport properties...")
        transport = hetero.calculate_transport_properties()
        
        # Plot results
        print("   Generating plots...")
        hetero.plot_band_structure('gaas_algaas_bands.png')
        hetero.plot_transport_properties('gaas_algaas_transport.png')
        
        # Generate report
        report = hetero.generate_structure_report()
        print(f"\n{report}")
        
        print("   ✅ GaAs/AlGaAs heterostructure simulation completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ GaAs/AlGaAs heterostructure simulation failed: {e}")
        return False

def example_gan_algan_heterostructure():
    """Example: GaN/AlGaN heterostructure for high-power applications"""
    print("\n🔬 GaN/AlGaN HIGH-POWER HETEROSTRUCTURE EXAMPLE")
    print("=" * 60)
    
    try:
        from heterostructure_simulation import (HeterostructureDevice, LayerStructure, 
                                              SemiconductorMaterial)
        
        # Define GaN/AlGaN heterostructure (HEMT for high power)
        layers = [
            # GaN buffer
            LayerStructure(
                material=SemiconductorMaterial.GAN,
                thickness=2000.0,  # 2 μm buffer
                composition=0.0,
                doping_type="intrinsic",
                doping_concentration=1e15,
                position=0.0
            ),
            # GaN channel
            LayerStructure(
                material=SemiconductorMaterial.GAN,
                thickness=100.0,   # 100 nm channel
                composition=0.0,
                doping_type="intrinsic",
                doping_concentration=1e15,
                position=2000.0
            ),
            # AlGaN barrier
            LayerStructure(
                material=SemiconductorMaterial.ALGAN,
                thickness=25.0,    # 25 nm barrier
                composition=0.25,  # Al₀.₂₅Ga₀.₇₅N
                doping_type="intrinsic",
                doping_concentration=1e15,
                position=2100.0
            ),
            # GaN cap
            LayerStructure(
                material=SemiconductorMaterial.GAN,
                thickness=5.0,     # 5 nm cap
                composition=0.0,
                doping_type="n",
                doping_concentration=1e18,
                position=2125.0
            )
        ]
        
        print("   Creating GaN/AlGaN heterostructure...")
        gan_hetero = HeterostructureDevice(layers, temperature=300.0)
        
        print(f"   Total thickness: {gan_hetero.total_thickness:.1f} nm")
        
        # Calculate band structure
        print("   Calculating band structure...")
        band_structure = gan_hetero.calculate_band_structure()
        carrier_densities = gan_hetero.calculate_carrier_densities()
        
        # Analyze quantum wells
        quantum_wells = gan_hetero.calculate_quantum_wells()
        
        print(f"\n⚡ HIGH-POWER DEVICE ANALYSIS:")
        print(f"   Quantum wells: {len(quantum_wells)}")
        
        # Calculate 2DEG properties (if quantum well exists)
        if quantum_wells:
            qw = quantum_wells[0]  # First quantum well
            print(f"   2DEG channel width: {qw['width']:.1f} nm")
            print(f"   2DEG confinement energy: {qw['depth']:.3f} eV")
            print(f"   Energy levels: {len(qw['energy_levels'])}")
        
        # Transport properties
        transport = gan_hetero.calculate_transport_properties()
        max_conductivity = np.max(transport['conductivity'])
        
        print(f"   Maximum conductivity: {max_conductivity:.2e} S/m")
        print(f"   Breakdown field capability: >1 MV/cm (GaN)")
        
        # Plot results
        print("   Generating plots...")
        gan_hetero.plot_band_structure('gan_algan_bands.png')
        gan_hetero.plot_transport_properties('gan_algan_transport.png')
        
        # Generate report
        report = gan_hetero.generate_structure_report()
        print(f"\n{report}")
        
        print("   ✅ GaN/AlGaN heterostructure simulation completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ GaN/AlGaN heterostructure simulation failed: {e}")
        return False

def example_cmos_inverter_analysis():
    """Example: CMOS inverter analysis using NMOS and PMOS"""
    print("\n🔬 CMOS INVERTER ANALYSIS EXAMPLE")
    print("=" * 60)
    
    try:
        from mosfet_simulation import (MOSFETDevice, MOSFETType, DeviceGeometry, 
                                     DopingProfile)
        
        # Common geometry for both devices
        geometry = DeviceGeometry(
            length=0.18, width=10.0, tox=4.0, xj=0.15,
            channel_length=0.18, source_length=0.5, drain_length=0.5
        )
        
        # NMOS doping
        nmos_doping = DopingProfile(
            substrate_doping=1e17, source_drain_doping=1e20,
            channel_doping=5e16, gate_doping=1e20, profile_type="uniform"
        )
        
        # PMOS doping (opposite polarity)
        pmos_doping = DopingProfile(
            substrate_doping=1e17, source_drain_doping=1e20,
            channel_doping=5e16, gate_doping=1e20, profile_type="uniform"
        )
        
        print("   Creating CMOS devices...")
        nmos = MOSFETDevice(MOSFETType.NMOS, geometry, nmos_doping)
        pmos = MOSFETDevice(MOSFETType.PMOS, geometry, pmos_doping)
        
        print(f"   NMOS VTH: {nmos.vth:.3f} V")
        print(f"   PMOS VTH: {pmos.vth:.3f} V")
        
        # Calculate transfer characteristics for inverter
        print("   Calculating CMOS inverter transfer function...")
        
        vdd = 3.0  # Supply voltage
        vin_range = np.linspace(0, vdd, 31)
        vout_values = []
        
        for vin in vin_range:
            # For inverter: VGS_NMOS = VIN, VGS_PMOS = VIN - VDD
            vgs_nmos = vin
            vgs_pmos = vin - vdd
            
            # Assume VDS = VOUT for both (simplified)
            # In practice, would solve iteratively
            vout_guess = vdd / 2  # Initial guess
            
            # Calculate currents (simplified analysis)
            ids_nmos = nmos.calculate_drain_current(vgs_nmos, vout_guess)
            ids_pmos = pmos.calculate_drain_current(vgs_pmos, vout_guess - vdd)
            
            # For steady state: IDS_NMOS = IDS_PMOS
            # Simplified: use voltage divider approximation
            if vin < nmos.vth:
                vout = vdd  # NMOS off, PMOS on
            elif vin > vdd + pmos.vth:
                vout = 0.0  # NMOS on, PMOS off
            else:
                # Transition region - simplified calculation
                vout = vdd * (1 - (vin - nmos.vth) / (vdd - nmos.vth))
                vout = max(0, min(vdd, vout))
            
            vout_values.append(vout)
        
        # Plot inverter transfer function
        plt.figure(figsize=(10, 6))
        plt.plot(vin_range, vout_values, 'b-', linewidth=2, label='VOUT')
        plt.plot([0, vdd], [0, vdd], 'r--', alpha=0.5, label='VIN = VOUT')
        plt.xlabel('Input Voltage (V)')
        plt.ylabel('Output Voltage (V)')
        plt.title('CMOS Inverter Transfer Function')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('cmos_inverter_transfer.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate noise margins
        # Find switching threshold
        switching_threshold = vin_range[np.argmin(np.abs(np.array(vout_values) - vdd/2))]
        
        print(f"\n📊 CMOS INVERTER ANALYSIS:")
        print(f"   Supply voltage: {vdd:.1f} V")
        print(f"   Switching threshold: {switching_threshold:.3f} V")
        print(f"   Logic high: {max(vout_values):.3f} V")
        print(f"   Logic low: {min(vout_values):.3f} V")
        
        # Estimate noise margins (simplified)
        vil = switching_threshold * 0.8  # Input low threshold
        vih = switching_threshold * 1.2  # Input high threshold
        vol = min(vout_values)           # Output low
        voh = max(vout_values)           # Output high
        
        nm_low = vil - vol   # Low noise margin
        nm_high = voh - vih  # High noise margin
        
        print(f"   Noise margin low: {nm_low:.3f} V")
        print(f"   Noise margin high: {nm_high:.3f} V")
        
        print("   ✅ CMOS inverter analysis completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ CMOS inverter analysis failed: {e}")
        return False

def main():
    """Main function to run all complex device examples"""
    print("🚀 COMPLEX DEVICE EXAMPLES: MOSFET AND HETEROSTRUCTURE SIMULATIONS")
    print("=" * 80)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    examples_passed = 0
    total_examples = 5
    
    # Run all examples
    if example_nmos_characterization():
        examples_passed += 1
    
    if example_pmos_characterization():
        examples_passed += 1
    
    if example_gaas_algaas_heterostructure():
        examples_passed += 1
    
    if example_gan_algan_heterostructure():
        examples_passed += 1
    
    if example_cmos_inverter_analysis():
        examples_passed += 1
    
    # Print summary
    print(f"\n📊 EXAMPLES SUMMARY")
    print("=" * 30)
    print(f"Examples completed: {examples_passed}/{total_examples}")
    print(f"Success rate: {examples_passed/total_examples*100:.1f}%")
    
    if examples_passed == total_examples:
        print("\n🎉 ALL COMPLEX DEVICE EXAMPLES COMPLETED SUCCESSFULLY!")
        print("   MOSFET and heterostructure simulations are working correctly")
        return 0
    else:
        print(f"\n⚠️  {total_examples - examples_passed} EXAMPLES FAILED")
        print("   Some complex device features may not be working")
        return 1

if __name__ == "__main__":
    sys.exit(main())
