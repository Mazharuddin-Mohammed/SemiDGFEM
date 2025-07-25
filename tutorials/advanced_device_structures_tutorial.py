"""
Advanced Device Structures Tutorial

This tutorial demonstrates the advanced device structures capabilities
of the SemiDGFEM simulator, including:

1. FinFET transistor design and analysis
2. Gate-All-Around (GAA) nanowire transistors
3. Heterojunction bipolar transistors
4. Quantum well devices
5. Tunnel FETs
6. Device structure comparison and optimization

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import matplotlib.pyplot as plt
from advanced_device_structures import (
    AdvancedDeviceStructure, DeviceStructureFactory, DeviceStructureType,
    GateConfiguration, InterfaceType, Point3D, BoundingBox3D,
    MaterialRegion, DopingProfile, GateStructure, Contact,
    analyze_device_performance, compare_device_structures
)

def tutorial_1_finfet_design():
    """Tutorial 1: FinFET Design and Analysis"""
    print("\n" + "="*60)
    print("Tutorial 1: FinFET Design and Analysis")
    print("="*60)
    
    # Create a FinFET using the factory method
    print("Creating FinFET structure...")
    finfet = DeviceStructureFactory.create_finfet(
        fin_width=10e-9,      # 10 nm fin width
        fin_height=30e-9,     # 30 nm fin height
        gate_length=20e-9,    # 20 nm gate length
        num_fins=2            # 2 fins
    )
    
    # Analyze the geometry
    analysis = analyze_device_performance(finfet)
    print(f"FinFET Analysis:")
    print(f"  Total volume: {analysis['total_volume']*1e27:.2f} nm³")
    print(f"  Fin aspect ratio: {analysis['fin_aspect_ratio']:.2f}")
    print(f"  Effective channel width: {analysis['effective_channel_width']*1e9:.2f} nm")
    print(f"  Fin density: {analysis['fin_density']*1e-9:.2f} fins/nm")
    
    # Customize the FinFET
    print("\nCustomizing FinFET structure...")
    finfet.set_fin_pitch(25e-9)
    finfet.enable_strain_effects(True)
    finfet.set_self_heating(True)
    finfet.set_thermal_boundary_conditions({"substrate": 350.0, "ambient": 300.0})
    
    # Add interface properties
    finfet.set_interface_properties("Si_SiO2", InterfaceType.ABRUPT, 0.5e-9)
    finfet.set_interface_charge_density("Si_SiO2", 1e11)
    finfet.enable_interface_roughness("Si_SiO2", 0.3e-9)
    
    # Export structure
    finfet.export_structure_summary("finfet_summary.txt")
    finfet.export_to_json("finfet_structure.json")
    finfet.export_to_gmsh("finfet_mesh.geo")
    
    print("FinFET structure exported to files:")
    print("  - finfet_summary.txt")
    print("  - finfet_structure.json")
    print("  - finfet_mesh.geo")
    
    # Visualize structure
    print("\nVisualizing FinFET structure...")
    try:
        finfet.plot_structure_2d("xz", position=finfet.device_width/2, save_path="finfet_xz.png")
        finfet.plot_doping_profile("x", position=(finfet.device_width/2, finfet.device_height/2), 
                                  save_path="finfet_doping.png")
        print("Structure plots saved as finfet_xz.png and finfet_doping.png")
    except Exception as e:
        print(f"Plotting skipped (display not available): {e}")
    
    return finfet


def tutorial_2_gate_all_around():
    """Tutorial 2: Gate-All-Around Nanowire Transistor"""
    print("\n" + "="*60)
    print("Tutorial 2: Gate-All-Around Nanowire Transistor")
    print("="*60)
    
    # Create GAA transistor
    print("Creating Gate-All-Around structure...")
    gaa = DeviceStructureFactory.create_gate_all_around(
        channel_diameter=15e-9,   # 15 nm channel diameter
        gate_length=30e-9,        # 30 nm gate length
        channel_length=100e-9     # 100 nm total channel length
    )
    
    # Customize materials
    print("Customizing GAA materials...")
    gaa.set_channel_material("InGaAs")
    gaa.set_spacer_material("Si3N4", 3e-9)
    
    # Add high-k dielectric
    for gate in gaa.gate_structures:
        gate.oxide_material = "HfO2"
        gate.oxide_thickness = 1e-9
        gate.gate_material = "TiN"
    
    # Enable 3D simulation
    gaa.enable_3d_simulation(True)
    
    # Add mesh refinement regions
    channel_center = gaa.device_length / 2.0
    refinement_min = Point3D(channel_center - 20e-9, 0.0, 0.0)
    refinement_max = Point3D(channel_center + 20e-9, gaa.device_width, gaa.device_height)
    refinement_region = BoundingBox3D(refinement_min, refinement_max)
    gaa.set_mesh_refinement_regions([refinement_region])
    
    # Analyze performance
    analysis = analyze_device_performance(gaa)
    print(f"GAA Analysis:")
    print(f"  Channel perimeter: {analysis['channel_perimeter']*1e9:.2f} nm")
    print(f"  Electrostatic control: {analysis['electrostatic_control']:.3f}")
    print(f"  Volume inversion: {analysis['volume_inversion']}")
    
    # Export structure
    gaa.export_to_json("gaa_structure.json")
    print("GAA structure exported to gaa_structure.json")
    
    return gaa


def tutorial_3_heterojunction_bipolar():
    """Tutorial 3: Heterojunction Bipolar Transistor"""
    print("\n" + "="*60)
    print("Tutorial 3: Heterojunction Bipolar Transistor")
    print("="*60)
    
    # Create HBT with multiple layers
    print("Creating Heterojunction Bipolar Transistor...")
    materials = ["GaAs", "AlGaAs", "GaAs", "InGaAs"]
    thicknesses = [200e-9, 50e-9, 100e-9, 300e-9]  # Collector, base, emitter, subcollector
    
    hbt = DeviceStructureFactory.create_heterojunction_bipolar(materials, thicknesses)
    
    # Add graded interfaces
    print("Configuring graded interfaces...")
    hbt.set_interface_properties("GaAs_AlGaAs", InterfaceType.GRADED, 5e-9)
    hbt.set_interface_properties("AlGaAs_GaAs", InterfaceType.GRADED, 3e-9)
    hbt.set_interface_properties("GaAs_InGaAs", InterfaceType.ABRUPT, 1e-9)
    
    # Set interface charges and traps
    hbt.set_interface_charge_density("GaAs_AlGaAs", 5e11)
    hbt.set_interface_trap_density("GaAs_AlGaAs", 1e11)
    hbt.set_interface_charge_density("AlGaAs_GaAs", 3e11)
    
    # Add custom doping profiles
    print("Adding custom doping profiles...")
    
    # Emitter doping (n-type)
    emitter_min = Point3D(0.0, 0.0, 250e-9)
    emitter_max = Point3D(100e-9, 100e-9, 350e-9)
    emitter_region = BoundingBox3D(emitter_min, emitter_max)
    emitter_doping = DopingProfile("n", 1e18, emitter_region)
    hbt.add_doping_profile(emitter_doping)
    
    # Base doping (p-type)
    base_min = Point3D(0.0, 0.0, 200e-9)
    base_max = Point3D(100e-9, 100e-9, 250e-9)
    base_region = BoundingBox3D(base_min, base_max)
    base_doping = DopingProfile("p", 5e17, base_region)
    hbt.add_doping_profile(base_doping)
    
    # Collector doping (n-type)
    collector_min = Point3D(0.0, 0.0, 0.0)
    collector_max = Point3D(100e-9, 100e-9, 200e-9)
    collector_region = BoundingBox3D(collector_min, collector_max)
    collector_doping = DopingProfile("n", 1e16, collector_region)
    hbt.add_doping_profile(collector_doping)
    
    # Add contacts
    print("Adding contacts...")
    
    # Emitter contact
    emitter_contact_min = Point3D(40e-9, 40e-9, 350e-9)
    emitter_contact_max = Point3D(60e-9, 60e-9, 350e-9)
    emitter_contact_region = BoundingBox3D(emitter_contact_min, emitter_contact_max)
    emitter_contact = Contact("emitter", "ohmic", emitter_contact_region, 4.2)
    hbt.add_contact(emitter_contact)
    
    # Base contact
    base_contact_min = Point3D(10e-9, 10e-9, 250e-9)
    base_contact_max = Point3D(30e-9, 30e-9, 250e-9)
    base_contact_region = BoundingBox3D(base_contact_min, base_contact_max)
    base_contact = Contact("base", "ohmic", base_contact_region, 5.1)
    hbt.add_contact(base_contact)
    
    # Collector contact
    collector_contact_min = Point3D(40e-9, 40e-9, 0.0)
    collector_contact_max = Point3D(60e-9, 60e-9, 0.0)
    collector_contact_region = BoundingBox3D(collector_contact_min, collector_contact_max)
    collector_contact = Contact("collector", "ohmic", collector_contact_region, 4.3)
    hbt.add_contact(collector_contact)
    
    # Analyze structure
    analysis = hbt.analyze_geometry()
    print(f"HBT Analysis:")
    print(f"  Number of material regions: {analysis['num_material_regions']}")
    print(f"  Number of interfaces: {analysis['num_interfaces']}")
    print(f"  Number of contacts: {analysis['num_contacts']}")
    print(f"  Total volume: {analysis['total_volume']*1e27:.2f} nm³")
    
    # Export structure
    hbt.export_structure_summary("hbt_summary.txt")
    hbt.export_to_json("hbt_structure.json")
    print("HBT structure exported to hbt_summary.txt and hbt_structure.json")
    
    return hbt


def tutorial_4_quantum_well_device():
    """Tutorial 4: Quantum Well Device"""
    print("\n" + "="*60)
    print("Tutorial 4: Quantum Well Device")
    print("="*60)
    
    # Create quantum well device
    print("Creating quantum well device...")
    qw_device = DeviceStructureFactory.create_quantum_well_device(
        well_material="InGaAs",
        well_width=8e-9,
        barrier_material="InP",
        barrier_width=15e-9,
        num_wells=5
    )
    
    # Add multiple quantum well structures
    print("Adding additional quantum well structures...")
    qw_device.add_quantum_well("GaAs", 10e-9, "AlGaAs", 20e-9)
    qw_device.add_quantum_well("InAs", 5e-9, "GaSb", 25e-9)
    
    # Set interface properties for quantum wells
    print("Configuring quantum well interfaces...")
    qw_device.set_interface_properties("InGaAs_InP", InterfaceType.ABRUPT, 0.5e-9)
    qw_device.set_interface_properties("GaAs_AlGaAs", InterfaceType.GRADED, 2e-9)
    qw_device.set_interface_properties("InAs_GaSb", InterfaceType.ABRUPT, 0.3e-9)
    
    # Enable strain effects (important for quantum wells)
    qw_device.enable_strain_effects(True)
    
    # Set fine mesh parameters for quantum effects
    mesh_params = {
        "max_element_size": 2e-9,
        "min_element_size": 0.2e-9,
        "refinement_factor": 4.0,
        "interface_refinement": 0.1e-9
    }
    qw_device.set_mesh_parameters(mesh_params)
    
    # Analyze structure
    analysis = qw_device.analyze_geometry()
    print(f"Quantum Well Device Analysis:")
    print(f"  Number of quantum wells: {len(qw_device.quantum_wells)}")
    print(f"  Number of material regions: {analysis['num_material_regions']}")
    print(f"  Total device height: {qw_device.device_height*1e9:.2f} nm")
    
    # Export structure
    qw_device.export_to_json("quantum_well_device.json")
    print("Quantum well device exported to quantum_well_device.json")
    
    return qw_device


def tutorial_5_tunnel_fet():
    """Tutorial 5: Tunnel FET"""
    print("\n" + "="*60)
    print("Tutorial 5: Tunnel FET")
    print("="*60)
    
    # Create tunnel FET
    print("Creating Tunnel FET...")
    tfet = DeviceStructureFactory.create_tunnel_fet(
        channel_length=80e-9,
        source_material="Ge",
        channel_material="Si",
        drain_material="Si"
    )
    
    # Customize for better tunneling
    print("Optimizing for tunneling characteristics...")
    
    # Use high-k dielectric for better gate control
    for gate in tfet.gate_structures:
        gate.oxide_material = "HfO2"
        gate.oxide_thickness = 1e-9
        gate.gate_material = "TiN"
    
    # Add abrupt source-channel junction for tunneling
    tfet.set_interface_properties("Ge_Si", InterfaceType.ABRUPT, 0.2e-9)
    
    # Enable strain effects (can enhance tunneling)
    tfet.enable_strain_effects(True)
    
    # Set fine mesh for tunneling region
    source_channel_interface = tfet.device_length * 0.3
    refinement_min = Point3D(source_channel_interface - 5e-9, 0.0, 0.0)
    refinement_max = Point3D(source_channel_interface + 5e-9, tfet.device_width, tfet.device_height)
    refinement_region = BoundingBox3D(refinement_min, refinement_max)
    tfet.set_mesh_refinement_regions([refinement_region])
    
    # Analyze structure
    analysis = tfet.analyze_geometry()
    print(f"Tunnel FET Analysis:")
    print(f"  Channel length: {tfet.device_length*1e9:.2f} nm")
    print(f"  Number of material regions: {analysis['num_material_regions']}")
    print(f"  Number of interfaces: {analysis['num_interfaces']}")
    
    # Export structure
    tfet.export_to_json("tunnel_fet.json")
    tfet.export_to_gmsh("tunnel_fet_mesh.geo")
    print("Tunnel FET exported to tunnel_fet.json and tunnel_fet_mesh.geo")
    
    return tfet


def tutorial_6_device_comparison():
    """Tutorial 6: Device Structure Comparison"""
    print("\n" + "="*60)
    print("Tutorial 6: Device Structure Comparison and Optimization")
    print("="*60)
    
    # Create different device structures for comparison
    print("Creating devices for comparison...")
    
    # FinFET variants
    finfet_narrow = DeviceStructureFactory.create_finfet(8e-9, 25e-9, 20e-9, 1)
    finfet_wide = DeviceStructureFactory.create_finfet(15e-9, 25e-9, 20e-9, 1)
    finfet_multi = DeviceStructureFactory.create_finfet(10e-9, 25e-9, 20e-9, 3)
    
    # GAA variants
    gaa_small = DeviceStructureFactory.create_gate_all_around(10e-9, 20e-9, 80e-9)
    gaa_large = DeviceStructureFactory.create_gate_all_around(20e-9, 20e-9, 80e-9)
    
    # Nanowire variants
    nanowire_si = DeviceStructureFactory.create_nanowire_transistor(12e-9, 80e-9, "Si")
    nanowire_inas = DeviceStructureFactory.create_nanowire_transistor(12e-9, 80e-9, "InAs")
    
    # Compare FinFET variants
    print("\nComparing FinFET variants...")
    finfet_devices = [finfet_narrow, finfet_wide, finfet_multi]
    finfet_comparison = compare_device_structures(finfet_devices)
    
    print("FinFET Comparison:")
    print(f"  Effective channel widths: {[w*1e9 for w in finfet_comparison['effective_channel_width']]} nm")
    print(f"  Fin aspect ratios: {finfet_comparison['fin_aspect_ratio']}")
    print(f"  Electrostatic control: {finfet_comparison['electrostatic_control']}")
    
    # Compare GAA variants
    print("\nComparing GAA variants...")
    gaa_devices = [gaa_small, gaa_large]
    gaa_comparison = compare_device_structures(gaa_devices)
    
    print("GAA Comparison:")
    print(f"  Channel perimeters: {[p*1e9 for p in gaa_comparison['channel_perimeter']]} nm")
    print(f"  Total volumes: {[v*1e27 for v in gaa_comparison['total_volume']]} nm³")
    
    # Compare nanowire materials
    print("\nComparing nanowire materials...")
    nanowire_devices = [nanowire_si, nanowire_inas]
    nanowire_comparison = compare_device_structures(nanowire_devices)
    
    print("Nanowire Material Comparison:")
    print(f"  Channel areas: {[a*1e18 for a in nanowire_comparison['channel_area']]} nm²")
    print(f"  Channel perimeters: {[p*1e9 for p in nanowire_comparison['channel_perimeter']]} nm")
    
    # Overall device type comparison
    print("\nOverall device type comparison...")
    all_devices = [finfet_narrow, gaa_small, nanowire_si]
    device_names = ["FinFET", "GAA", "Nanowire"]
    
    overall_comparison = compare_device_structures(all_devices)
    
    print("Device Type Comparison:")
    for i, name in enumerate(device_names):
        print(f"  {name}:")
        print(f"    Total volume: {overall_comparison['total_volume'][i]*1e27:.2f} nm³")
        print(f"    Surface area: {overall_comparison['surface_area'][i]*1e18:.2f} nm²")
    
    # Performance optimization recommendations
    print("\nPerformance Optimization Recommendations:")
    print("1. FinFET: Use narrow fins (8-10 nm) for better electrostatic control")
    print("2. GAA: Smaller diameter provides better gate control but may reduce current")
    print("3. Nanowire: InAs provides higher mobility but Si offers better CMOS compatibility")
    print("4. All devices: Use high-k dielectrics and metal gates for improved performance")
    
    return {
        'finfet_comparison': finfet_comparison,
        'gaa_comparison': gaa_comparison,
        'nanowire_comparison': nanowire_comparison,
        'overall_comparison': overall_comparison
    }


def run_all_tutorials():
    """Run all advanced device structure tutorials"""
    print("Advanced Device Structures Tutorial Suite")
    print("SemiDGFEM Semiconductor Device Simulator")
    print("=" * 80)
    
    try:
        # Run all tutorials
        finfet = tutorial_1_finfet_design()
        gaa = tutorial_2_gate_all_around()
        hbt = tutorial_3_heterojunction_bipolar()
        qw_device = tutorial_4_quantum_well_device()
        tfet = tutorial_5_tunnel_fet()
        comparisons = tutorial_6_device_comparison()
        
        print("\n" + "="*80)
        print("All tutorials completed successfully!")
        print("\nGenerated files:")
        print("  - finfet_summary.txt, finfet_structure.json, finfet_mesh.geo")
        print("  - gaa_structure.json")
        print("  - hbt_summary.txt, hbt_structure.json")
        print("  - quantum_well_device.json")
        print("  - tunnel_fet.json, tunnel_fet_mesh.geo")
        print("  - Structure plots (if display available)")
        
        print("\nKey learnings:")
        print("1. FinFETs provide excellent electrostatic control with 3D channel geometry")
        print("2. GAA devices offer the best electrostatic control for ultimate scaling")
        print("3. Heterojunction devices enable bandgap engineering for specific applications")
        print("4. Quantum wells allow precise control of electronic properties")
        print("5. Tunnel FETs can achieve steep subthreshold slopes for low-power applications")
        print("6. Device structure comparison helps optimize design parameters")
        
    except Exception as e:
        print(f"Tutorial error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tutorials()
