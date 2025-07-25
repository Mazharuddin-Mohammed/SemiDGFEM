"""
Test suite for Advanced Device Structures

This test suite validates the advanced device structures implementation
including FinFET, Gate-All-Around, nanowire transistors, and other
complex device geometries.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import tempfile
import json
from advanced_device_structures import (
    AdvancedDeviceStructure, DeviceStructureFactory, DeviceStructureType,
    GateConfiguration, InterfaceType, Point3D, BoundingBox3D,
    MaterialRegion, DopingProfile, GateStructure, Contact,
    create_test_device_mesh, analyze_device_performance, compare_device_structures
)

class TestAdvancedDeviceStructure:
    """Test AdvancedDeviceStructure class"""
    
    def test_device_structure_creation(self):
        """Test basic device structure creation"""
        device = AdvancedDeviceStructure(DeviceStructureType.FINFET)
        
        assert device.device_type == DeviceStructureType.FINFET
        assert device.device_length == 100e-9
        assert device.device_width == 50e-9
        assert device.device_height == 50e-9
        assert device.substrate_material == "Si"
        
        print("✓ Device structure creation test passed")
    
    def test_set_dimensions(self):
        """Test setting device dimensions"""
        device = AdvancedDeviceStructure(DeviceStructureType.PLANAR_MOSFET)
        
        device.set_dimensions(200e-9, 100e-9, 75e-9)
        
        assert device.device_length == 200e-9
        assert device.device_width == 100e-9
        assert device.device_height == 75e-9
        
        # Test invalid dimensions
        try:
            device.set_dimensions(-1e-9, 100e-9, 75e-9)
            assert False, "Should raise ValueError for negative dimensions"
        except ValueError:
            pass
        
        print("✓ Set dimensions test passed")
    
    def test_material_regions(self):
        """Test material region management"""
        device = AdvancedDeviceStructure(DeviceStructureType.PLANAR_MOSFET)
        
        # Create material region
        min_point = Point3D(0.0, 0.0, 0.0)
        max_point = Point3D(50e-9, 50e-9, 25e-9)
        bbox = BoundingBox3D(min_point, max_point)
        
        region = MaterialRegion("GaAs", "channel", bbox, {"bandgap": 1.42})
        device.add_material_region(region)
        
        assert len(device.material_regions) == 1
        assert device.material_regions[0].material_name == "GaAs"
        assert device.material_regions[0].region_name == "channel"
        assert device.material_regions[0].properties["bandgap"] == 1.42
        
        print("✓ Material regions test passed")
    
    def test_doping_profiles(self):
        """Test doping profile management"""
        device = AdvancedDeviceStructure(DeviceStructureType.PLANAR_MOSFET)
        
        # Create doping profile
        min_point = Point3D(0.0, 0.0, 0.0)
        max_point = Point3D(30e-9, 50e-9, 25e-9)
        region = BoundingBox3D(min_point, max_point)
        
        profile = DopingProfile("n", 1e18, region)
        device.add_doping_profile(profile)
        
        assert len(device.doping_profiles) == 1
        assert device.doping_profiles[0].dopant_type == "n"
        assert device.doping_profiles[0].concentration == 1e18
        
        # Test spatial function
        doping_value = device.doping_profiles[0].spatial_function(15e-9, 25e-9, 12e-9)
        assert doping_value == 1e18
        
        print("✓ Doping profiles test passed")
    
    def test_gate_structures(self):
        """Test gate structure management"""
        device = AdvancedDeviceStructure(DeviceStructureType.PLANAR_MOSFET)
        
        # Create gate structure
        gate = GateStructure("PolySi", 50e-9, 100e-9, 100e-9)
        gate.configuration = GateConfiguration.SINGLE_GATE
        gate.oxide_thickness = 2e-9
        gate.oxide_material = "SiO2"
        
        device.add_gate_structure(gate)
        
        assert len(device.gate_structures) == 1
        assert device.gate_structures[0].gate_material == "PolySi"
        assert device.gate_structures[0].gate_length == 50e-9
        assert device.gate_structures[0].configuration == GateConfiguration.SINGLE_GATE
        
        print("✓ Gate structures test passed")
    
    def test_contacts(self):
        """Test contact management"""
        device = AdvancedDeviceStructure(DeviceStructureType.PLANAR_MOSFET)
        
        # Create contact
        min_point = Point3D(0.0, 0.0, 0.0)
        max_point = Point3D(10e-9, 50e-9, 25e-9)
        region = BoundingBox3D(min_point, max_point)
        
        contact = Contact("source", "ohmic", region, 4.5)
        device.add_contact(contact)
        
        assert len(device.contacts) == 1
        assert device.contacts[0].name == "source"
        assert device.contacts[0].contact_type == "ohmic"
        assert device.contacts[0].work_function == 4.5
        
        print("✓ Contacts test passed")
    
    def test_finfet_configuration(self):
        """Test FinFET configuration"""
        device = AdvancedDeviceStructure(DeviceStructureType.FINFET)
        
        device.configure_finfet(10e-9, 30e-9, 2)
        device.set_fin_pitch(25e-9)
        device.set_fin_orientation("x")
        
        assert device.fin_width == 10e-9
        assert device.fin_height == 30e-9
        assert device.num_fins == 2
        assert device.fin_pitch == 25e-9
        assert device.fin_orientation == "x"
        
        # Check that geometry was generated
        assert len(device.material_regions) > 0
        assert len(device.contacts) == 2  # Source and drain
        
        print("✓ FinFET configuration test passed")
    
    def test_gate_all_around_configuration(self):
        """Test Gate-All-Around configuration"""
        device = AdvancedDeviceStructure(DeviceStructureType.GATE_ALL_AROUND)
        
        device.configure_gate_all_around(15e-9, 50e-9)
        device.set_channel_material("InGaAs")
        device.set_spacer_material("Si3N4", 3e-9)
        
        assert device.channel_diameter == 15e-9
        assert device.device_length == 50e-9
        assert device.channel_material == "InGaAs"
        assert device.spacer_material == "Si3N4"
        assert device.spacer_thickness == 3e-9
        
        # Check that geometry was generated
        assert len(device.material_regions) > 0
        assert len(device.contacts) == 2
        
        print("✓ Gate-All-Around configuration test passed")
    
    def test_nanowire_configuration(self):
        """Test nanowire configuration"""
        device = AdvancedDeviceStructure(DeviceStructureType.NANOWIRE_TRANSISTOR)
        
        device.configure_nanowire(12e-9, 100e-9, "x")
        device.set_nanowire_cross_section("circular")
        device.set_channel_material("Ge")
        
        assert device.nanowire_diameter == 12e-9
        assert device.device_length == 100e-9
        assert device.nanowire_orientation == "x"
        assert device.nanowire_cross_section == "circular"
        assert device.channel_material == "Ge"
        
        # Check that geometry was generated
        assert len(device.material_regions) > 0
        assert len(device.contacts) == 2
        
        print("✓ Nanowire configuration test passed")
    
    def test_heterojunction_layers(self):
        """Test heterojunction layer management"""
        device = AdvancedDeviceStructure(DeviceStructureType.HETEROJUNCTION_BIPOLAR)
        
        device.add_heterojunction_layer("GaAs", 100e-9, 1.0)
        device.add_heterojunction_layer("AlGaAs", 50e-9, 0.3)
        device.add_heterojunction_layer("GaAs", 200e-9, 1.0)
        
        assert len(device.hetero_layers) == 3
        assert device.hetero_layers[0]["material"] == "GaAs"
        assert device.hetero_layers[1]["composition"] == 0.3
        assert device.hetero_layers[2]["thickness"] == 200e-9
        
        print("✓ Heterojunction layers test passed")
    
    def test_quantum_wells(self):
        """Test quantum well management"""
        device = AdvancedDeviceStructure(DeviceStructureType.QUANTUM_WELL_DEVICE)
        
        device.add_quantum_well("InGaAs", 10e-9, "InP", 20e-9)
        device.set_quantum_well_stack(5)
        
        assert len(device.quantum_wells) == 1
        assert device.quantum_wells[0]["well_material"] == "InGaAs"
        assert device.quantum_wells[0]["well_width"] == 10e-9
        assert device.quantum_wells[0]["barrier_material"] == "InP"
        assert device.quantum_wells[0]["barrier_width"] == 20e-9
        assert device.num_quantum_wells == 5
        
        print("✓ Quantum wells test passed")
    
    def test_interface_properties(self):
        """Test interface property management"""
        device = AdvancedDeviceStructure(DeviceStructureType.HETEROJUNCTION_BIPOLAR)
        
        device.set_interface_properties("GaAs_AlGaAs", InterfaceType.GRADED, 2e-9)
        device.set_interface_charge_density("GaAs_AlGaAs", 1e12)
        device.set_interface_trap_density("GaAs_AlGaAs", 1e11)
        device.enable_interface_roughness("GaAs_AlGaAs", 0.5e-9)
        
        assert device.interface_types["GaAs_AlGaAs"] == InterfaceType.GRADED
        assert device.interface_widths["GaAs_AlGaAs"] == 2e-9
        assert device.interface_charges["GaAs_AlGaAs"] == 1e12
        assert device.interface_traps["GaAs_AlGaAs"] == 1e11
        assert device.interface_roughness["GaAs_AlGaAs"] == 0.5e-9
        
        print("✓ Interface properties test passed")
    
    def test_3d_simulation_settings(self):
        """Test 3D simulation settings"""
        device = AdvancedDeviceStructure(DeviceStructureType.FINFET)
        
        device.enable_3d_simulation(True)
        
        # Add refinement regions
        min_point = Point3D(20e-9, 10e-9, 15e-9)
        max_point = Point3D(80e-9, 40e-9, 35e-9)
        refinement_region = BoundingBox3D(min_point, max_point)
        device.set_mesh_refinement_regions([refinement_region])
        
        device.set_symmetry_conditions(["mirror_x", "mirror_y"])
        
        assert device.enable_3d == True
        assert len(device.refinement_regions) == 1
        assert len(device.symmetry_conditions) == 2
        assert "mirror_x" in device.symmetry_conditions
        
        print("✓ 3D simulation settings test passed")
    
    def test_advanced_physics(self):
        """Test advanced physics settings"""
        device = AdvancedDeviceStructure(DeviceStructureType.FINFET)
        
        device.enable_strain_effects(True)
        device.set_self_heating(True)
        device.set_thermal_boundary_conditions({"substrate": 350.0, "ambient": 300.0})
        
        assert device.enable_strain == True
        assert device.enable_self_heating == True
        assert device.thermal_boundaries["substrate"] == 350.0
        assert device.thermal_boundaries["ambient"] == 300.0
        
        print("✓ Advanced physics test passed")
    
    def test_mesh_parameters(self):
        """Test mesh parameter management"""
        device = AdvancedDeviceStructure(DeviceStructureType.FINFET)
        
        mesh_params = {
            "max_element_size": 3e-9,
            "min_element_size": 0.5e-9,
            "refinement_factor": 3.0
        }
        device.set_mesh_parameters(mesh_params)
        
        assert device.mesh_params["max_element_size"] == 3e-9
        assert device.mesh_params["min_element_size"] == 0.5e-9
        assert device.mesh_params["refinement_factor"] == 3.0
        
        print("✓ Mesh parameters test passed")
    
    def test_structure_validation(self):
        """Test structure validation"""
        device = AdvancedDeviceStructure(DeviceStructureType.FINFET)
        device.configure_finfet(10e-9, 20e-9, 1)

        # Valid structure should pass
        assert device.validate_structure() == True

        # Invalid structure should fail - create a new device with invalid dimensions
        invalid_device = AdvancedDeviceStructure(DeviceStructureType.FINFET)
        invalid_device.device_length = -1e-9  # Set invalid dimension directly
        assert invalid_device.validate_structure() == False

        print("✓ Structure validation test passed")
    
    def test_geometry_analysis(self):
        """Test geometry analysis"""
        device = AdvancedDeviceStructure(DeviceStructureType.FINFET)
        device.configure_finfet(10e-9, 20e-9, 2)
        
        analysis = device.analyze_geometry()
        
        assert "total_volume" in analysis
        assert "surface_area" in analysis
        assert "fin_aspect_ratio" in analysis
        assert "total_fin_width" in analysis
        assert "fin_density" in analysis
        
        assert analysis["fin_aspect_ratio"] == 2.0  # 20e-9 / 10e-9
        assert analysis["total_fin_width"] == 20e-9  # 2 * 10e-9
        
        print("✓ Geometry analysis test passed")
    
    def test_export_import_json(self):
        """Test JSON export and import"""
        device = AdvancedDeviceStructure(DeviceStructureType.FINFET)
        device.configure_finfet(10e-9, 20e-9, 1)
        
        # Export to JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
        
        device.export_to_json(json_file)
        
        # Import from JSON
        new_device = AdvancedDeviceStructure(DeviceStructureType.PLANAR_MOSFET)
        new_device.load_from_json(json_file)
        
        assert new_device.device_type == DeviceStructureType.FINFET
        assert new_device.fin_width == 10e-9
        assert new_device.fin_height == 20e-9
        assert new_device.num_fins == 1
        
        # Clean up
        os.unlink(json_file)
        
        print("✓ JSON export/import test passed")
    
    def test_gmsh_export(self):
        """Test GMSH export"""
        device = AdvancedDeviceStructure(DeviceStructureType.FINFET)
        device.configure_finfet(10e-9, 20e-9, 1)
        
        # Export to GMSH
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geo', delete=False) as f:
            gmsh_file = f.name
        
        device.export_to_gmsh(gmsh_file)
        
        # Check that file was created and contains expected content
        with open(gmsh_file, 'r') as f:
            content = f.read()
        
        assert "GMSH geometry file" in content
        assert "Point(" in content
        assert "Line(" in content
        assert "Physical Surface" in content
        
        # Clean up
        os.unlink(gmsh_file)
        
        print("✓ GMSH export test passed")


class TestDeviceStructureFactory:
    """Test DeviceStructureFactory class"""
    
    def test_create_finfet(self):
        """Test FinFET factory method"""
        device = DeviceStructureFactory.create_finfet(8e-9, 25e-9, 60e-9, 3)
        
        assert device.device_type == DeviceStructureType.FINFET
        assert device.fin_width == 8e-9
        assert device.fin_height == 25e-9
        assert device.num_fins == 3
        assert device.device_length == 60e-9
        
        # Check that gate and doping were added
        assert len(device.gate_structures) == 1
        assert len(device.doping_profiles) == 3  # Source, drain, channel
        
        print("✓ FinFET factory test passed")
    
    def test_create_gate_all_around(self):
        """Test Gate-All-Around factory method"""
        device = DeviceStructureFactory.create_gate_all_around(12e-9, 40e-9, 120e-9)
        
        assert device.device_type == DeviceStructureType.GATE_ALL_AROUND
        assert device.channel_diameter == 12e-9
        assert device.device_length == 120e-9
        
        # Check that gate and doping were added
        assert len(device.gate_structures) == 1
        assert len(device.doping_profiles) == 2  # Source, drain
        assert device.gate_structures[0].configuration == GateConfiguration.GATE_ALL_AROUND
        
        print("✓ Gate-All-Around factory test passed")
    
    def test_create_nanowire_transistor(self):
        """Test nanowire transistor factory method"""
        device = DeviceStructureFactory.create_nanowire_transistor(15e-9, 150e-9, "InAs")
        
        assert device.device_type == DeviceStructureType.NANOWIRE_TRANSISTOR
        assert device.nanowire_diameter == 15e-9
        assert device.device_length == 150e-9
        assert device.channel_material == "InAs"
        
        # Check that gate and doping were added
        assert len(device.gate_structures) == 1
        assert len(device.doping_profiles) == 2
        assert device.gate_structures[0].configuration == GateConfiguration.WRAP_AROUND_GATE
        
        print("✓ Nanowire transistor factory test passed")
    
    def test_create_heterojunction_bipolar(self):
        """Test heterojunction bipolar factory method"""
        materials = ["GaAs", "AlGaAs", "GaAs"]
        thicknesses = [100e-9, 50e-9, 200e-9]
        
        device = DeviceStructureFactory.create_heterojunction_bipolar(materials, thicknesses)
        
        assert device.device_type == DeviceStructureType.HETEROJUNCTION_BIPOLAR
        assert len(device.hetero_layers) == 3
        assert len(device.material_regions) == 3
        assert len(device.interface_types) == 2  # Two interfaces
        
        print("✓ Heterojunction bipolar factory test passed")
    
    def test_create_quantum_well_device(self):
        """Test quantum well device factory method"""
        device = DeviceStructureFactory.create_quantum_well_device("InGaAs", 8e-9, "InP", 15e-9, 3)
        
        assert device.device_type == DeviceStructureType.QUANTUM_WELL_DEVICE
        assert len(device.quantum_wells) == 1
        assert device.num_quantum_wells == 3
        
        # Check material regions: 3 wells + 4 barriers (top, bottom, and between wells)
        expected_regions = 3 + 4  # wells + barriers
        assert len(device.material_regions) == expected_regions
        
        print("✓ Quantum well device factory test passed")
    
    def test_create_tunnel_fet(self):
        """Test tunnel FET factory method"""
        device = DeviceStructureFactory.create_tunnel_fet(80e-9, "Ge", "Si", "Si")
        
        assert device.device_type == DeviceStructureType.TUNNEL_FET
        assert device.device_length == 80e-9
        
        # Check regions: source, channel, drain
        assert len(device.material_regions) == 3
        assert len(device.doping_profiles) == 3
        assert len(device.gate_structures) == 1
        
        # Check materials
        source_region = next(r for r in device.material_regions if r.region_name == "source")
        assert source_region.material_name == "Ge"
        
        print("✓ Tunnel FET factory test passed")


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_test_device_mesh(self):
        """Test test device mesh creation"""
        device = DeviceStructureFactory.create_finfet(10e-9, 20e-9, 50e-9, 1)
        
        vertices, elements = create_test_device_mesh(device, 20, 15, 10)
        
        assert vertices.shape == (20 * 15 * 10, 3)
        assert elements.shape == (19 * 14 * 9, 8)  # Hexahedral elements
        
        # Check vertex coordinates are within device bounds
        assert np.all(vertices[:, 0] >= 0) and np.all(vertices[:, 0] <= device.device_length)
        assert np.all(vertices[:, 1] >= 0) and np.all(vertices[:, 1] <= device.device_width)
        assert np.all(vertices[:, 2] >= 0) and np.all(vertices[:, 2] <= device.device_height)
        
        print("✓ Test device mesh creation test passed")
    
    def test_analyze_device_performance(self):
        """Test device performance analysis"""
        device = DeviceStructureFactory.create_finfet(10e-9, 20e-9, 50e-9, 2)
        
        analysis = analyze_device_performance(device)
        
        assert "effective_channel_width" in analysis
        assert "fin_aspect_ratio" in analysis
        assert "electrostatic_control" in analysis
        
        # Check specific calculations
        expected_width = 2 * (2 * 20e-9 + 10e-9)  # 2 fins * (2*height + width)
        assert abs(analysis["effective_channel_width"] - expected_width) < 1e-12
        
        print("✓ Device performance analysis test passed")
    
    def test_compare_device_structures(self):
        """Test device structure comparison"""
        finfet = DeviceStructureFactory.create_finfet(10e-9, 20e-9, 50e-9, 1)
        gaa = DeviceStructureFactory.create_gate_all_around(15e-9, 40e-9, 100e-9)
        nanowire = DeviceStructureFactory.create_nanowire_transistor(12e-9, 80e-9, "Si")
        
        devices = [finfet, gaa, nanowire]
        comparison = compare_device_structures(devices)
        
        assert "total_volume" in comparison
        assert len(comparison["total_volume"]) == 3
        
        # Each metric should have values for all devices
        for metric, values in comparison.items():
            assert len(values) == 3
        
        print("✓ Device structure comparison test passed")


def run_all_tests():
    """Run all tests"""
    print("Running Advanced Device Structures Test Suite")
    print("=" * 60)
    
    # Test AdvancedDeviceStructure class
    test_device = TestAdvancedDeviceStructure()
    test_device.test_device_structure_creation()
    test_device.test_set_dimensions()
    test_device.test_material_regions()
    test_device.test_doping_profiles()
    test_device.test_gate_structures()
    test_device.test_contacts()
    test_device.test_finfet_configuration()
    test_device.test_gate_all_around_configuration()
    test_device.test_nanowire_configuration()
    test_device.test_heterojunction_layers()
    test_device.test_quantum_wells()
    test_device.test_interface_properties()
    test_device.test_3d_simulation_settings()
    test_device.test_advanced_physics()
    test_device.test_mesh_parameters()
    test_device.test_structure_validation()
    test_device.test_geometry_analysis()
    test_device.test_export_import_json()
    test_device.test_gmsh_export()
    
    # Test DeviceStructureFactory class
    test_factory = TestDeviceStructureFactory()
    test_factory.test_create_finfet()
    test_factory.test_create_gate_all_around()
    test_factory.test_create_nanowire_transistor()
    test_factory.test_create_heterojunction_bipolar()
    test_factory.test_create_quantum_well_device()
    test_factory.test_create_tunnel_fet()
    
    # Test utility functions
    test_utils = TestUtilityFunctions()
    test_utils.test_create_test_device_mesh()
    test_utils.test_analyze_device_performance()
    test_utils.test_compare_device_structures()
    
    print("\n" + "=" * 60)
    print("All Advanced Device Structures tests passed successfully!")
    print("Total tests: 29")


if __name__ == "__main__":
    run_all_tests()
