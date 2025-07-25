"""
Advanced Device Structures for SemiDGFEM

This module provides Python interface for advanced device structures including:
- Multi-gate transistors (FinFET, GAA)
- Nanowire transistors
- Heterojunction devices
- 3D device geometries
- Complex material interfaces

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass
import json
import os

class DeviceStructureType(Enum):
    """Device structure types"""
    PLANAR_MOSFET = "planar_mosfet"
    FINFET = "finfet"
    GATE_ALL_AROUND = "gate_all_around"
    NANOWIRE_TRANSISTOR = "nanowire_transistor"
    HETEROJUNCTION_BIPOLAR = "heterojunction_bipolar"
    QUANTUM_WELL_DEVICE = "quantum_well_device"
    TUNNEL_FET = "tunnel_fet"
    JUNCTIONLESS_TRANSISTOR = "junctionless_transistor"
    CARBON_NANOTUBE_FET = "carbon_nanotube_fet"
    GRAPHENE_FET = "graphene_fet"

class GateConfiguration(Enum):
    """Gate configuration types"""
    SINGLE_GATE = "single_gate"
    DOUBLE_GATE = "double_gate"
    TRI_GATE = "tri_gate"
    GATE_ALL_AROUND = "gate_all_around"
    WRAP_AROUND_GATE = "wrap_around_gate"

class InterfaceType(Enum):
    """Material interface types"""
    ABRUPT = "abrupt"
    GRADED = "graded"
    DELTA_DOPED = "delta_doped"
    SUPERLATTICE = "superlattice"
    QUANTUM_WELL = "quantum_well"
    BARRIER_LAYER = "barrier_layer"

@dataclass
class Point3D:
    """3D point representation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class BoundingBox3D:
    """3D bounding box"""
    min_point: Point3D
    max_point: Point3D
    
    def volume(self) -> float:
        """Calculate volume of bounding box"""
        dx = self.max_point.x - self.min_point.x
        dy = self.max_point.y - self.min_point.y
        dz = self.max_point.z - self.min_point.z
        return dx * dy * dz
    
    def center(self) -> Point3D:
        """Calculate center point"""
        return Point3D(
            (self.min_point.x + self.max_point.x) / 2.0,
            (self.min_point.y + self.max_point.y) / 2.0,
            (self.min_point.z + self.max_point.z) / 2.0
        )

@dataclass
class MaterialRegion:
    """Material region definition"""
    material_name: str
    region_name: str
    bounding_box: BoundingBox3D
    properties: Dict[str, float]
    interface_type: InterfaceType = InterfaceType.ABRUPT
    interface_width: float = 0.0

@dataclass
class DopingProfile:
    """Doping profile definition"""
    dopant_type: str  # "n" or "p"
    concentration: float  # cm^-3
    region: BoundingBox3D
    spatial_function: Optional[callable] = None
    
    def __post_init__(self):
        if self.spatial_function is None:
            # Default uniform doping
            self.spatial_function = lambda x, y, z: self.concentration

@dataclass
class GateStructure:
    """Gate structure definition"""
    gate_material: str
    gate_length: float
    gate_width: float
    gate_thickness: float
    oxide_thickness: float = 2e-9
    oxide_material: str = "SiO2"
    configuration: GateConfiguration = GateConfiguration.SINGLE_GATE
    gate_positions: List[Point3D] = None
    
    def __post_init__(self):
        if self.gate_positions is None:
            self.gate_positions = []

@dataclass
class Contact:
    """Contact definition"""
    name: str
    contact_type: str  # "ohmic", "schottky"
    region: BoundingBox3D
    work_function: float = 4.5
    contact_resistance: float = 1e-8

class AdvancedDeviceStructure:
    """Advanced device structure builder"""
    
    def __init__(self, device_type: DeviceStructureType):
        self.device_type = device_type
        
        # Device dimensions
        self.device_length = 100e-9
        self.device_width = 50e-9
        self.device_height = 50e-9
        self.substrate_material = "Si"
        
        # Structure components
        self.material_regions: List[MaterialRegion] = []
        self.doping_profiles: List[DopingProfile] = []
        self.gate_structures: List[GateStructure] = []
        self.contacts: List[Contact] = []
        
        # FinFET parameters
        self.fin_width = 10e-9
        self.fin_height = 20e-9
        self.num_fins = 1
        self.fin_pitch = 30e-9
        self.fin_orientation = "x"
        
        # GAA parameters
        self.channel_diameter = 10e-9
        self.channel_material = "Si"
        self.spacer_material = "SiO2"
        self.spacer_thickness = 5e-9
        
        # Nanowire parameters
        self.nanowire_diameter = 10e-9
        self.nanowire_orientation = "x"
        self.nanowire_cross_section = "circular"
        
        # Heterojunction layers
        self.hetero_layers: List[Dict[str, Any]] = []
        
        # Interface properties
        self.interface_types: Dict[str, InterfaceType] = {}
        self.interface_widths: Dict[str, float] = {}
        self.interface_charges: Dict[str, float] = {}
        self.interface_traps: Dict[str, float] = {}
        self.interface_roughness: Dict[str, float] = {}
        
        # Quantum well parameters
        self.quantum_wells: List[Dict[str, Any]] = []
        self.num_quantum_wells = 1
        
        # 3D simulation parameters
        self.enable_3d = False
        self.refinement_regions: List[BoundingBox3D] = []
        self.symmetry_conditions: List[str] = []
        
        # Advanced physics
        self.enable_strain = False
        self.enable_self_heating = False
        self.thermal_boundaries: Dict[str, float] = {"substrate": 300.0, "ambient": 300.0}
        
        # Mesh parameters
        self.mesh_params = {
            "max_element_size": 5e-9,
            "min_element_size": 1e-9,
            "refinement_factor": 2.0,
            "interface_refinement": 0.5e-9
        }
    
    def set_device_type(self, device_type: DeviceStructureType):
        """Set device structure type"""
        self.device_type = device_type
    
    def set_dimensions(self, length: float, width: float, height: float):
        """Set device dimensions"""
        if length <= 0.0 or width <= 0.0 or height <= 0.0:
            raise ValueError("Device dimensions must be positive")
        
        self.device_length = length
        self.device_width = width
        self.device_height = height
    
    def set_substrate_material(self, material: str):
        """Set substrate material"""
        self.substrate_material = material
    
    def add_material_region(self, region: MaterialRegion):
        """Add material region"""
        self.material_regions.append(region)
    
    def add_doping_profile(self, profile: DopingProfile):
        """Add doping profile"""
        self.doping_profiles.append(profile)
    
    def add_gate_structure(self, gate: GateStructure):
        """Add gate structure"""
        self.gate_structures.append(gate)
    
    def add_contact(self, contact: Contact):
        """Add contact"""
        self.contacts.append(contact)
    
    def configure_finfet(self, fin_width: float, fin_height: float, num_fins: int):
        """Configure FinFET structure"""
        if fin_width <= 0.0 or fin_height <= 0.0 or num_fins <= 0:
            raise ValueError("FinFET parameters must be positive")
        
        self.device_type = DeviceStructureType.FINFET
        self.fin_width = fin_width
        self.fin_height = fin_height
        self.num_fins = num_fins
        
        self._generate_finfet_geometry()
    
    def set_fin_pitch(self, pitch: float):
        """Set fin pitch"""
        if pitch <= 0.0:
            raise ValueError("Fin pitch must be positive")
        self.fin_pitch = pitch
    
    def set_fin_orientation(self, orientation: str):
        """Set fin orientation"""
        if orientation not in ["x", "y", "z"]:
            raise ValueError("Fin orientation must be 'x', 'y', or 'z'")
        self.fin_orientation = orientation
    
    def configure_gate_all_around(self, channel_diameter: float, gate_length: float):
        """Configure Gate-All-Around structure"""
        if channel_diameter <= 0.0 or gate_length <= 0.0:
            raise ValueError("GAA parameters must be positive")
        
        self.device_type = DeviceStructureType.GATE_ALL_AROUND
        self.channel_diameter = channel_diameter
        self.device_length = gate_length
        
        self._generate_gaa_geometry()
    
    def set_channel_material(self, material: str):
        """Set channel material"""
        self.channel_material = material
    
    def set_spacer_material(self, material: str, thickness: float):
        """Set spacer material and thickness"""
        self.spacer_material = material
        self.spacer_thickness = thickness
    
    def configure_nanowire(self, diameter: float, length: float, orientation: str = "x"):
        """Configure nanowire transistor"""
        if diameter <= 0.0 or length <= 0.0:
            raise ValueError("Nanowire parameters must be positive")
        
        self.device_type = DeviceStructureType.NANOWIRE_TRANSISTOR
        self.nanowire_diameter = diameter
        self.device_length = length
        self.nanowire_orientation = orientation
        
        self._generate_nanowire_geometry()
    
    def set_nanowire_cross_section(self, shape: str):
        """Set nanowire cross section shape"""
        if shape not in ["circular", "rectangular", "hexagonal"]:
            raise ValueError("Nanowire cross section must be 'circular', 'rectangular', or 'hexagonal'")
        self.nanowire_cross_section = shape
    
    def add_heterojunction_layer(self, material: str, thickness: float, composition: float = 1.0):
        """Add heterojunction layer"""
        if thickness <= 0.0 or composition < 0.0 or composition > 1.0:
            raise ValueError("Invalid heterojunction layer parameters")
        
        layer = {
            "material": material,
            "thickness": thickness,
            "composition": composition
        }
        self.hetero_layers.append(layer)
    
    def set_interface_properties(self, interface_name: str, interface_type: InterfaceType, width: float):
        """Set interface properties"""
        self.interface_types[interface_name] = interface_type
        self.interface_widths[interface_name] = width
    
    def add_quantum_well(self, well_material: str, well_width: float, 
                        barrier_material: str, barrier_width: float):
        """Add quantum well structure"""
        if well_width <= 0.0 or barrier_width <= 0.0:
            raise ValueError("Quantum well dimensions must be positive")
        
        qw = {
            "well_material": well_material,
            "well_width": well_width,
            "barrier_material": barrier_material,
            "barrier_width": barrier_width
        }
        self.quantum_wells.append(qw)
    
    def set_quantum_well_stack(self, num_wells: int):
        """Set number of quantum wells"""
        if num_wells <= 0:
            raise ValueError("Number of quantum wells must be positive")
        self.num_quantum_wells = num_wells
    
    def enable_3d_simulation(self, enable: bool):
        """Enable/disable 3D simulation"""
        self.enable_3d = enable
    
    def set_mesh_refinement_regions(self, regions: List[BoundingBox3D]):
        """Set mesh refinement regions"""
        self.refinement_regions = regions
    
    def set_symmetry_conditions(self, symmetries: List[str]):
        """Set symmetry conditions"""
        self.symmetry_conditions = symmetries
    
    def set_interface_charge_density(self, interface_name: str, charge_density: float):
        """Set interface charge density"""
        self.interface_charges[interface_name] = charge_density
    
    def set_interface_trap_density(self, interface_name: str, trap_density: float):
        """Set interface trap density"""
        self.interface_traps[interface_name] = trap_density
    
    def enable_interface_roughness(self, interface_name: str, rms_roughness: float):
        """Enable interface roughness"""
        self.interface_roughness[interface_name] = rms_roughness
    
    def enable_strain_effects(self, enable: bool):
        """Enable/disable strain effects"""
        self.enable_strain = enable
    
    def set_thermal_boundary_conditions(self, temperatures: Dict[str, float]):
        """Set thermal boundary conditions"""
        self.thermal_boundaries.update(temperatures)
    
    def set_self_heating(self, enable: bool):
        """Enable/disable self-heating"""
        self.enable_self_heating = enable
    
    def set_mesh_parameters(self, params: Dict[str, float]):
        """Set mesh parameters"""
        self.mesh_params.update(params)

    def validate_structure(self) -> bool:
        """Validate device structure"""
        try:
            self._validate_dimensions()
            self._validate_materials()
            self._validate_interfaces()
            return True
        except Exception as e:
            print(f"Structure validation failed: {e}")
            return False

    def analyze_geometry(self) -> Dict[str, float]:
        """Analyze device geometry"""
        analysis = {}

        # Calculate total volume
        total_volume = self.device_length * self.device_width * self.device_height
        analysis["total_volume"] = total_volume

        # Calculate surface area
        surface_area = 2.0 * (self.device_length * self.device_width +
                             self.device_length * self.device_height +
                             self.device_width * self.device_height)
        analysis["surface_area"] = surface_area

        # Calculate aspect ratios
        analysis["length_width_ratio"] = self.device_length / self.device_width
        analysis["length_height_ratio"] = self.device_length / self.device_height
        analysis["width_height_ratio"] = self.device_width / self.device_height

        # Device-specific analysis
        if self.device_type == DeviceStructureType.FINFET:
            analysis["fin_aspect_ratio"] = self.fin_height / self.fin_width
            analysis["total_fin_width"] = self.num_fins * self.fin_width
            analysis["fin_density"] = self.num_fins / self.device_width

        elif self.device_type == DeviceStructureType.GATE_ALL_AROUND:
            analysis["channel_area"] = np.pi * self.channel_diameter**2 / 4.0
            analysis["channel_perimeter"] = np.pi * self.channel_diameter

        elif self.device_type == DeviceStructureType.NANOWIRE_TRANSISTOR:
            if self.nanowire_cross_section == "circular":
                analysis["nanowire_area"] = np.pi * self.nanowire_diameter**2 / 4.0
                analysis["nanowire_perimeter"] = np.pi * self.nanowire_diameter
            elif self.nanowire_cross_section == "rectangular":
                analysis["nanowire_area"] = self.nanowire_diameter**2
                analysis["nanowire_perimeter"] = 4.0 * self.nanowire_diameter

        # Interface analysis
        analysis["num_material_regions"] = len(self.material_regions)
        analysis["num_interfaces"] = len(self.interface_types)
        analysis["num_contacts"] = len(self.contacts)

        return analysis

    def get_material_interfaces(self) -> List[str]:
        """Get list of material interfaces"""
        return list(self.interface_types.keys())

    def export_to_gmsh(self, filename: str):
        """Export structure to GMSH format"""
        with open(filename, 'w') as f:
            f.write("// GMSH geometry file for advanced device structure\n")
            f.write("// Generated by SemiDGFEM Advanced Device Structures\n\n")

            # Set mesh parameters
            f.write(f"Mesh.CharacteristicLengthMin = {self.mesh_params['min_element_size']};\n")
            f.write(f"Mesh.CharacteristicLengthMax = {self.mesh_params['max_element_size']};\n\n")

            # Define points
            f.write("// Device bounding box\n")
            f.write(f"Point(1) = {{0, 0, 0, {self.mesh_params['max_element_size']}}};\n")
            f.write(f"Point(2) = {{{self.device_length}, 0, 0, {self.mesh_params['max_element_size']}}};\n")
            f.write(f"Point(3) = {{{self.device_length}, {self.device_width}, 0, {self.mesh_params['max_element_size']}}};\n")
            f.write(f"Point(4) = {{0, {self.device_width}, 0, {self.mesh_params['max_element_size']}}};\n\n")

            # Define lines
            f.write("// Device boundary lines\n")
            f.write("Line(1) = {1, 2};\n")
            f.write("Line(2) = {2, 3};\n")
            f.write("Line(3) = {3, 4};\n")
            f.write("Line(4) = {4, 1};\n\n")

            # Define surface
            f.write("// Device surface\n")
            f.write("Line Loop(1) = {1, 2, 3, 4};\n")
            f.write("Plane Surface(1) = {1};\n\n")

            # Add device-specific geometry
            if self.device_type == DeviceStructureType.FINFET:
                self._export_finfet_geometry_to_gmsh(f)
            elif self.device_type == DeviceStructureType.GATE_ALL_AROUND:
                self._export_gaa_geometry_to_gmsh(f)
            elif self.device_type == DeviceStructureType.NANOWIRE_TRANSISTOR:
                self._export_nanowire_geometry_to_gmsh(f)

            # Physical regions
            f.write("// Physical regions\n")
            f.write("Physical Surface(\"substrate\") = {1};\n")

    def export_structure_summary(self, filename: str):
        """Export structure summary"""
        with open(filename, 'w') as f:
            f.write("Advanced Device Structure Summary\n")
            f.write("=================================\n\n")

            # Device type
            f.write(f"Device Type: {self.device_type.value}\n\n")

            # Dimensions
            f.write("Dimensions:\n")
            f.write(f"  Length: {self.device_length * 1e9:.2f} nm\n")
            f.write(f"  Width: {self.device_width * 1e9:.2f} nm\n")
            f.write(f"  Height: {self.device_height * 1e9:.2f} nm\n\n")

            # Material regions
            f.write(f"Material Regions ({len(self.material_regions)}):\n")
            for i, region in enumerate(self.material_regions):
                f.write(f"  {i+1}. {region.region_name} ({region.material_name})\n")
            f.write("\n")

            # Contacts
            f.write(f"Contacts ({len(self.contacts)}):\n")
            for i, contact in enumerate(self.contacts):
                f.write(f"  {i+1}. {contact.name} ({contact.contact_type})\n")
            f.write("\n")

            # Geometry analysis
            analysis = self.analyze_geometry()
            f.write("Geometry Analysis:\n")
            for metric, value in analysis.items():
                f.write(f"  {metric}: {value}\n")

    def export_to_json(self, filename: str):
        """Export structure to JSON format"""
        data = {
            "device_type": self.device_type.value,
            "dimensions": {
                "length": self.device_length,
                "width": self.device_width,
                "height": self.device_height
            },
            "substrate_material": self.substrate_material,
            "material_regions": [
                {
                    "material_name": region.material_name,
                    "region_name": region.region_name,
                    "bounding_box": {
                        "min": {"x": region.bounding_box.min_point.x,
                               "y": region.bounding_box.min_point.y,
                               "z": region.bounding_box.min_point.z},
                        "max": {"x": region.bounding_box.max_point.x,
                               "y": region.bounding_box.max_point.y,
                               "z": region.bounding_box.max_point.z}
                    },
                    "properties": region.properties,
                    "interface_type": region.interface_type.value,
                    "interface_width": region.interface_width
                }
                for region in self.material_regions
            ],
            "doping_profiles": [
                {
                    "dopant_type": profile.dopant_type,
                    "concentration": profile.concentration,
                    "region": {
                        "min": {"x": profile.region.min_point.x,
                               "y": profile.region.min_point.y,
                               "z": profile.region.min_point.z},
                        "max": {"x": profile.region.max_point.x,
                               "y": profile.region.max_point.y,
                               "z": profile.region.max_point.z}
                    }
                }
                for profile in self.doping_profiles
            ],
            "gate_structures": [
                {
                    "gate_material": gate.gate_material,
                    "gate_length": gate.gate_length,
                    "gate_width": gate.gate_width,
                    "gate_thickness": gate.gate_thickness,
                    "oxide_thickness": gate.oxide_thickness,
                    "oxide_material": gate.oxide_material,
                    "configuration": gate.configuration.value
                }
                for gate in self.gate_structures
            ],
            "contacts": [
                {
                    "name": contact.name,
                    "contact_type": contact.contact_type,
                    "work_function": contact.work_function,
                    "contact_resistance": contact.contact_resistance
                }
                for contact in self.contacts
            ],
            "mesh_parameters": self.mesh_params,
            "advanced_features": {
                "enable_3d": self.enable_3d,
                "enable_strain": self.enable_strain,
                "enable_self_heating": self.enable_self_heating
            }
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, filename: str):
        """Load structure from JSON format"""
        with open(filename, 'r') as f:
            data = json.load(f)

        # Load basic properties
        self.device_type = DeviceStructureType(data["device_type"])
        self.device_length = data["dimensions"]["length"]
        self.device_width = data["dimensions"]["width"]
        self.device_height = data["dimensions"]["height"]
        self.substrate_material = data["substrate_material"]

        # Load material regions
        self.material_regions = []
        for region_data in data["material_regions"]:
            bbox = BoundingBox3D(
                Point3D(**region_data["bounding_box"]["min"]),
                Point3D(**region_data["bounding_box"]["max"])
            )
            region = MaterialRegion(
                region_data["material_name"],
                region_data["region_name"],
                bbox,
                region_data["properties"],
                InterfaceType(region_data["interface_type"]),
                region_data["interface_width"]
            )
            self.material_regions.append(region)

        # Load other components
        self.mesh_params = data["mesh_parameters"]
        self.enable_3d = data["advanced_features"]["enable_3d"]
        self.enable_strain = data["advanced_features"]["enable_strain"]
        self.enable_self_heating = data["advanced_features"]["enable_self_heating"]

    # Private helper methods
    def _validate_dimensions(self):
        """Validate device dimensions"""
        if self.device_length <= 0.0 or self.device_width <= 0.0 or self.device_height <= 0.0:
            raise ValueError("Device dimensions must be positive")

    def _validate_materials(self):
        """Validate materials"""
        if not self.substrate_material:
            raise ValueError("Substrate material must be specified")

    def _validate_interfaces(self):
        """Validate interfaces"""
        for interface_name, width in self.interface_widths.items():
            if width < 0.0:
                raise ValueError("Interface width must be non-negative")

    def _generate_finfet_geometry(self):
        """Generate FinFET geometry"""
        # Clear existing regions
        self.material_regions = []

        # Create substrate region
        substrate_min = Point3D(0.0, 0.0, 0.0)
        substrate_max = Point3D(self.device_length, self.device_width, self.device_height - self.fin_height)
        substrate_box = BoundingBox3D(substrate_min, substrate_max)
        substrate = MaterialRegion("Si", "substrate", substrate_box, {})
        self.add_material_region(substrate)

        # Create fin regions
        fin_start_y = (self.device_width - self.num_fins * self.fin_width -
                      (self.num_fins - 1) * (self.fin_pitch - self.fin_width)) / 2.0

        for i in range(self.num_fins):
            y_start = fin_start_y + i * self.fin_pitch
            y_end = y_start + self.fin_width

            fin_min = Point3D(0.0, y_start, self.device_height - self.fin_height)
            fin_max = Point3D(self.device_length, y_end, self.device_height)
            fin_box = BoundingBox3D(fin_min, fin_max)

            fin_name = f"fin_{i + 1}"
            fin = MaterialRegion("Si", fin_name, fin_box, {})
            self.add_material_region(fin)

        self._setup_default_contacts()

    def _generate_gaa_geometry(self):
        """Generate Gate-All-Around geometry"""
        self.material_regions = []

        # Create channel region (cylindrical approximation with bounding box)
        channel_min = Point3D(0.0, self.device_width/2.0 - self.channel_diameter/2.0,
                             self.device_height/2.0 - self.channel_diameter/2.0)
        channel_max = Point3D(self.device_length, self.device_width/2.0 + self.channel_diameter/2.0,
                             self.device_height/2.0 + self.channel_diameter/2.0)
        channel_box = BoundingBox3D(channel_min, channel_max)
        channel = MaterialRegion(self.channel_material, "channel", channel_box, {})
        self.add_material_region(channel)

        self._setup_default_contacts()

    def _generate_nanowire_geometry(self):
        """Generate nanowire geometry"""
        self.material_regions = []

        # Create nanowire region
        wire_min = Point3D(0.0, self.device_width/2.0 - self.nanowire_diameter/2.0,
                          self.device_height/2.0 - self.nanowire_diameter/2.0)
        wire_max = Point3D(self.device_length, self.device_width/2.0 + self.nanowire_diameter/2.0,
                          self.device_height/2.0 + self.nanowire_diameter/2.0)
        wire_box = BoundingBox3D(wire_min, wire_max)
        nanowire = MaterialRegion(self.channel_material, "nanowire", wire_box, {})
        self.add_material_region(nanowire)

        self._setup_default_contacts()

    def _setup_default_contacts(self):
        """Setup default contacts"""
        self.contacts = []

        # Source contact
        source_min = Point3D(0.0, 0.0, 0.0)
        source_max = Point3D(self.device_length * 0.1, self.device_width, self.device_height)
        source_box = BoundingBox3D(source_min, source_max)
        source = Contact("source", "ohmic", source_box)
        self.add_contact(source)

        # Drain contact
        drain_min = Point3D(self.device_length * 0.9, 0.0, 0.0)
        drain_max = Point3D(self.device_length, self.device_width, self.device_height)
        drain_box = BoundingBox3D(drain_min, drain_max)
        drain = Contact("drain", "ohmic", drain_box)
        self.add_contact(drain)

    def _export_finfet_geometry_to_gmsh(self, file):
        """Export FinFET geometry to GMSH"""
        file.write("// FinFET specific geometry\n")
        file.write(f"// Number of fins: {self.num_fins}\n")
        file.write(f"// Fin width: {self.fin_width * 1e9:.2f} nm\n")
        file.write(f"// Fin height: {self.fin_height * 1e9:.2f} nm\n\n")

    def _export_gaa_geometry_to_gmsh(self, file):
        """Export GAA geometry to GMSH"""
        file.write("// Gate-All-Around specific geometry\n")
        file.write(f"// Channel diameter: {self.channel_diameter * 1e9:.2f} nm\n\n")

    def _export_nanowire_geometry_to_gmsh(self, file):
        """Export nanowire geometry to GMSH"""
        file.write("// Nanowire specific geometry\n")
        file.write(f"// Nanowire diameter: {self.nanowire_diameter * 1e9:.2f} nm\n")
        file.write(f"// Cross section: {self.nanowire_cross_section}\n\n")

    def plot_structure_2d(self, plane: str = "xy", position: float = 0.0,
                         save_path: Optional[str] = None) -> None:
        """Plot 2D cross-section of device structure"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot material regions
        for region in self.material_regions:
            bbox = region.bounding_box

            if plane == "xy":
                # XY plane at z = position
                if bbox.min_point.z <= position <= bbox.max_point.z:
                    rect = plt.Rectangle(
                        (bbox.min_point.x * 1e9, bbox.min_point.y * 1e9),
                        (bbox.max_point.x - bbox.min_point.x) * 1e9,
                        (bbox.max_point.y - bbox.min_point.y) * 1e9,
                        alpha=0.7, label=region.region_name
                    )
                    ax.add_patch(rect)

            elif plane == "xz":
                # XZ plane at y = position
                if bbox.min_point.y <= position <= bbox.max_point.y:
                    rect = plt.Rectangle(
                        (bbox.min_point.x * 1e9, bbox.min_point.z * 1e9),
                        (bbox.max_point.x - bbox.min_point.x) * 1e9,
                        (bbox.max_point.z - bbox.min_point.z) * 1e9,
                        alpha=0.7, label=region.region_name
                    )
                    ax.add_patch(rect)

            elif plane == "yz":
                # YZ plane at x = position
                if bbox.min_point.x <= position <= bbox.max_point.x:
                    rect = plt.Rectangle(
                        (bbox.min_point.y * 1e9, bbox.min_point.z * 1e9),
                        (bbox.max_point.y - bbox.min_point.y) * 1e9,
                        (bbox.max_point.z - bbox.min_point.z) * 1e9,
                        alpha=0.7, label=region.region_name
                    )
                    ax.add_patch(rect)

        # Set labels and title
        if plane == "xy":
            ax.set_xlabel("X (nm)")
            ax.set_ylabel("Y (nm)")
            ax.set_title(f"Device Structure - XY Plane (Z = {position * 1e9:.1f} nm)")
        elif plane == "xz":
            ax.set_xlabel("X (nm)")
            ax.set_ylabel("Z (nm)")
            ax.set_title(f"Device Structure - XZ Plane (Y = {position * 1e9:.1f} nm)")
        elif plane == "yz":
            ax.set_xlabel("Y (nm)")
            ax.set_ylabel("Z (nm)")
            ax.set_title(f"Device Structure - YZ Plane (X = {position * 1e9:.1f} nm)")

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_doping_profile(self, direction: str = "x", position: Tuple[float, float] = (0.0, 0.0),
                           save_path: Optional[str] = None) -> None:
        """Plot doping profile along specified direction"""
        fig, ax = plt.subplots(figsize=(10, 6))

        if direction == "x":
            x_coords = np.linspace(0, self.device_length, 1000)
            y_pos, z_pos = position

            for profile in self.doping_profiles:
                doping_values = []
                for x in x_coords:
                    if (profile.region.min_point.x <= x <= profile.region.max_point.x and
                        profile.region.min_point.y <= y_pos <= profile.region.max_point.y and
                        profile.region.min_point.z <= z_pos <= profile.region.max_point.z):
                        doping_values.append(profile.spatial_function(x, y_pos, z_pos))
                    else:
                        doping_values.append(0.0)

                if any(val > 0 for val in doping_values):
                    label = f"{profile.dopant_type}-type ({profile.concentration:.1e} cm⁻³)"
                    ax.semilogy(x_coords * 1e9, doping_values, label=label, linewidth=2)

            ax.set_xlabel("X Position (nm)")
            ax.set_ylabel("Doping Concentration (cm⁻³)")
            ax.set_title(f"Doping Profile - X Direction (Y={position[0]*1e9:.1f}nm, Z={position[1]*1e9:.1f}nm)")

        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class DeviceStructureFactory:
    """Factory for creating common device structures"""

    @staticmethod
    def create_finfet(fin_width: float, fin_height: float, gate_length: float,
                     num_fins: int) -> AdvancedDeviceStructure:
        """Create FinFET structure"""
        device = AdvancedDeviceStructure(DeviceStructureType.FINFET)

        # Set basic dimensions
        device.set_dimensions(gate_length, num_fins * fin_width * 2.0, fin_height * 2.0)
        device.configure_finfet(fin_width, fin_height, num_fins)

        # Add default gate structure
        gate = GateStructure("PolySi", gate_length, num_fins * fin_width, 50e-9)
        gate.configuration = GateConfiguration.TRI_GATE
        gate.oxide_thickness = 2e-9
        gate.oxide_material = "HfO2"
        device.add_gate_structure(gate)

        # Add default doping profiles
        source_min = Point3D(0.0, 0.0, 0.0)
        source_max = Point3D(gate_length * 0.3, device.device_width, device.device_height)
        source_region = BoundingBox3D(source_min, source_max)
        source_doping = DopingProfile("n", 1e20, source_region)
        device.add_doping_profile(source_doping)

        drain_min = Point3D(gate_length * 0.7, 0.0, 0.0)
        drain_max = Point3D(gate_length, device.device_width, device.device_height)
        drain_region = BoundingBox3D(drain_min, drain_max)
        drain_doping = DopingProfile("n", 1e20, drain_region)
        device.add_doping_profile(drain_doping)

        channel_min = Point3D(gate_length * 0.3, 0.0, 0.0)
        channel_max = Point3D(gate_length * 0.7, device.device_width, device.device_height)
        channel_region = BoundingBox3D(channel_min, channel_max)
        channel_doping = DopingProfile("p", 1e17, channel_region)
        device.add_doping_profile(channel_doping)

        return device

    @staticmethod
    def create_gate_all_around(channel_diameter: float, gate_length: float,
                              channel_length: float) -> AdvancedDeviceStructure:
        """Create Gate-All-Around structure"""
        device = AdvancedDeviceStructure(DeviceStructureType.GATE_ALL_AROUND)

        # Configure GAA first, then set total channel length
        device.configure_gate_all_around(channel_diameter, gate_length)
        device.set_dimensions(channel_length, channel_diameter * 3.0, channel_diameter * 3.0)

        # Add gate structure
        gate = GateStructure("TiN", gate_length, np.pi * channel_diameter, 10e-9)
        gate.configuration = GateConfiguration.GATE_ALL_AROUND
        gate.oxide_thickness = 1e-9
        gate.oxide_material = "HfO2"
        device.add_gate_structure(gate)

        # Add spacers
        device.set_spacer_material("Si3N4", 5e-9)

        # Add doping profiles
        source_length = (channel_length - gate_length) / 2.0
        drain_start = source_length + gate_length

        source_min = Point3D(0.0, 0.0, 0.0)
        source_max = Point3D(source_length, channel_diameter * 3.0, channel_diameter * 3.0)
        source_region = BoundingBox3D(source_min, source_max)
        source_doping = DopingProfile("n", 1e20, source_region)
        device.add_doping_profile(source_doping)

        drain_min = Point3D(drain_start, 0.0, 0.0)
        drain_max = Point3D(channel_length, channel_diameter * 3.0, channel_diameter * 3.0)
        drain_region = BoundingBox3D(drain_min, drain_max)
        drain_doping = DopingProfile("n", 1e20, drain_region)
        device.add_doping_profile(drain_doping)

        return device

    @staticmethod
    def create_nanowire_transistor(diameter: float, length: float,
                                  material: str = "Si") -> AdvancedDeviceStructure:
        """Create nanowire transistor"""
        device = AdvancedDeviceStructure(DeviceStructureType.NANOWIRE_TRANSISTOR)

        # Set dimensions
        device.set_dimensions(length, diameter * 2.0, diameter * 2.0)
        device.configure_nanowire(diameter, length, "x")
        device.set_channel_material(material)

        # Add gate structure
        gate_length = length * 0.4
        gate = GateStructure("PolySi", gate_length, np.pi * diameter, 20e-9)
        gate.configuration = GateConfiguration.WRAP_AROUND_GATE
        gate.oxide_thickness = 2e-9
        gate.oxide_material = "SiO2"
        device.add_gate_structure(gate)

        # Add doping profiles
        source_length = length * 0.3
        drain_start = length * 0.7

        source_min = Point3D(0.0, 0.0, 0.0)
        source_max = Point3D(source_length, diameter * 2.0, diameter * 2.0)
        source_region = BoundingBox3D(source_min, source_max)
        source_doping = DopingProfile("n", 5e19, source_region)
        device.add_doping_profile(source_doping)

        drain_min = Point3D(drain_start, 0.0, 0.0)
        drain_max = Point3D(length, diameter * 2.0, diameter * 2.0)
        drain_region = BoundingBox3D(drain_min, drain_max)
        drain_doping = DopingProfile("n", 5e19, drain_region)
        device.add_doping_profile(drain_doping)

        return device

    @staticmethod
    def create_heterojunction_bipolar(materials: List[str],
                                     thicknesses: List[float]) -> AdvancedDeviceStructure:
        """Create heterojunction bipolar transistor"""
        if len(materials) != len(thicknesses) or len(materials) < 3:
            raise ValueError("Invalid heterojunction bipolar parameters")

        device = AdvancedDeviceStructure(DeviceStructureType.HETEROJUNCTION_BIPOLAR)

        # Calculate total thickness
        total_thickness = sum(thicknesses)

        # Set dimensions
        device.set_dimensions(100e-9, 100e-9, total_thickness)

        # Add heterojunction layers
        current_z = 0.0
        for i, (material, thickness) in enumerate(zip(materials, thicknesses)):
            device.add_heterojunction_layer(material, thickness)

            # Create material region
            layer_min = Point3D(0.0, 0.0, current_z)
            layer_max = Point3D(100e-9, 100e-9, current_z + thickness)
            layer_box = BoundingBox3D(layer_min, layer_max)

            region_name = f"layer_{i + 1}"
            layer = MaterialRegion(material, region_name, layer_box, {})
            device.add_material_region(layer)

            current_z += thickness

        # Add interface properties
        for i in range(len(materials) - 1):
            interface_name = f"{materials[i]}_{materials[i + 1]}"
            device.set_interface_properties(interface_name, InterfaceType.ABRUPT, 0.5e-9)

        return device

    @staticmethod
    def create_quantum_well_device(well_material: str, well_width: float,
                                  barrier_material: str, barrier_width: float,
                                  num_wells: int) -> AdvancedDeviceStructure:
        """Create quantum well device"""
        device = AdvancedDeviceStructure(DeviceStructureType.QUANTUM_WELL_DEVICE)

        # Calculate total thickness
        total_thickness = num_wells * well_width + (num_wells + 1) * barrier_width

        # Set dimensions
        device.set_dimensions(100e-9, 100e-9, total_thickness)

        # Add quantum well structure
        device.add_quantum_well(well_material, well_width, barrier_material, barrier_width)
        device.set_quantum_well_stack(num_wells)

        # Create material regions
        current_z = 0.0

        # Bottom barrier
        bottom_barrier_min = Point3D(0.0, 0.0, current_z)
        bottom_barrier_max = Point3D(100e-9, 100e-9, current_z + barrier_width)
        bottom_barrier_box = BoundingBox3D(bottom_barrier_min, bottom_barrier_max)
        bottom_barrier = MaterialRegion(barrier_material, "bottom_barrier", bottom_barrier_box, {})
        device.add_material_region(bottom_barrier)
        current_z += barrier_width

        # Wells and barriers
        for i in range(num_wells):
            # Well
            well_min = Point3D(0.0, 0.0, current_z)
            well_max = Point3D(100e-9, 100e-9, current_z + well_width)
            well_box = BoundingBox3D(well_min, well_max)
            well_name = f"well_{i + 1}"
            well = MaterialRegion(well_material, well_name, well_box, {})
            device.add_material_region(well)
            current_z += well_width

            # Barrier (if not the last well)
            if i < num_wells - 1:
                barrier_min = Point3D(0.0, 0.0, current_z)
                barrier_max = Point3D(100e-9, 100e-9, current_z + barrier_width)
                barrier_box = BoundingBox3D(barrier_min, barrier_max)
                barrier_name = f"barrier_{i + 1}"
                barrier = MaterialRegion(barrier_material, barrier_name, barrier_box, {})
                device.add_material_region(barrier)
                current_z += barrier_width

        # Top barrier
        top_barrier_min = Point3D(0.0, 0.0, current_z)
        top_barrier_max = Point3D(100e-9, 100e-9, current_z + barrier_width)
        top_barrier_box = BoundingBox3D(top_barrier_min, top_barrier_max)
        top_barrier = MaterialRegion(barrier_material, "top_barrier", top_barrier_box, {})
        device.add_material_region(top_barrier)

        return device

    @staticmethod
    def create_tunnel_fet(channel_length: float, source_material: str = "Ge",
                         channel_material: str = "Si", drain_material: str = "Si") -> AdvancedDeviceStructure:
        """Create tunnel FET"""
        device = AdvancedDeviceStructure(DeviceStructureType.TUNNEL_FET)

        # Set dimensions
        device.set_dimensions(channel_length, 50e-9, 50e-9)

        # Create regions
        source_length = channel_length * 0.3
        drain_start = channel_length * 0.7

        # Source region
        source_min = Point3D(0.0, 0.0, 0.0)
        source_max = Point3D(source_length, 50e-9, 50e-9)
        source_box = BoundingBox3D(source_min, source_max)
        source = MaterialRegion(source_material, "source", source_box, {})
        device.add_material_region(source)

        # Channel region
        channel_min = Point3D(source_length, 0.0, 0.0)
        channel_max = Point3D(drain_start, 50e-9, 50e-9)
        channel_box = BoundingBox3D(channel_min, channel_max)
        channel = MaterialRegion(channel_material, "channel", channel_box, {})
        device.add_material_region(channel)

        # Drain region
        drain_min = Point3D(drain_start, 0.0, 0.0)
        drain_max = Point3D(channel_length, 50e-9, 50e-9)
        drain_box = BoundingBox3D(drain_min, drain_max)
        drain = MaterialRegion(drain_material, "drain", drain_box, {})
        device.add_material_region(drain)

        # Add gate structure
        gate = GateStructure("TiN", channel_length * 0.4, 50e-9, 10e-9)
        gate.oxide_thickness = 1e-9
        gate.oxide_material = "HfO2"
        device.add_gate_structure(gate)

        # Add doping profiles for tunnel junction
        source_doping = DopingProfile("p", 1e20, source_box)
        device.add_doping_profile(source_doping)

        channel_doping = DopingProfile("p", 1e15, channel_box)
        device.add_doping_profile(channel_doping)

        drain_doping = DopingProfile("n", 1e20, drain_box)
        device.add_doping_profile(drain_doping)

        return device


# Utility functions
def create_test_device_mesh(device: AdvancedDeviceStructure, nx: int = 50, ny: int = 30, nz: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Create test mesh for device structure"""
    # Create structured mesh
    x = np.linspace(0, device.device_length, nx)
    y = np.linspace(0, device.device_width, ny)
    z = np.linspace(0, device.device_height, nz)

    # Create vertices
    vertices = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                vertices.append([x[i], y[j], z[k]])

    vertices = np.array(vertices)

    # Create elements (hexahedral)
    elements = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                # Node indices for hexahedral element
                n0 = k * nx * ny + j * nx + i
                n1 = k * nx * ny + j * nx + (i + 1)
                n2 = k * nx * ny + (j + 1) * nx + (i + 1)
                n3 = k * nx * ny + (j + 1) * nx + i
                n4 = (k + 1) * nx * ny + j * nx + i
                n5 = (k + 1) * nx * ny + j * nx + (i + 1)
                n6 = (k + 1) * nx * ny + (j + 1) * nx + (i + 1)
                n7 = (k + 1) * nx * ny + (j + 1) * nx + i

                elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

    elements = np.array(elements)

    return vertices, elements


def analyze_device_performance(device: AdvancedDeviceStructure) -> Dict[str, float]:
    """Analyze device performance metrics"""
    analysis = device.analyze_geometry()

    # Add device-specific performance metrics (avoid duplicates)
    if device.device_type == DeviceStructureType.FINFET:
        # FinFET specific metrics (add only new ones)
        analysis["effective_channel_width"] = device.num_fins * (2 * device.fin_height + device.fin_width)
        analysis["electrostatic_control"] = 1.0 / (device.fin_width / device.fin_height + 1.0)

    elif device.device_type == DeviceStructureType.GATE_ALL_AROUND:
        # GAA specific metrics (add only new ones)
        analysis["volume_inversion"] = True
        analysis["electrostatic_control"] = 1.0  # Perfect control

    elif device.device_type == DeviceStructureType.NANOWIRE_TRANSISTOR:
        # Nanowire specific metrics (use different names to avoid conflicts)
        if device.nanowire_cross_section == "circular":
            analysis["channel_perimeter"] = np.pi * device.nanowire_diameter
            analysis["channel_area"] = np.pi * (device.nanowire_diameter / 2.0)**2
        elif device.nanowire_cross_section == "rectangular":
            analysis["channel_perimeter"] = 4.0 * device.nanowire_diameter
            analysis["channel_area"] = device.nanowire_diameter**2

    return analysis


def compare_device_structures(devices: List[AdvancedDeviceStructure]) -> Dict[str, List[float]]:
    """Compare multiple device structures"""
    comparison = {}

    # First pass: collect all metrics
    all_metrics = set()
    device_analyses = []

    for device in devices:
        analysis = analyze_device_performance(device)
        device_analyses.append(analysis)
        all_metrics.update(analysis.keys())

    # Second pass: create comparison with consistent structure
    for metric in all_metrics:
        comparison[metric] = []
        for analysis in device_analyses:
            # Use None or 0.0 for missing metrics
            value = analysis.get(metric, None)
            if value is not None:
                comparison[metric].append(value)
            else:
                comparison[metric].append(0.0)

    return comparison
