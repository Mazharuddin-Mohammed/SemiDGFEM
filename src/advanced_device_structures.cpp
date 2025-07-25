/**
 * Advanced Device Structures Implementation
 * 
 * This file implements advanced device structures including:
 * - Multi-gate transistors (FinFET, GAA)
 * - Nanowire transistors
 * - Heterojunction devices
 * - 3D device geometries
 * 
 * Author: Dr. Mazharuddin Mohammed
 */

#include "advanced_device_structures.hpp"
#include "mesh.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>

namespace simulator {

AdvancedDeviceStructure::AdvancedDeviceStructure(DeviceStructureType type)
    : device_type_(type), device_length_(100e-9), device_width_(50e-9), device_height_(50e-9),
      substrate_material_("Si"), fin_width_(10e-9), fin_height_(20e-9), num_fins_(1),
      fin_pitch_(30e-9), fin_orientation_("x"), channel_diameter_(10e-9),
      channel_material_("Si"), spacer_material_("SiO2"), spacer_thickness_(5e-9),
      nanowire_diameter_(10e-9), nanowire_orientation_("x"), nanowire_cross_section_("circular"),
      num_quantum_wells_(1), enable_3d_(false), enable_strain_(false), enable_self_heating_(false) {
    
    initialize_default_parameters();
}

void AdvancedDeviceStructure::initialize_default_parameters() {
    // Set default mesh parameters
    mesh_params_["max_element_size"] = 5e-9;
    mesh_params_["min_element_size"] = 1e-9;
    mesh_params_["refinement_factor"] = 2.0;
    mesh_params_["interface_refinement"] = 0.5e-9;
    
    // Set default thermal boundaries
    thermal_boundaries_["substrate"] = 300.0;  // K
    thermal_boundaries_["ambient"] = 300.0;    // K
}

void AdvancedDeviceStructure::set_device_type(DeviceStructureType type) {
    device_type_ = type;
}

void AdvancedDeviceStructure::set_dimensions(double length, double width, double height) {
    if (length <= 0.0 || width <= 0.0 || height <= 0.0) {
        throw std::invalid_argument("Device dimensions must be positive");
    }
    device_length_ = length;
    device_width_ = width;
    device_height_ = height;
}

void AdvancedDeviceStructure::set_substrate_material(const std::string& material) {
    substrate_material_ = material;
}

void AdvancedDeviceStructure::add_material_region(const MaterialRegion& region) {
    material_regions_.push_back(region);
}

void AdvancedDeviceStructure::add_doping_profile(const DopingProfile& profile) {
    doping_profiles_.push_back(profile);
}

void AdvancedDeviceStructure::add_gate_structure(const GateStructure& gate) {
    gate_structures_.push_back(gate);
}

void AdvancedDeviceStructure::add_contact(const Contact& contact) {
    contacts_.push_back(contact);
}

void AdvancedDeviceStructure::configure_finfet(double fin_width, double fin_height, int num_fins) {
    if (fin_width <= 0.0 || fin_height <= 0.0 || num_fins <= 0) {
        throw std::invalid_argument("FinFET parameters must be positive");
    }
    
    device_type_ = DeviceStructureType::FINFET;
    fin_width_ = fin_width;
    fin_height_ = fin_height;
    num_fins_ = num_fins;
    
    generate_finfet_geometry();
}

void AdvancedDeviceStructure::set_fin_pitch(double pitch) {
    if (pitch <= 0.0) {
        throw std::invalid_argument("Fin pitch must be positive");
    }
    fin_pitch_ = pitch;
}

void AdvancedDeviceStructure::set_fin_orientation(const std::string& orientation) {
    if (orientation != "x" && orientation != "y" && orientation != "z") {
        throw std::invalid_argument("Fin orientation must be 'x', 'y', or 'z'");
    }
    fin_orientation_ = orientation;
}

void AdvancedDeviceStructure::configure_gate_all_around(double channel_diameter, double gate_length) {
    if (channel_diameter <= 0.0 || gate_length <= 0.0) {
        throw std::invalid_argument("GAA parameters must be positive");
    }
    
    device_type_ = DeviceStructureType::GATE_ALL_AROUND;
    channel_diameter_ = channel_diameter;
    device_length_ = gate_length;
    
    generate_gaa_geometry();
}

void AdvancedDeviceStructure::set_channel_material(const std::string& material) {
    channel_material_ = material;
}

void AdvancedDeviceStructure::set_spacer_material(const std::string& material, double thickness) {
    spacer_material_ = material;
    spacer_thickness_ = thickness;
}

void AdvancedDeviceStructure::configure_nanowire(double diameter, double length, const std::string& orientation) {
    if (diameter <= 0.0 || length <= 0.0) {
        throw std::invalid_argument("Nanowire parameters must be positive");
    }
    
    device_type_ = DeviceStructureType::NANOWIRE_TRANSISTOR;
    nanowire_diameter_ = diameter;
    device_length_ = length;
    nanowire_orientation_ = orientation;
    
    generate_nanowire_geometry();
}

void AdvancedDeviceStructure::set_nanowire_cross_section(const std::string& shape) {
    if (shape != "circular" && shape != "rectangular" && shape != "hexagonal") {
        throw std::invalid_argument("Nanowire cross section must be 'circular', 'rectangular', or 'hexagonal'");
    }
    nanowire_cross_section_ = shape;
}

void AdvancedDeviceStructure::add_heterojunction_layer(const std::string& material, double thickness, double composition) {
    if (thickness <= 0.0 || composition < 0.0 || composition > 1.0) {
        throw std::invalid_argument("Invalid heterojunction layer parameters");
    }
    
    HeterojunctionLayer layer;
    layer.material = material;
    layer.thickness = thickness;
    layer.composition = composition;
    hetero_layers_.push_back(layer);
}

void AdvancedDeviceStructure::set_interface_properties(const std::string& interface_name, 
                                                      InterfaceType type, double width) {
    interface_types_[interface_name] = type;
    interface_widths_[interface_name] = width;
}

void AdvancedDeviceStructure::add_quantum_well(const std::string& well_material, double well_width,
                                              const std::string& barrier_material, double barrier_width) {
    if (well_width <= 0.0 || barrier_width <= 0.0) {
        throw std::invalid_argument("Quantum well dimensions must be positive");
    }
    
    QuantumWellLayer qw;
    qw.well_material = well_material;
    qw.well_width = well_width;
    qw.barrier_material = barrier_material;
    qw.barrier_width = barrier_width;
    quantum_wells_.push_back(qw);
}

void AdvancedDeviceStructure::set_quantum_well_stack(int num_wells) {
    if (num_wells <= 0) {
        throw std::invalid_argument("Number of quantum wells must be positive");
    }
    num_quantum_wells_ = num_wells;
}

void AdvancedDeviceStructure::enable_3d_simulation(bool enable) {
    enable_3d_ = enable;
}

void AdvancedDeviceStructure::set_mesh_refinement_regions(const std::vector<BoundingBox3D>& regions) {
    refinement_regions_ = regions;
}

void AdvancedDeviceStructure::set_symmetry_conditions(const std::vector<std::string>& symmetries) {
    symmetry_conditions_ = symmetries;
}

void AdvancedDeviceStructure::set_interface_charge_density(const std::string& interface_name, double charge_density) {
    interface_charges_[interface_name] = charge_density;
}

void AdvancedDeviceStructure::set_interface_trap_density(const std::string& interface_name, double trap_density) {
    interface_traps_[interface_name] = trap_density;
}

void AdvancedDeviceStructure::enable_interface_roughness(const std::string& interface_name, double rms_roughness) {
    interface_roughness_[interface_name] = rms_roughness;
}

void AdvancedDeviceStructure::enable_strain_effects(bool enable) {
    enable_strain_ = enable;
}

void AdvancedDeviceStructure::set_thermal_boundary_conditions(const std::map<std::string, double>& temperatures) {
    thermal_boundaries_ = temperatures;
}

void AdvancedDeviceStructure::enable_self_heating(bool enable) {
    enable_self_heating_ = enable;
}

void AdvancedDeviceStructure::set_mesh_parameters(const std::map<std::string, double>& params) {
    for (const auto& param : params) {
        mesh_params_[param.first] = param.second;
    }
}

bool AdvancedDeviceStructure::validate_structure() const {
    try {
        validate_dimensions();
        validate_materials();
        validate_interfaces();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Structure validation failed: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, double> AdvancedDeviceStructure::analyze_geometry() const {
    std::map<std::string, double> analysis;
    
    // Calculate total volume
    double total_volume = device_length_ * device_width_ * device_height_;
    analysis["total_volume"] = total_volume;
    
    // Calculate surface area
    double surface_area = 2.0 * (device_length_ * device_width_ + 
                                device_length_ * device_height_ + 
                                device_width_ * device_height_);
    analysis["surface_area"] = surface_area;
    
    // Calculate aspect ratios
    analysis["length_width_ratio"] = device_length_ / device_width_;
    analysis["length_height_ratio"] = device_length_ / device_height_;
    analysis["width_height_ratio"] = device_width_ / device_height_;
    
    // Device-specific analysis
    switch (device_type_) {
        case DeviceStructureType::FINFET:
            analysis["fin_aspect_ratio"] = fin_height_ / fin_width_;
            analysis["total_fin_width"] = num_fins_ * fin_width_;
            analysis["fin_density"] = num_fins_ / device_width_;
            break;
            
        case DeviceStructureType::GATE_ALL_AROUND:
            analysis["channel_area"] = M_PI * channel_diameter_ * channel_diameter_ / 4.0;
            analysis["channel_perimeter"] = M_PI * channel_diameter_;
            break;
            
        case DeviceStructureType::NANOWIRE_TRANSISTOR:
            if (nanowire_cross_section_ == "circular") {
                analysis["nanowire_area"] = M_PI * nanowire_diameter_ * nanowire_diameter_ / 4.0;
                analysis["nanowire_perimeter"] = M_PI * nanowire_diameter_;
            } else if (nanowire_cross_section_ == "rectangular") {
                analysis["nanowire_area"] = nanowire_diameter_ * nanowire_diameter_;
                analysis["nanowire_perimeter"] = 4.0 * nanowire_diameter_;
            }
            break;
            
        default:
            break;
    }
    
    // Interface analysis
    analysis["num_material_regions"] = static_cast<double>(material_regions_.size());
    analysis["num_interfaces"] = static_cast<double>(interface_types_.size());
    analysis["num_contacts"] = static_cast<double>(contacts_.size());
    
    return analysis;
}

std::vector<std::string> AdvancedDeviceStructure::get_material_interfaces() const {
    std::vector<std::string> interfaces;
    
    for (const auto& interface : interface_types_) {
        interfaces.push_back(interface.first);
    }
    
    return interfaces;
}

void AdvancedDeviceStructure::export_to_gmsh(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    file << "// GMSH geometry file for advanced device structure\n";
    file << "// Generated by SemiDGFEM Advanced Device Structures\n\n";
    
    // Set mesh parameters
    file << "Mesh.CharacteristicLengthMin = " << mesh_params_.at("min_element_size") << ";\n";
    file << "Mesh.CharacteristicLengthMax = " << mesh_params_.at("max_element_size") << ";\n\n";
    
    // Define points
    file << "// Device bounding box\n";
    file << "Point(1) = {0, 0, 0, " << mesh_params_.at("max_element_size") << "};\n";
    file << "Point(2) = {" << device_length_ << ", 0, 0, " << mesh_params_.at("max_element_size") << "};\n";
    file << "Point(3) = {" << device_length_ << ", " << device_width_ << ", 0, " << mesh_params_.at("max_element_size") << "};\n";
    file << "Point(4) = {0, " << device_width_ << ", 0, " << mesh_params_.at("max_element_size") << "};\n\n";
    
    // Define lines
    file << "// Device boundary lines\n";
    file << "Line(1) = {1, 2};\n";
    file << "Line(2) = {2, 3};\n";
    file << "Line(3) = {3, 4};\n";
    file << "Line(4) = {4, 1};\n\n";
    
    // Define surface
    file << "// Device surface\n";
    file << "Line Loop(1) = {1, 2, 3, 4};\n";
    file << "Plane Surface(1) = {1};\n\n";
    
    // Add device-specific geometry
    switch (device_type_) {
        case DeviceStructureType::FINFET:
            export_finfet_geometry_to_gmsh(file);
            break;
        case DeviceStructureType::GATE_ALL_AROUND:
            export_gaa_geometry_to_gmsh(file);
            break;
        case DeviceStructureType::NANOWIRE_TRANSISTOR:
            export_nanowire_geometry_to_gmsh(file);
            break;
        default:
            break;
    }
    
    // Physical regions
    file << "// Physical regions\n";
    file << "Physical Surface(\"substrate\") = {1};\n";
    
    file.close();
}

void AdvancedDeviceStructure::export_structure_summary(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    file << "Advanced Device Structure Summary\n";
    file << "=================================\n\n";
    
    // Device type
    file << "Device Type: ";
    switch (device_type_) {
        case DeviceStructureType::PLANAR_MOSFET: file << "Planar MOSFET"; break;
        case DeviceStructureType::FINFET: file << "FinFET"; break;
        case DeviceStructureType::GATE_ALL_AROUND: file << "Gate-All-Around"; break;
        case DeviceStructureType::NANOWIRE_TRANSISTOR: file << "Nanowire Transistor"; break;
        case DeviceStructureType::HETEROJUNCTION_BIPOLAR: file << "Heterojunction Bipolar"; break;
        case DeviceStructureType::QUANTUM_WELL_DEVICE: file << "Quantum Well Device"; break;
        case DeviceStructureType::TUNNEL_FET: file << "Tunnel FET"; break;
        case DeviceStructureType::JUNCTIONLESS_TRANSISTOR: file << "Junctionless Transistor"; break;
        default: file << "Unknown"; break;
    }
    file << "\n\n";
    
    // Dimensions
    file << "Dimensions:\n";
    file << "  Length: " << device_length_ * 1e9 << " nm\n";
    file << "  Width: " << device_width_ * 1e9 << " nm\n";
    file << "  Height: " << device_height_ * 1e9 << " nm\n\n";
    
    // Material regions
    file << "Material Regions (" << material_regions_.size() << "):\n";
    for (size_t i = 0; i < material_regions_.size(); ++i) {
        const auto& region = material_regions_[i];
        file << "  " << i+1 << ". " << region.region_name << " (" << region.material_name << ")\n";
    }
    file << "\n";
    
    // Contacts
    file << "Contacts (" << contacts_.size() << "):\n";
    for (size_t i = 0; i < contacts_.size(); ++i) {
        const auto& contact = contacts_[i];
        file << "  " << i+1 << ". " << contact.name << " (" << contact.contact_type << ")\n";
    }
    file << "\n";
    
    // Geometry analysis
    auto analysis = analyze_geometry();
    file << "Geometry Analysis:\n";
    for (const auto& metric : analysis) {
        file << "  " << metric.first << ": " << metric.second << "\n";
    }
    
    file.close();
}

// Private helper methods
void AdvancedDeviceStructure::validate_dimensions() const {
    if (device_length_ <= 0.0 || device_width_ <= 0.0 || device_height_ <= 0.0) {
        throw std::invalid_argument("Device dimensions must be positive");
    }
}

void AdvancedDeviceStructure::validate_materials() const {
    if (substrate_material_.empty()) {
        throw std::invalid_argument("Substrate material must be specified");
    }
}

void AdvancedDeviceStructure::validate_interfaces() const {
    // Check that interface widths are reasonable
    for (const auto& interface : interface_widths_) {
        if (interface.second < 0.0) {
            throw std::invalid_argument("Interface width must be non-negative");
        }
    }
}

BoundingBox3D AdvancedDeviceStructure::calculate_device_bounding_box() const {
    Point3D min_point(0.0, 0.0, 0.0);
    Point3D max_point(device_length_, device_width_, device_height_);
    return BoundingBox3D(min_point, max_point);
}

void AdvancedDeviceStructure::generate_finfet_geometry() {
    // Clear existing regions
    material_regions_.clear();
    
    // Create substrate region
    Point3D substrate_min(0.0, 0.0, 0.0);
    Point3D substrate_max(device_length_, device_width_, device_height_ - fin_height_);
    BoundingBox3D substrate_box(substrate_min, substrate_max);
    MaterialRegion substrate("Si", "substrate", substrate_box);
    add_material_region(substrate);
    
    // Create fin regions
    double fin_start_y = (device_width_ - num_fins_ * fin_width_ - (num_fins_ - 1) * (fin_pitch_ - fin_width_)) / 2.0;
    
    for (int i = 0; i < num_fins_; ++i) {
        double y_start = fin_start_y + i * fin_pitch_;
        double y_end = y_start + fin_width_;
        
        Point3D fin_min(0.0, y_start, device_height_ - fin_height_);
        Point3D fin_max(device_length_, y_end, device_height_);
        BoundingBox3D fin_box(fin_min, fin_max);
        
        std::string fin_name = "fin_" + std::to_string(i + 1);
        MaterialRegion fin("Si", fin_name, fin_box);
        add_material_region(fin);
    }
    
    setup_default_contacts();
}

void AdvancedDeviceStructure::generate_gaa_geometry() {
    // Implementation for Gate-All-Around geometry
    material_regions_.clear();
    
    // Create channel region (cylindrical)
    Point3D channel_min(0.0, device_width_/2.0 - channel_diameter_/2.0, device_height_/2.0 - channel_diameter_/2.0);
    Point3D channel_max(device_length_, device_width_/2.0 + channel_diameter_/2.0, device_height_/2.0 + channel_diameter_/2.0);
    BoundingBox3D channel_box(channel_min, channel_max);
    MaterialRegion channel(channel_material_, "channel", channel_box);
    add_material_region(channel);
    
    setup_default_contacts();
}

void AdvancedDeviceStructure::generate_nanowire_geometry() {
    // Implementation for nanowire geometry
    material_regions_.clear();
    
    // Create nanowire region
    Point3D wire_min(0.0, device_width_/2.0 - nanowire_diameter_/2.0, device_height_/2.0 - nanowire_diameter_/2.0);
    Point3D wire_max(device_length_, device_width_/2.0 + nanowire_diameter_/2.0, device_height_/2.0 + nanowire_diameter_/2.0);
    BoundingBox3D wire_box(wire_min, wire_max);
    MaterialRegion nanowire(channel_material_, "nanowire", wire_box);
    add_material_region(nanowire);
    
    setup_default_contacts();
}

void AdvancedDeviceStructure::setup_default_contacts() {
    contacts_.clear();
    
    // Source contact
    Point3D source_min(0.0, 0.0, 0.0);
    Point3D source_max(device_length_ * 0.1, device_width_, device_height_);
    BoundingBox3D source_box(source_min, source_max);
    Contact source("source", "ohmic", source_box);
    add_contact(source);
    
    // Drain contact
    Point3D drain_min(device_length_ * 0.9, 0.0, 0.0);
    Point3D drain_max(device_length_, device_width_, device_height_);
    BoundingBox3D drain_box(drain_min, drain_max);
    Contact drain("drain", "ohmic", drain_box);
    add_contact(drain);
}

void AdvancedDeviceStructure::export_finfet_geometry_to_gmsh(std::ofstream& file) const {
    file << "// FinFET specific geometry\n";
    file << "// Number of fins: " << num_fins_ << "\n";
    file << "// Fin width: " << fin_width_ * 1e9 << " nm\n";
    file << "// Fin height: " << fin_height_ * 1e9 << " nm\n\n";
}

void AdvancedDeviceStructure::export_gaa_geometry_to_gmsh(std::ofstream& file) const {
    file << "// Gate-All-Around specific geometry\n";
    file << "// Channel diameter: " << channel_diameter_ * 1e9 << " nm\n\n";
}

void AdvancedDeviceStructure::export_nanowire_geometry_to_gmsh(std::ofstream& file) const {
    file << "// Nanowire specific geometry\n";
    file << "// Nanowire diameter: " << nanowire_diameter_ * 1e9 << " nm\n";
    file << "// Cross section: " << nanowire_cross_section_ << "\n\n";
}

// Device Structure Factory Implementation
std::unique_ptr<AdvancedDeviceStructure> DeviceStructureFactory::create_finfet(
    double fin_width, double fin_height, double gate_length, int num_fins) {

    auto device = std::make_unique<AdvancedDeviceStructure>(DeviceStructureType::FINFET);

    // Set basic dimensions
    device->set_dimensions(gate_length, num_fins * fin_width * 2.0, fin_height * 2.0);
    device->configure_finfet(fin_width, fin_height, num_fins);

    // Add default gate structure
    GateStructure gate("PolySi", gate_length, num_fins * fin_width, 50e-9);
    gate.configuration = GateConfiguration::TRI_GATE;
    gate.oxide_thickness = 2e-9;
    gate.oxide_material = "HfO2";
    device->add_gate_structure(gate);

    // Add default doping profiles
    Point3D source_min(0.0, 0.0, 0.0);
    Point3D source_max(gate_length * 0.3, device->get_device_width(), device->get_device_height());
    BoundingBox3D source_region(source_min, source_max);
    DopingProfile source_doping("n", 1e20, source_region);
    device->add_doping_profile(source_doping);

    Point3D drain_min(gate_length * 0.7, 0.0, 0.0);
    Point3D drain_max(gate_length, device->get_device_width(), device->get_device_height());
    BoundingBox3D drain_region(drain_min, drain_max);
    DopingProfile drain_doping("n", 1e20, drain_region);
    device->add_doping_profile(drain_doping);

    Point3D channel_min(gate_length * 0.3, 0.0, 0.0);
    Point3D channel_max(gate_length * 0.7, device->get_device_width(), device->get_device_height());
    BoundingBox3D channel_region(channel_min, channel_max);
    DopingProfile channel_doping("p", 1e17, channel_region);
    device->add_doping_profile(channel_doping);

    return device;
}

std::unique_ptr<AdvancedDeviceStructure> DeviceStructureFactory::create_gate_all_around(
    double channel_diameter, double gate_length, double channel_length) {

    auto device = std::make_unique<AdvancedDeviceStructure>(DeviceStructureType::GATE_ALL_AROUND);

    // Set dimensions
    device->set_dimensions(channel_length, channel_diameter * 3.0, channel_diameter * 3.0);
    device->configure_gate_all_around(channel_diameter, gate_length);

    // Add gate structure
    GateStructure gate("TiN", gate_length, M_PI * channel_diameter, 10e-9);
    gate.configuration = GateConfiguration::GATE_ALL_AROUND;
    gate.oxide_thickness = 1e-9;
    gate.oxide_material = "HfO2";
    device->add_gate_structure(gate);

    // Add spacers
    device->set_spacer_material("Si3N4", 5e-9);

    // Add doping profiles
    double source_length = (channel_length - gate_length) / 2.0;
    double drain_start = source_length + gate_length;

    Point3D source_min(0.0, 0.0, 0.0);
    Point3D source_max(source_length, channel_diameter * 3.0, channel_diameter * 3.0);
    BoundingBox3D source_region(source_min, source_max);
    DopingProfile source_doping("n", 1e20, source_region);
    device->add_doping_profile(source_doping);

    Point3D drain_min(drain_start, 0.0, 0.0);
    Point3D drain_max(channel_length, channel_diameter * 3.0, channel_diameter * 3.0);
    BoundingBox3D drain_region(drain_min, drain_max);
    DopingProfile drain_doping("n", 1e20, drain_region);
    device->add_doping_profile(drain_doping);

    return device;
}

std::unique_ptr<AdvancedDeviceStructure> DeviceStructureFactory::create_nanowire_transistor(
    double diameter, double length, const std::string& material) {

    auto device = std::make_unique<AdvancedDeviceStructure>(DeviceStructureType::NANOWIRE_TRANSISTOR);

    // Set dimensions
    device->set_dimensions(length, diameter * 2.0, diameter * 2.0);
    device->configure_nanowire(diameter, length, "x");
    device->set_channel_material(material);

    // Add gate structure
    double gate_length = length * 0.4;
    GateStructure gate("PolySi", gate_length, M_PI * diameter, 20e-9);
    gate.configuration = GateConfiguration::WRAP_AROUND_GATE;
    gate.oxide_thickness = 2e-9;
    gate.oxide_material = "SiO2";
    device->add_gate_structure(gate);

    // Add doping profiles
    double source_length = length * 0.3;
    double drain_start = length * 0.7;

    Point3D source_min(0.0, 0.0, 0.0);
    Point3D source_max(source_length, diameter * 2.0, diameter * 2.0);
    BoundingBox3D source_region(source_min, source_max);
    DopingProfile source_doping("n", 5e19, source_region);
    device->add_doping_profile(source_doping);

    Point3D drain_min(drain_start, 0.0, 0.0);
    Point3D drain_max(length, diameter * 2.0, diameter * 2.0);
    BoundingBox3D drain_region(drain_min, drain_max);
    DopingProfile drain_doping("n", 5e19, drain_region);
    device->add_doping_profile(drain_doping);

    return device;
}

std::unique_ptr<AdvancedDeviceStructure> DeviceStructureFactory::create_heterojunction_bipolar(
    const std::vector<std::string>& materials, const std::vector<double>& thicknesses) {

    if (materials.size() != thicknesses.size() || materials.size() < 3) {
        throw std::invalid_argument("Invalid heterojunction bipolar parameters");
    }

    auto device = std::make_unique<AdvancedDeviceStructure>(DeviceStructureType::HETEROJUNCTION_BIPOLAR);

    // Calculate total thickness
    double total_thickness = 0.0;
    for (double thickness : thicknesses) {
        total_thickness += thickness;
    }

    // Set dimensions
    device->set_dimensions(100e-9, 100e-9, total_thickness);

    // Add heterojunction layers
    double current_z = 0.0;
    for (size_t i = 0; i < materials.size(); ++i) {
        device->add_heterojunction_layer(materials[i], thicknesses[i]);

        // Create material region
        Point3D layer_min(0.0, 0.0, current_z);
        Point3D layer_max(100e-9, 100e-9, current_z + thicknesses[i]);
        BoundingBox3D layer_box(layer_min, layer_max);

        std::string region_name = "layer_" + std::to_string(i + 1);
        MaterialRegion layer(materials[i], region_name, layer_box);
        device->add_material_region(layer);

        current_z += thicknesses[i];
    }

    // Add interface properties
    for (size_t i = 0; i < materials.size() - 1; ++i) {
        std::string interface_name = materials[i] + "_" + materials[i + 1];
        device->set_interface_properties(interface_name, InterfaceType::ABRUPT, 0.5e-9);
    }

    return device;
}

std::unique_ptr<AdvancedDeviceStructure> DeviceStructureFactory::create_quantum_well_device(
    const std::string& well_material, double well_width,
    const std::string& barrier_material, double barrier_width, int num_wells) {

    auto device = std::make_unique<AdvancedDeviceStructure>(DeviceStructureType::QUANTUM_WELL_DEVICE);

    // Calculate total thickness
    double total_thickness = num_wells * well_width + (num_wells + 1) * barrier_width;

    // Set dimensions
    device->set_dimensions(100e-9, 100e-9, total_thickness);

    // Add quantum well structure
    device->add_quantum_well(well_material, well_width, barrier_material, barrier_width);
    device->set_quantum_well_stack(num_wells);

    // Create material regions
    double current_z = 0.0;

    // Bottom barrier
    Point3D bottom_barrier_min(0.0, 0.0, current_z);
    Point3D bottom_barrier_max(100e-9, 100e-9, current_z + barrier_width);
    BoundingBox3D bottom_barrier_box(bottom_barrier_min, bottom_barrier_max);
    MaterialRegion bottom_barrier(barrier_material, "bottom_barrier", bottom_barrier_box);
    device->add_material_region(bottom_barrier);
    current_z += barrier_width;

    // Wells and barriers
    for (int i = 0; i < num_wells; ++i) {
        // Well
        Point3D well_min(0.0, 0.0, current_z);
        Point3D well_max(100e-9, 100e-9, current_z + well_width);
        BoundingBox3D well_box(well_min, well_max);
        std::string well_name = "well_" + std::to_string(i + 1);
        MaterialRegion well(well_material, well_name, well_box);
        device->add_material_region(well);
        current_z += well_width;

        // Barrier (if not the last well)
        if (i < num_wells - 1) {
            Point3D barrier_min(0.0, 0.0, current_z);
            Point3D barrier_max(100e-9, 100e-9, current_z + barrier_width);
            BoundingBox3D barrier_box(barrier_min, barrier_max);
            std::string barrier_name = "barrier_" + std::to_string(i + 1);
            MaterialRegion barrier(barrier_material, barrier_name, barrier_box);
            device->add_material_region(barrier);
            current_z += barrier_width;
        }
    }

    // Top barrier
    Point3D top_barrier_min(0.0, 0.0, current_z);
    Point3D top_barrier_max(100e-9, 100e-9, current_z + barrier_width);
    BoundingBox3D top_barrier_box(top_barrier_min, top_barrier_max);
    MaterialRegion top_barrier(barrier_material, "top_barrier", top_barrier_box);
    device->add_material_region(top_barrier);

    return device;
}

std::unique_ptr<AdvancedDeviceStructure> DeviceStructureFactory::create_tunnel_fet(
    double channel_length, const std::string& source_material,
    const std::string& channel_material, const std::string& drain_material) {

    auto device = std::make_unique<AdvancedDeviceStructure>(DeviceStructureType::TUNNEL_FET);

    // Set dimensions
    device->set_dimensions(channel_length, 50e-9, 50e-9);

    // Create regions
    double source_length = channel_length * 0.3;
    double drain_start = channel_length * 0.7;

    // Source region
    Point3D source_min(0.0, 0.0, 0.0);
    Point3D source_max(source_length, 50e-9, 50e-9);
    BoundingBox3D source_box(source_min, source_max);
    MaterialRegion source(source_material, "source", source_box);
    device->add_material_region(source);

    // Channel region
    Point3D channel_min(source_length, 0.0, 0.0);
    Point3D channel_max(drain_start, 50e-9, 50e-9);
    BoundingBox3D channel_box(channel_min, channel_max);
    MaterialRegion channel(channel_material, "channel", channel_box);
    device->add_material_region(channel);

    // Drain region
    Point3D drain_min(drain_start, 0.0, 0.0);
    Point3D drain_max(channel_length, 50e-9, 50e-9);
    BoundingBox3D drain_box(drain_min, drain_max);
    MaterialRegion drain(drain_material, "drain", drain_box);
    device->add_material_region(drain);

    // Add gate structure
    GateStructure gate("TiN", channel_length * 0.4, 50e-9, 10e-9);
    gate.oxide_thickness = 1e-9;
    gate.oxide_material = "HfO2";
    device->add_gate_structure(gate);

    // Add doping profiles for tunnel junction
    DopingProfile source_doping("p", 1e20, source_box);
    device->add_doping_profile(source_doping);

    DopingProfile channel_doping("p", 1e15, channel_box);
    device->add_doping_profile(channel_doping);

    DopingProfile drain_doping("n", 1e20, drain_box);
    device->add_doping_profile(drain_doping);

    return device;
}

} // namespace simulator
