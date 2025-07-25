/**
 * Advanced Device Structures for SemiDGFEM
 * 
 * This header defines advanced device structures including:
 * - Multi-gate transistors (FinFET, GAA)
 * - Nanowire transistors
 * - Heterojunction devices
 * - 3D device geometries
 * - Complex material interfaces
 * 
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <array>

namespace simulator {

// Forward declarations
class Device;
class Mesh;

/**
 * Device structure types
 */
enum class DeviceStructureType {
    PLANAR_MOSFET,
    FINFET,
    GATE_ALL_AROUND,
    NANOWIRE_TRANSISTOR,
    HETEROJUNCTION_BIPOLAR,
    QUANTUM_WELL_DEVICE,
    TUNNEL_FET,
    JUNCTIONLESS_TRANSISTOR,
    CARBON_NANOTUBE_FET,
    GRAPHENE_FET
};

/**
 * Gate configuration types
 */
enum class GateConfiguration {
    SINGLE_GATE,
    DOUBLE_GATE,
    TRI_GATE,
    GATE_ALL_AROUND,
    WRAP_AROUND_GATE
};

/**
 * Material interface types
 */
enum class InterfaceType {
    ABRUPT,
    GRADED,
    DELTA_DOPED,
    SUPERLATTICE,
    QUANTUM_WELL,
    BARRIER_LAYER
};

/**
 * 3D geometric primitives
 */
struct Point3D {
    double x, y, z;
    Point3D(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}
};

struct BoundingBox3D {
    Point3D min_point, max_point;
    BoundingBox3D(const Point3D& min_pt, const Point3D& max_pt) 
        : min_point(min_pt), max_point(max_pt) {}
};

/**
 * Material region definition
 */
struct MaterialRegion {
    std::string material_name;
    std::string region_name;
    BoundingBox3D bounding_box;
    std::map<std::string, double> properties;
    InterfaceType interface_type;
    double interface_width;
    
    MaterialRegion(const std::string& mat_name, const std::string& reg_name,
                  const BoundingBox3D& bbox, InterfaceType iface_type = InterfaceType::ABRUPT)
        : material_name(mat_name), region_name(reg_name), bounding_box(bbox),
          interface_type(iface_type), interface_width(0.0) {}
};

/**
 * Doping profile definition
 */
struct DopingProfile {
    std::string dopant_type;  // "n" or "p"
    double concentration;     // cm^-3
    std::function<double(double, double, double)> spatial_function;
    BoundingBox3D region;
    
    DopingProfile(const std::string& type, double conc, const BoundingBox3D& reg)
        : dopant_type(type), concentration(conc), region(reg) {
        // Default uniform doping
        spatial_function = [conc](double x, double y, double z) { return conc; };
    }
};

/**
 * Gate structure definition
 */
struct GateStructure {
    std::string gate_material;
    double gate_length;
    double gate_width;
    double gate_thickness;
    double oxide_thickness;
    std::string oxide_material;
    GateConfiguration configuration;
    std::vector<Point3D> gate_positions;
    
    GateStructure(const std::string& mat, double length, double width, double thickness)
        : gate_material(mat), gate_length(length), gate_width(width), 
          gate_thickness(thickness), oxide_thickness(2e-9), 
          oxide_material("SiO2"), configuration(GateConfiguration::SINGLE_GATE) {}
};

/**
 * Contact definition
 */
struct Contact {
    std::string name;
    std::string contact_type;  // "ohmic", "schottky"
    BoundingBox3D region;
    double work_function;
    double contact_resistance;
    
    Contact(const std::string& contact_name, const std::string& type, 
           const BoundingBox3D& reg, double work_func = 4.5)
        : name(contact_name), contact_type(type), region(reg), 
          work_function(work_func), contact_resistance(1e-8) {}
};

/**
 * Advanced device structure builder
 */
class AdvancedDeviceStructure {
public:
    AdvancedDeviceStructure(DeviceStructureType type);
    ~AdvancedDeviceStructure() = default;
    
    // Device structure configuration
    void set_device_type(DeviceStructureType type);
    void set_dimensions(double length, double width, double height);
    void set_substrate_material(const std::string& material);
    
    // Material region management
    void add_material_region(const MaterialRegion& region);
    void add_doping_profile(const DopingProfile& profile);
    void add_gate_structure(const GateStructure& gate);
    void add_contact(const Contact& contact);
    
    // FinFET specific methods
    void configure_finfet(double fin_width, double fin_height, int num_fins);
    void set_fin_pitch(double pitch);
    void set_fin_orientation(const std::string& orientation);
    
    // Gate-All-Around specific methods
    void configure_gate_all_around(double channel_diameter, double gate_length);
    void set_channel_material(const std::string& material);
    void set_spacer_material(const std::string& material, double thickness);
    
    // Nanowire specific methods
    void configure_nanowire(double diameter, double length, const std::string& orientation);
    void set_nanowire_cross_section(const std::string& shape);  // "circular", "rectangular", "hexagonal"
    
    // Heterojunction methods
    void add_heterojunction_layer(const std::string& material, double thickness, 
                                 double composition = 1.0);
    void set_interface_properties(const std::string& interface_name, 
                                 InterfaceType type, double width);
    
    // Quantum well methods
    void add_quantum_well(const std::string& well_material, double well_width,
                         const std::string& barrier_material, double barrier_width);
    void set_quantum_well_stack(int num_wells);
    
    // 3D geometry methods
    void enable_3d_simulation(bool enable);
    void set_mesh_refinement_regions(const std::vector<BoundingBox3D>& regions);
    void set_symmetry_conditions(const std::vector<std::string>& symmetries);
    
    // Material interface handling
    void set_interface_charge_density(const std::string& interface_name, double charge_density);
    void set_interface_trap_density(const std::string& interface_name, double trap_density);
    void enable_interface_roughness(const std::string& interface_name, double rms_roughness);
    
    // Advanced features
    void enable_strain_effects(bool enable);
    void set_thermal_boundary_conditions(const std::map<std::string, double>& temperatures);
    void enable_self_heating(bool enable);
    
    // Mesh generation
    std::shared_ptr<Mesh> generate_mesh(int refinement_level = 0);
    void set_mesh_parameters(const std::map<std::string, double>& params);
    
    // Validation and analysis
    bool validate_structure() const;
    std::map<std::string, double> analyze_geometry() const;
    std::vector<std::string> get_material_interfaces() const;
    
    // Export methods
    void export_to_gmsh(const std::string& filename) const;
    void export_to_tcad(const std::string& filename) const;
    void export_structure_summary(const std::string& filename) const;
    
    // Getters
    DeviceStructureType get_device_type() const { return device_type_; }
    std::vector<MaterialRegion> get_material_regions() const { return material_regions_; }
    std::vector<DopingProfile> get_doping_profiles() const { return doping_profiles_; }
    std::vector<GateStructure> get_gate_structures() const { return gate_structures_; }
    std::vector<Contact> get_contacts() const { return contacts_; }
    double get_device_width() const { return device_width_; }
    double get_device_height() const { return device_height_; }
    
private:
    DeviceStructureType device_type_;
    
    // Device dimensions
    double device_length_;
    double device_width_;
    double device_height_;
    std::string substrate_material_;
    
    // Structure components
    std::vector<MaterialRegion> material_regions_;
    std::vector<DopingProfile> doping_profiles_;
    std::vector<GateStructure> gate_structures_;
    std::vector<Contact> contacts_;
    
    // FinFET parameters
    double fin_width_;
    double fin_height_;
    int num_fins_;
    double fin_pitch_;
    std::string fin_orientation_;
    
    // GAA parameters
    double channel_diameter_;
    std::string channel_material_;
    std::string spacer_material_;
    double spacer_thickness_;
    
    // Nanowire parameters
    double nanowire_diameter_;
    std::string nanowire_orientation_;
    std::string nanowire_cross_section_;
    
    // Heterojunction layers
    struct HeterojunctionLayer {
        std::string material;
        double thickness;
        double composition;
    };
    std::vector<HeterojunctionLayer> hetero_layers_;
    
    // Interface properties
    std::map<std::string, InterfaceType> interface_types_;
    std::map<std::string, double> interface_widths_;
    std::map<std::string, double> interface_charges_;
    std::map<std::string, double> interface_traps_;
    std::map<std::string, double> interface_roughness_;
    
    // Quantum well parameters
    struct QuantumWellLayer {
        std::string well_material;
        double well_width;
        std::string barrier_material;
        double barrier_width;
    };
    std::vector<QuantumWellLayer> quantum_wells_;
    int num_quantum_wells_;
    
    // 3D simulation parameters
    bool enable_3d_;
    std::vector<BoundingBox3D> refinement_regions_;
    std::vector<std::string> symmetry_conditions_;
    
    // Advanced physics
    bool enable_strain_;
    bool enable_self_heating_;
    std::map<std::string, double> thermal_boundaries_;
    
    // Mesh parameters
    std::map<std::string, double> mesh_params_;
    
    // Helper methods
    void initialize_default_parameters();
    void validate_dimensions() const;
    void validate_materials() const;
    void validate_interfaces() const;
    BoundingBox3D calculate_device_bounding_box() const;
    std::vector<Point3D> generate_gate_positions() const;
    void setup_default_contacts();
    void apply_symmetry_conditions();
    
    // Geometry generation helpers
    void generate_finfet_geometry();
    void generate_gaa_geometry();
    void generate_nanowire_geometry();
    void generate_heterojunction_geometry();
    void generate_quantum_well_geometry();
    
    // Material property interpolation
    double interpolate_material_property(const std::string& property,
                                       const Point3D& position) const;
    double calculate_interface_profile(const std::string& interface_name,
                                     double distance) const;

    // GMSH export helpers
    void export_finfet_geometry_to_gmsh(std::ofstream& file) const;
    void export_gaa_geometry_to_gmsh(std::ofstream& file) const;
    void export_nanowire_geometry_to_gmsh(std::ofstream& file) const;
};

/**
 * Device structure factory
 */
class DeviceStructureFactory {
public:
    static std::unique_ptr<AdvancedDeviceStructure> create_finfet(
        double fin_width, double fin_height, double gate_length, int num_fins);
    
    static std::unique_ptr<AdvancedDeviceStructure> create_gate_all_around(
        double channel_diameter, double gate_length, double channel_length);
    
    static std::unique_ptr<AdvancedDeviceStructure> create_nanowire_transistor(
        double diameter, double length, const std::string& material);
    
    static std::unique_ptr<AdvancedDeviceStructure> create_heterojunction_bipolar(
        const std::vector<std::string>& materials, const std::vector<double>& thicknesses);
    
    static std::unique_ptr<AdvancedDeviceStructure> create_quantum_well_device(
        const std::string& well_material, double well_width,
        const std::string& barrier_material, double barrier_width, int num_wells);
    
    static std::unique_ptr<AdvancedDeviceStructure> create_tunnel_fet(
        double channel_length, const std::string& source_material,
        const std::string& channel_material, const std::string& drain_material);
};

} // namespace simulator
