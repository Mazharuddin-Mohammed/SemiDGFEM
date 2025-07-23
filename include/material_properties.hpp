/**
 * Comprehensive Material Properties Database
 * Temperature-dependent parameters for semiconductor device simulation
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include <functional>
#include <memory>

namespace simulator {
namespace materials {

/**
 * @brief Material types supported by the simulator
 */
enum class MaterialType {
    SILICON = 0,
    GERMANIUM = 1,
    GALLIUM_ARSENIDE = 2,
    SILICON_CARBIDE = 3,
    GALLIUM_NITRIDE = 4,
    INDIUM_GALLIUM_ARSENIDE = 5,
    CUSTOM = 99
};

/**
 * @brief Temperature-dependent parameter function type
 */
using TemperatureDependentParameter = std::function<double(double)>;

/**
 * @brief Comprehensive material properties structure
 */
struct MaterialProperties {
    // Basic properties
    std::string name;
    MaterialType type;
    double lattice_constant;        // Lattice constant (m)
    double density;                 // Density (kg/m³)
    double atomic_mass;             // Atomic mass (kg)
    
    // Electronic properties
    double bandgap_300K;            // Bandgap at 300K (eV)
    double electron_affinity;       // Electron affinity (eV)
    double dielectric_constant;     // Relative permittivity
    double intrinsic_concentration_300K; // Intrinsic carrier concentration at 300K (m⁻³)
    
    // Effective masses (in units of free electron mass)
    double electron_effective_mass_density_of_states;
    double hole_effective_mass_density_of_states;
    double electron_effective_mass_conductivity;
    double hole_effective_mass_conductivity;
    
    // Mobility parameters (Caughey-Thomas model)
    double electron_mobility_min;   // Minimum mobility (m²/V·s)
    double electron_mobility_max;   // Maximum mobility (m²/V·s)
    double electron_mobility_ref_doping; // Reference doping (m⁻³)
    double electron_mobility_alpha; // Mobility exponent
    
    double hole_mobility_min;       // Minimum mobility (m²/V·s)
    double hole_mobility_max;       // Maximum mobility (m²/V·s)
    double hole_mobility_ref_doping; // Reference doping (m⁻³)
    double hole_mobility_alpha;     // Mobility exponent
    
    // Velocity saturation
    double electron_saturation_velocity; // Saturation velocity (m/s)
    double hole_saturation_velocity;     // Saturation velocity (m/s)
    double electron_critical_field;     // Critical field for velocity saturation (V/m)
    double hole_critical_field;         // Critical field for velocity saturation (V/m)
    
    // Thermal properties
    double thermal_conductivity;    // Thermal conductivity (W/m·K)
    double specific_heat;           // Specific heat capacity (J/kg·K)
    double thermal_expansion;       // Thermal expansion coefficient (1/K)
    
    // Recombination parameters
    double srh_electron_lifetime;   // SRH electron lifetime (s)
    double srh_hole_lifetime;       // SRH hole lifetime (s)
    double radiative_recombination_coefficient; // Radiative recombination (m³/s)
    double auger_electron_coefficient; // Auger recombination for electrons (m⁶/s)
    double auger_hole_coefficient;     // Auger recombination for holes (m⁶/s)
    
    // Impact ionization parameters
    double electron_impact_ionization_a; // Impact ionization coefficient a (1/m)
    double electron_impact_ionization_b; // Impact ionization coefficient b (V/m)
    double hole_impact_ionization_a;     // Impact ionization coefficient a (1/m)
    double hole_impact_ionization_b;     // Impact ionization coefficient b (V/m)
    
    // Temperature-dependent functions
    TemperatureDependentParameter bandgap_temperature_dependence;
    TemperatureDependentParameter intrinsic_concentration_temperature_dependence;
    TemperatureDependentParameter electron_mobility_temperature_dependence;
    TemperatureDependentParameter hole_mobility_temperature_dependence;
    
    MaterialProperties() = default;
    MaterialProperties(MaterialType mat_type);
};

/**
 * @brief Material properties database and manager
 */
class MaterialDatabase {
public:
    MaterialDatabase();
    ~MaterialDatabase() = default;
    
    // Database access
    MaterialProperties get_material_properties(MaterialType type) const;
    MaterialProperties get_material_properties(const std::string& name) const;
    void add_custom_material(const std::string& name, const MaterialProperties& properties);
    
    // Temperature-dependent calculations
    double get_bandgap(MaterialType type, double temperature) const;
    double get_intrinsic_concentration(MaterialType type, double temperature) const;
    double get_electron_mobility(MaterialType type, double temperature, double doping) const;
    double get_hole_mobility(MaterialType type, double temperature, double doping) const;
    
    // Mobility models
    double calculate_caughey_thomas_mobility(double mu_min, double mu_max, 
                                           double N_ref, double alpha, 
                                           double doping, double temperature) const;
    
    double calculate_velocity_saturation_mobility(double mu_low_field, double v_sat, 
                                                 double E_field) const;
    
    // Recombination calculations
    double calculate_srh_recombination(double n, double p, double ni, 
                                     double tau_n, double tau_p) const;
    
    double calculate_radiative_recombination(double n, double p, double ni, 
                                           double B_rad) const;
    
    double calculate_auger_recombination(double n, double p, double ni,
                                       double C_n, double C_p) const;
    
    // Impact ionization
    double calculate_impact_ionization_rate(double E_field, double a, double b) const;
    
    // Utility methods
    std::vector<std::string> get_available_materials() const;
    bool is_material_available(const std::string& name) const;
    void print_material_properties(MaterialType type) const;

private:
    std::map<MaterialType, MaterialProperties> database_;
    std::map<std::string, MaterialProperties> custom_materials_;
    
    // Initialization methods
    void initialize_silicon_properties();
    void initialize_germanium_properties();
    void initialize_gallium_arsenide_properties();
    void initialize_silicon_carbide_properties();
    void initialize_gallium_nitride_properties();
    void initialize_indium_gallium_arsenide_properties();
    
    // Temperature dependence functions
    static double silicon_bandgap_temperature(double T);
    static double silicon_intrinsic_concentration_temperature(double T);
    static double silicon_mobility_temperature(double T);
    
    static double gallium_arsenide_bandgap_temperature(double T);
    static double gallium_arsenide_intrinsic_concentration_temperature(double T);
    static double gallium_arsenide_mobility_temperature(double T);
};

/**
 * @brief Material property interpolation for alloys and composites
 */
class MaterialInterpolator {
public:
    MaterialInterpolator() = default;
    ~MaterialInterpolator() = default;
    
    // Linear interpolation between two materials
    MaterialProperties interpolate_linear(const MaterialProperties& mat1,
                                        const MaterialProperties& mat2,
                                        double fraction) const;
    
    // Vegard's law for lattice constant
    double interpolate_lattice_constant_vegard(double a1, double a2, double x) const;
    
    // Bowing parameter interpolation for bandgap
    double interpolate_bandgap_bowing(double Eg1, double Eg2, double bowing, double x) const;
    
    // Create ternary alloy (e.g., InGaAs)
    MaterialProperties create_ternary_alloy(const MaterialProperties& mat1,
                                          const MaterialProperties& mat2,
                                          double composition,
                                          double bowing_parameter = 0.0) const;
    
    // Create quaternary alloy (e.g., InGaAsP)
    MaterialProperties create_quaternary_alloy(const MaterialProperties& mat1,
                                             const MaterialProperties& mat2,
                                             const MaterialProperties& mat3,
                                             const MaterialProperties& mat4,
                                             double x, double y) const;
};

/**
 * @brief Global material database instance
 */
extern MaterialDatabase& get_material_database();

} // namespace materials
} // namespace simulator

// C interface for Cython bindings
extern "C" {
    void* create_material_database();
    void destroy_material_database(void* db);
    
    int material_database_get_properties(void* db, int material_type,
                                        double* bandgap, double* dielectric_constant,
                                        double* electron_mobility, double* hole_mobility);
    
    double material_database_get_bandgap(void* db, int material_type, double temperature);
    double material_database_get_intrinsic_concentration(void* db, int material_type, double temperature);
    double material_database_get_electron_mobility(void* db, int material_type, 
                                                   double temperature, double doping);
    double material_database_get_hole_mobility(void* db, int material_type, 
                                              double temperature, double doping);
    
    double material_database_calculate_srh_recombination(void* db, double n, double p, double ni,
                                                        double tau_n, double tau_p);
    double material_database_calculate_impact_ionization(void* db, double E_field, 
                                                        double a, double b);
}
