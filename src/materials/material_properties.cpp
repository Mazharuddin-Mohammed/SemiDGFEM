/**
 * Implementation of Material Properties Database
 * Temperature-dependent parameters for semiconductor device simulation
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "material_properties.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace simulator {
namespace materials {

// Global material database instance
static std::unique_ptr<MaterialDatabase> global_database = nullptr;

MaterialDatabase& get_material_database() {
    if (!global_database) {
        global_database = std::make_unique<MaterialDatabase>();
    }
    return *global_database;
}

MaterialProperties::MaterialProperties(MaterialType mat_type) : type(mat_type) {
    // Initialize with default values based on material type
    switch (mat_type) {
        case MaterialType::SILICON:
            name = "Silicon";
            break;
        case MaterialType::GERMANIUM:
            name = "Germanium";
            break;
        case MaterialType::GALLIUM_ARSENIDE:
            name = "Gallium Arsenide";
            break;
        case MaterialType::SILICON_CARBIDE:
            name = "Silicon Carbide";
            break;
        case MaterialType::GALLIUM_NITRIDE:
            name = "Gallium Nitride";
            break;
        case MaterialType::INDIUM_GALLIUM_ARSENIDE:
            name = "Indium Gallium Arsenide";
            break;
        default:
            name = "Custom Material";
            break;
    }
}

MaterialDatabase::MaterialDatabase() {
    // Initialize all material properties
    initialize_silicon_properties();
    initialize_germanium_properties();
    initialize_gallium_arsenide_properties();
    initialize_silicon_carbide_properties();
    initialize_gallium_nitride_properties();
    initialize_indium_gallium_arsenide_properties();
}

void MaterialDatabase::initialize_silicon_properties() {
    MaterialProperties si(MaterialType::SILICON);
    
    // Basic properties
    si.lattice_constant = 5.431e-10;           // m
    si.density = 2329.0;                       // kg/m³
    si.atomic_mass = 28.0855 * 1.66054e-27;    // kg
    
    // Electronic properties
    si.bandgap_300K = 1.12;                    // eV
    si.electron_affinity = 4.05;               // eV
    si.dielectric_constant = 11.7;             // Relative permittivity
    si.intrinsic_concentration_300K = 1.45e16; // m⁻³
    
    // Effective masses
    si.electron_effective_mass_density_of_states = 1.08;
    si.hole_effective_mass_density_of_states = 0.81;
    si.electron_effective_mass_conductivity = 0.26;
    si.hole_effective_mass_conductivity = 0.39;
    
    // Mobility parameters (Caughey-Thomas model)
    si.electron_mobility_min = 0.0068;         // m²/V·s
    si.electron_mobility_max = 0.1350;         // m²/V·s
    si.electron_mobility_ref_doping = 1.26e23; // m⁻³
    si.electron_mobility_alpha = 0.88;
    
    si.hole_mobility_min = 0.0044;             // m²/V·s
    si.hole_mobility_max = 0.0480;             // m²/V·s
    si.hole_mobility_ref_doping = 2.35e23;     // m⁻³
    si.hole_mobility_alpha = 0.719;
    
    // Velocity saturation
    si.electron_saturation_velocity = 1.07e5;  // m/s
    si.hole_saturation_velocity = 8.37e4;      // m/s
    si.electron_critical_field = 1.01e4;       // V/m
    si.hole_critical_field = 1.24e4;           // V/m
    
    // Thermal properties
    si.thermal_conductivity = 150.0;           // W/m·K
    si.specific_heat = 700.0;                  // J/kg·K
    si.thermal_expansion = 2.6e-6;             // 1/K
    
    // Recombination parameters
    si.srh_electron_lifetime = 1e-6;           // s
    si.srh_hole_lifetime = 1e-6;               // s
    si.radiative_recombination_coefficient = 1.1e-21; // m³/s
    si.auger_electron_coefficient = 2.8e-43;   // m⁶/s
    si.auger_hole_coefficient = 9.9e-44;       // m⁶/s
    
    // Impact ionization parameters
    si.electron_impact_ionization_a = 7.03e5;  // 1/m
    si.electron_impact_ionization_b = 1.231e6; // V/m
    si.hole_impact_ionization_a = 1.582e6;     // 1/m
    si.hole_impact_ionization_b = 2.036e6;     // V/m
    
    // Temperature-dependent functions
    si.bandgap_temperature_dependence = silicon_bandgap_temperature;
    si.intrinsic_concentration_temperature_dependence = silicon_intrinsic_concentration_temperature;
    si.electron_mobility_temperature_dependence = silicon_mobility_temperature;
    si.hole_mobility_temperature_dependence = silicon_mobility_temperature;
    
    database_[MaterialType::SILICON] = si;
}

void MaterialDatabase::initialize_gallium_arsenide_properties() {
    MaterialProperties gaas(MaterialType::GALLIUM_ARSENIDE);
    
    // Basic properties
    gaas.lattice_constant = 5.653e-10;         // m
    gaas.density = 5317.0;                     // kg/m³
    gaas.atomic_mass = 144.645 * 1.66054e-27;  // kg (average of Ga and As)
    
    // Electronic properties
    gaas.bandgap_300K = 1.424;                 // eV
    gaas.electron_affinity = 4.07;             // eV
    gaas.dielectric_constant = 13.1;           // Relative permittivity
    gaas.intrinsic_concentration_300K = 1.79e12; // m⁻³
    
    // Effective masses
    gaas.electron_effective_mass_density_of_states = 0.063;
    gaas.hole_effective_mass_density_of_states = 0.51;
    gaas.electron_effective_mass_conductivity = 0.063;
    gaas.hole_effective_mass_conductivity = 0.51;
    
    // Mobility parameters
    gaas.electron_mobility_min = 0.0050;       // m²/V·s
    gaas.electron_mobility_max = 0.8500;       // m²/V·s
    gaas.electron_mobility_ref_doping = 1.0e23; // m⁻³
    gaas.electron_mobility_alpha = 1.0;
    
    gaas.hole_mobility_min = 0.0020;           // m²/V·s
    gaas.hole_mobility_max = 0.0400;           // m²/V·s
    gaas.hole_mobility_ref_doping = 1.0e23;    // m⁻³
    gaas.hole_mobility_alpha = 2.1;
    
    // Velocity saturation
    gaas.electron_saturation_velocity = 1.2e5; // m/s
    gaas.hole_saturation_velocity = 6.0e4;     // m/s
    gaas.electron_critical_field = 4.0e3;      // V/m
    gaas.hole_critical_field = 4.0e3;          // V/m
    
    // Thermal properties
    gaas.thermal_conductivity = 55.0;          // W/m·K
    gaas.specific_heat = 350.0;                // J/kg·K
    gaas.thermal_expansion = 5.7e-6;           // 1/K
    
    // Recombination parameters
    gaas.srh_electron_lifetime = 1e-9;         // s
    gaas.srh_hole_lifetime = 1e-9;             // s
    gaas.radiative_recombination_coefficient = 7.2e-16; // m³/s
    gaas.auger_electron_coefficient = 1e-42;   // m⁶/s
    gaas.auger_hole_coefficient = 1e-42;       // m⁶/s
    
    // Impact ionization parameters
    gaas.electron_impact_ionization_a = 5.0e5; // 1/m
    gaas.electron_impact_ionization_b = 9.5e5; // V/m
    gaas.hole_impact_ionization_a = 5.0e5;     // 1/m
    gaas.hole_impact_ionization_b = 9.5e5;     // V/m
    
    // Temperature-dependent functions
    gaas.bandgap_temperature_dependence = gallium_arsenide_bandgap_temperature;
    gaas.intrinsic_concentration_temperature_dependence = gallium_arsenide_intrinsic_concentration_temperature;
    gaas.electron_mobility_temperature_dependence = gallium_arsenide_mobility_temperature;
    gaas.hole_mobility_temperature_dependence = gallium_arsenide_mobility_temperature;
    
    database_[MaterialType::GALLIUM_ARSENIDE] = gaas;
}

void MaterialDatabase::initialize_germanium_properties() {
    MaterialProperties ge(MaterialType::GERMANIUM);
    
    // Basic properties - simplified implementation
    ge.lattice_constant = 5.658e-10;
    ge.density = 5323.0;
    ge.bandgap_300K = 0.66;
    ge.dielectric_constant = 16.0;
    ge.intrinsic_concentration_300K = 2.4e19;
    
    // Simplified mobility values
    ge.electron_mobility_max = 0.3900;
    ge.hole_mobility_max = 0.1900;
    
    database_[MaterialType::GERMANIUM] = ge;
}

void MaterialDatabase::initialize_silicon_carbide_properties() {
    MaterialProperties sic(MaterialType::SILICON_CARBIDE);
    
    // Basic properties - simplified implementation
    sic.lattice_constant = 3.073e-10;
    sic.density = 3210.0;
    sic.bandgap_300K = 3.26;
    sic.dielectric_constant = 9.7;
    sic.intrinsic_concentration_300K = 8.2e8;
    
    // High temperature, wide bandgap properties
    sic.electron_mobility_max = 0.0950;
    sic.hole_mobility_max = 0.0120;
    sic.thermal_conductivity = 490.0;
    
    database_[MaterialType::SILICON_CARBIDE] = sic;
}

void MaterialDatabase::initialize_gallium_nitride_properties() {
    MaterialProperties gan(MaterialType::GALLIUM_NITRIDE);
    
    // Basic properties - simplified implementation
    gan.lattice_constant = 3.189e-10;
    gan.density = 6150.0;
    gan.bandgap_300K = 3.39;
    gan.dielectric_constant = 9.0;
    gan.intrinsic_concentration_300K = 1.9e6;
    
    // Wide bandgap properties
    gan.electron_mobility_max = 0.1300;
    gan.hole_mobility_max = 0.0030;
    gan.thermal_conductivity = 130.0;
    
    database_[MaterialType::GALLIUM_NITRIDE] = gan;
}

void MaterialDatabase::initialize_indium_gallium_arsenide_properties() {
    MaterialProperties ingaas(MaterialType::INDIUM_GALLIUM_ARSENIDE);
    
    // Basic properties - simplified implementation for In0.53Ga0.47As
    ingaas.lattice_constant = 5.869e-10;
    ingaas.density = 5500.0;
    ingaas.bandgap_300K = 0.75;
    ingaas.dielectric_constant = 13.9;
    ingaas.intrinsic_concentration_300K = 5.7e18;
    
    // High mobility properties
    ingaas.electron_mobility_max = 1.2000;
    ingaas.hole_mobility_max = 0.0450;
    
    database_[MaterialType::INDIUM_GALLIUM_ARSENIDE] = ingaas;
}

MaterialProperties MaterialDatabase::get_material_properties(MaterialType type) const {
    auto it = database_.find(type);
    if (it != database_.end()) {
        return it->second;
    }
    throw std::runtime_error("Material type not found in database");
}

MaterialProperties MaterialDatabase::get_material_properties(const std::string& name) const {
    // First check custom materials
    auto custom_it = custom_materials_.find(name);
    if (custom_it != custom_materials_.end()) {
        return custom_it->second;
    }
    
    // Then check standard materials by name
    for (const auto& pair : database_) {
        if (pair.second.name == name) {
            return pair.second;
        }
    }
    
    throw std::runtime_error("Material '" + name + "' not found in database");
}

void MaterialDatabase::add_custom_material(const std::string& name, const MaterialProperties& properties) {
    custom_materials_[name] = properties;
}

double MaterialDatabase::get_bandgap(MaterialType type, double temperature) const {
    auto props = get_material_properties(type);
    if (props.bandgap_temperature_dependence) {
        return props.bandgap_temperature_dependence(temperature);
    }
    return props.bandgap_300K;
}

double MaterialDatabase::get_intrinsic_concentration(MaterialType type, double temperature) const {
    auto props = get_material_properties(type);
    if (props.intrinsic_concentration_temperature_dependence) {
        return props.intrinsic_concentration_temperature_dependence(temperature);
    }
    return props.intrinsic_concentration_300K;
}

double MaterialDatabase::get_electron_mobility(MaterialType type, double temperature, double doping) const {
    auto props = get_material_properties(type);
    
    // Calculate doping-dependent mobility
    double mu_doping = calculate_caughey_thomas_mobility(
        props.electron_mobility_min, props.electron_mobility_max,
        props.electron_mobility_ref_doping, props.electron_mobility_alpha,
        doping, temperature);
    
    // Apply temperature dependence
    if (props.electron_mobility_temperature_dependence) {
        double temp_factor = props.electron_mobility_temperature_dependence(temperature);
        return mu_doping * temp_factor;
    }
    
    return mu_doping;
}

double MaterialDatabase::get_hole_mobility(MaterialType type, double temperature, double doping) const {
    auto props = get_material_properties(type);
    
    // Calculate doping-dependent mobility
    double mu_doping = calculate_caughey_thomas_mobility(
        props.hole_mobility_min, props.hole_mobility_max,
        props.hole_mobility_ref_doping, props.hole_mobility_alpha,
        doping, temperature);
    
    // Apply temperature dependence
    if (props.hole_mobility_temperature_dependence) {
        double temp_factor = props.hole_mobility_temperature_dependence(temperature);
        return mu_doping * temp_factor;
    }
    
    return mu_doping;
}

double MaterialDatabase::calculate_caughey_thomas_mobility(double mu_min, double mu_max, 
                                                         double N_ref, double alpha, 
                                                         double doping, double temperature) const {
    // Caughey-Thomas mobility model
    double mu_lattice = mu_max * std::pow(300.0 / temperature, 2.4);
    return mu_min + (mu_lattice - mu_min) / (1.0 + std::pow(doping / N_ref, alpha));
}

double MaterialDatabase::calculate_velocity_saturation_mobility(double mu_low_field, double v_sat, 
                                                               double E_field) const {
    // Velocity saturation model
    return mu_low_field / (1.0 + mu_low_field * E_field / v_sat);
}

double MaterialDatabase::calculate_srh_recombination(double n, double p, double ni, 
                                                    double tau_n, double tau_p) const {
    // Shockley-Read-Hall recombination
    return (n * p - ni * ni) / (tau_p * (n + ni) + tau_n * (p + ni));
}

double MaterialDatabase::calculate_radiative_recombination(double n, double p, double ni, 
                                                          double B_rad) const {
    // Radiative recombination
    return B_rad * (n * p - ni * ni);
}

double MaterialDatabase::calculate_auger_recombination(double n, double p, double ni,
                                                      double C_n, double C_p) const {
    // Auger recombination
    return (C_n * n + C_p * p) * (n * p - ni * ni);
}

double MaterialDatabase::calculate_impact_ionization_rate(double E_field, double a, double b) const {
    // Impact ionization rate (Chynoweth's law)
    if (E_field <= 0.0) return 0.0;
    return a * std::exp(-b / E_field);
}

std::vector<std::string> MaterialDatabase::get_available_materials() const {
    std::vector<std::string> materials;
    
    // Add standard materials
    for (const auto& pair : database_) {
        materials.push_back(pair.second.name);
    }
    
    // Add custom materials
    for (const auto& pair : custom_materials_) {
        materials.push_back(pair.first);
    }
    
    return materials;
}

bool MaterialDatabase::is_material_available(const std::string& name) const {
    // Check custom materials
    if (custom_materials_.find(name) != custom_materials_.end()) {
        return true;
    }
    
    // Check standard materials
    for (const auto& pair : database_) {
        if (pair.second.name == name) {
            return true;
        }
    }
    
    return false;
}

void MaterialDatabase::print_material_properties(MaterialType type) const {
    try {
        auto props = get_material_properties(type);
        std::cout << "Material: " << props.name << std::endl;
        std::cout << "  Bandgap (300K): " << props.bandgap_300K << " eV" << std::endl;
        std::cout << "  Dielectric constant: " << props.dielectric_constant << std::endl;
        std::cout << "  Electron mobility (max): " << props.electron_mobility_max << " m²/V·s" << std::endl;
        std::cout << "  Hole mobility (max): " << props.hole_mobility_max << " m²/V·s" << std::endl;
        std::cout << "  Intrinsic concentration (300K): " << props.intrinsic_concentration_300K << " m⁻³" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
}

// Temperature dependence functions for Silicon
double MaterialDatabase::silicon_bandgap_temperature(double T) {
    // Varshni equation for Silicon
    double Eg0 = 1.17;  // eV at 0K
    double alpha = 4.73e-4; // eV/K
    double beta = 636.0; // K
    return Eg0 - (alpha * T * T) / (T + beta);
}

double MaterialDatabase::silicon_intrinsic_concentration_temperature(double T) {
    // Temperature dependence of intrinsic concentration
    double Eg = silicon_bandgap_temperature(T);
    double kT = 8.617e-5 * T; // eV
    double ni_300 = 1.45e16; // m⁻³
    return ni_300 * std::pow(T / 300.0, 1.5) * std::exp(-Eg / (2 * kT) + 1.12 / (2 * 8.617e-5 * 300.0));
}

double MaterialDatabase::silicon_mobility_temperature(double T) {
    // Temperature dependence of mobility
    return std::pow(300.0 / T, 2.4);
}

// Temperature dependence functions for GaAs
double MaterialDatabase::gallium_arsenide_bandgap_temperature(double T) {
    // Varshni equation for GaAs
    double Eg0 = 1.519;  // eV at 0K
    double alpha = 5.405e-4; // eV/K
    double beta = 204.0; // K
    return Eg0 - (alpha * T * T) / (T + beta);
}

double MaterialDatabase::gallium_arsenide_intrinsic_concentration_temperature(double T) {
    // Temperature dependence of intrinsic concentration for GaAs
    double Eg = gallium_arsenide_bandgap_temperature(T);
    double kT = 8.617e-5 * T; // eV
    double ni_300 = 1.79e12; // m⁻³
    return ni_300 * std::pow(T / 300.0, 1.5) * std::exp(-Eg / (2 * kT) + 1.424 / (2 * 8.617e-5 * 300.0));
}

double MaterialDatabase::gallium_arsenide_mobility_temperature(double T) {
    // Temperature dependence of mobility for GaAs
    return std::pow(300.0 / T, 1.0);
}

} // namespace materials
} // namespace simulator

// C interface implementation
extern "C" {
    void* create_material_database() {
        try {
            return new simulator::materials::MaterialDatabase();
        } catch (...) {
            return nullptr;
        }
    }

    void destroy_material_database(void* db) {
        delete static_cast<simulator::materials::MaterialDatabase*>(db);
    }

    int material_database_get_properties(void* db, int material_type,
                                        double* bandgap, double* dielectric_constant,
                                        double* electron_mobility, double* hole_mobility) {
        if (!db || !bandgap || !dielectric_constant || !electron_mobility || !hole_mobility) {
            return -1;
        }

        try {
            auto* database = static_cast<simulator::materials::MaterialDatabase*>(db);
            auto props = database->get_material_properties(
                static_cast<simulator::materials::MaterialType>(material_type));

            *bandgap = props.bandgap_300K;
            *dielectric_constant = props.dielectric_constant;
            *electron_mobility = props.electron_mobility_max;
            *hole_mobility = props.hole_mobility_max;

            return 0;
        } catch (...) {
            return -1;
        }
    }

    double material_database_get_bandgap(void* db, int material_type, double temperature) {
        if (!db) return 0.0;
        try {
            auto* database = static_cast<simulator::materials::MaterialDatabase*>(db);
            return database->get_bandgap(
                static_cast<simulator::materials::MaterialType>(material_type), temperature);
        } catch (...) {
            return 0.0;
        }
    }

    double material_database_get_intrinsic_concentration(void* db, int material_type, double temperature) {
        if (!db) return 0.0;
        try {
            auto* database = static_cast<simulator::materials::MaterialDatabase*>(db);
            return database->get_intrinsic_concentration(
                static_cast<simulator::materials::MaterialType>(material_type), temperature);
        } catch (...) {
            return 0.0;
        }
    }

    double material_database_get_electron_mobility(void* db, int material_type,
                                                   double temperature, double doping) {
        if (!db) return 0.0;
        try {
            auto* database = static_cast<simulator::materials::MaterialDatabase*>(db);
            return database->get_electron_mobility(
                static_cast<simulator::materials::MaterialType>(material_type), temperature, doping);
        } catch (...) {
            return 0.0;
        }
    }

    double material_database_get_hole_mobility(void* db, int material_type,
                                              double temperature, double doping) {
        if (!db) return 0.0;
        try {
            auto* database = static_cast<simulator::materials::MaterialDatabase*>(db);
            return database->get_hole_mobility(
                static_cast<simulator::materials::MaterialType>(material_type), temperature, doping);
        } catch (...) {
            return 0.0;
        }
    }

    double material_database_calculate_srh_recombination(void* db, double n, double p, double ni,
                                                        double tau_n, double tau_p) {
        if (!db) return 0.0;
        try {
            auto* database = static_cast<simulator::materials::MaterialDatabase*>(db);
            return database->calculate_srh_recombination(n, p, ni, tau_n, tau_p);
        } catch (...) {
            return 0.0;
        }
    }

    double material_database_calculate_impact_ionization(void* db, double E_field,
                                                        double a, double b) {
        if (!db) return 0.0;
        try {
            auto* database = static_cast<simulator::materials::MaterialDatabase*>(db);
            return database->calculate_impact_ionization_rate(E_field, a, b);
        } catch (...) {
            return 0.0;
        }
    }
}
