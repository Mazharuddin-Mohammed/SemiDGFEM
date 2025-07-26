/**
 * Advanced Physics Models for Semiconductor Device Simulation
 * 
 * This header defines advanced physics models including:
 * - Strain effects and mechanical stress
 * - Thermal transport and self-heating
 * - Piezoelectric effects
 * - Thermoelectric coupling
 * - Optical properties and photogeneration
 * 
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>

namespace SemiDGFEM {
namespace Physics {

// Forward declarations
struct MaterialProperties;
struct MeshGeometry;

// ============================================================================
// Physical Constants
// ============================================================================

struct PhysicalConstants {
    static constexpr double k = 1.380649e-23;    // Boltzmann constant (J/K)
    static constexpr double q = 1.602176634e-19; // Elementary charge (C)
    static constexpr double h = 6.62607015e-34;  // Planck constant (J·s)
    static constexpr double hbar = h / (2.0 * M_PI); // Reduced Planck constant
    static constexpr double m0 = 9.1093837015e-31;  // Electron rest mass (kg)
    static constexpr double eps0 = 8.8541878128e-12; // Vacuum permittivity (F/m)
    static constexpr double c = 299792458.0;      // Speed of light (m/s)
};

// ============================================================================
// Strain Effects Model
// ============================================================================

/**
 * @brief Configuration for strain effects model
 */
struct StrainConfig {
    bool enable_strain_effects = true;
    bool enable_piezoresistance = true;
    bool enable_band_modification = true;
    
    // Deformation potential parameters
    double shear_deformation_potential = 9.0;  // eV
    double mass_deformation_factor = 0.1;      // Dimensionless
    
    // Piezoresistance factors
    double electron_piezoresistance_factor = -100.0;  // Typical for Silicon
    double hole_piezoresistance_factor = 70.0;        // Typical for Silicon
    
    // Numerical parameters
    double strain_tolerance = 1e-8;
    int max_strain_iterations = 50;
};

/**
 * @brief Deformation potentials for different bands
 */
struct DeformationPotentials {
    double conduction_band;        // eV
    double valence_band_heavy;     // eV
    double valence_band_light;     // eV
    double valence_band_split;     // eV
};

/**
 * @brief Elastic constants for stress-strain relationship
 */
struct ElasticConstants {
    double c11, c12, c44;  // Pa
};

/**
 * @brief Strain tensor components
 */
struct StrainTensor {
    std::vector<double> exx, eyy, ezz;  // Normal strain components
    std::vector<double> exy, exz, eyz;  // Shear strain components
};

/**
 * @brief Band structure modifications due to strain
 */
struct BandStructureModification {
    std::vector<double> conduction_band_shift;      // eV
    std::vector<double> valence_band_shift_heavy;   // eV
    std::vector<double> valence_band_shift_light;   // eV
    std::vector<double> valence_band_shift_split;   // eV
    std::vector<double> effective_mass_modification; // Dimensionless factor
};

/**
 * @brief Mobility modifications due to strain
 */
struct MobilityModification {
    std::vector<double> electron_mobility_factor;  // Dimensionless
    std::vector<double> hole_mobility_factor;      // Dimensionless
};

/**
 * @brief Strain effects model class
 */
class StrainEffectsModel {
private:
    StrainConfig config_;
    MaterialProperties material_;
    DeformationPotentials deformation_potentials_;
    ElasticConstants elastic_constants_;
    
public:
    StrainEffectsModel(const StrainConfig& config = StrainConfig{},
                      const MaterialProperties& material = MaterialProperties{});
    
    // Strain calculation methods
    StrainTensor calculate_strain_tensor(
        const std::vector<double>& displacement_x,
        const std::vector<double>& displacement_y,
        const std::vector<double>& displacement_z,
        const MeshGeometry& mesh) const;
    
    // Band structure modification
    BandStructureModification calculate_band_modification(
        const StrainTensor& strain) const;
    
    // Mobility modification
    MobilityModification calculate_mobility_modification(
        const StrainTensor& strain) const;
    
    // Configuration methods
    void set_config(const StrainConfig& config) { config_ = config; }
    StrainConfig get_config() const { return config_; }
    
private:
    void initialize_strain_parameters();
    double calculate_gradient_component(
        const std::vector<double>& field,
        const std::vector<size_t>& neighbors,
        const MeshGeometry& mesh,
        int field_component,
        int spatial_component) const;
};

// ============================================================================
// Thermal Transport Model
// ============================================================================

/**
 * @brief Configuration for thermal transport model
 */
struct ThermalConfig {
    bool enable_thermal_transport = true;
    bool enable_joule_heating = true;
    bool enable_thermal_coupling = true;
    bool enable_temperature_dependent_properties = true;
    
    // Thermal parameters
    double thermal_diffusivity = 8.8e-5;  // m²/s for Silicon
    double ambient_temperature = 300.0;    // K
    double thermal_time_constant = 1e-6;   // s
    
    // Boundary conditions
    double thermal_boundary_resistance = 1e-4;  // K·m²/W
    
    // Numerical parameters
    double thermal_tolerance = 1e-6;
    int max_thermal_iterations = 100;
    double thermal_time_step = 1e-9;  // s
};

/**
 * @brief Thermal properties of materials
 */
struct ThermalProperties {
    double thermal_conductivity;  // W/m·K
    double specific_heat;         // J/kg·K
    double density;              // kg/m³
    double thermal_expansion;    // 1/K
};

/**
 * @brief Thermal boundary conditions
 */
class ThermalBoundaryConditions {
private:
    std::map<size_t, double> boundary_temperatures_;
    std::map<size_t, double> heat_flux_boundaries_;
    
public:
    void set_temperature_boundary(size_t point_id, double temperature) {
        boundary_temperatures_[point_id] = temperature;
    }
    
    void set_heat_flux_boundary(size_t point_id, double heat_flux) {
        heat_flux_boundaries_[point_id] = heat_flux;
    }
    
    bool is_boundary_point(size_t point_id) const {
        return boundary_temperatures_.find(point_id) != boundary_temperatures_.end() ||
               heat_flux_boundaries_.find(point_id) != heat_flux_boundaries_.end();
    }
    
    double get_temperature(size_t point_id) const {
        auto it = boundary_temperatures_.find(point_id);
        return (it != boundary_temperatures_.end()) ? it->second : 300.0;
    }
    
    double get_heat_flux(size_t point_id) const {
        auto it = heat_flux_boundaries_.find(point_id);
        return (it != heat_flux_boundaries_.end()) ? it->second : 0.0;
    }
};

/**
 * @brief Thermal coupling effects
 */
struct ThermalCoupling {
    std::vector<double> bandgap_modification;      // eV
    std::vector<double> mobility_modification;     // Dimensionless factor
    std::vector<double> thermal_voltage;           // V
    std::vector<double> thermal_diffusion_length;  // m
};

/**
 * @brief Thermal transport model class
 */
class ThermalTransportModel {
private:
    ThermalConfig config_;
    MaterialProperties material_;
    ThermalProperties thermal_properties_;
    
public:
    ThermalTransportModel(const ThermalConfig& config = ThermalConfig{},
                         const MaterialProperties& material = MaterialProperties{});
    
    // Heat equation solver
    std::vector<double> solve_heat_equation(
        const std::vector<double>& initial_temperature,
        const std::vector<double>& heat_generation,
        const ThermalBoundaryConditions& boundary_conditions,
        const MeshGeometry& mesh,
        double time_step,
        int num_time_steps) const;
    
    // Joule heating calculation
    std::vector<double> calculate_joule_heating(
        const std::vector<double>& current_density_x,
        const std::vector<double>& current_density_y,
        const std::vector<double>& electric_field_x,
        const std::vector<double>& electric_field_y) const;
    
    // Thermal coupling effects
    ThermalCoupling calculate_thermal_coupling(
        const std::vector<double>& temperature,
        const std::vector<double>& carrier_density_n,
        const std::vector<double>& carrier_density_p) const;
    
    // Configuration methods
    void set_config(const ThermalConfig& config) { config_ = config; }
    ThermalConfig get_config() const { return config_; }
    
    ThermalProperties get_thermal_properties() const { return thermal_properties_; }
    
private:
    void initialize_thermal_parameters();
    double calculate_thermal_diffusion(
        const std::vector<double>& temperature,
        size_t point_index,
        const MeshGeometry& mesh) const;
};

// ============================================================================
// Piezoelectric Effects Model
// ============================================================================

/**
 * @brief Configuration for piezoelectric effects
 */
struct PiezoelectricConfig {
    bool enable_piezoelectric_effects = true;
    bool enable_spontaneous_polarization = true;
    bool enable_piezoelectric_polarization = true;
    
    // Piezoelectric constants (C/m² for wurtzite structure)
    double e31 = -0.49;  // C/m²
    double e33 = 0.73;   // C/m²
    double e15 = -0.40;  // C/m²
    
    // Spontaneous polarization
    double spontaneous_polarization = -0.029;  // C/m² for GaN
    
    // Numerical parameters
    double piezo_tolerance = 1e-8;
    int max_piezo_iterations = 50;
};

/**
 * @brief Piezoelectric tensor components
 */
struct PiezoelectricTensor {
    std::vector<double> e31, e33, e15;  // Piezoelectric constants
};

/**
 * @brief Polarization fields
 */
struct PolarizationField {
    std::vector<double> Px, Py, Pz;  // Polarization components (C/m²)
    std::vector<double> bound_charge_density;  // C/m³
};

/**
 * @brief Piezoelectric effects model class
 */
class PiezoelectricModel {
private:
    PiezoelectricConfig config_;
    MaterialProperties material_;
    PiezoelectricTensor piezo_tensor_;
    
public:
    PiezoelectricModel(const PiezoelectricConfig& config = PiezoelectricConfig{},
                      const MaterialProperties& material = MaterialProperties{});
    
    // Polarization calculation
    PolarizationField calculate_polarization(
        const StrainTensor& strain,
        const std::vector<double>& electric_field_x,
        const std::vector<double>& electric_field_y,
        const std::vector<double>& electric_field_z) const;
    
    // Bound charge calculation
    std::vector<double> calculate_bound_charge_density(
        const PolarizationField& polarization,
        const MeshGeometry& mesh) const;
    
    // Interface charge calculation
    std::vector<double> calculate_interface_charge(
        const PolarizationField& polarization_1,
        const PolarizationField& polarization_2,
        const std::vector<size_t>& interface_points) const;
    
    // Configuration methods
    void set_config(const PiezoelectricConfig& config) { config_ = config; }
    PiezoelectricConfig get_config() const { return config_; }
    
private:
    void initialize_piezoelectric_parameters();
};

// ============================================================================
// Material Properties Structure
// ============================================================================

/**
 * @brief Material properties for advanced physics models
 */
struct MaterialProperties {
    std::string name = "Silicon";
    
    // Basic properties
    double bandgap = 1.12;           // eV at 300K
    double electron_affinity = 4.05; // eV
    double dielectric_constant = 11.7; // Relative permittivity
    
    // Effective masses (in units of m0)
    double electron_effective_mass = 0.26;
    double hole_effective_mass_heavy = 0.49;
    double hole_effective_mass_light = 0.16;
    
    // Mobility parameters
    double electron_mobility = 1350.0;  // cm²/V·s at 300K
    double hole_mobility = 480.0;       // cm²/V·s at 300K
    
    // Thermal properties
    double thermal_conductivity = 148.0;  // W/m·K
    double specific_heat = 700.0;         // J/kg·K
    double density = 2330.0;              // kg/m³
    
    // Mechanical properties
    double youngs_modulus = 170e9;        // Pa
    double poisson_ratio = 0.28;          // Dimensionless
    
    // Optical properties
    double refractive_index = 3.5;        // At 1550 nm
    double absorption_coefficient = 0.0;   // cm⁻¹ (for indirect bandgap)
};

/**
 * @brief Mesh geometry interface for advanced physics calculations
 */
struct MeshGeometry {
    virtual ~MeshGeometry() = default;
    
    virtual std::vector<size_t> get_neighbors(size_t point_index) const = 0;
    virtual double get_distance(size_t point1, size_t point2) const = 0;
    virtual double get_coordinate_difference(size_t point1, size_t point2, int component) const = 0;
    virtual size_t get_num_points() const = 0;
    virtual std::vector<double> get_coordinates(size_t point_index) const = 0;
};

} // namespace Physics
} // namespace SemiDGFEM
