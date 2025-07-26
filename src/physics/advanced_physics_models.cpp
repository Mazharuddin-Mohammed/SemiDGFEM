/**
 * Advanced Physics Models for Semiconductor Device Simulation
 * 
 * This module implements advanced physics models including:
 * - Strain effects and mechanical stress
 * - Thermal transport and self-heating
 * - Piezoelectric effects
 * - Thermoelectric coupling
 * - Optical properties and photogeneration
 * 
 * Author: Dr. Mazharuddin Mohammed
 */

#include "advanced_physics_models.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace SemiDGFEM {
namespace Physics {

// ============================================================================
// Strain Effects Model Implementation
// ============================================================================

StrainEffectsModel::StrainEffectsModel(const StrainConfig& config, 
                                     const MaterialProperties& material)
    : config_(config), material_(material) {
    initialize_strain_parameters();
}

void StrainEffectsModel::initialize_strain_parameters() {
    // Initialize deformation potentials for different materials
    if (material_.name == "Silicon") {
        deformation_potentials_.conduction_band = -8.6;  // eV
        deformation_potentials_.valence_band_heavy = -4.8;  // eV
        deformation_potentials_.valence_band_light = -4.8;  // eV
        deformation_potentials_.valence_band_split = -5.1;  // eV
        
        elastic_constants_.c11 = 165.8e9;  // Pa
        elastic_constants_.c12 = 63.9e9;   // Pa
        elastic_constants_.c44 = 79.6e9;   // Pa
    } else if (material_.name == "GaAs") {
        deformation_potentials_.conduction_band = -7.17;  // eV
        deformation_potentials_.valence_band_heavy = -1.16;  // eV
        deformation_potentials_.valence_band_light = -1.16;  // eV
        deformation_potentials_.valence_band_split = -1.7;   // eV
        
        elastic_constants_.c11 = 118.8e9;  // Pa
        elastic_constants_.c12 = 53.8e9;   // Pa
        elastic_constants_.c44 = 59.4e9;   // Pa
    } else {
        // Default to Silicon values
        initialize_strain_parameters();
    }
}

StrainTensor StrainEffectsModel::calculate_strain_tensor(
    const std::vector<double>& displacement_x,
    const std::vector<double>& displacement_y,
    const std::vector<double>& displacement_z,
    const MeshGeometry& mesh) const {
    
    StrainTensor strain;
    size_t num_points = displacement_x.size();
    
    strain.exx.resize(num_points);
    strain.eyy.resize(num_points);
    strain.ezz.resize(num_points);
    strain.exy.resize(num_points);
    strain.exz.resize(num_points);
    strain.eyz.resize(num_points);
    
    // Calculate strain components using finite differences
    for (size_t i = 0; i < num_points; ++i) {
        // Get neighboring points for gradient calculation
        auto neighbors = mesh.get_neighbors(i);
        
        if (neighbors.size() >= 4) {  // Ensure sufficient neighbors
            // Calculate gradients using least squares fit
            double dux_dx = calculate_gradient_component(displacement_x, neighbors, mesh, 0, 0);
            double duy_dy = calculate_gradient_component(displacement_y, neighbors, mesh, 1, 1);
            double duz_dz = calculate_gradient_component(displacement_z, neighbors, mesh, 2, 2);
            
            double dux_dy = calculate_gradient_component(displacement_x, neighbors, mesh, 0, 1);
            double duy_dx = calculate_gradient_component(displacement_y, neighbors, mesh, 1, 0);
            
            double dux_dz = calculate_gradient_component(displacement_x, neighbors, mesh, 0, 2);
            double duz_dx = calculate_gradient_component(displacement_z, neighbors, mesh, 2, 0);
            
            double duy_dz = calculate_gradient_component(displacement_y, neighbors, mesh, 1, 2);
            double duz_dy = calculate_gradient_component(displacement_z, neighbors, mesh, 2, 1);
            
            // Calculate strain tensor components
            strain.exx[i] = dux_dx;
            strain.eyy[i] = duy_dy;
            strain.ezz[i] = duz_dz;
            strain.exy[i] = 0.5 * (dux_dy + duy_dx);
            strain.exz[i] = 0.5 * (dux_dz + duz_dx);
            strain.eyz[i] = 0.5 * (duy_dz + duz_dy);
        } else {
            // Insufficient neighbors - set to zero strain
            strain.exx[i] = strain.eyy[i] = strain.ezz[i] = 0.0;
            strain.exy[i] = strain.exz[i] = strain.eyz[i] = 0.0;
        }
    }
    
    return strain;
}

BandStructureModification StrainEffectsModel::calculate_band_modification(
    const StrainTensor& strain) const {
    
    BandStructureModification modification;
    size_t num_points = strain.exx.size();
    
    modification.conduction_band_shift.resize(num_points);
    modification.valence_band_shift_heavy.resize(num_points);
    modification.valence_band_shift_light.resize(num_points);
    modification.valence_band_shift_split.resize(num_points);
    modification.effective_mass_modification.resize(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        // Calculate hydrostatic strain
        double hydrostatic_strain = strain.exx[i] + strain.eyy[i] + strain.ezz[i];
        
        // Calculate uniaxial strain components
        double uniaxial_strain_xx = strain.exx[i] - hydrostatic_strain / 3.0;
        double uniaxial_strain_yy = strain.eyy[i] - hydrostatic_strain / 3.0;
        double uniaxial_strain_zz = strain.ezz[i] - hydrostatic_strain / 3.0;
        
        // Conduction band shift (hydrostatic deformation potential)
        modification.conduction_band_shift[i] = 
            deformation_potentials_.conduction_band * hydrostatic_strain;
        
        // Valence band shifts (more complex due to degeneracy lifting)
        double shear_strain = std::sqrt(
            uniaxial_strain_xx * uniaxial_strain_xx +
            uniaxial_strain_yy * uniaxial_strain_yy +
            uniaxial_strain_zz * uniaxial_strain_zz +
            6.0 * (strain.exy[i] * strain.exy[i] + 
                   strain.exz[i] * strain.exz[i] + 
                   strain.eyz[i] * strain.eyz[i])
        );
        
        modification.valence_band_shift_heavy[i] = 
            deformation_potentials_.valence_band_heavy * hydrostatic_strain +
            config_.shear_deformation_potential * shear_strain;
            
        modification.valence_band_shift_light[i] = 
            deformation_potentials_.valence_band_light * hydrostatic_strain -
            config_.shear_deformation_potential * shear_strain;
            
        modification.valence_band_shift_split[i] = 
            deformation_potentials_.valence_band_split * hydrostatic_strain;
        
        // Effective mass modification (simplified model)
        modification.effective_mass_modification[i] = 
            1.0 + config_.mass_deformation_factor * std::abs(hydrostatic_strain);
    }
    
    return modification;
}

MobilityModification StrainEffectsModel::calculate_mobility_modification(
    const StrainTensor& strain) const {
    
    MobilityModification modification;
    size_t num_points = strain.exx.size();
    
    modification.electron_mobility_factor.resize(num_points);
    modification.hole_mobility_factor.resize(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        // Calculate strain magnitude
        double strain_magnitude = std::sqrt(
            strain.exx[i] * strain.exx[i] + strain.eyy[i] * strain.eyy[i] + 
            strain.ezz[i] * strain.ezz[i] + 2.0 * (strain.exy[i] * strain.exy[i] + 
            strain.exz[i] * strain.exz[i] + strain.eyz[i] * strain.eyz[i])
        );
        
        // Piezoresistance effect on mobility
        // Simplified model: mobility changes with strain
        double electron_piezoresistance = config_.electron_piezoresistance_factor;
        double hole_piezoresistance = config_.hole_piezoresistance_factor;
        
        modification.electron_mobility_factor[i] = 
            1.0 + electron_piezoresistance * strain_magnitude;
            
        modification.hole_mobility_factor[i] = 
            1.0 + hole_piezoresistance * strain_magnitude;
        
        // Ensure positive mobility factors
        modification.electron_mobility_factor[i] = 
            std::max(0.1, modification.electron_mobility_factor[i]);
        modification.hole_mobility_factor[i] = 
            std::max(0.1, modification.hole_mobility_factor[i]);
    }
    
    return modification;
}

double StrainEffectsModel::calculate_gradient_component(
    const std::vector<double>& field,
    const std::vector<size_t>& neighbors,
    const MeshGeometry& mesh,
    int field_component,
    int spatial_component) const {
    
    // Simplified gradient calculation using finite differences
    // In a real implementation, this would use proper DG gradient operators
    
    if (neighbors.size() < 2) return 0.0;
    
    // Use central difference approximation
    size_t left_neighbor = neighbors[0];
    size_t right_neighbor = neighbors[1];
    
    double dx = mesh.get_coordinate_difference(left_neighbor, right_neighbor, spatial_component);
    if (std::abs(dx) < 1e-12) return 0.0;
    
    return (field[right_neighbor] - field[left_neighbor]) / dx;
}

// ============================================================================
// Thermal Transport Model Implementation
// ============================================================================

ThermalTransportModel::ThermalTransportModel(const ThermalConfig& config,
                                           const MaterialProperties& material)
    : config_(config), material_(material) {
    initialize_thermal_parameters();
}

void ThermalTransportModel::initialize_thermal_parameters() {
    // Initialize thermal properties for different materials
    if (material_.name == "Silicon") {
        thermal_properties_.thermal_conductivity = 148.0;  // W/m·K at 300K
        thermal_properties_.specific_heat = 700.0;         // J/kg·K
        thermal_properties_.density = 2330.0;              // kg/m³
        thermal_properties_.thermal_expansion = 2.6e-6;    // 1/K
    } else if (material_.name == "GaAs") {
        thermal_properties_.thermal_conductivity = 55.0;   // W/m·K at 300K
        thermal_properties_.specific_heat = 350.0;         // J/kg·K
        thermal_properties_.density = 5320.0;              // kg/m³
        thermal_properties_.thermal_expansion = 5.7e-6;    // 1/K
    } else {
        // Default to Silicon values
        thermal_properties_.thermal_conductivity = 148.0;
        thermal_properties_.specific_heat = 700.0;
        thermal_properties_.density = 2330.0;
        thermal_properties_.thermal_expansion = 2.6e-6;
    }
}

std::vector<double> ThermalTransportModel::solve_heat_equation(
    const std::vector<double>& initial_temperature,
    const std::vector<double>& heat_generation,
    const ThermalBoundaryConditions& boundary_conditions,
    const MeshGeometry& mesh,
    double time_step,
    int num_time_steps) const {
    
    size_t num_points = initial_temperature.size();
    std::vector<double> temperature = initial_temperature;
    std::vector<double> temperature_new(num_points);
    
    // Time stepping for heat equation
    for (int step = 0; step < num_time_steps; ++step) {
        // Solve: ρcp ∂T/∂t = ∇·(k∇T) + Q
        
        for (size_t i = 0; i < num_points; ++i) {
            // Calculate thermal diffusion term
            double thermal_diffusion = calculate_thermal_diffusion(temperature, i, mesh);
            
            // Heat capacity term
            double heat_capacity = thermal_properties_.density * thermal_properties_.specific_heat;
            
            // Update temperature using explicit time stepping
            temperature_new[i] = temperature[i] + time_step * (
                thermal_diffusion / heat_capacity + heat_generation[i] / heat_capacity
            );
            
            // Apply boundary conditions
            if (boundary_conditions.is_boundary_point(i)) {
                temperature_new[i] = boundary_conditions.get_temperature(i);
            }
        }
        
        temperature = temperature_new;
    }
    
    return temperature;
}

std::vector<double> ThermalTransportModel::calculate_joule_heating(
    const std::vector<double>& current_density_x,
    const std::vector<double>& current_density_y,
    const std::vector<double>& electric_field_x,
    const std::vector<double>& electric_field_y) const {
    
    size_t num_points = current_density_x.size();
    std::vector<double> joule_heating(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        // Joule heating: Q = J·E
        joule_heating[i] = current_density_x[i] * electric_field_x[i] +
                          current_density_y[i] * electric_field_y[i];
        
        // Ensure non-negative heating
        joule_heating[i] = std::max(0.0, joule_heating[i]);
    }
    
    return joule_heating;
}

ThermalCoupling ThermalTransportModel::calculate_thermal_coupling(
    const std::vector<double>& temperature,
    const std::vector<double>& carrier_density_n,
    const std::vector<double>& carrier_density_p) const {
    
    ThermalCoupling coupling;
    size_t num_points = temperature.size();
    
    coupling.bandgap_modification.resize(num_points);
    coupling.mobility_modification.resize(num_points);
    coupling.thermal_voltage.resize(num_points);
    coupling.thermal_diffusion_length.resize(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        double T = temperature[i];
        double T_ref = 300.0;  // Reference temperature
        
        // Bandgap temperature dependence (Varshni model)
        double alpha = 4.73e-4;  // eV/K for Silicon
        double beta = 636.0;     // K for Silicon
        coupling.bandgap_modification[i] = -alpha * T * T / (T + beta);
        
        // Mobility temperature dependence
        double mobility_temp_exponent = -2.4;  // For electrons in Silicon
        coupling.mobility_modification[i] = std::pow(T / T_ref, mobility_temp_exponent);
        
        // Thermal voltage
        coupling.thermal_voltage[i] = PhysicalConstants::k * T / PhysicalConstants::q;
        
        // Thermal diffusion length (simplified)
        double diffusion_coefficient = config_.thermal_diffusivity;
        double recombination_time = 1e-6;  // Typical value
        coupling.thermal_diffusion_length[i] = 
            std::sqrt(diffusion_coefficient * recombination_time);
    }
    
    return coupling;
}

double ThermalTransportModel::calculate_thermal_diffusion(
    const std::vector<double>& temperature,
    size_t point_index,
    const MeshGeometry& mesh) const {
    
    // Simplified thermal diffusion calculation
    // In a real implementation, this would use proper DG operators
    
    auto neighbors = mesh.get_neighbors(point_index);
    if (neighbors.empty()) return 0.0;
    
    double laplacian = 0.0;
    double T_center = temperature[point_index];
    
    for (size_t neighbor : neighbors) {
        double distance = mesh.get_distance(point_index, neighbor);
        if (distance > 1e-12) {
            laplacian += (temperature[neighbor] - T_center) / (distance * distance);
        }
    }
    
    return thermal_properties_.thermal_conductivity * laplacian;
}

// ============================================================================
// Piezoelectric Effects Model Implementation
// ============================================================================

PiezoelectricModel::PiezoelectricModel(const PiezoelectricConfig& config,
                                     const MaterialProperties& material)
    : config_(config), material_(material) {
    initialize_piezoelectric_parameters();
}

void PiezoelectricModel::initialize_piezoelectric_parameters() {
    // Initialize piezoelectric constants for different materials
    if (material_.name == "GaN") {
        piezo_tensor_.e31.resize(1, -0.49);  // C/m²
        piezo_tensor_.e33.resize(1, 0.73);   // C/m²
        piezo_tensor_.e15.resize(1, -0.40);  // C/m²
    } else if (material_.name == "AlN") {
        piezo_tensor_.e31.resize(1, -0.60);  // C/m²
        piezo_tensor_.e33.resize(1, 1.46);   // C/m²
        piezo_tensor_.e15.resize(1, -0.48);  // C/m²
    } else if (material_.name == "ZnO") {
        piezo_tensor_.e31.resize(1, -0.57);  // C/m²
        piezo_tensor_.e33.resize(1, 1.22);   // C/m²
        piezo_tensor_.e15.resize(1, -0.48);  // C/m²
    } else {
        // Default to GaN values
        piezo_tensor_.e31.resize(1, -0.49);
        piezo_tensor_.e33.resize(1, 0.73);
        piezo_tensor_.e15.resize(1, -0.40);
    }
}

PolarizationField PiezoelectricModel::calculate_polarization(
    const StrainTensor& strain,
    const std::vector<double>& electric_field_x,
    const std::vector<double>& electric_field_y,
    const std::vector<double>& electric_field_z) const {

    PolarizationField polarization;
    size_t num_points = strain.exx.size();

    polarization.Px.resize(num_points);
    polarization.Py.resize(num_points);
    polarization.Pz.resize(num_points);
    polarization.bound_charge_density.resize(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        // Spontaneous polarization (along z-axis for wurtzite)
        double P_sp_z = config_.spontaneous_polarization;

        // Piezoelectric polarization
        double e31 = (i < piezo_tensor_.e31.size()) ? piezo_tensor_.e31[i] : piezo_tensor_.e31[0];
        double e33 = (i < piezo_tensor_.e33.size()) ? piezo_tensor_.e33[i] : piezo_tensor_.e33[0];
        double e15 = (i < piezo_tensor_.e15.size()) ? piezo_tensor_.e15[i] : piezo_tensor_.e15[0];

        // Piezoelectric polarization components
        double P_pz_x = 2.0 * e15 * strain.exz[i];
        double P_pz_y = 2.0 * e15 * strain.eyz[i];
        double P_pz_z = e31 * (strain.exx[i] + strain.eyy[i]) + e33 * strain.ezz[i];

        // Total polarization
        polarization.Px[i] = P_pz_x;
        polarization.Py[i] = P_pz_y;
        polarization.Pz[i] = P_sp_z + P_pz_z;

        // Bound charge density: ρ_bound = -∇·P
        // Simplified calculation - in real implementation would use proper divergence
        polarization.bound_charge_density[i] = 0.0;  // Will be calculated separately
    }

    return polarization;
}

std::vector<double> PiezoelectricModel::calculate_bound_charge_density(
    const PolarizationField& polarization,
    const MeshGeometry& mesh) const {

    size_t num_points = polarization.Px.size();
    std::vector<double> bound_charge(num_points, 0.0);

    for (size_t i = 0; i < num_points; ++i) {
        // Calculate divergence of polarization: ∇·P
        auto neighbors = mesh.get_neighbors(i);

        if (neighbors.size() >= 3) {  // Need sufficient neighbors for gradient
            double div_P = 0.0;

            // Simplified divergence calculation using finite differences
            for (size_t neighbor : neighbors) {
                double distance = mesh.get_distance(i, neighbor);
                if (distance > 1e-12) {
                    // Calculate contribution to divergence
                    double dx = mesh.get_coordinate_difference(i, neighbor, 0);
                    double dy = mesh.get_coordinate_difference(i, neighbor, 1);
                    double dz = mesh.get_coordinate_difference(i, neighbor, 2);

                    double dPx = polarization.Px[neighbor] - polarization.Px[i];
                    double dPy = polarization.Py[neighbor] - polarization.Py[i];
                    double dPz = polarization.Pz[neighbor] - polarization.Pz[i];

                    div_P += (dPx * dx + dPy * dy + dPz * dz) / (distance * distance);
                }
            }

            // Bound charge density: ρ_bound = -∇·P
            bound_charge[i] = -div_P / neighbors.size();
        }
    }

    return bound_charge;
}

std::vector<double> PiezoelectricModel::calculate_interface_charge(
    const PolarizationField& polarization_1,
    const PolarizationField& polarization_2,
    const std::vector<size_t>& interface_points) const {

    std::vector<double> interface_charge(interface_points.size());

    for (size_t idx = 0; idx < interface_points.size(); ++idx) {
        size_t point = interface_points[idx];

        // Interface charge density: σ = (P2 - P1)·n
        // Assuming normal vector is along z-direction for simplicity
        double delta_Pz = polarization_2.Pz[point] - polarization_1.Pz[point];
        interface_charge[idx] = delta_Pz;
    }

    return interface_charge;
}

// ============================================================================
// Optical Properties Model Implementation
// ============================================================================

/**
 * @brief Configuration for optical properties model
 */
struct OpticalConfig {
    bool enable_optical_generation = true;
    bool enable_photon_recycling = false;
    bool enable_stimulated_emission = false;

    // Optical parameters
    double photon_energy = 1.24;        // eV (1 μm wavelength)
    double optical_power = 1e-3;        // W
    double absorption_coefficient = 1e4; // cm⁻¹
    double quantum_efficiency = 0.8;     // Dimensionless

    // Numerical parameters
    double optical_tolerance = 1e-8;
    int max_optical_iterations = 50;
};

/**
 * @brief Optical generation and recombination model
 */
class OpticalModel {
private:
    OpticalConfig config_;
    MaterialProperties material_;

public:
    OpticalModel(const OpticalConfig& config = OpticalConfig{},
                const MaterialProperties& material = MaterialProperties{})
        : config_(config), material_(material) {}

    /**
     * @brief Calculate optical generation rate
     */
    std::vector<double> calculate_optical_generation(
        const std::vector<double>& photon_flux,
        const std::vector<double>& position_z) const {

        size_t num_points = photon_flux.size();
        std::vector<double> generation_rate(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            // Beer-Lambert law: I(z) = I0 * exp(-α*z)
            double absorption = std::exp(-config_.absorption_coefficient * position_z[i] * 1e-2); // Convert to cm

            // Generation rate: G = α * Φ * η / hν
            generation_rate[i] = config_.absorption_coefficient * 1e2 * // Convert back to m⁻¹
                               photon_flux[i] * absorption * config_.quantum_efficiency;
        }

        return generation_rate;
    }

    /**
     * @brief Calculate radiative recombination rate
     */
    std::vector<double> calculate_radiative_recombination(
        const std::vector<double>& electron_density,
        const std::vector<double>& hole_density) const {

        size_t num_points = electron_density.size();
        std::vector<double> recombination_rate(num_points);

        // Radiative recombination coefficient for Silicon (very low due to indirect bandgap)
        double B_rad = 1e-15;  // cm³/s

        for (size_t i = 0; i < num_points; ++i) {
            // Radiative recombination: R = B * n * p
            recombination_rate[i] = B_rad * 1e6 * // Convert to m³/s
                                  electron_density[i] * hole_density[i];
        }

        return recombination_rate;
    }

    // Configuration methods
    void set_config(const OpticalConfig& config) { config_ = config; }
    OpticalConfig get_config() const { return config_; }
};

} // namespace Physics
} // namespace SemiDGFEM
