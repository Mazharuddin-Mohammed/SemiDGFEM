/**
 * Advanced Physics Models for Semiconductor Device Simulation
 * Includes effective mass models, advanced mobility, and SRH recombination
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include <vector>
#include <cmath>
#include <memory>
#include <string>
#include <array>

namespace SemiDGFEM {
namespace Physics {

/**
 * @brief Physical constants
 */
struct PhysicalConstants {
    static constexpr double q = 1.602176634e-19;  // Elementary charge (C)
    static constexpr double k = 1.380649e-23;     // Boltzmann constant (J/K)
    static constexpr double h = 6.62607015e-34;   // Planck constant (J⋅s)
    static constexpr double hbar = h / (2.0 * M_PI); // Reduced Planck constant
    static constexpr double m0 = 9.1093837015e-31; // Free electron mass (kg)
    static constexpr double epsilon_0 = 8.8541878128e-12; // Vacuum permittivity (F/m)
};

/**
 * @brief Advanced silicon material properties with temperature dependence
 */
struct SiliconProperties {
    // Basic properties
    double epsilon_r = 11.7;           // Relative permittivity
    double ni_300K = 1.45e16;          // Intrinsic carrier concentration (/m³) at 300K
    double Eg_300K = 1.12;             // Bandgap (eV) at 300K
    
    // Effective masses (in units of m0)
    double m_eff_n_dos = 1.08;         // Electron DOS effective mass
    double m_eff_p_dos = 0.81;         // Hole DOS effective mass
    double m_eff_n_cond = 0.26;        // Electron conductivity effective mass
    double m_eff_p_cond = 0.39;        // Hole conductivity effective mass
    
    // Mobility parameters for Caughey-Thomas model
    double mu_n_max = 0.135;           // Maximum electron mobility (m²/V⋅s)
    double mu_p_max = 0.048;           // Maximum hole mobility (m²/V⋅s)
    double N_ref_n = 8.5e23;           // Reference doping for electrons (/m³)
    double N_ref_p = 8.5e23;           // Reference doping for holes (/m³)
    double alpha_n = 0.88;             // Mobility exponent for electrons
    double alpha_p = 0.76;             // Mobility exponent for holes
    
    // Temperature coefficients
    double alpha_ni = 1.5;             // Temperature exponent for ni
    double beta_Eg = 4.73e-4;          // Bandgap temperature coefficient (eV/K)
    double gamma_Eg = 636.0;           // Bandgap temperature parameter (K)
    
    // SRH parameters
    double tau_n0_default = 1e-6;      // Default electron SRH lifetime (s)
    double tau_p0_default = 1e-6;      // Default hole SRH lifetime (s)
    double Et_Ei_default = 0.0;        // Default trap level relative to intrinsic (eV)
    
    /**
     * @brief Calculate temperature-dependent intrinsic carrier concentration
     */
    double calculate_ni(double T) const {
        double Eg_T = Eg_300K - (beta_Eg * T * T) / (T + gamma_Eg);
        double ni_T = ni_300K * std::pow(T / 300.0, alpha_ni) * 
                      std::exp(-PhysicalConstants::q * (Eg_T - Eg_300K) / (2.0 * PhysicalConstants::k * T));
        return ni_T;
    }
    
    /**
     * @brief Calculate temperature-dependent bandgap
     */
    double calculate_bandgap(double T) const {
        return Eg_300K - (beta_Eg * T * T) / (T + gamma_Eg);
    }
};

/**
 * @brief Advanced mobility model interface
 */
class MobilityModel {
public:
    virtual ~MobilityModel() = default;
    virtual double calculate_electron_mobility(double N_total, double T = 300.0, double E_field = 0.0) const = 0;
    virtual double calculate_hole_mobility(double N_total, double T = 300.0, double E_field = 0.0) const = 0;
    virtual std::string get_model_name() const = 0;
};

/**
 * @brief Caughey-Thomas mobility model with temperature and field dependence
 */
class CaugheyThomasMobility : public MobilityModel {
private:
    SiliconProperties props_;
    
public:
    CaugheyThomasMobility(const SiliconProperties& props = SiliconProperties{}) 
        : props_(props) {}
    
    double calculate_electron_mobility(double N_total, double T = 300.0, double E_field = 0.0) const override {
        // Temperature dependence
        double mu_max_T = props_.mu_n_max * std::pow(T / 300.0, -2.4);
        
        // Doping dependence (Caughey-Thomas)
        double mu_doping = mu_max_T / (1.0 + std::pow(N_total / props_.N_ref_n, props_.alpha_n));
        
        // High-field saturation (simplified)
        if (E_field > 1e3) {
            double v_sat = 1e5; // Saturation velocity (m/s)
            double mu_field = v_sat / E_field;
            mu_doping = 1.0 / (1.0/mu_doping + 1.0/mu_field);
        }
        
        return mu_doping;
    }
    
    double calculate_hole_mobility(double N_total, double T = 300.0, double E_field = 0.0) const override {
        // Temperature dependence
        double mu_max_T = props_.mu_p_max * std::pow(T / 300.0, -2.2);
        
        // Doping dependence (Caughey-Thomas)
        double mu_doping = mu_max_T / (1.0 + std::pow(N_total / props_.N_ref_p, props_.alpha_p));
        
        // High-field saturation (simplified)
        if (E_field > 1e3) {
            double v_sat = 8e4; // Saturation velocity for holes (m/s)
            double mu_field = v_sat / E_field;
            mu_doping = 1.0 / (1.0/mu_doping + 1.0/mu_field);
        }
        
        return mu_doping;
    }
    
    std::string get_model_name() const override {
        return "Caughey-Thomas with temperature and field dependence";
    }
};

/**
 * @brief SRH recombination model
 */
class SRHRecombination {
private:
    double tau_n0_, tau_p0_;  // SRH lifetimes
    double Et_Ei_;            // Trap level relative to intrinsic level (eV)
    SiliconProperties props_;
    
public:
    SRHRecombination(double tau_n0, double tau_p0, double Et_Ei = 0.0,
                     const SiliconProperties& props = SiliconProperties{})
        : tau_n0_(tau_n0), tau_p0_(tau_p0), Et_Ei_(Et_Ei), props_(props) {}
    
    /**
     * @brief Calculate SRH recombination rate
     */
    double calculate_recombination_rate(double n, double p, double T = 300.0) const {
        double ni = props_.calculate_ni(T);
        double Vt = PhysicalConstants::k * T / PhysicalConstants::q;
        
        // Trap densities
        double n_trap = ni * std::exp(Et_Ei_ / Vt);
        double p_trap = ni * std::exp(-Et_Ei_ / Vt);
        
        // SRH recombination rate
        double R_srh = (n * p - ni * ni) / 
                       (tau_p0_ * (n + n_trap) + tau_n0_ * (p + p_trap));
        
        return R_srh;
    }
    
    /**
     * @brief Calculate generation rate (negative recombination)
     */
    double calculate_generation_rate(double n, double p, double T = 300.0) const {
        return -calculate_recombination_rate(n, p, T);
    }
};

/**
 * @brief Effective mass model for quantum effects
 */
class EffectiveMassModel {
private:
    SiliconProperties props_;

public:
    EffectiveMassModel(const SiliconProperties& props = SiliconProperties{})
        : props_(props) {}

    /**
     * @brief Calculate density of states effective mass for electrons
     */
    double get_electron_dos_mass() const {
        return props_.m_eff_n_dos * PhysicalConstants::m0;
    }

    /**
     * @brief Calculate density of states effective mass for holes
     */
    double get_hole_dos_mass() const {
        return props_.m_eff_p_dos * PhysicalConstants::m0;
    }

    /**
     * @brief Calculate conductivity effective mass for electrons
     */
    double get_electron_conductivity_mass() const {
        return props_.m_eff_n_cond * PhysicalConstants::m0;
    }

    /**
     * @brief Calculate conductivity effective mass for holes
     */
    double get_hole_conductivity_mass() const {
        return props_.m_eff_p_cond * PhysicalConstants::m0;
    }

    /**
     * @brief Calculate effective density of states in conduction band
     */
    double calculate_Nc(double T = 300.0) const {
        double m_eff = get_electron_dos_mass();
        return 2.0 * std::pow(2.0 * M_PI * m_eff * PhysicalConstants::k * T /
                             (PhysicalConstants::h * PhysicalConstants::h), 1.5);
    }

    /**
     * @brief Calculate effective density of states in valence band
     */
    double calculate_Nv(double T = 300.0) const {
        double m_eff = get_hole_dos_mass();
        return 2.0 * std::pow(2.0 * M_PI * m_eff * PhysicalConstants::k * T /
                             (PhysicalConstants::h * PhysicalConstants::h), 1.5);
    }
};

/**
 * @brief Non-equilibrium carrier statistics model
 */
class NonEquilibriumStatistics {
private:
    SiliconProperties props_;
    std::unique_ptr<EffectiveMassModel> effective_mass_model_;

public:
    NonEquilibriumStatistics(const SiliconProperties& props = SiliconProperties{})
        : props_(props) {
        effective_mass_model_ = std::make_unique<EffectiveMassModel>(props);
    }

    /**
     * @brief Calculate carrier densities using Fermi-Dirac statistics
     */
    void calculate_fermi_dirac_densities(
        const std::vector<double>& potential,
        const std::vector<double>& quasi_fermi_n,
        const std::vector<double>& quasi_fermi_p,
        const std::vector<double>& Nd,
        const std::vector<double>& Na,
        std::vector<double>& n,
        std::vector<double>& p,
        double T = 300.0) const;

    /**
     * @brief Calculate bandgap narrowing in heavily doped regions
     */
    double calculate_bandgap_narrowing(double N_total) const {
        // Slotboom model for bandgap narrowing
        double bandgap_narrowing_factor = 1e-3; // eV⋅m^(1/3)
        double delta_Eg = bandgap_narrowing_factor * std::pow(N_total, 1.0/3.0);
        return std::min(delta_Eg, 0.1); // Limit to 100 meV
    }

    /**
     * @brief Calculate incomplete ionization effects
     */
    double calculate_ionization_fraction(double N_doping, double T = 300.0) const {
        // Simplified incomplete ionization model
        double E_ionization = 0.045; // eV for phosphorus in silicon
        double Vt = PhysicalConstants::k * T / PhysicalConstants::q;

        return 1.0 / (1.0 + 2.0 * std::exp(E_ionization / Vt));
    }

    /**
     * @brief Calculate degeneracy factor for statistics
     */
    double calculate_degeneracy_factor(double n, double Nc, double T = 300.0) const {
        double eta = n / Nc;
        if (eta < 0.1) return 1.0; // Non-degenerate case

        // Joyce-Dixon approximation for Fermi-Dirac integral
        return 1.0 + eta / (2.0 + eta);
    }
};

/**
 * @brief Transport model types for advanced physics
 */
enum class TransportModel {
    DRIFT_DIFFUSION,           // Classical drift-diffusion
    ENERGY_TRANSPORT,          // Energy transport model
    HYDRODYNAMIC,             // Hydrodynamic model
    NON_EQUILIBRIUM_STATISTICS // Non-equilibrium carrier statistics
};

/**
 * @brief Non-equilibrium statistics configuration
 */
struct NonEquilibriumConfig {
    bool enable_fermi_dirac = true;           // Use Fermi-Dirac statistics
    bool enable_degeneracy_effects = true;    // Include degeneracy effects
    bool enable_bandgap_narrowing = true;     // Bandgap narrowing in heavily doped regions
    bool enable_incomplete_ionization = false; // Incomplete dopant ionization
    double degeneracy_factor = 2.0;           // Degeneracy factor for statistics
    double bandgap_narrowing_factor = 1e-3;   // Bandgap narrowing coefficient (eV⋅m^(1/3))
};

/**
 * @brief Energy transport model configuration
 */
struct EnergyTransportConfig {
    bool enable_energy_relaxation = true;     // Energy relaxation effects
    bool enable_velocity_overshoot = true;    // Velocity overshoot effects
    double energy_relaxation_time_n = 0.1e-12; // Electron energy relaxation time (s)
    double energy_relaxation_time_p = 0.1e-12; // Hole energy relaxation time (s)
    double saturation_velocity_n = 1e5;       // Electron saturation velocity (m/s)
    double saturation_velocity_p = 8e4;       // Hole saturation velocity (m/s)
};

/**
 * @brief Hydrodynamic model configuration
 */
struct HydrodynamicConfig {
    bool enable_momentum_relaxation = true;   // Momentum relaxation effects
    bool enable_pressure_gradient = true;     // Pressure gradient effects
    bool enable_heat_flow = true;             // Heat flow effects
    double momentum_relaxation_time_n = 0.1e-12; // Electron momentum relaxation time (s)
    double momentum_relaxation_time_p = 0.1e-12; // Hole momentum relaxation time (s)
    double thermal_conductivity = 150.0;      // Thermal conductivity (W/m⋅K)
    double specific_heat = 700.0;             // Specific heat capacity (J/kg⋅K)
};

/**
 * @brief Advanced physics configuration structure
 */
struct PhysicsConfig {
    // Temperature
    double temperature = 300.0;  // K

    // Material properties
    SiliconProperties silicon_props;

    // Transport model selection
    TransportModel transport_model = TransportModel::DRIFT_DIFFUSION;

    // Mobility model selection
    std::string mobility_model = "CaugheyThomas";

    // SRH parameters
    double tau_n0 = 1e-6;        // s
    double tau_p0 = 1e-6;        // s
    double Et_Ei = 0.0;          // eV

    // Advanced transport configurations
    NonEquilibriumConfig non_equilibrium_config;
    EnergyTransportConfig energy_transport_config;
    HydrodynamicConfig hydrodynamic_config;

    // Numerical parameters
    double poisson_tolerance = 1e-12;
    double dd_tolerance = 1e-10;
    double energy_tolerance = 1e-10;
    double momentum_tolerance = 1e-10;
    int max_iterations = 100;

    // Enable/disable physics models
    bool enable_srh_recombination = true;
    bool enable_field_dependent_mobility = true;
    bool enable_temperature_dependence = true;
    bool enable_quantum_effects = false;
    bool enable_impact_ionization = false;
    bool enable_tunneling = false;
};

/**
 * @brief Energy transport model for hot carrier effects
 */
class EnergyTransportModel {
private:
    EnergyTransportConfig config_;
    SiliconProperties props_;

public:
    EnergyTransportModel(const EnergyTransportConfig& config = EnergyTransportConfig{},
                        const SiliconProperties& props = SiliconProperties{})
        : config_(config), props_(props) {}

    /**
     * @brief Calculate carrier temperatures from energy densities
     */
    void calculate_carrier_temperatures(
        const std::vector<double>& energy_density_n,
        const std::vector<double>& energy_density_p,
        const std::vector<double>& n,
        const std::vector<double>& p,
        std::vector<double>& T_n,
        std::vector<double>& T_p,
        double lattice_T = 300.0) const;

    /**
     * @brief Calculate energy relaxation rates
     */
    void calculate_energy_relaxation(
        const std::vector<double>& T_n,
        const std::vector<double>& T_p,
        const std::vector<double>& n,
        const std::vector<double>& p,
        std::vector<double>& energy_relaxation_n,
        std::vector<double>& energy_relaxation_p,
        double lattice_T = 300.0) const;

    /**
     * @brief Calculate velocity overshoot effects
     */
    double calculate_velocity_overshoot(double E_field, double carrier_temp,
                                      bool is_electron = true) const {
        if (!config_.enable_velocity_overshoot) return 1.0;

        double v_sat = is_electron ? config_.saturation_velocity_n : config_.saturation_velocity_p;
        double tau_energy = is_electron ? config_.energy_relaxation_time_n : config_.energy_relaxation_time_p;

        // Simplified overshoot model
        double overshoot_factor = 1.0 + (carrier_temp - 300.0) / 300.0 * 0.5;
        return std::min(overshoot_factor, 2.0); // Limit overshoot
    }
};

/**
 * @brief Hydrodynamic transport model
 */
class HydrodynamicModel {
private:
    HydrodynamicConfig config_;
    SiliconProperties props_;

public:
    HydrodynamicModel(const HydrodynamicConfig& config = HydrodynamicConfig{},
                     const SiliconProperties& props = SiliconProperties{})
        : config_(config), props_(props) {}

    /**
     * @brief Calculate momentum relaxation rates
     */
    void calculate_momentum_relaxation(
        const std::vector<double>& velocity_n,
        const std::vector<double>& velocity_p,
        const std::vector<double>& n,
        const std::vector<double>& p,
        std::vector<double>& momentum_relaxation_n,
        std::vector<double>& momentum_relaxation_p) const;

    /**
     * @brief Calculate pressure gradient effects
     */
    void calculate_pressure_gradients(
        const std::vector<double>& n,
        const std::vector<double>& p,
        const std::vector<double>& T_n,
        const std::vector<double>& T_p,
        std::vector<double>& pressure_grad_n,
        std::vector<double>& pressure_grad_p) const;

    /**
     * @brief Calculate heat flow effects
     */
    void calculate_heat_flow(
        const std::vector<double>& T_n,
        const std::vector<double>& T_p,
        const std::vector<double>& lattice_temp,
        std::vector<double>& heat_flow_n,
        std::vector<double>& heat_flow_p) const;

    /**
     * @brief Calculate thermal conductivity tensor
     */
    std::array<std::array<double, 2>, 2> calculate_thermal_conductivity(
        double carrier_density, double carrier_temp) const {
        double kappa = config_.thermal_conductivity * (carrier_density / 1e16);
        return {{{{kappa, 0.0}}, {{0.0, kappa}}}};
    }
};

} // namespace Physics
} // namespace SemiDGFEM
