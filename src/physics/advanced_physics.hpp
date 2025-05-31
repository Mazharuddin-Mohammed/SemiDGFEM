#pragma once

#include <vector>
#include <cmath>
#include <memory>
#include <string>

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
 * @brief Advanced physics configuration structure
 */
struct PhysicsConfig {
    // Temperature
    double temperature = 300.0;  // K
    
    // Material properties
    SiliconProperties silicon_props;
    
    // Mobility model selection
    std::string mobility_model = "CaugheyThomas";
    
    // SRH parameters
    double tau_n0 = 1e-6;        // s
    double tau_p0 = 1e-6;        // s
    double Et_Ei = 0.0;          // eV
    
    // Numerical parameters
    double poisson_tolerance = 1e-12;
    double dd_tolerance = 1e-10;
    int max_iterations = 100;
    
    // Enable/disable physics models
    bool enable_srh_recombination = true;
    bool enable_field_dependent_mobility = true;
    bool enable_temperature_dependence = true;
    bool enable_quantum_effects = false;
};

} // namespace Physics
} // namespace SemiDGFEM
