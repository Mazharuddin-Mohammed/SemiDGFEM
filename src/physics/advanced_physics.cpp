/**
 * Implementation of Advanced Physics Models
 * Realistic semiconductor device physics with temperature dependence
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "advanced_physics.hpp"
#include <algorithm>
#include <stdexcept>

namespace SemiDGFEM {
namespace Physics {

/**
 * @brief Advanced physics solver with realistic models
 */
class AdvancedPhysicsSolver {
private:
    PhysicsConfig config_;
    std::unique_ptr<MobilityModel> mobility_model_;
    std::unique_ptr<SRHRecombination> srh_model_;
    std::unique_ptr<EffectiveMassModel> effective_mass_model_;
    
public:
    AdvancedPhysicsSolver(const PhysicsConfig& config) : config_(config) {
        initialize_models();
    }
    
    void initialize_models() {
        // Initialize mobility model
        if (config_.mobility_model == "CaugheyThomas") {
            mobility_model_ = std::make_unique<CaugheyThomasMobility>(config_.silicon_props);
        } else {
            throw std::invalid_argument("Unknown mobility model: " + config_.mobility_model);
        }
        
        // Initialize SRH model
        if (config_.enable_srh_recombination) {
            srh_model_ = std::make_unique<SRHRecombination>(
                config_.tau_n0, config_.tau_p0, config_.Et_Ei, config_.silicon_props);
        }
        
        // Initialize effective mass model
        effective_mass_model_ = std::make_unique<EffectiveMassModel>(config_.silicon_props);
    }
    
    /**
     * @brief Calculate carrier densities using advanced models
     */
    void calculate_carrier_densities(
        const std::vector<double>& potential,
        const std::vector<double>& Nd,
        const std::vector<double>& Na,
        std::vector<double>& n,
        std::vector<double>& p) const {
        
        double ni = config_.silicon_props.calculate_ni(config_.temperature);
        double Vt = PhysicalConstants::k * config_.temperature / PhysicalConstants::q;
        
        size_t num_points = potential.size();
        n.resize(num_points);
        p.resize(num_points);
        
        for (size_t i = 0; i < num_points; ++i) {
            double phi = potential[i];
            double Nd_local = Nd[i];
            double Na_local = Na[i];
            
            if (Nd_local > Na_local) {
                // N-type region
                double N_net = Nd_local - Na_local;
                
                if (config_.enable_temperature_dependence) {
                    // Use Fermi-Dirac statistics approximation
                    double Nc = effective_mass_model_->calculate_Nc(config_.temperature);
                    n[i] = Nc * std::exp((phi - calculate_conduction_band_edge(N_net)) / Vt);
                } else {
                    // Simplified Boltzmann statistics
                    n[i] = N_net * std::exp(phi / Vt);
                }
                
                p[i] = ni * ni / n[i];
            } else {
                // P-type region
                double N_net = Na_local - Nd_local;
                
                if (config_.enable_temperature_dependence) {
                    // Use Fermi-Dirac statistics approximation
                    double Nv = effective_mass_model_->calculate_Nv(config_.temperature);
                    p[i] = Nv * std::exp(-(phi - calculate_valence_band_edge(N_net)) / Vt);
                } else {
                    // Simplified Boltzmann statistics
                    p[i] = N_net * std::exp(-phi / Vt);
                }
                
                n[i] = ni * ni / p[i];
            }
            
            // Ensure minimum carrier concentrations
            n[i] = std::max(n[i], ni / 1000.0);
            p[i] = std::max(p[i], ni / 1000.0);
        }
    }
    
    /**
     * @brief Calculate current densities with advanced mobility models
     */
    void calculate_current_densities(
        const std::vector<double>& potential,
        const std::vector<double>& n,
        const std::vector<double>& p,
        const std::vector<double>& Nd,
        const std::vector<double>& Na,
        int nx, int ny,
        std::vector<double>& Jn_x,
        std::vector<double>& Jn_y,
        std::vector<double>& Jp_x,
        std::vector<double>& Jp_y) const {
        
        size_t num_points = potential.size();
        Jn_x.resize(num_points, 0.0);
        Jn_y.resize(num_points, 0.0);
        Jp_x.resize(num_points, 0.0);
        Jp_y.resize(num_points, 0.0);
        
        // Calculate electric field
        std::vector<double> Ex(num_points, 0.0);
        std::vector<double> Ey(num_points, 0.0);
        calculate_electric_field(potential, nx, ny, Ex, Ey);
        
        for (size_t i = 0; i < num_points; ++i) {
            double N_total = Nd[i] + Na[i];
            double E_magnitude = std::sqrt(Ex[i] * Ex[i] + Ey[i] * Ey[i]);
            
            // Calculate mobilities using advanced models
            double mu_n = mobility_model_->calculate_electron_mobility(
                N_total, config_.temperature, 
                config_.enable_field_dependent_mobility ? E_magnitude : 0.0);
            double mu_p = mobility_model_->calculate_hole_mobility(
                N_total, config_.temperature,
                config_.enable_field_dependent_mobility ? E_magnitude : 0.0);
            
            // Calculate current densities (drift component)
            Jn_x[i] = PhysicalConstants::q * mu_n * n[i] * Ex[i];
            Jn_y[i] = PhysicalConstants::q * mu_n * n[i] * Ey[i];
            
            Jp_x[i] = -PhysicalConstants::q * mu_p * p[i] * Ex[i];  // Opposite direction
            Jp_y[i] = -PhysicalConstants::q * mu_p * p[i] * Ey[i];
        }
    }
    
    /**
     * @brief Calculate SRH recombination rates
     */
    void calculate_recombination_rates(
        const std::vector<double>& n,
        const std::vector<double>& p,
        std::vector<double>& R_srh) const {
        
        if (!config_.enable_srh_recombination || !srh_model_) {
            R_srh.assign(n.size(), 0.0);
            return;
        }
        
        size_t num_points = n.size();
        R_srh.resize(num_points);
        
        for (size_t i = 0; i < num_points; ++i) {
            R_srh[i] = srh_model_->calculate_recombination_rate(n[i], p[i], config_.temperature);
        }
    }
    
    /**
     * @brief Calculate threshold voltage using advanced models
     */
    double calculate_threshold_voltage(double Na_substrate, double tox) const {
        double ni = config_.silicon_props.calculate_ni(config_.temperature);
        double Vt = PhysicalConstants::k * config_.temperature / PhysicalConstants::q;
        double epsilon_si = config_.silicon_props.epsilon_r * PhysicalConstants::epsilon_0;
        double epsilon_ox = 3.9 * PhysicalConstants::epsilon_0;  // SiO2
        
        // Calculate Fermi potential
        double phi_F = Vt * std::log(Na_substrate / ni);
        
        // Calculate oxide capacitance
        double Cox = epsilon_ox / tox;
        
        // Calculate depletion charge
        double gamma = std::sqrt(2.0 * PhysicalConstants::q * epsilon_si * Na_substrate) / Cox;
        
        // Threshold voltage
        double Vth = 2.0 * phi_F + gamma * std::sqrt(2.0 * phi_F);
        
        return Vth;
    }
    
    /**
     * @brief Get physics configuration
     */
    const PhysicsConfig& get_config() const {
        return config_;
    }
    
    /**
     * @brief Update physics configuration
     */
    void update_config(const PhysicsConfig& new_config) {
        config_ = new_config;
        initialize_models();
    }

private:
    /**
     * @brief Calculate electric field from potential
     */
    void calculate_electric_field(
        const std::vector<double>& potential,
        int nx, int ny,
        std::vector<double>& Ex,
        std::vector<double>& Ey) const {
        
        // Simple finite difference for electric field calculation
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = j * nx + i;
                
                // Ex = -dV/dx
                if (i > 0 && i < nx - 1) {
                    int idx_left = j * nx + (i - 1);
                    int idx_right = j * nx + (i + 1);
                    Ex[idx] = -(potential[idx_right] - potential[idx_left]) / 2.0;
                } else if (i == 0) {
                    int idx_right = j * nx + (i + 1);
                    Ex[idx] = -(potential[idx_right] - potential[idx]);
                } else {
                    int idx_left = j * nx + (i - 1);
                    Ex[idx] = -(potential[idx] - potential[idx_left]);
                }
                
                // Ey = -dV/dy
                if (j > 0 && j < ny - 1) {
                    int idx_bottom = (j - 1) * nx + i;
                    int idx_top = (j + 1) * nx + i;
                    Ey[idx] = -(potential[idx_top] - potential[idx_bottom]) / 2.0;
                } else if (j == 0) {
                    int idx_top = (j + 1) * nx + i;
                    Ey[idx] = -(potential[idx_top] - potential[idx]);
                } else {
                    int idx_bottom = (j - 1) * nx + i;
                    Ey[idx] = -(potential[idx] - potential[idx_bottom]);
                }
            }
        }
    }
    
    /**
     * @brief Calculate conduction band edge for doped semiconductor
     */
    double calculate_conduction_band_edge(double N_doping) const {
        double Vt = PhysicalConstants::k * config_.temperature / PhysicalConstants::q;
        double ni = config_.silicon_props.calculate_ni(config_.temperature);
        
        // Simplified band edge calculation
        return Vt * std::log(N_doping / ni);
    }
    
    /**
     * @brief Calculate valence band edge for doped semiconductor
     */
    double calculate_valence_band_edge(double N_doping) const {
        double Vt = PhysicalConstants::k * config_.temperature / PhysicalConstants::q;
        double ni = config_.silicon_props.calculate_ni(config_.temperature);
        
        // Simplified band edge calculation
        return -Vt * std::log(N_doping / ni);
    }
};

// Implementation of NonEquilibriumStatistics methods
void NonEquilibriumStatistics::calculate_fermi_dirac_densities(
    const std::vector<double>& potential,
    const std::vector<double>& quasi_fermi_n,
    const std::vector<double>& quasi_fermi_p,
    const std::vector<double>& Nd,
    const std::vector<double>& Na,
    std::vector<double>& n,
    std::vector<double>& p,
    double T) const {

    double ni = props_.calculate_ni(T);
    double Vt = PhysicalConstants::k * T / PhysicalConstants::q;

    size_t num_points = potential.size();
    n.resize(num_points);
    p.resize(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        double Nc = effective_mass_model_->calculate_Nc(T);
        double Nv = effective_mass_model_->calculate_Nv(T);

        // Calculate bandgap narrowing
        double N_total = Nd[i] + Na[i];
        double delta_Eg = calculate_bandgap_narrowing(N_total);

        // Calculate ionization fractions
        double ionization_n = calculate_ionization_fraction(Nd[i], T);
        double ionization_p = calculate_ionization_fraction(Na[i], T);

        // Effective doping concentrations
        double Nd_eff = Nd[i] * ionization_n;
        double Na_eff = Na[i] * ionization_p;

        // Use Fermi-Dirac statistics in non-equilibrium model
        // Fermi-Dirac statistics
        double eta_n = (quasi_fermi_n[i] - potential[i] + delta_Eg/2.0) / Vt;
        double eta_p = -(quasi_fermi_p[i] - potential[i] - delta_Eg/2.0) / Vt;

        // Joyce-Dixon approximation for F_{1/2}
        auto fermi_half = [](double eta) -> double {
            if (eta < -5.0) return std::exp(eta);
            if (eta > 5.0) return (2.0/3.0) * std::pow(eta, 1.5);

            // Rational approximation
            double eta2 = eta * eta;
            double num = eta + 0.24 * eta2 + 0.056 * eta2 * eta;
            double den = 1.0 + 0.43 * eta + 0.18 * eta2;
            return std::exp(eta) / (1.0 + std::exp(-eta)) * num / den;
        };

        n[i] = Nc * fermi_half(eta_n);
        p[i] = Nv * fermi_half(eta_p);

        // Apply degeneracy corrections
        {
            double deg_factor_n = calculate_degeneracy_factor(n[i], Nc, T);
            double deg_factor_p = calculate_degeneracy_factor(p[i], Nv, T);

            n[i] *= deg_factor_n;
            p[i] *= deg_factor_p;
        }

        // Ensure minimum concentrations
        n[i] = std::max(n[i], ni / 1000.0);
        p[i] = std::max(p[i], ni / 1000.0);
    }
}

// Implementation of EnergyTransportModel methods
void EnergyTransportModel::calculate_carrier_temperatures(
    const std::vector<double>& energy_density_n,
    const std::vector<double>& energy_density_p,
    const std::vector<double>& n,
    const std::vector<double>& p,
    std::vector<double>& T_n,
    std::vector<double>& T_p,
    double lattice_T) const {

    size_t num_points = energy_density_n.size();
    T_n.resize(num_points);
    T_p.resize(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        // Calculate carrier temperatures from energy densities
        // W_n = (3/2) * k * T_n * n  =>  T_n = (2/3) * W_n / (k * n)

        if (n[i] > 1e10) {
            T_n[i] = (2.0/3.0) * energy_density_n[i] / (PhysicalConstants::k * n[i]);
        } else {
            T_n[i] = lattice_T;
        }

        if (p[i] > 1e10) {
            T_p[i] = (2.0/3.0) * energy_density_p[i] / (PhysicalConstants::k * p[i]);
        } else {
            T_p[i] = lattice_T;
        }

        // Ensure reasonable temperature bounds
        T_n[i] = std::max(lattice_T, std::min(T_n[i], 2000.0));
        T_p[i] = std::max(lattice_T, std::min(T_p[i], 2000.0));
    }
}

void EnergyTransportModel::calculate_energy_relaxation(
    const std::vector<double>& T_n,
    const std::vector<double>& T_p,
    const std::vector<double>& n,
    const std::vector<double>& p,
    std::vector<double>& energy_relaxation_n,
    std::vector<double>& energy_relaxation_p,
    double lattice_T) const {

    size_t num_points = T_n.size();
    energy_relaxation_n.resize(num_points);
    energy_relaxation_p.resize(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        // Energy relaxation: R_W = (3/2) * k * n * (T_carrier - T_lattice) / tau_energy

        double tau_energy_n = 0.1e-12; // Default energy relaxation time
        double tau_energy_p = 0.1e-12;

        energy_relaxation_n[i] = (3.0/2.0) * PhysicalConstants::k * n[i] *
                                 (T_n[i] - lattice_T) / tau_energy_n;

        energy_relaxation_p[i] = (3.0/2.0) * PhysicalConstants::k * p[i] *
                                 (T_p[i] - lattice_T) / tau_energy_p;
    }
}

// Implementation of HydrodynamicModel methods
void HydrodynamicModel::calculate_momentum_relaxation(
    const std::vector<double>& velocity_n,
    const std::vector<double>& velocity_p,
    const std::vector<double>& n,
    const std::vector<double>& p,
    std::vector<double>& momentum_relaxation_n,
    std::vector<double>& momentum_relaxation_p) const {

    size_t num_points = velocity_n.size();
    momentum_relaxation_n.resize(num_points);
    momentum_relaxation_p.resize(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        // Momentum relaxation: R_momentum = m_eff * n * v / tau_momentum

        double m_eff_n = 0.26 * PhysicalConstants::m0; // Electron effective mass
        double m_eff_p = 0.39 * PhysicalConstants::m0; // Hole effective mass

        double tau_momentum_n = 0.1e-12; // Default momentum relaxation time
        double tau_momentum_p = 0.1e-12;

        momentum_relaxation_n[i] = m_eff_n * n[i] * velocity_n[i] / tau_momentum_n;
        momentum_relaxation_p[i] = m_eff_p * p[i] * velocity_p[i] / tau_momentum_p;
    }
}

void HydrodynamicModel::calculate_pressure_gradients(
    const std::vector<double>& n,
    const std::vector<double>& p,
    const std::vector<double>& T_n,
    const std::vector<double>& T_p,
    std::vector<double>& pressure_grad_n,
    std::vector<double>& pressure_grad_p) const {

    size_t num_points = n.size();
    pressure_grad_n.resize(num_points);
    pressure_grad_p.resize(num_points);

    // Simplified finite difference for pressure gradients
    for (size_t i = 0; i < num_points; ++i) {
        if (i > 0 && i < num_points - 1) {
            // Pressure = n * k * T
            double P_n_left = n[i-1] * PhysicalConstants::k * T_n[i-1];
            double P_n_right = n[i+1] * PhysicalConstants::k * T_n[i+1];
            double P_p_left = p[i-1] * PhysicalConstants::k * T_p[i-1];
            double P_p_right = p[i+1] * PhysicalConstants::k * T_p[i+1];

            // Gradient (simplified, assuming unit spacing)
            pressure_grad_n[i] = -(P_n_right - P_n_left) / 2.0;
            pressure_grad_p[i] = -(P_p_right - P_p_left) / 2.0;
        } else {
            pressure_grad_n[i] = 0.0;
            pressure_grad_p[i] = 0.0;
        }
    }
}

void HydrodynamicModel::calculate_heat_flow(
    const std::vector<double>& T_n,
    const std::vector<double>& T_p,
    const std::vector<double>& lattice_temp,
    std::vector<double>& heat_flow_n,
    std::vector<double>& heat_flow_p) const {

    size_t num_points = T_n.size();
    heat_flow_n.resize(num_points);
    heat_flow_p.resize(num_points);

    // Simplified finite difference for heat flow
    for (size_t i = 0; i < num_points; ++i) {
        if (i > 0 && i < num_points - 1) {
            // Heat flow: q = -kappa * grad(T)
            double grad_T_n = (T_n[i+1] - T_n[i-1]) / 2.0;
            double grad_T_p = (T_p[i+1] - T_p[i-1]) / 2.0;

            double thermal_conductivity = 150.0; // Default thermal conductivity
            heat_flow_n[i] = -thermal_conductivity * grad_T_n;
            heat_flow_p[i] = -thermal_conductivity * grad_T_p;
        } else {
            heat_flow_n[i] = 0.0;
            heat_flow_p[i] = 0.0;
        }
    }
}

} // namespace Physics
} // namespace SemiDGFEM
