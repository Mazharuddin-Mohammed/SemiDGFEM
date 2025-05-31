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

} // namespace Physics
} // namespace SemiDGFEM
