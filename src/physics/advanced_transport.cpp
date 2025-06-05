/**
 * Implementation of Advanced Transport Models
 * Non-equilibrium statistics, energy transport, and hydrodynamics
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "advanced_transport.hpp"
#include "performance_optimization.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace simulator {
namespace transport {

AdvancedTransportSolver::AdvancedTransportSolver(const Device& device, Method method, 
                                                MeshType mesh_type, 
                                                SemiDGFEM::Physics::TransportModel transport_model,
                                                int order)
    : device_(device), method_(method), mesh_type_(mesh_type), 
      transport_model_(transport_model), order_(order), convergence_residual_(0.0) {
    
    validate_order();
    
    // Initialize base solvers
    poisson_ = std::make_unique<Poisson>(device, method, mesh_type);
    drift_diffusion_ = std::make_unique<DriftDiffusion>(device, method, mesh_type);
    
    // Initialize physics models
    initialize_physics_models();
}

AdvancedTransportSolver::~AdvancedTransportSolver() = default;

AdvancedTransportSolver::AdvancedTransportSolver(const AdvancedTransportSolver& other)
    : device_(other.device_), method_(other.method_), mesh_type_(other.mesh_type_),
      transport_model_(other.transport_model_), order_(other.order_),
      physics_config_(other.physics_config_), Nd_(other.Nd_), Na_(other.Na_), Et_(other.Et_),
      convergence_residual_(other.convergence_residual_) {
    
    // Deep copy base solvers
    poisson_ = std::make_unique<Poisson>(*other.poisson_);
    drift_diffusion_ = std::make_unique<DriftDiffusion>(*other.drift_diffusion_);
    
    // Reinitialize physics models
    initialize_physics_models();
}

AdvancedTransportSolver& AdvancedTransportSolver::operator=(const AdvancedTransportSolver& other) {
    if (this != &other) {
        // Note: device_ is const reference, cannot be reassigned
        method_ = other.method_;
        mesh_type_ = other.mesh_type_;
        transport_model_ = other.transport_model_;
        order_ = other.order_;
        physics_config_ = other.physics_config_;
        Nd_ = other.Nd_;
        Na_ = other.Na_;
        Et_ = other.Et_;
        convergence_residual_ = other.convergence_residual_;

        // Deep copy base solvers
        poisson_ = std::make_unique<Poisson>(*other.poisson_);
        drift_diffusion_ = std::make_unique<DriftDiffusion>(*other.drift_diffusion_);

        // Reinitialize physics models
        initialize_physics_models();
    }
    return *this;
}

AdvancedTransportSolver::AdvancedTransportSolver(AdvancedTransportSolver&& other) noexcept
    : device_(other.device_), method_(other.method_), mesh_type_(other.mesh_type_),
      transport_model_(other.transport_model_), order_(other.order_),
      physics_config_(std::move(other.physics_config_)),
      non_eq_stats_(std::move(other.non_eq_stats_)),
      energy_transport_(std::move(other.energy_transport_)),
      hydrodynamic_(std::move(other.hydrodynamic_)),
      poisson_(std::move(other.poisson_)),
      drift_diffusion_(std::move(other.drift_diffusion_)),
      Nd_(std::move(other.Nd_)), Na_(std::move(other.Na_)), Et_(std::move(other.Et_)),
      convergence_residual_(other.convergence_residual_) {
}

AdvancedTransportSolver& AdvancedTransportSolver::operator=(AdvancedTransportSolver&& other) noexcept {
    if (this != &other) {
        // Note: device_ is const reference, cannot be reassigned
        method_ = other.method_;
        mesh_type_ = other.mesh_type_;
        transport_model_ = other.transport_model_;
        order_ = other.order_;
        physics_config_ = std::move(other.physics_config_);
        non_eq_stats_ = std::move(other.non_eq_stats_);
        energy_transport_ = std::move(other.energy_transport_);
        hydrodynamic_ = std::move(other.hydrodynamic_);
        poisson_ = std::move(other.poisson_);
        drift_diffusion_ = std::move(other.drift_diffusion_);
        Nd_ = std::move(other.Nd_);
        Na_ = std::move(other.Na_);
        Et_ = std::move(other.Et_);
        convergence_residual_ = other.convergence_residual_;
    }
    return *this;
}

void AdvancedTransportSolver::set_physics_config(const SemiDGFEM::Physics::PhysicsConfig& config) {
    physics_config_ = config;
    initialize_physics_models();
}

void AdvancedTransportSolver::set_doping(const std::vector<double>& Nd, const std::vector<double>& Na) {
    Nd_ = Nd;
    Na_ = Na;
    
    // Also set doping in base drift-diffusion solver
    if (drift_diffusion_) {
        drift_diffusion_->set_doping(Nd, Na);
    }
}

void AdvancedTransportSolver::set_trap_level(const std::vector<double>& Et) {
    Et_ = Et;
    
    // Also set trap levels in base drift-diffusion solver
    if (drift_diffusion_) {
        drift_diffusion_->set_trap_level(Et);
    }
}

std::map<std::string, std::vector<double>> AdvancedTransportSolver::solve_transport(
    const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
    int poisson_max_iter, double poisson_tol) {
    
    validate();
    validate_inputs(bc);
    validate_doping();
    
    switch (transport_model_) {
        case SemiDGFEM::Physics::TransportModel::DRIFT_DIFFUSION:
            return solve_drift_diffusion_transport(bc, Vg, max_steps, use_amr, poisson_max_iter, poisson_tol);
            
        case SemiDGFEM::Physics::TransportModel::ENERGY_TRANSPORT:
            return solve_energy_transport(bc, Vg, max_steps, use_amr, poisson_max_iter, poisson_tol);
            
        case SemiDGFEM::Physics::TransportModel::HYDRODYNAMIC:
            return solve_hydrodynamic_transport(bc, Vg, max_steps, use_amr, poisson_max_iter, poisson_tol);
            
        case SemiDGFEM::Physics::TransportModel::NON_EQUILIBRIUM_STATISTICS:
            return solve_non_equilibrium_transport(bc, Vg, max_steps, use_amr, poisson_max_iter, poisson_tol);
            
        default:
            throw std::invalid_argument("Unknown transport model");
    }
}

bool AdvancedTransportSolver::is_valid() const {
    return device_.is_valid() && poisson_ && drift_diffusion_ && 
           poisson_->is_valid() && drift_diffusion_->is_valid();
}

void AdvancedTransportSolver::validate() const {
    if (!is_valid()) {
        throw std::runtime_error("AdvancedTransportSolver is in invalid state");
    }
}

size_t AdvancedTransportSolver::get_dof_count() const {
    return drift_diffusion_ ? drift_diffusion_->get_dof_count() : 0;
}

double AdvancedTransportSolver::get_convergence_residual() const {
    return convergence_residual_;
}

void AdvancedTransportSolver::initialize_physics_models() {
    // Initialize physics models based on configuration
    non_eq_stats_ = std::make_unique<SemiDGFEM::Physics::NonEquilibriumStatistics>(
        physics_config_.silicon_props);

    energy_transport_ = std::make_unique<SemiDGFEM::Physics::EnergyTransportModel>(
        physics_config_.energy_transport_config, physics_config_.silicon_props);

    hydrodynamic_ = std::make_unique<SemiDGFEM::Physics::HydrodynamicModel>(
        physics_config_.hydrodynamic_config, physics_config_.silicon_props);
}

void AdvancedTransportSolver::validate_inputs(const std::vector<double>& bc) const {
    if (bc.size() != 4) {
        throw std::invalid_argument("Boundary conditions must have exactly 4 elements");
    }
    
    for (double val : bc) {
        if (!std::isfinite(val)) {
            throw std::invalid_argument("Boundary conditions must be finite");
        }
    }
}

void AdvancedTransportSolver::validate_doping() const {
    if (Nd_.empty() || Na_.empty()) {
        throw std::runtime_error("Doping concentrations must be set before solving");
    }
    
    if (Nd_.size() != Na_.size()) {
        throw std::runtime_error("Nd and Na arrays must have the same size");
    }
    
    for (size_t i = 0; i < Nd_.size(); ++i) {
        if (Nd_[i] < 0.0 || Na_[i] < 0.0) {
            throw std::invalid_argument("Doping concentrations must be non-negative");
        }
        if (!std::isfinite(Nd_[i]) || !std::isfinite(Na_[i])) {
            throw std::invalid_argument("Doping concentrations must be finite");
        }
    }
}

void AdvancedTransportSolver::validate_order() const {
    if (order_ < 1 || order_ > 3) {
        throw std::invalid_argument("Polynomial order must be between 1 and 3");
    }
}

std::map<std::string, std::vector<double>> AdvancedTransportSolver::solve_drift_diffusion_transport(
    const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
    int poisson_max_iter, double poisson_tol) {

    // Use the base drift-diffusion solver with advanced physics
    auto results = drift_diffusion_->solve_drift_diffusion(bc, Vg, max_steps, use_amr,
                                                          poisson_max_iter, poisson_tol);

    // Apply advanced physics corrections if enabled
    if (physics_config_.enable_temperature_dependence) {

        std::vector<double> n_corrected, p_corrected;
        // Use non-equilibrium statistics for carrier density calculation
        std::vector<double> quasi_fermi_n = results["potential"]; // Simplified
        std::vector<double> quasi_fermi_p = results["potential"]; // Simplified

        non_eq_stats_->calculate_fermi_dirac_densities(
            results["potential"], quasi_fermi_n, quasi_fermi_p, Nd_, Na_,
            n_corrected, p_corrected, physics_config_.temperature);

        results["n"] = n_corrected;
        results["p"] = p_corrected;
    }

    convergence_residual_ = drift_diffusion_->get_convergence_residual();
    return results;
}

std::map<std::string, std::vector<double>> AdvancedTransportSolver::solve_energy_transport(
    const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
    int poisson_max_iter, double poisson_tol) {

    performance::PROFILE_FUNCTION();

    try {
        size_t dof_count = get_dof_count();

        // Initialize solution vectors
        std::vector<double> V(dof_count, 0.0);
        std::vector<double> n(dof_count, 1e10);
        std::vector<double> p(dof_count, 1e10);
        std::vector<double> energy_n(dof_count, 0.0);
        std::vector<double> energy_p(dof_count, 0.0);
        std::vector<double> T_n(dof_count, physics_config_.temperature);
        std::vector<double> T_p(dof_count, physics_config_.temperature);
        std::vector<double> Jn(dof_count, 0.0);
        std::vector<double> Jp(dof_count, 0.0);

        // Initialize energy densities
        for (size_t i = 0; i < dof_count; ++i) {
            energy_n[i] = (3.0/2.0) * SemiDGFEM::Physics::PhysicalConstants::k *
                         physics_config_.temperature * n[i];
            energy_p[i] = (3.0/2.0) * SemiDGFEM::Physics::PhysicalConstants::k *
                         physics_config_.temperature * p[i];
        }

        // Store previous solutions for convergence checking
        std::vector<double> V_old = V;
        std::vector<double> n_old = n;
        std::vector<double> p_old = p;
        std::vector<double> energy_n_old = energy_n;
        std::vector<double> energy_p_old = energy_p;

        // Main iteration loop
        for (int step = 0; step < max_steps; ++step) {
            // Update carrier temperatures from energy densities
            energy_transport_->calculate_carrier_temperatures(
                energy_n, energy_p, n, p, T_n, T_p, physics_config_.temperature);

            // Update carrier densities with temperature-dependent statistics
            std::vector<double> quasi_fermi_n = V; // Simplified
            std::vector<double> quasi_fermi_p = V; // Simplified
            non_eq_stats_->calculate_fermi_dirac_densities(
                V, quasi_fermi_n, quasi_fermi_p, Nd_, Na_, n, p, physics_config_.temperature);

            // Solve Poisson equation with updated charge density
            std::vector<double> rho(dof_count);
            const double q = SemiDGFEM::Physics::PhysicalConstants::q;
            for (size_t i = 0; i < dof_count; ++i) {
                double Nd_val = (i < Nd_.size()) ? Nd_[i] : 0.0;
                double Na_val = (i < Na_.size()) ? Na_[i] : 0.0;
                rho[i] = q * (p[i] - n[i] + Nd_val - Na_val);
            }

            poisson_->set_charge_density(rho);
            V = poisson_->solve_2d(bc);

            // Update energy densities
            compute_energy_densities(V, n, p, energy_n, energy_p);

            // Calculate energy relaxation
            std::vector<double> energy_relaxation_n, energy_relaxation_p;
            energy_transport_->calculate_energy_relaxation(
                T_n, T_p, n, p, energy_relaxation_n, energy_relaxation_p, physics_config_.temperature);

            // Update energy densities with relaxation
            for (size_t i = 0; i < dof_count; ++i) {
                energy_n[i] -= energy_relaxation_n[i] * 1e-12; // Time step
                energy_p[i] -= energy_relaxation_p[i] * 1e-12;

                // Ensure minimum energy
                double min_energy = (3.0/2.0) * SemiDGFEM::Physics::PhysicalConstants::k *
                                   physics_config_.temperature * n[i];
                energy_n[i] = std::max(energy_n[i], min_energy);
                energy_p[i] = std::max(energy_p[i], min_energy);
            }

            // Check convergence
            bool converged = check_convergence(V_old, V, physics_config_.dd_tolerance) &&
                           check_convergence(energy_n_old, energy_n, physics_config_.energy_tolerance) &&
                           check_convergence(energy_p_old, energy_p, physics_config_.energy_tolerance);

            if (converged) {
                convergence_residual_ = 0.0;
                for (size_t i = 0; i < V.size(); ++i) {
                    convergence_residual_ += std::abs(V[i] - V_old[i]);
                }
                convergence_residual_ /= V.size();
                break;
            }

            // Update old solutions
            V_old = V;
            n_old = n;
            p_old = p;
            energy_n_old = energy_n;
            energy_p_old = energy_p;
        }

        // Prepare results
        std::map<std::string, std::vector<double>> results;
        results["potential"] = V;
        results["n"] = n;
        results["p"] = p;
        results["Jn"] = Jn;
        results["Jp"] = Jp;
        results["energy_n"] = energy_n;
        results["energy_p"] = energy_p;
        results["T_n"] = T_n;
        results["T_p"] = T_p;

        return results;

    } catch (const std::exception& e) {
        throw std::runtime_error("Energy transport solve failed: " + std::string(e.what()));
    }
}

std::map<std::string, std::vector<double>> AdvancedTransportSolver::solve_hydrodynamic_transport(
    const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
    int poisson_max_iter, double poisson_tol) {

    performance::PROFILE_FUNCTION();

    try {
        size_t dof_count = get_dof_count();

        // Initialize solution vectors for hydrodynamic model
        std::vector<double> V(dof_count, 0.0);
        std::vector<double> n(dof_count, 1e10);
        std::vector<double> p(dof_count, 1e10);
        std::vector<double> velocity_n(dof_count, 0.0);
        std::vector<double> velocity_p(dof_count, 0.0);
        std::vector<double> T_n(dof_count, physics_config_.temperature);
        std::vector<double> T_p(dof_count, physics_config_.temperature);
        std::vector<double> momentum_n(dof_count, 0.0);
        std::vector<double> momentum_p(dof_count, 0.0);
        std::vector<double> Jn(dof_count, 0.0);
        std::vector<double> Jp(dof_count, 0.0);

        // Store previous solutions for convergence checking
        std::vector<double> V_old = V;
        std::vector<double> velocity_n_old = velocity_n;
        std::vector<double> velocity_p_old = velocity_p;
        std::vector<double> T_n_old = T_n;
        std::vector<double> T_p_old = T_p;

        // Main iteration loop
        for (int step = 0; step < max_steps; ++step) {
            // Update carrier densities
            std::vector<double> quasi_fermi_n = V; // Simplified
            std::vector<double> quasi_fermi_p = V; // Simplified
            non_eq_stats_->calculate_fermi_dirac_densities(
                V, quasi_fermi_n, quasi_fermi_p, Nd_, Na_, n, p, physics_config_.temperature);

            // Calculate momentum densities
            compute_momentum_densities(V, n, p, momentum_n, momentum_p);

            // Update velocities from momentum densities
            const double m_eff_n = 0.26 * SemiDGFEM::Physics::PhysicalConstants::m0;
            const double m_eff_p = 0.39 * SemiDGFEM::Physics::PhysicalConstants::m0;

            for (size_t i = 0; i < dof_count; ++i) {
                if (n[i] > 1e10) {
                    velocity_n[i] = momentum_n[i] / (m_eff_n * n[i]);
                }
                if (p[i] > 1e10) {
                    velocity_p[i] = momentum_p[i] / (m_eff_p * p[i]);
                }
            }

            // Calculate momentum relaxation
            std::vector<double> momentum_relaxation_n, momentum_relaxation_p;
            hydrodynamic_->calculate_momentum_relaxation(
                velocity_n, velocity_p, n, p, momentum_relaxation_n, momentum_relaxation_p);

            // Calculate pressure gradients
            std::vector<double> pressure_grad_n, pressure_grad_p;
            hydrodynamic_->calculate_pressure_gradients(
                n, p, T_n, T_p, pressure_grad_n, pressure_grad_p);

            // Calculate heat flow
            std::vector<double> lattice_temp(dof_count, physics_config_.temperature);
            std::vector<double> heat_flow_n, heat_flow_p;
            hydrodynamic_->calculate_heat_flow(T_n, T_p, lattice_temp, heat_flow_n, heat_flow_p);

            // Update momentum densities with relaxation and pressure effects
            for (size_t i = 0; i < dof_count; ++i) {
                momentum_n[i] -= momentum_relaxation_n[i] * 1e-12; // Time step
                momentum_p[i] -= momentum_relaxation_p[i] * 1e-12;

                // Add pressure gradient effects
                momentum_n[i] += pressure_grad_n[i] * 1e-12;
                momentum_p[i] += pressure_grad_p[i] * 1e-12;
            }

            // Solve Poisson equation
            std::vector<double> rho(dof_count);
            const double q = SemiDGFEM::Physics::PhysicalConstants::q;
            for (size_t i = 0; i < dof_count; ++i) {
                double Nd_val = (i < Nd_.size()) ? Nd_[i] : 0.0;
                double Na_val = (i < Na_.size()) ? Na_[i] : 0.0;
                rho[i] = q * (p[i] - n[i] + Nd_val - Na_val);
            }

            poisson_->set_charge_density(rho);
            V = poisson_->solve_2d(bc);

            // Check convergence
            bool converged = check_convergence(V_old, V, physics_config_.dd_tolerance) &&
                           check_convergence(velocity_n_old, velocity_n, physics_config_.momentum_tolerance) &&
                           check_convergence(velocity_p_old, velocity_p, physics_config_.momentum_tolerance);

            if (converged) {
                convergence_residual_ = 0.0;
                for (size_t i = 0; i < V.size(); ++i) {
                    convergence_residual_ += std::abs(V[i] - V_old[i]);
                }
                convergence_residual_ /= V.size();
                break;
            }

            // Update old solutions
            V_old = V;
            velocity_n_old = velocity_n;
            velocity_p_old = velocity_p;
            T_n_old = T_n;
            T_p_old = T_p;
        }

        // Prepare results
        std::map<std::string, std::vector<double>> results;
        results["potential"] = V;
        results["n"] = n;
        results["p"] = p;
        results["Jn"] = Jn;
        results["Jp"] = Jp;
        results["velocity_n"] = velocity_n;
        results["velocity_p"] = velocity_p;
        results["momentum_n"] = momentum_n;
        results["momentum_p"] = momentum_p;
        results["T_n"] = T_n;
        results["T_p"] = T_p;

        return results;

    } catch (const std::exception& e) {
        throw std::runtime_error("Hydrodynamic transport solve failed: " + std::string(e.what()));
    }
}

std::map<std::string, std::vector<double>> AdvancedTransportSolver::solve_non_equilibrium_transport(
    const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
    int poisson_max_iter, double poisson_tol) {

    performance::PROFILE_FUNCTION();

    try {
        size_t dof_count = get_dof_count();

        // Initialize solution vectors
        std::vector<double> V(dof_count, 0.0);
        std::vector<double> n(dof_count, 1e10);
        std::vector<double> p(dof_count, 1e10);
        std::vector<double> quasi_fermi_n(dof_count, 0.0);
        std::vector<double> quasi_fermi_p(dof_count, 0.0);
        std::vector<double> Jn(dof_count, 0.0);
        std::vector<double> Jp(dof_count, 0.0);

        // Store previous solutions for convergence checking
        std::vector<double> V_old = V;
        std::vector<double> quasi_fermi_n_old = quasi_fermi_n;
        std::vector<double> quasi_fermi_p_old = quasi_fermi_p;

        // Main iteration loop
        for (int step = 0; step < max_steps; ++step) {
            // Update carrier densities using Fermi-Dirac statistics
            non_eq_stats_->calculate_fermi_dirac_densities(
                V, quasi_fermi_n, quasi_fermi_p, Nd_, Na_, n, p, physics_config_.temperature);

            // Solve Poisson equation
            std::vector<double> rho(dof_count);
            const double q = SemiDGFEM::Physics::PhysicalConstants::q;
            for (size_t i = 0; i < dof_count; ++i) {
                double Nd_val = (i < Nd_.size()) ? Nd_[i] : 0.0;
                double Na_val = (i < Na_.size()) ? Na_[i] : 0.0;
                rho[i] = q * (p[i] - n[i] + Nd_val - Na_val);
            }

            poisson_->set_charge_density(rho);
            V = poisson_->solve_2d(bc);

            // Update quasi-Fermi levels (simplified approach)
            const double Vt = SemiDGFEM::Physics::PhysicalConstants::k * physics_config_.temperature /
                             SemiDGFEM::Physics::PhysicalConstants::q;

            for (size_t i = 0; i < dof_count; ++i) {
                // Simplified quasi-Fermi level calculation
                double ni = 1e16; // Intrinsic concentration
                quasi_fermi_n[i] = V[i] + Vt * std::log(n[i] / ni);
                quasi_fermi_p[i] = V[i] - Vt * std::log(p[i] / ni);
            }

            // Check convergence
            bool converged = check_convergence(V_old, V, physics_config_.dd_tolerance) &&
                           check_convergence(quasi_fermi_n_old, quasi_fermi_n, physics_config_.dd_tolerance) &&
                           check_convergence(quasi_fermi_p_old, quasi_fermi_p, physics_config_.dd_tolerance);

            if (converged) {
                convergence_residual_ = 0.0;
                for (size_t i = 0; i < V.size(); ++i) {
                    convergence_residual_ += std::abs(V[i] - V_old[i]);
                }
                convergence_residual_ /= V.size();
                break;
            }

            // Update old solutions
            V_old = V;
            quasi_fermi_n_old = quasi_fermi_n;
            quasi_fermi_p_old = quasi_fermi_p;
        }

        // Prepare results
        std::map<std::string, std::vector<double>> results;
        results["potential"] = V;
        results["n"] = n;
        results["p"] = p;
        results["Jn"] = Jn;
        results["Jp"] = Jp;
        results["quasi_fermi_n"] = quasi_fermi_n;
        results["quasi_fermi_p"] = quasi_fermi_p;

        return results;

    } catch (const std::exception& e) {
        throw std::runtime_error("Non-equilibrium transport solve failed: " + std::string(e.what()));
    }
}

// Helper method implementations
void AdvancedTransportSolver::compute_advanced_carrier_densities(
    const std::vector<double>& V,
    const std::vector<double>& quasi_fermi_n,
    const std::vector<double>& quasi_fermi_p,
    std::vector<double>& n,
    std::vector<double>& p) const {

    non_eq_stats_->calculate_fermi_dirac_densities(
        V, quasi_fermi_n, quasi_fermi_p, Nd_, Na_, n, p, physics_config_.temperature);
}

void AdvancedTransportSolver::compute_energy_densities(
    const std::vector<double>& V,
    const std::vector<double>& n,
    const std::vector<double>& p,
    std::vector<double>& energy_n,
    std::vector<double>& energy_p) const {

    // Simplified energy density calculation
    // In a full implementation, this would solve the energy transport equations
    for (size_t i = 0; i < V.size(); ++i) {
        energy_n[i] = (3.0/2.0) * SemiDGFEM::Physics::PhysicalConstants::k *
                     physics_config_.temperature * n[i];
        energy_p[i] = (3.0/2.0) * SemiDGFEM::Physics::PhysicalConstants::k *
                     physics_config_.temperature * p[i];
    }
}

void AdvancedTransportSolver::compute_momentum_densities(
    const std::vector<double>& V,
    const std::vector<double>& n,
    const std::vector<double>& p,
    std::vector<double>& momentum_n,
    std::vector<double>& momentum_p) const {

    // Simplified momentum density calculation
    const double m_eff_n = 0.26 * SemiDGFEM::Physics::PhysicalConstants::m0;
    const double m_eff_p = 0.39 * SemiDGFEM::Physics::PhysicalConstants::m0;

    for (size_t i = 0; i < V.size(); ++i) {
        // Initial momentum is zero (equilibrium)
        momentum_n[i] = 0.0;
        momentum_p[i] = 0.0;
    }
}

bool AdvancedTransportSolver::check_convergence(
    const std::vector<double>& old_solution,
    const std::vector<double>& new_solution,
    double tolerance) const {

    if (old_solution.size() != new_solution.size()) return false;

    double max_diff = 0.0;
    for (size_t i = 0; i < old_solution.size(); ++i) {
        double diff = std::abs(new_solution[i] - old_solution[i]);
        max_diff = std::max(max_diff, diff);
    }

    return max_diff < tolerance;
}

// Factory class implementation
std::unique_ptr<AdvancedTransportSolver> TransportSolverFactory::create_solver(
    const Device& device, Method method, MeshType mesh_type,
    SemiDGFEM::Physics::TransportModel transport_model, int order) {

    return std::make_unique<AdvancedTransportSolver>(device, method, mesh_type, transport_model, order);
}

std::vector<std::string> TransportSolverFactory::get_available_transport_models() {
    return {
        "DRIFT_DIFFUSION",
        "ENERGY_TRANSPORT",
        "HYDRODYNAMIC",
        "NON_EQUILIBRIUM_STATISTICS"
    };
}

std::string TransportSolverFactory::get_transport_model_description(SemiDGFEM::Physics::TransportModel model) {
    switch (model) {
        case SemiDGFEM::Physics::TransportModel::DRIFT_DIFFUSION:
            return "Classical drift-diffusion transport with Boltzmann statistics";
        case SemiDGFEM::Physics::TransportModel::ENERGY_TRANSPORT:
            return "Energy transport model with hot carrier effects";
        case SemiDGFEM::Physics::TransportModel::HYDRODYNAMIC:
            return "Hydrodynamic model with momentum and energy conservation";
        case SemiDGFEM::Physics::TransportModel::NON_EQUILIBRIUM_STATISTICS:
            return "Non-equilibrium transport with Fermi-Dirac statistics";
        default:
            return "Unknown transport model";
    }
}

} // namespace transport

// C interface implementations
extern "C" {
    simulator::transport::AdvancedTransportSolver* create_advanced_transport_solver(
        simulator::Device* device, int method, int mesh_type, int transport_model, int order) {

        if (!device) return nullptr;

        try {
            auto transport_model_enum = static_cast<SemiDGFEM::Physics::TransportModel>(transport_model);
            auto method_enum = static_cast<simulator::Method>(method);
            auto mesh_type_enum = static_cast<simulator::MeshType>(mesh_type);

            return new simulator::transport::AdvancedTransportSolver(
                *device, method_enum, mesh_type_enum, transport_model_enum, order);
        } catch (...) {
            return nullptr;
        }
    }

    void destroy_advanced_transport_solver(simulator::transport::AdvancedTransportSolver* solver) {
        delete solver;
    }

    int advanced_transport_solver_is_valid(simulator::transport::AdvancedTransportSolver* solver) {
        if (!solver) return 0;
        try {
            return solver->is_valid() ? 1 : 0;
        } catch (...) {
            return 0;
        }
    }

    int advanced_transport_solver_set_doping(simulator::transport::AdvancedTransportSolver* solver,
                                            double* Nd, double* Na, int size) {
        if (!solver || !Nd || !Na || size <= 0) return -1;

        try {
            std::vector<double> Nd_vec(Nd, Nd + size);
            std::vector<double> Na_vec(Na, Na + size);
            solver->set_doping(Nd_vec, Na_vec);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int advanced_transport_solver_set_trap_level(simulator::transport::AdvancedTransportSolver* solver,
                                                double* Et, int size) {
        if (!solver || !Et || size <= 0) return -1;

        try {
            std::vector<double> Et_vec(Et, Et + size);
            solver->set_trap_level(Et_vec);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    size_t advanced_transport_solver_get_dof_count(simulator::transport::AdvancedTransportSolver* solver) {
        if (!solver) return 0;
        try {
            return solver->get_dof_count();
        } catch (...) {
            return 0;
        }
    }

    double advanced_transport_solver_get_convergence_residual(simulator::transport::AdvancedTransportSolver* solver) {
        if (!solver) return -1.0;
        try {
            return solver->get_convergence_residual();
        } catch (...) {
            return -1.0;
        }
    }

    int advanced_transport_solver_get_order(simulator::transport::AdvancedTransportSolver* solver) {
        if (!solver) return -1;
        try {
            return solver->get_order();
        } catch (...) {
            return -1;
        }
    }

    int advanced_transport_solver_get_transport_model(simulator::transport::AdvancedTransportSolver* solver) {
        if (!solver) return -1;
        try {
            return static_cast<int>(solver->get_transport_model());
        } catch (...) {
            return -1;
        }
    }
}

} // namespace simulator
