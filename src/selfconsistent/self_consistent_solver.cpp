/**
 * Implementation of Self-Consistent Solver Framework
 * Coupled solving of Poisson, drift-diffusion, and transport equations
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "self_consistent_solver.hpp"
#include "performance_optimization.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <chrono>

namespace simulator {
namespace selfconsistent {

SelfConsistentSolver::SelfConsistentSolver(const Device& device, Method method, MeshType mesh_type, int order)
    : device_(device), method_(method), mesh_type_(mesh_type), order_(order),
      energy_transport_enabled_(false), hydrodynamic_enabled_(false), quantum_corrections_enabled_(false),
      transport_model_(SemiDGFEM::Physics::TransportModel::DRIFT_DIFFUSION),
      last_iteration_count_(0), last_residual_(0.0) {
    
    validate();
    initialize_solvers();
}

SelfConsistentSolver::~SelfConsistentSolver() = default;

SelfConsistentSolver::SelfConsistentSolver(SelfConsistentSolver&& other) noexcept
    : device_(other.device_), method_(other.method_), mesh_type_(other.mesh_type_), order_(other.order_),
      criteria_(other.criteria_), energy_transport_enabled_(other.energy_transport_enabled_),
      hydrodynamic_enabled_(other.hydrodynamic_enabled_), quantum_corrections_enabled_(other.quantum_corrections_enabled_),
      transport_model_(other.transport_model_), poisson_(std::move(other.poisson_)),
      drift_diffusion_(std::move(other.drift_diffusion_)), advanced_transport_(std::move(other.advanced_transport_)),
      transient_solver_(std::move(other.transient_solver_)), Nd_(std::move(other.Nd_)), Na_(std::move(other.Na_)),
      Et_(std::move(other.Et_)), material_properties_(std::move(other.material_properties_)),
      last_iteration_count_(other.last_iteration_count_), last_residual_(other.last_residual_),
      convergence_history_(std::move(other.convergence_history_)) {
}

SelfConsistentSolver& SelfConsistentSolver::operator=(SelfConsistentSolver&& other) noexcept {
    if (this != &other) {
        criteria_ = other.criteria_;
        energy_transport_enabled_ = other.energy_transport_enabled_;
        hydrodynamic_enabled_ = other.hydrodynamic_enabled_;
        quantum_corrections_enabled_ = other.quantum_corrections_enabled_;
        transport_model_ = other.transport_model_;
        poisson_ = std::move(other.poisson_);
        drift_diffusion_ = std::move(other.drift_diffusion_);
        advanced_transport_ = std::move(other.advanced_transport_);
        transient_solver_ = std::move(other.transient_solver_);
        Nd_ = std::move(other.Nd_);
        Na_ = std::move(other.Na_);
        Et_ = std::move(other.Et_);
        material_properties_ = std::move(other.material_properties_);
        last_iteration_count_ = other.last_iteration_count_;
        last_residual_ = other.last_residual_;
        convergence_history_ = std::move(other.convergence_history_);
    }
    return *this;
}

void SelfConsistentSolver::set_convergence_criteria(const ConvergenceCriteria& criteria) {
    criteria_ = criteria;
    
    // Validate criteria
    if (criteria_.potential_tolerance <= 0.0 || criteria_.density_tolerance <= 0.0 ||
        criteria_.current_tolerance <= 0.0 || criteria_.temperature_tolerance <= 0.0) {
        throw std::invalid_argument("All tolerances must be positive");
    }
    
    if (criteria_.max_iterations <= 0 || criteria_.min_iterations < 0) {
        throw std::invalid_argument("Invalid iteration limits");
    }
    
    if (criteria_.damping_factor <= 0.0 || criteria_.damping_factor > 1.0) {
        throw std::invalid_argument("Damping factor must be in (0, 1]");
    }
}

void SelfConsistentSolver::enable_energy_transport(bool enable) {
    energy_transport_enabled_ = enable;
    if (enable) {
        transport_model_ = SemiDGFEM::Physics::TransportModel::ENERGY_TRANSPORT;
    }
}

void SelfConsistentSolver::enable_hydrodynamic_transport(bool enable) {
    hydrodynamic_enabled_ = enable;
    if (enable) {
        transport_model_ = SemiDGFEM::Physics::TransportModel::HYDRODYNAMIC;
    }
}

void SelfConsistentSolver::enable_quantum_corrections(bool enable) {
    quantum_corrections_enabled_ = enable;
}

void SelfConsistentSolver::set_transport_model(SemiDGFEM::Physics::TransportModel model) {
    transport_model_ = model;
    
    // Update flags based on model
    energy_transport_enabled_ = (model == SemiDGFEM::Physics::TransportModel::ENERGY_TRANSPORT);
    hydrodynamic_enabled_ = (model == SemiDGFEM::Physics::TransportModel::HYDRODYNAMIC);
}

void SelfConsistentSolver::set_doping(const std::vector<double>& Nd, const std::vector<double>& Na) {
    if (Nd.size() != Na.size()) {
        throw std::invalid_argument("Nd and Na arrays must have the same size");
    }
    if (std::any_of(Nd.begin(), Nd.end(), [](double x) { return x < 0; }) ||
        std::any_of(Na.begin(), Na.end(), [](double x) { return x < 0; })) {
        throw std::invalid_argument("Doping concentrations must be non-negative");
    }
    
    Nd_ = Nd;
    Na_ = Na;
    
    // Set doping in all solvers
    if (drift_diffusion_) {
        drift_diffusion_->set_doping(Nd_, Na_);
    }
    if (advanced_transport_) {
        advanced_transport_->set_doping(Nd_, Na_);
    }
}

void SelfConsistentSolver::set_trap_levels(const std::vector<double>& Et) {
    Et_ = Et;
    
    if (drift_diffusion_) {
        drift_diffusion_->set_trap_level(Et_);
    }
    if (advanced_transport_) {
        advanced_transport_->set_trap_level(Et_);
    }
}

void SelfConsistentSolver::set_material_properties(const std::map<std::string, double>& properties) {
    material_properties_ = properties;
    
    // Apply material properties to solvers
    // This would be extended to set specific material parameters
    std::cout << "Material properties set: " << properties.size() << " parameters" << std::endl;
}

bool SelfConsistentSolver::is_valid() const {
    return device_.is_valid() && poisson_ && (drift_diffusion_ || advanced_transport_);
}

void SelfConsistentSolver::validate() const {
    if (!device_.is_valid()) {
        throw std::runtime_error("Device is invalid");
    }
    if (criteria_.potential_tolerance <= 0.0) {
        throw std::runtime_error("Potential tolerance must be positive");
    }
}

size_t SelfConsistentSolver::get_dof_count() const {
    if (advanced_transport_) {
        return advanced_transport_->get_dof_count();
    } else if (drift_diffusion_) {
        return drift_diffusion_->get_dof_count();
    } else {
        throw std::runtime_error("No solver initialized");
    }
}

void SelfConsistentSolver::initialize_solvers() {
    try {
        // Always create Poisson solver
        poisson_ = std::make_unique<Poisson>(device_, method_, mesh_type_);
        
        // Create appropriate transport solver based on model
        if (transport_model_ == SemiDGFEM::Physics::TransportModel::DRIFT_DIFFUSION) {
            drift_diffusion_ = std::make_unique<DriftDiffusion>(device_, method_, mesh_type_, order_);
        } else {
            advanced_transport_ = std::make_unique<transport::AdvancedTransportSolver>(
                device_, method_, mesh_type_, transport_model_, order_);
        }
        
        // Create transient solver for time-dependent problems
        transient_solver_ = std::make_unique<transient::TransientSolver>(device_, method_, mesh_type_, order_);
        
        // Set doping if available
        if (!Nd_.empty() && !Na_.empty()) {
            set_doping(Nd_, Na_);
        }
        
        // Set trap levels if available
        if (!Et_.empty()) {
            set_trap_levels(Et_);
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize solvers: " + std::string(e.what()));
    }
}

void SelfConsistentSolver::validate_solution_state(const SolutionState& state) const {
    size_t dof_count = get_dof_count();
    
    if (state.potential.size() != dof_count ||
        state.electron_density.size() != dof_count ||
        state.hole_density.size() != dof_count) {
        throw std::invalid_argument("Solution state arrays must match DOF count");
    }
    
    if (energy_transport_enabled_) {
        if (!state.has_energy_transport ||
            state.electron_temperature.size() != dof_count ||
            state.hole_temperature.size() != dof_count) {
            throw std::invalid_argument("Energy transport state required");
        }
    }
    
    if (hydrodynamic_enabled_) {
        if (!state.has_hydrodynamic ||
            state.electron_momentum_x.size() != dof_count ||
            state.electron_momentum_y.size() != dof_count ||
            state.hole_momentum_x.size() != dof_count ||
            state.hole_momentum_y.size() != dof_count) {
            throw std::invalid_argument("Hydrodynamic state required");
        }
    }
    
    // Check for physical validity
    if (std::any_of(state.electron_density.begin(), state.electron_density.end(), [](double x) { return x <= 0; }) ||
        std::any_of(state.hole_density.begin(), state.hole_density.end(), [](double x) { return x <= 0; })) {
        throw std::invalid_argument("Carrier densities must be positive");
    }
    
    if (energy_transport_enabled_) {
        if (std::any_of(state.electron_temperature.begin(), state.electron_temperature.end(), [](double x) { return x <= 0; }) ||
            std::any_of(state.hole_temperature.begin(), state.hole_temperature.end(), [](double x) { return x <= 0; })) {
            throw std::invalid_argument("Temperatures must be positive");
        }
    }
}

SolutionState SelfConsistentSolver::solve_steady_state(const std::vector<double>& boundary_conditions,
                                                      const SolutionState& initial_guess) {
    validate_solution_state(initial_guess);
    
    if (boundary_conditions.size() != 4) {
        throw std::invalid_argument("Boundary conditions must have 4 values");
    }
    
    std::cout << "Starting self-consistent steady-state solution..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SolutionState current_state = initial_guess.copy();
    SolutionState previous_state = current_state.copy();
    
    convergence_history_.clear();
    last_iteration_count_ = 0;
    last_residual_ = 1e10;
    
    for (int iter = 0; iter < criteria_.max_iterations; ++iter) {
        // Store previous state
        previous_state = current_state.copy();
        
        // Perform self-consistent iteration
        current_state = perform_self_consistent_iteration(current_state, boundary_conditions);
        
        // Apply damping if enabled
        if (criteria_.use_damping && iter > 0) {
            current_state = apply_damping(previous_state, current_state);
        }
        
        // Compute residual and check convergence
        double residual = compute_residual(previous_state, current_state);
        convergence_history_.push_back(residual);
        last_residual_ = residual;
        last_iteration_count_ = iter + 1;
        
        // Output progress
        if (iter % 10 == 0 || iter < 5) {
            std::cout << "  Iteration " << iter + 1 << ", residual = " << residual << std::endl;
        }
        
        // Check convergence
        if (iter >= criteria_.min_iterations && check_convergence(previous_state, current_state)) {
            std::cout << "  Converged after " << iter + 1 << " iterations" << std::endl;
            break;
        }
        
        if (iter == criteria_.max_iterations - 1) {
            std::cout << "  Warning: Maximum iterations reached without convergence" << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Self-consistent solution completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Final residual: " << last_residual_ << std::endl;
    
    return current_state;
}

SolutionState SelfConsistentSolver::perform_self_consistent_iteration(const SolutionState& current_state,
                                                                   const std::vector<double>& boundary_conditions) {
    SolutionState new_state = current_state.copy();

    try {
        // Step 1: Update charge density based on current carrier densities
        update_charge_density(new_state);

        // Step 2: Solve Poisson equation with updated charge density
        solve_poisson_step(new_state, boundary_conditions);

        // Step 3: Solve transport equations with updated potential
        solve_transport_step(new_state, boundary_conditions);

        // Step 4: Update generation-recombination rates
        update_generation_recombination(new_state);

        // Step 5: Apply quantum corrections if enabled
        if (quantum_corrections_enabled_) {
            apply_quantum_corrections(new_state);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in self-consistent iteration: " << e.what() << std::endl;
        throw;
    }

    return new_state;
}

bool SelfConsistentSolver::check_convergence(const SolutionState& old_state, const SolutionState& new_state) {
    size_t dof_count = old_state.potential.size();

    // Check potential convergence
    double max_potential_change = 0.0;
    for (size_t i = 0; i < dof_count; ++i) {
        double change = std::abs(new_state.potential[i] - old_state.potential[i]);
        max_potential_change = std::max(max_potential_change, change);
    }

    if (max_potential_change > criteria_.potential_tolerance) {
        return false;
    }

    // Check carrier density convergence
    double max_n_rel_change = 0.0;
    double max_p_rel_change = 0.0;

    for (size_t i = 0; i < dof_count; ++i) {
        double n_rel_change = std::abs(new_state.electron_density[i] - old_state.electron_density[i]) /
                              std::max(new_state.electron_density[i], old_state.electron_density[i]);
        double p_rel_change = std::abs(new_state.hole_density[i] - old_state.hole_density[i]) /
                              std::max(new_state.hole_density[i], old_state.hole_density[i]);

        max_n_rel_change = std::max(max_n_rel_change, n_rel_change);
        max_p_rel_change = std::max(max_p_rel_change, p_rel_change);
    }

    if (max_n_rel_change > criteria_.density_tolerance || max_p_rel_change > criteria_.density_tolerance) {
        return false;
    }

    // Check temperature convergence for energy transport
    if (energy_transport_enabled_) {
        double max_Tn_rel_change = 0.0;
        double max_Tp_rel_change = 0.0;

        for (size_t i = 0; i < dof_count; ++i) {
            double Tn_rel_change = std::abs(new_state.electron_temperature[i] - old_state.electron_temperature[i]) /
                                   std::max(new_state.electron_temperature[i], old_state.electron_temperature[i]);
            double Tp_rel_change = std::abs(new_state.hole_temperature[i] - old_state.hole_temperature[i]) /
                                   std::max(new_state.hole_temperature[i], old_state.hole_temperature[i]);

            max_Tn_rel_change = std::max(max_Tn_rel_change, Tn_rel_change);
            max_Tp_rel_change = std::max(max_Tp_rel_change, Tp_rel_change);
        }

        if (max_Tn_rel_change > criteria_.temperature_tolerance || max_Tp_rel_change > criteria_.temperature_tolerance) {
            return false;
        }
    }

    return true;
}

double SelfConsistentSolver::compute_residual(const SolutionState& old_state, const SolutionState& new_state) {
    size_t dof_count = old_state.potential.size();
    double residual = 0.0;

    // Potential residual
    for (size_t i = 0; i < dof_count; ++i) {
        double diff = new_state.potential[i] - old_state.potential[i];
        residual += diff * diff;
    }

    // Carrier density residual
    for (size_t i = 0; i < dof_count; ++i) {
        double n_diff = (new_state.electron_density[i] - old_state.electron_density[i]) /
                       std::max(new_state.electron_density[i], old_state.electron_density[i]);
        double p_diff = (new_state.hole_density[i] - old_state.hole_density[i]) /
                       std::max(new_state.hole_density[i], old_state.hole_density[i]);

        residual += n_diff * n_diff + p_diff * p_diff;
    }

    // Temperature residual for energy transport
    if (energy_transport_enabled_) {
        for (size_t i = 0; i < dof_count; ++i) {
            double Tn_diff = (new_state.electron_temperature[i] - old_state.electron_temperature[i]) /
                            std::max(new_state.electron_temperature[i], old_state.electron_temperature[i]);
            double Tp_diff = (new_state.hole_temperature[i] - old_state.hole_temperature[i]) /
                            std::max(new_state.hole_temperature[i], old_state.hole_temperature[i]);

            residual += Tn_diff * Tn_diff + Tp_diff * Tp_diff;
        }
    }

    return std::sqrt(residual / (dof_count * (energy_transport_enabled_ ? 5 : 3)));
}

SolutionState SelfConsistentSolver::apply_damping(const SolutionState& old_state, const SolutionState& new_state) {
    SolutionState damped_state = new_state.copy();
    double alpha = criteria_.damping_factor;
    double beta = 1.0 - alpha;

    size_t dof_count = old_state.potential.size();

    // Damp potential
    for (size_t i = 0; i < dof_count; ++i) {
        damped_state.potential[i] = alpha * new_state.potential[i] + beta * old_state.potential[i];
    }

    // Damp carrier densities
    for (size_t i = 0; i < dof_count; ++i) {
        damped_state.electron_density[i] = alpha * new_state.electron_density[i] + beta * old_state.electron_density[i];
        damped_state.hole_density[i] = alpha * new_state.hole_density[i] + beta * old_state.hole_density[i];
    }

    // Damp temperatures for energy transport
    if (energy_transport_enabled_) {
        for (size_t i = 0; i < dof_count; ++i) {
            damped_state.electron_temperature[i] = alpha * new_state.electron_temperature[i] + beta * old_state.electron_temperature[i];
            damped_state.hole_temperature[i] = alpha * new_state.hole_temperature[i] + beta * old_state.hole_temperature[i];
        }
    }

    return damped_state;
}

void SelfConsistentSolver::update_charge_density(SolutionState& state) {
    // Update charge density: rho = q(p - n + Nd - Na)
    // This will be used by the Poisson solver

    size_t dof_count = state.potential.size();
    std::vector<double> charge_density(dof_count);

    const double q = 1.602e-19; // Elementary charge

    for (size_t i = 0; i < dof_count; ++i) {
        double Nd = (i < Nd_.size()) ? Nd_[i] : 0.0;
        double Na = (i < Na_.size()) ? Na_[i] : 0.0;

        charge_density[i] = q * (state.hole_density[i] - state.electron_density[i] + Nd - Na);
    }

    // Set charge density in Poisson solver
    if (poisson_) {
        poisson_->set_charge_density(charge_density);
    }
}

void SelfConsistentSolver::solve_poisson_step(SolutionState& state, const std::vector<double>& boundary_conditions) {
    if (!poisson_) {
        throw std::runtime_error("Poisson solver not initialized");
    }

    try {
        // Solve Poisson equation
        auto potential_result = poisson_->solve_2d(boundary_conditions);

        // Convert to map format for consistency
        std::map<std::string, std::vector<double>> poisson_results;
        poisson_results["potential"] = potential_result;

        // Update potential in state
        if (poisson_results.find("potential") != poisson_results.end()) {
            state.potential = poisson_results.at("potential");
        }

        // Update electric field if available
        if (poisson_results.find("electric_field") != poisson_results.end()) {
            // Electric field would be stored in state if needed
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Poisson step failed: " + std::string(e.what()));
    }
}

void SelfConsistentSolver::solve_transport_step(SolutionState& state, const std::vector<double>& boundary_conditions) {
    try {
        std::map<std::string, std::vector<double>> transport_results;

        if (advanced_transport_) {
            // Use advanced transport solver
            transport_results = advanced_transport_->solve_transport(boundary_conditions, 0.0, 10, false, 20, 1e-6);
        } else if (drift_diffusion_) {
            // Use basic drift-diffusion solver
            transport_results = drift_diffusion_->solve_drift_diffusion(boundary_conditions, 0.0, 10, false, 20, 1e-6);
        } else {
            throw std::runtime_error("No transport solver available");
        }

        // Update state with transport results
        copy_solution_data(transport_results, state);

    } catch (const std::exception& e) {
        throw std::runtime_error("Transport step failed: " + std::string(e.what()));
    }
}

void SelfConsistentSolver::update_generation_recombination(SolutionState& state) {
    // Update generation and recombination rates
    size_t dof_count = state.potential.size();

    const double ni = 1.45e16; // Intrinsic carrier concentration (Si at 300K) in m^-3
    const double tau_n = 1e-6; // Electron lifetime (s)
    const double tau_p = 1e-6; // Hole lifetime (s)

    for (size_t i = 0; i < dof_count; ++i) {
        // SRH recombination
        double n = state.electron_density[i];
        double p = state.hole_density[i];

        double R_srh = (n * p - ni * ni) / (tau_p * (n + ni) + tau_n * (p + ni));
        state.recombination_rate[i] = R_srh;

        // Impact ionization (simplified)
        // Would require electric field calculation
        state.generation_rate[i] = 0.0; // Placeholder
    }
}

void SelfConsistentSolver::apply_quantum_corrections(SolutionState& state) {
    if (!quantum_corrections_enabled_) {
        return;
    }

    // Apply quantum corrections to potential
    // This is a simplified implementation
    size_t dof_count = state.potential.size();

    const double hbar = 1.055e-34; // Reduced Planck constant
    const double m_n = 9.11e-31 * 0.26; // Effective electron mass
    const double m_p = 9.11e-31 * 0.39; // Effective hole mass
    const double q = 1.602e-19; // Elementary charge

    for (size_t i = 0; i < dof_count; ++i) {
        // Quantum potential for electrons (simplified)
        double n = state.electron_density[i];
        if (n > 1e12) { // Only apply for significant densities
            state.quantum_potential_n[i] = (hbar * hbar) / (2 * m_n * q) * std::log(n);
        }

        // Quantum potential for holes (simplified)
        double p = state.hole_density[i];
        if (p > 1e12) {
            state.quantum_potential_p[i] = (hbar * hbar) / (2 * m_p * q) * std::log(p);
        }
    }
}

void SelfConsistentSolver::copy_solution_data(const std::map<std::string, std::vector<double>>& solver_results,
                                             SolutionState& state) const {
    if (solver_results.find("potential") != solver_results.end()) {
        state.potential = solver_results.at("potential");
    }
    if (solver_results.find("n") != solver_results.end()) {
        state.electron_density = solver_results.at("n");
    }
    if (solver_results.find("p") != solver_results.end()) {
        state.hole_density = solver_results.at("p");
    }
    if (solver_results.find("Jn") != solver_results.end()) {
        const auto& Jn = solver_results.at("Jn");
        // Split current into x and y components (simplified)
        state.electron_current_x.resize(Jn.size());
        state.electron_current_y.resize(Jn.size());
        for (size_t i = 0; i < Jn.size(); ++i) {
            state.electron_current_x[i] = Jn[i] * 0.7; // Simplified split
            state.electron_current_y[i] = Jn[i] * 0.3;
        }
    }
    if (solver_results.find("Jp") != solver_results.end()) {
        const auto& Jp = solver_results.at("Jp");
        state.hole_current_x.resize(Jp.size());
        state.hole_current_y.resize(Jp.size());
        for (size_t i = 0; i < Jp.size(); ++i) {
            state.hole_current_x[i] = Jp[i] * 0.7; // Simplified split
            state.hole_current_y[i] = Jp[i] * 0.3;
        }
    }

    // Energy transport results
    if (energy_transport_enabled_) {
        if (solver_results.find("T_n") != solver_results.end()) {
            state.electron_temperature = solver_results.at("T_n");
        }
        if (solver_results.find("T_p") != solver_results.end()) {
            state.hole_temperature = solver_results.at("T_p");
        }
    }
}

} // namespace selfconsistent
} // namespace simulator

// C interface implementation
extern "C" {
    simulator::selfconsistent::SelfConsistentSolver* create_self_consistent_solver(
        simulator::Device* device, int method, int mesh_type, int order) {
        if (!device) return nullptr;
        try {
            return new simulator::selfconsistent::SelfConsistentSolver(
                *device,
                static_cast<simulator::Method>(method),
                static_cast<simulator::MeshType>(mesh_type),
                order);
        } catch (...) {
            return nullptr;
        }
    }

    void destroy_self_consistent_solver(simulator::selfconsistent::SelfConsistentSolver* solver) {
        delete solver;
    }

    int self_consistent_solver_set_convergence_criteria(
        simulator::selfconsistent::SelfConsistentSolver* solver,
        double potential_tol, double density_tol, double current_tol, int max_iter) {
        if (!solver) return -1;
        try {
            simulator::selfconsistent::ConvergenceCriteria criteria;
            criteria.potential_tolerance = potential_tol;
            criteria.density_tolerance = density_tol;
            criteria.current_tolerance = current_tol;
            criteria.max_iterations = max_iter;
            solver->set_convergence_criteria(criteria);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int self_consistent_solver_set_doping(simulator::selfconsistent::SelfConsistentSolver* solver,
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

    int self_consistent_solver_solve_steady_state(
        simulator::selfconsistent::SelfConsistentSolver* solver,
        double* bc, int bc_size,
        double* initial_potential, double* initial_n, double* initial_p, int state_size,
        double* result_potential, double* result_n, double* result_p,
        int* iterations, double* residual) {

        if (!solver || !bc || !initial_potential || !initial_n || !initial_p ||
            !result_potential || !result_n || !result_p || !iterations || !residual) {
            return -1;
        }

        try {
            // Set up boundary conditions
            std::vector<double> bc_vec(bc, bc + std::min(bc_size, 4));

            // Set up initial state
            simulator::selfconsistent::SolutionState initial_state;
            initial_state.resize(state_size);
            initial_state.potential.assign(initial_potential, initial_potential + state_size);
            initial_state.electron_density.assign(initial_n, initial_n + state_size);
            initial_state.hole_density.assign(initial_p, initial_p + state_size);

            // Solve
            auto result_state = solver->solve_steady_state(bc_vec, initial_state);

            // Copy results
            for (int i = 0; i < state_size; ++i) {
                result_potential[i] = result_state.potential[i];
                result_n[i] = result_state.electron_density[i];
                result_p[i] = result_state.hole_density[i];
            }

            *iterations = solver->get_last_iteration_count();
            *residual = solver->get_last_residual();

            return 0;
        } catch (...) {
            return -1;
        }
    }

    size_t self_consistent_solver_get_dof_count(simulator::selfconsistent::SelfConsistentSolver* solver) {
        if (!solver) return 0;
        try {
            return solver->get_dof_count();
        } catch (...) {
            return 0;
        }
    }

    int self_consistent_solver_is_valid(simulator::selfconsistent::SelfConsistentSolver* solver) {
        if (!solver) return 0;
        try {
            return solver->is_valid() ? 1 : 0;
        } catch (...) {
            return 0;
        }
    }
}
