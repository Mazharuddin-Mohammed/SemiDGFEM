/**
 * Self-Consistent Solver Framework for Semiconductor Device Simulation
 * Provides coupled solving of Poisson, drift-diffusion, and transport equations
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include "device.hpp"
#include "mesh.hpp"
#include "poisson.hpp"
#include "driftdiffusion.hpp"
#include "advanced_transport.hpp"
#include "transient_solver.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <stdexcept>

namespace simulator {
namespace selfconsistent {

/**
 * @brief Convergence criteria for self-consistent iterations
 */
struct ConvergenceCriteria {
    double potential_tolerance = 1e-6;      // Potential convergence tolerance (V)
    double density_tolerance = 1e-3;        // Carrier density relative tolerance
    double current_tolerance = 1e-3;        // Current density relative tolerance
    double temperature_tolerance = 1e-2;    // Temperature relative tolerance (for energy transport)
    int max_iterations = 100;               // Maximum self-consistent iterations
    int min_iterations = 3;                 // Minimum iterations before checking convergence
    bool use_damping = true;                // Use damping for stability
    double damping_factor = 0.7;            // Damping factor (0 < factor <= 1)
    
    ConvergenceCriteria() = default;
};

/**
 * @brief Solution state for self-consistent iterations
 */
struct SolutionState {
    std::vector<double> potential;
    std::vector<double> electron_density;
    std::vector<double> hole_density;
    std::vector<double> electron_current_x;
    std::vector<double> electron_current_y;
    std::vector<double> hole_current_x;
    std::vector<double> hole_current_y;
    
    // Extended variables for advanced transport
    std::vector<double> electron_temperature;
    std::vector<double> hole_temperature;
    std::vector<double> electron_energy;
    std::vector<double> hole_energy;
    std::vector<double> electron_momentum_x;
    std::vector<double> electron_momentum_y;
    std::vector<double> hole_momentum_x;
    std::vector<double> hole_momentum_y;
    
    // Quantum corrections
    std::vector<double> quantum_potential_n;
    std::vector<double> quantum_potential_p;
    
    // Generation-recombination rates
    std::vector<double> generation_rate;
    std::vector<double> recombination_rate;
    
    bool has_energy_transport = false;
    bool has_hydrodynamic = false;
    bool has_quantum_corrections = false;
    
    void resize(size_t size) {
        potential.resize(size, 0.0);
        electron_density.resize(size, 1e10);
        hole_density.resize(size, 1e10);
        electron_current_x.resize(size, 0.0);
        electron_current_y.resize(size, 0.0);
        hole_current_x.resize(size, 0.0);
        hole_current_y.resize(size, 0.0);
        generation_rate.resize(size, 0.0);
        recombination_rate.resize(size, 0.0);
        
        if (has_energy_transport) {
            electron_temperature.resize(size, 300.0);
            hole_temperature.resize(size, 300.0);
            electron_energy.resize(size, 0.0);
            hole_energy.resize(size, 0.0);
        }
        
        if (has_hydrodynamic) {
            electron_momentum_x.resize(size, 0.0);
            electron_momentum_y.resize(size, 0.0);
            hole_momentum_x.resize(size, 0.0);
            hole_momentum_y.resize(size, 0.0);
        }
        
        if (has_quantum_corrections) {
            quantum_potential_n.resize(size, 0.0);
            quantum_potential_p.resize(size, 0.0);
        }
    }
    
    SolutionState copy() const {
        SolutionState copy_state = *this;
        return copy_state;
    }
};

/**
 * @brief Self-consistent solver for coupled semiconductor equations
 */
class SelfConsistentSolver {
public:
    SelfConsistentSolver(const Device& device, Method method, MeshType mesh_type, int order = 3);
    ~SelfConsistentSolver();

    // Copy and move constructors
    SelfConsistentSolver(const SelfConsistentSolver& other) = delete;
    SelfConsistentSolver& operator=(const SelfConsistentSolver& other) = delete;
    SelfConsistentSolver(SelfConsistentSolver&& other) noexcept;
    SelfConsistentSolver& operator=(SelfConsistentSolver&& other) noexcept;

    // Configuration methods
    void set_convergence_criteria(const ConvergenceCriteria& criteria);
    void enable_energy_transport(bool enable = true);
    void enable_hydrodynamic_transport(bool enable = true);
    void enable_quantum_corrections(bool enable = true);
    void set_transport_model(SemiDGFEM::Physics::TransportModel model);
    
    // Material and doping setup
    void set_doping(const std::vector<double>& Nd, const std::vector<double>& Na);
    void set_trap_levels(const std::vector<double>& Et);
    void set_material_properties(const std::map<std::string, double>& properties);
    
    // Main solving methods
    SolutionState solve_steady_state(const std::vector<double>& boundary_conditions,
                                   const SolutionState& initial_guess);
    
    std::vector<SolutionState> solve_transient(const transient::TransientBoundaryConditions& bc,
                                              const SolutionState& initial_state,
                                              double final_time, double time_step);
    
    // Utility methods
    bool is_valid() const;
    void validate() const;
    size_t get_dof_count() const;
    ConvergenceCriteria get_convergence_criteria() const { return criteria_; }
    
    // Convergence information
    int get_last_iteration_count() const { return last_iteration_count_; }
    double get_last_residual() const { return last_residual_; }
    std::vector<double> get_convergence_history() const { return convergence_history_; }

private:
    const Device& device_;
    Method method_;
    MeshType mesh_type_;
    int order_;
    
    // Convergence criteria
    ConvergenceCriteria criteria_;
    
    // Physics configuration
    bool energy_transport_enabled_;
    bool hydrodynamic_enabled_;
    bool quantum_corrections_enabled_;
    SemiDGFEM::Physics::TransportModel transport_model_;
    
    // Solvers
    std::unique_ptr<Poisson> poisson_;
    std::unique_ptr<DriftDiffusion> drift_diffusion_;
    std::unique_ptr<transport::AdvancedTransportSolver> advanced_transport_;
    std::unique_ptr<transient::TransientSolver> transient_solver_;
    
    // Material data
    std::vector<double> Nd_, Na_, Et_;
    std::map<std::string, double> material_properties_;
    
    // Convergence tracking
    int last_iteration_count_;
    double last_residual_;
    std::vector<double> convergence_history_;
    
    // Helper methods
    void initialize_solvers();
    void validate_solution_state(const SolutionState& state) const;
    
    // Self-consistent iteration methods
    SolutionState perform_self_consistent_iteration(const SolutionState& current_state,
                                                   const std::vector<double>& boundary_conditions);
    
    bool check_convergence(const SolutionState& old_state, const SolutionState& new_state);
    double compute_residual(const SolutionState& old_state, const SolutionState& new_state);
    SolutionState apply_damping(const SolutionState& old_state, const SolutionState& new_state);
    
    // Physics coupling methods
    void update_charge_density(SolutionState& state);
    void solve_poisson_step(SolutionState& state, const std::vector<double>& boundary_conditions);
    void solve_transport_step(SolutionState& state, const std::vector<double>& boundary_conditions);
    void update_generation_recombination(SolutionState& state);
    void apply_quantum_corrections(SolutionState& state);
    
    // Utility methods
    void copy_solution_data(const std::map<std::string, std::vector<double>>& solver_results,
                           SolutionState& state) const;
    std::map<std::string, std::vector<double>> extract_solution_data(const SolutionState& state) const;
};

} // namespace selfconsistent
} // namespace simulator

// C interface for Cython bindings
extern "C" {
    simulator::selfconsistent::SelfConsistentSolver* create_self_consistent_solver(
        simulator::Device* device, int method, int mesh_type, int order);
    void destroy_self_consistent_solver(simulator::selfconsistent::SelfConsistentSolver* solver);
    
    int self_consistent_solver_set_convergence_criteria(
        simulator::selfconsistent::SelfConsistentSolver* solver,
        double potential_tol, double density_tol, double current_tol, int max_iter);
    
    int self_consistent_solver_set_doping(simulator::selfconsistent::SelfConsistentSolver* solver,
                                         double* Nd, double* Na, int size);
    
    int self_consistent_solver_solve_steady_state(
        simulator::selfconsistent::SelfConsistentSolver* solver,
        double* bc, int bc_size,
        double* initial_potential, double* initial_n, double* initial_p, int state_size,
        double* result_potential, double* result_n, double* result_p,
        int* iterations, double* residual);
    
    size_t self_consistent_solver_get_dof_count(simulator::selfconsistent::SelfConsistentSolver* solver);
    int self_consistent_solver_is_valid(simulator::selfconsistent::SelfConsistentSolver* solver);
}
