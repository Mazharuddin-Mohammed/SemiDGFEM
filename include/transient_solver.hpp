/**
 * Transient Simulation Framework for Semiconductor Device Simulation
 * Implements time-dependent solvers with multiple time integration schemes
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include "device.hpp"
#include "mesh.hpp"
#include "poisson.hpp"
#include "driftdiffusion.hpp"
#include "advanced_transport.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <stdexcept>

namespace simulator {
namespace transient {

/**
 * @brief Time integration schemes for transient simulation
 */
enum class TimeIntegrator {
    BACKWARD_EULER = 0,
    FORWARD_EULER = 1,
    CRANK_NICOLSON = 2,
    RK4 = 3,
    BDF2 = 4,
    ADAPTIVE_RK45 = 5
};

/**
 * @brief Time-dependent boundary condition function type
 */
using TimeDependentBC = std::function<double(double)>;

/**
 * @brief Structure for time-dependent boundary conditions
 */
struct TransientBoundaryConditions {
    std::vector<TimeDependentBC> voltage_functions;  // [left, right, bottom, top]
    std::vector<double> static_voltages;             // Static values if no function provided
    bool has_time_dependent_bc = false;
    
    TransientBoundaryConditions() : voltage_functions(4), static_voltages(4, 0.0) {}
    
    void set_static_voltage(int boundary, double voltage) {
        static_voltages[boundary] = voltage;
    }
    
    void set_time_dependent_voltage(int boundary, TimeDependentBC func) {
        voltage_functions[boundary] = func;
        has_time_dependent_bc = true;
    }
    
    std::vector<double> evaluate_at_time(double t) const {
        std::vector<double> bc(4);
        for (int i = 0; i < 4; ++i) {
            if (voltage_functions[i]) {
                bc[i] = voltage_functions[i](t);
            } else {
                bc[i] = static_voltages[i];
            }
        }
        return bc;
    }
};

/**
 * @brief Initial conditions for transient simulation
 */
struct InitialConditions {
    std::vector<double> potential;
    std::vector<double> electron_density;
    std::vector<double> hole_density;
    std::vector<double> electron_temperature;  // For energy transport
    std::vector<double> hole_temperature;      // For energy transport
    std::vector<double> electron_momentum_x;   // For hydrodynamic
    std::vector<double> electron_momentum_y;   // For hydrodynamic
    std::vector<double> hole_momentum_x;       // For hydrodynamic
    std::vector<double> hole_momentum_y;       // For hydrodynamic
    
    bool has_energy_transport = false;
    bool has_hydrodynamic = false;
    
    void resize(size_t size) {
        potential.resize(size, 0.0);
        electron_density.resize(size, 1e10);
        hole_density.resize(size, 1e10);
        if (has_energy_transport) {
            electron_temperature.resize(size, 300.0);
            hole_temperature.resize(size, 300.0);
        }
        if (has_hydrodynamic) {
            electron_momentum_x.resize(size, 0.0);
            electron_momentum_y.resize(size, 0.0);
            hole_momentum_x.resize(size, 0.0);
            hole_momentum_y.resize(size, 0.0);
        }
    }
};

/**
 * @brief Transient simulation results at a single time point
 */
struct TransientSolution {
    double time;
    std::vector<double> potential;
    std::vector<double> electron_density;
    std::vector<double> hole_density;
    std::vector<double> electron_current_x;
    std::vector<double> electron_current_y;
    std::vector<double> hole_current_x;
    std::vector<double> hole_current_y;
    
    // Extended results for advanced transport
    std::vector<double> electron_temperature;
    std::vector<double> hole_temperature;
    std::vector<double> electron_momentum_x;
    std::vector<double> electron_momentum_y;
    std::vector<double> hole_momentum_x;
    std::vector<double> hole_momentum_y;
    
    bool has_energy_transport = false;
    bool has_hydrodynamic = false;
    
    void resize(size_t size) {
        potential.resize(size);
        electron_density.resize(size);
        hole_density.resize(size);
        electron_current_x.resize(size);
        electron_current_y.resize(size);
        hole_current_x.resize(size);
        hole_current_y.resize(size);
        
        if (has_energy_transport) {
            electron_temperature.resize(size);
            hole_temperature.resize(size);
        }
        if (has_hydrodynamic) {
            electron_momentum_x.resize(size);
            electron_momentum_y.resize(size);
            hole_momentum_x.resize(size);
            hole_momentum_y.resize(size);
        }
    }
};

/**
 * @brief Main transient solver class
 */
class TransientSolver {
public:
    TransientSolver(const Device& device, Method method, MeshType mesh_type, int order = 3);
    ~TransientSolver();

    // Copy and move constructors
    TransientSolver(const TransientSolver& other) = delete;
    TransientSolver& operator=(const TransientSolver& other) = delete;
    TransientSolver(TransientSolver&& other) noexcept;
    TransientSolver& operator=(TransientSolver&& other) noexcept;

    // Configuration methods
    void set_time_step(double dt);
    void set_final_time(double t_final);
    void set_time_integrator(TimeIntegrator integrator);
    void set_adaptive_tolerance(double abs_tol, double rel_tol);
    void set_max_time_steps(int max_steps);
    
    // Physics configuration
    void enable_energy_transport(bool enable = true);
    void enable_hydrodynamic_transport(bool enable = true);
    void set_transport_model(SemiDGFEM::Physics::TransportModel model);
    
    // Doping and material setup
    void set_doping(const std::vector<double>& Nd, const std::vector<double>& Na);
    void set_trap_levels(const std::vector<double>& Et);
    
    // Main solver method
    std::vector<TransientSolution> solve(const TransientBoundaryConditions& bc,
                                       const InitialConditions& ic);
    
    // Utility methods
    bool is_valid() const;
    void validate() const;
    size_t get_dof_count() const;
    double get_current_time() const { return current_time_; }
    double get_time_step() const { return dt_; }
    TimeIntegrator get_time_integrator() const { return integrator_; }
    
    // Convergence and error information
    double get_last_time_step_error() const { return last_time_step_error_; }
    int get_total_time_steps() const { return total_time_steps_; }
    int get_rejected_steps() const { return rejected_steps_; }

private:
    const Device& device_;
    Method method_;
    MeshType mesh_type_;
    int order_;
    
    // Time integration parameters
    double dt_;
    double t_final_;
    double current_time_;
    TimeIntegrator integrator_;
    double abs_tolerance_;
    double rel_tolerance_;
    int max_time_steps_;
    
    // Physics configuration
    bool energy_transport_enabled_;
    bool hydrodynamic_enabled_;
    SemiDGFEM::Physics::TransportModel transport_model_;
    
    // Solvers
    std::unique_ptr<Poisson> poisson_;
    std::unique_ptr<DriftDiffusion> drift_diffusion_;
    std::unique_ptr<transport::AdvancedTransportSolver> advanced_transport_;
    
    // Material data
    std::vector<double> Nd_, Na_, Et_;
    
    // Time stepping state
    TransientSolution current_solution_;
    TransientSolution previous_solution_;
    double last_time_step_error_;
    int total_time_steps_;
    int rejected_steps_;
    
    // Helper methods
    void initialize_solvers();
    void validate_initial_conditions(const InitialConditions& ic) const;
    void validate_boundary_conditions(const TransientBoundaryConditions& bc) const;
    
    // Time integration methods
    TransientSolution step_backward_euler(const TransientSolution& current,
                                        const TransientBoundaryConditions& bc,
                                        double dt);
    
    TransientSolution step_forward_euler(const TransientSolution& current,
                                       const TransientBoundaryConditions& bc,
                                       double dt);
    
    TransientSolution step_crank_nicolson(const TransientSolution& current,
                                        const TransientBoundaryConditions& bc,
                                        double dt);
    
    TransientSolution step_rk4(const TransientSolution& current,
                             const TransientBoundaryConditions& bc,
                             double dt);
    
    std::pair<TransientSolution, double> step_adaptive_rk45(const TransientSolution& current,
                                                           const TransientBoundaryConditions& bc,
                                                           double dt);
    
    // Physics evaluation
    TransientSolution evaluate_physics(const TransientSolution& state,
                                     const std::vector<double>& bc,
                                     double time);
    
    // Error estimation and adaptive time stepping
    double estimate_error(const TransientSolution& sol1, const TransientSolution& sol2) const;
    double compute_optimal_time_step(double current_dt, double error, double tolerance) const;
    
    // Utility methods
    void copy_solution_data(const std::map<std::string, std::vector<double>>& solver_results,
                           TransientSolution& solution) const;
};

} // namespace transient
} // namespace simulator

// C interface for Cython bindings
extern "C" {
    simulator::transient::TransientSolver* create_transient_solver(
        simulator::Device* device, int method, int mesh_type, int order);
    void destroy_transient_solver(simulator::transient::TransientSolver* solver);

    int transient_solver_set_time_step(simulator::transient::TransientSolver* solver, double dt);
    int transient_solver_set_final_time(simulator::transient::TransientSolver* solver, double t_final);
    int transient_solver_set_time_integrator(simulator::transient::TransientSolver* solver, int integrator);
    int transient_solver_set_doping(simulator::transient::TransientSolver* solver,
                                   double* Nd, double* Na, int size);

    int transient_solver_solve(simulator::transient::TransientSolver* solver,
                              double* static_bc, int bc_size,
                              double* initial_potential, double* initial_n, double* initial_p, int ic_size,
                              double* time_points, double* potential_results, double* n_results, double* p_results,
                              int max_time_points, int* actual_time_points);

    size_t transient_solver_get_dof_count(simulator::transient::TransientSolver* solver);
    double transient_solver_get_current_time(simulator::transient::TransientSolver* solver);
    int transient_solver_is_valid(simulator::transient::TransientSolver* solver);
}
