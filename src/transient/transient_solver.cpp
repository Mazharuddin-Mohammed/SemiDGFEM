/**
 * Implementation of Transient Simulation Framework
 * Time-dependent solvers with multiple integration schemes
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "transient_solver.hpp"
#include "performance_optimization.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <chrono>

namespace simulator {
namespace transient {

TransientSolver::TransientSolver(const Device& device, Method method, MeshType mesh_type, int order)
    : device_(device), method_(method), mesh_type_(mesh_type), order_(order),
      dt_(1e-12), t_final_(1e-9), current_time_(0.0),
      integrator_(TimeIntegrator::BACKWARD_EULER),
      abs_tolerance_(1e-6), rel_tolerance_(1e-3), max_time_steps_(10000),
      energy_transport_enabled_(false), hydrodynamic_enabled_(false),
      transport_model_(SemiDGFEM::Physics::TransportModel::DRIFT_DIFFUSION),
      last_time_step_error_(0.0), total_time_steps_(0), rejected_steps_(0) {
    
    validate();
    initialize_solvers();
}

TransientSolver::~TransientSolver() = default;

TransientSolver::TransientSolver(TransientSolver&& other) noexcept
    : device_(other.device_), method_(other.method_), mesh_type_(other.mesh_type_), order_(other.order_),
      dt_(other.dt_), t_final_(other.t_final_), current_time_(other.current_time_),
      integrator_(other.integrator_), abs_tolerance_(other.abs_tolerance_), rel_tolerance_(other.rel_tolerance_),
      max_time_steps_(other.max_time_steps_), energy_transport_enabled_(other.energy_transport_enabled_),
      hydrodynamic_enabled_(other.hydrodynamic_enabled_), transport_model_(other.transport_model_),
      poisson_(std::move(other.poisson_)), drift_diffusion_(std::move(other.drift_diffusion_)),
      advanced_transport_(std::move(other.advanced_transport_)),
      Nd_(std::move(other.Nd_)), Na_(std::move(other.Na_)), Et_(std::move(other.Et_)),
      current_solution_(std::move(other.current_solution_)), previous_solution_(std::move(other.previous_solution_)),
      last_time_step_error_(other.last_time_step_error_), total_time_steps_(other.total_time_steps_),
      rejected_steps_(other.rejected_steps_) {
}

TransientSolver& TransientSolver::operator=(TransientSolver&& other) noexcept {
    if (this != &other) {
        dt_ = other.dt_;
        t_final_ = other.t_final_;
        current_time_ = other.current_time_;
        integrator_ = other.integrator_;
        abs_tolerance_ = other.abs_tolerance_;
        rel_tolerance_ = other.rel_tolerance_;
        max_time_steps_ = other.max_time_steps_;
        energy_transport_enabled_ = other.energy_transport_enabled_;
        hydrodynamic_enabled_ = other.hydrodynamic_enabled_;
        transport_model_ = other.transport_model_;
        poisson_ = std::move(other.poisson_);
        drift_diffusion_ = std::move(other.drift_diffusion_);
        advanced_transport_ = std::move(other.advanced_transport_);
        Nd_ = std::move(other.Nd_);
        Na_ = std::move(other.Na_);
        Et_ = std::move(other.Et_);
        current_solution_ = std::move(other.current_solution_);
        previous_solution_ = std::move(other.previous_solution_);
        last_time_step_error_ = other.last_time_step_error_;
        total_time_steps_ = other.total_time_steps_;
        rejected_steps_ = other.rejected_steps_;
    }
    return *this;
}

void TransientSolver::set_time_step(double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument("Time step must be positive");
    }
    dt_ = dt;
}

void TransientSolver::set_final_time(double t_final) {
    if (t_final <= 0.0) {
        throw std::invalid_argument("Final time must be positive");
    }
    t_final_ = t_final;
}

void TransientSolver::set_time_integrator(TimeIntegrator integrator) {
    integrator_ = integrator;
}

void TransientSolver::set_adaptive_tolerance(double abs_tol, double rel_tol) {
    if (abs_tol <= 0.0 || rel_tol <= 0.0) {
        throw std::invalid_argument("Tolerances must be positive");
    }
    abs_tolerance_ = abs_tol;
    rel_tolerance_ = rel_tol;
}

void TransientSolver::set_max_time_steps(int max_steps) {
    if (max_steps <= 0) {
        throw std::invalid_argument("Maximum time steps must be positive");
    }
    max_time_steps_ = max_steps;
}

void TransientSolver::enable_energy_transport(bool enable) {
    energy_transport_enabled_ = enable;
    if (enable) {
        transport_model_ = SemiDGFEM::Physics::TransportModel::ENERGY_TRANSPORT;
    }
}

void TransientSolver::enable_hydrodynamic_transport(bool enable) {
    hydrodynamic_enabled_ = enable;
    if (enable) {
        transport_model_ = SemiDGFEM::Physics::TransportModel::HYDRODYNAMIC;
    }
}

void TransientSolver::set_transport_model(SemiDGFEM::Physics::TransportModel model) {
    transport_model_ = model;
    
    // Update flags based on model
    energy_transport_enabled_ = (model == SemiDGFEM::Physics::TransportModel::ENERGY_TRANSPORT);
    hydrodynamic_enabled_ = (model == SemiDGFEM::Physics::TransportModel::HYDRODYNAMIC);
}

void TransientSolver::set_doping(const std::vector<double>& Nd, const std::vector<double>& Na) {
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

void TransientSolver::set_trap_levels(const std::vector<double>& Et) {
    Et_ = Et;
    
    if (drift_diffusion_) {
        drift_diffusion_->set_trap_level(Et_);
    }
    if (advanced_transport_) {
        advanced_transport_->set_trap_level(Et_);
    }
}

bool TransientSolver::is_valid() const {
    return device_.is_valid() && dt_ > 0.0 && t_final_ > 0.0 && 
           poisson_ && (drift_diffusion_ || advanced_transport_);
}

void TransientSolver::validate() const {
    if (!device_.is_valid()) {
        throw std::runtime_error("Device is invalid");
    }
    if (dt_ <= 0.0) {
        throw std::runtime_error("Time step must be positive");
    }
    if (t_final_ <= 0.0) {
        throw std::runtime_error("Final time must be positive");
    }
    if (abs_tolerance_ <= 0.0 || rel_tolerance_ <= 0.0) {
        throw std::runtime_error("Tolerances must be positive");
    }
}

size_t TransientSolver::get_dof_count() const {
    if (advanced_transport_) {
        return advanced_transport_->get_dof_count();
    } else if (drift_diffusion_) {
        return drift_diffusion_->get_dof_count();
    } else {
        throw std::runtime_error("No solver initialized");
    }
}

void TransientSolver::initialize_solvers() {
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

void TransientSolver::validate_initial_conditions(const InitialConditions& ic) const {
    size_t dof_count = get_dof_count();
    
    if (ic.potential.size() != dof_count ||
        ic.electron_density.size() != dof_count ||
        ic.hole_density.size() != dof_count) {
        throw std::invalid_argument("Initial condition arrays must match DOF count");
    }
    
    if (energy_transport_enabled_) {
        if (!ic.has_energy_transport ||
            ic.electron_temperature.size() != dof_count ||
            ic.hole_temperature.size() != dof_count) {
            throw std::invalid_argument("Energy transport initial conditions required");
        }
    }
    
    if (hydrodynamic_enabled_) {
        if (!ic.has_hydrodynamic ||
            ic.electron_momentum_x.size() != dof_count ||
            ic.electron_momentum_y.size() != dof_count ||
            ic.hole_momentum_x.size() != dof_count ||
            ic.hole_momentum_y.size() != dof_count) {
            throw std::invalid_argument("Hydrodynamic initial conditions required");
        }
    }
    
    // Check for physical validity
    if (std::any_of(ic.electron_density.begin(), ic.electron_density.end(), [](double x) { return x <= 0; }) ||
        std::any_of(ic.hole_density.begin(), ic.hole_density.end(), [](double x) { return x <= 0; })) {
        throw std::invalid_argument("Carrier densities must be positive");
    }
    
    if (energy_transport_enabled_) {
        if (std::any_of(ic.electron_temperature.begin(), ic.electron_temperature.end(), [](double x) { return x <= 0; }) ||
            std::any_of(ic.hole_temperature.begin(), ic.hole_temperature.end(), [](double x) { return x <= 0; })) {
            throw std::invalid_argument("Temperatures must be positive");
        }
    }
}

void TransientSolver::validate_boundary_conditions(const TransientBoundaryConditions& bc) const {
    if (bc.static_voltages.size() != 4) {
        throw std::invalid_argument("Boundary conditions must have 4 values");
    }
    
    if (!std::all_of(bc.static_voltages.begin(), bc.static_voltages.end(), 
                     [](double x) { return std::isfinite(x); })) {
        throw std::invalid_argument("All boundary conditions must be finite");
    }
}

std::vector<TransientSolution> TransientSolver::solve(const TransientBoundaryConditions& bc,
                                                     const InitialConditions& ic) {
    validate_initial_conditions(ic);
    validate_boundary_conditions(bc);

    std::vector<TransientSolution> solutions;

    // Initialize current solution from initial conditions
    current_solution_.time = 0.0;
    current_solution_.resize(get_dof_count());
    current_solution_.potential = ic.potential;
    current_solution_.electron_density = ic.electron_density;
    current_solution_.hole_density = ic.hole_density;

    if (energy_transport_enabled_) {
        current_solution_.has_energy_transport = true;
        current_solution_.electron_temperature = ic.electron_temperature;
        current_solution_.hole_temperature = ic.hole_temperature;
    }

    if (hydrodynamic_enabled_) {
        current_solution_.has_hydrodynamic = true;
        current_solution_.electron_momentum_x = ic.electron_momentum_x;
        current_solution_.electron_momentum_y = ic.electron_momentum_y;
        current_solution_.hole_momentum_x = ic.hole_momentum_x;
        current_solution_.hole_momentum_y = ic.hole_momentum_y;
    }

    // Add initial solution
    solutions.push_back(current_solution_);

    // Reset counters
    current_time_ = 0.0;
    total_time_steps_ = 0;
    rejected_steps_ = 0;

    // Main time stepping loop
    double dt = dt_;
    while (current_time_ < t_final_ && total_time_steps_ < max_time_steps_) {
        // Ensure we don't overshoot final time
        if (current_time_ + dt > t_final_) {
            dt = t_final_ - current_time_;
        }

        TransientSolution next_solution;
        bool step_accepted = false;

        try {
            switch (integrator_) {
                case TimeIntegrator::BACKWARD_EULER:
                    next_solution = step_backward_euler(current_solution_, bc, dt);
                    step_accepted = true;
                    break;

                case TimeIntegrator::FORWARD_EULER:
                    next_solution = step_forward_euler(current_solution_, bc, dt);
                    step_accepted = true;
                    break;

                case TimeIntegrator::CRANK_NICOLSON:
                    next_solution = step_crank_nicolson(current_solution_, bc, dt);
                    step_accepted = true;
                    break;

                case TimeIntegrator::RK4:
                    next_solution = step_rk4(current_solution_, bc, dt);
                    step_accepted = true;
                    break;

                case TimeIntegrator::ADAPTIVE_RK45: {
                    auto [solution, error] = step_adaptive_rk45(current_solution_, bc, dt);
                    next_solution = solution;
                    last_time_step_error_ = error;

                    double tolerance = abs_tolerance_ + rel_tolerance_ *
                        std::max(estimate_error(current_solution_, current_solution_),
                                estimate_error(next_solution, next_solution));

                    if (error <= tolerance) {
                        step_accepted = true;
                        // Compute optimal next time step
                        dt = compute_optimal_time_step(dt, error, tolerance);
                    } else {
                        step_accepted = false;
                        rejected_steps_++;
                        dt = compute_optimal_time_step(dt, error, tolerance);
                        continue; // Retry with smaller time step
                    }
                    break;
                }

                default:
                    throw std::runtime_error("Unknown time integrator");
            }

            if (step_accepted) {
                current_time_ += dt;
                next_solution.time = current_time_;
                previous_solution_ = current_solution_;
                current_solution_ = next_solution;
                solutions.push_back(current_solution_);
                total_time_steps_++;

                // Output progress
                if (total_time_steps_ % 100 == 0) {
                    std::cout << "Time step " << total_time_steps_
                              << ", t = " << current_time_ * 1e12 << " ps"
                              << ", dt = " << dt * 1e12 << " ps" << std::endl;
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "Error in time step " << total_time_steps_
                      << " at t = " << current_time_ << ": " << e.what() << std::endl;

            // Try with smaller time step
            dt *= 0.5;
            rejected_steps_++;

            if (dt < 1e-18) { // Minimum time step
                throw std::runtime_error("Time step became too small, simulation failed");
            }
        }
    }

    std::cout << "Transient simulation completed:" << std::endl;
    std::cout << "  Total time steps: " << total_time_steps_ << std::endl;
    std::cout << "  Rejected steps: " << rejected_steps_ << std::endl;
    std::cout << "  Final time: " << current_time_ * 1e12 << " ps" << std::endl;

    return solutions;
}

TransientSolution TransientSolver::step_backward_euler(const TransientSolution& current,
                                                      const TransientBoundaryConditions& bc,
                                                      double dt) {
    // Backward Euler: u^{n+1} = u^n + dt * f(u^{n+1}, t^{n+1})
    // This requires solving a nonlinear system at each time step

    TransientSolution next = current;
    next.time = current.time + dt;

    // Get boundary conditions at next time
    auto bc_values = bc.evaluate_at_time(next.time);

    // For backward Euler, we need to solve the implicit system
    // This is simplified - in practice would use Newton iteration
    auto physics_result = evaluate_physics(next, bc_values, next.time);

    // Update solution with time derivative
    size_t dof_count = current.potential.size();
    for (size_t i = 0; i < dof_count; ++i) {
        next.potential[i] = current.potential[i] + dt *
            (physics_result.potential[i] - current.potential[i]) / dt;
        next.electron_density[i] = current.electron_density[i] + dt *
            (physics_result.electron_density[i] - current.electron_density[i]) / dt;
        next.hole_density[i] = current.hole_density[i] + dt *
            (physics_result.hole_density[i] - current.hole_density[i]) / dt;
    }

    return next;
}

TransientSolution TransientSolver::step_forward_euler(const TransientSolution& current,
                                                     const TransientBoundaryConditions& bc,
                                                     double dt) {
    // Forward Euler: u^{n+1} = u^n + dt * f(u^n, t^n)

    auto bc_values = bc.evaluate_at_time(current.time);
    auto physics_result = evaluate_physics(current, bc_values, current.time);

    TransientSolution next = current;
    next.time = current.time + dt;

    // Simple forward Euler update
    size_t dof_count = current.potential.size();
    for (size_t i = 0; i < dof_count; ++i) {
        double dpdt = (physics_result.potential[i] - current.potential[i]) / dt;
        double dndt = (physics_result.electron_density[i] - current.electron_density[i]) / dt;
        double dpdt_hole = (physics_result.hole_density[i] - current.hole_density[i]) / dt;

        next.potential[i] = current.potential[i] + dt * dpdt;
        next.electron_density[i] = current.electron_density[i] + dt * dndt;
        next.hole_density[i] = current.hole_density[i] + dt * dpdt_hole;

        // Ensure physical bounds
        next.electron_density[i] = std::max(next.electron_density[i], 1e6);
        next.hole_density[i] = std::max(next.hole_density[i], 1e6);
    }

    return next;
}

TransientSolution TransientSolver::step_crank_nicolson(const TransientSolution& current,
                                                      const TransientBoundaryConditions& bc,
                                                      double dt) {
    // Crank-Nicolson: u^{n+1} = u^n + dt/2 * (f(u^n, t^n) + f(u^{n+1}, t^{n+1}))

    // First, get explicit part
    auto bc_current = bc.evaluate_at_time(current.time);
    auto f_current = evaluate_physics(current, bc_current, current.time);

    // Predictor step (forward Euler)
    TransientSolution predictor = step_forward_euler(current, bc, dt);

    // Get implicit part
    auto bc_next = bc.evaluate_at_time(current.time + dt);
    auto f_next = evaluate_physics(predictor, bc_next, current.time + dt);

    // Crank-Nicolson update
    TransientSolution next = current;
    next.time = current.time + dt;

    size_t dof_count = current.potential.size();
    for (size_t i = 0; i < dof_count; ++i) {
        double dpdt_avg = 0.5 * ((f_current.potential[i] - current.potential[i]) / dt +
                                (f_next.potential[i] - predictor.potential[i]) / dt);
        double dndt_avg = 0.5 * ((f_current.electron_density[i] - current.electron_density[i]) / dt +
                                (f_next.electron_density[i] - predictor.electron_density[i]) / dt);
        double dpdt_hole_avg = 0.5 * ((f_current.hole_density[i] - current.hole_density[i]) / dt +
                                     (f_next.hole_density[i] - predictor.hole_density[i]) / dt);

        next.potential[i] = current.potential[i] + dt * dpdt_avg;
        next.electron_density[i] = current.electron_density[i] + dt * dndt_avg;
        next.hole_density[i] = current.hole_density[i] + dt * dpdt_hole_avg;

        // Ensure physical bounds
        next.electron_density[i] = std::max(next.electron_density[i], 1e6);
        next.hole_density[i] = std::max(next.hole_density[i], 1e6);
    }

    return next;
}

TransientSolution TransientSolver::step_rk4(const TransientSolution& current,
                                           const TransientBoundaryConditions& bc,
                                           double dt) {
    // Fourth-order Runge-Kutta method

    // k1 = f(u^n, t^n)
    auto bc1 = bc.evaluate_at_time(current.time);
    auto k1 = evaluate_physics(current, bc1, current.time);

    // k2 = f(u^n + dt/2 * k1, t^n + dt/2)
    TransientSolution u2 = current;
    size_t dof_count = current.potential.size();
    for (size_t i = 0; i < dof_count; ++i) {
        u2.potential[i] += 0.5 * dt * (k1.potential[i] - current.potential[i]) / dt;
        u2.electron_density[i] += 0.5 * dt * (k1.electron_density[i] - current.electron_density[i]) / dt;
        u2.hole_density[i] += 0.5 * dt * (k1.hole_density[i] - current.hole_density[i]) / dt;
    }
    auto bc2 = bc.evaluate_at_time(current.time + 0.5 * dt);
    auto k2 = evaluate_physics(u2, bc2, current.time + 0.5 * dt);

    // k3 = f(u^n + dt/2 * k2, t^n + dt/2)
    TransientSolution u3 = current;
    for (size_t i = 0; i < dof_count; ++i) {
        u3.potential[i] += 0.5 * dt * (k2.potential[i] - current.potential[i]) / dt;
        u3.electron_density[i] += 0.5 * dt * (k2.electron_density[i] - current.electron_density[i]) / dt;
        u3.hole_density[i] += 0.5 * dt * (k2.hole_density[i] - current.hole_density[i]) / dt;
    }
    auto k3 = evaluate_physics(u3, bc2, current.time + 0.5 * dt);

    // k4 = f(u^n + dt * k3, t^n + dt)
    TransientSolution u4 = current;
    for (size_t i = 0; i < dof_count; ++i) {
        u4.potential[i] += dt * (k3.potential[i] - current.potential[i]) / dt;
        u4.electron_density[i] += dt * (k3.electron_density[i] - current.electron_density[i]) / dt;
        u4.hole_density[i] += dt * (k3.hole_density[i] - current.hole_density[i]) / dt;
    }
    auto bc4 = bc.evaluate_at_time(current.time + dt);
    auto k4 = evaluate_physics(u4, bc4, current.time + dt);

    // Final RK4 update
    TransientSolution next = current;
    next.time = current.time + dt;

    for (size_t i = 0; i < dof_count; ++i) {
        double dpdt1 = (k1.potential[i] - current.potential[i]) / dt;
        double dpdt2 = (k2.potential[i] - current.potential[i]) / dt;
        double dpdt3 = (k3.potential[i] - current.potential[i]) / dt;
        double dpdt4 = (k4.potential[i] - current.potential[i]) / dt;

        double dndt1 = (k1.electron_density[i] - current.electron_density[i]) / dt;
        double dndt2 = (k2.electron_density[i] - current.electron_density[i]) / dt;
        double dndt3 = (k3.electron_density[i] - current.electron_density[i]) / dt;
        double dndt4 = (k4.electron_density[i] - current.electron_density[i]) / dt;

        double dpdt_hole1 = (k1.hole_density[i] - current.hole_density[i]) / dt;
        double dpdt_hole2 = (k2.hole_density[i] - current.hole_density[i]) / dt;
        double dpdt_hole3 = (k3.hole_density[i] - current.hole_density[i]) / dt;
        double dpdt_hole4 = (k4.hole_density[i] - current.hole_density[i]) / dt;

        next.potential[i] = current.potential[i] + dt/6.0 * (dpdt1 + 2*dpdt2 + 2*dpdt3 + dpdt4);
        next.electron_density[i] = current.electron_density[i] + dt/6.0 * (dndt1 + 2*dndt2 + 2*dndt3 + dndt4);
        next.hole_density[i] = current.hole_density[i] + dt/6.0 * (dpdt_hole1 + 2*dpdt_hole2 + 2*dpdt_hole3 + dpdt_hole4);

        // Ensure physical bounds
        next.electron_density[i] = std::max(next.electron_density[i], 1e6);
        next.hole_density[i] = std::max(next.hole_density[i], 1e6);
    }

    return next;
}

std::pair<TransientSolution, double> TransientSolver::step_adaptive_rk45(const TransientSolution& current,
                                                                        const TransientBoundaryConditions& bc,
                                                                        double dt) {
    // Adaptive Runge-Kutta 4/5 method (simplified)
    // Use RK4 as main method and forward Euler for error estimation

    auto rk4_solution = step_rk4(current, bc, dt);
    auto euler_solution = step_forward_euler(current, bc, dt);

    // Estimate error
    double error = estimate_error(rk4_solution, euler_solution);

    return {rk4_solution, error};
}

TransientSolution TransientSolver::evaluate_physics(const TransientSolution& state,
                                                   const std::vector<double>& bc,
                                                   double time) {
    TransientSolution result = state;

    try {
        if (advanced_transport_) {
            // Use advanced transport solver
            auto solver_results = advanced_transport_->solve_transport(bc, 0.0, 10, false, 20, 1e-6);
            copy_solution_data(solver_results, result);
        } else if (drift_diffusion_) {
            // Use basic drift-diffusion solver
            auto solver_results = drift_diffusion_->solve_drift_diffusion(bc, 0.0, 10, false, 20, 1e-6);
            copy_solution_data(solver_results, result);
        } else {
            throw std::runtime_error("No solver available");
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Physics evaluation failed at t=" << time << ": " << e.what() << std::endl;
        // Return current state as fallback
    }

    return result;
}

double TransientSolver::estimate_error(const TransientSolution& sol1, const TransientSolution& sol2) const {
    double error = 0.0;
    size_t dof_count = sol1.potential.size();

    for (size_t i = 0; i < dof_count; ++i) {
        double pot_error = std::abs(sol1.potential[i] - sol2.potential[i]);
        double n_error = std::abs(sol1.electron_density[i] - sol2.electron_density[i]) /
                        std::max(sol1.electron_density[i], sol2.electron_density[i]);
        double p_error = std::abs(sol1.hole_density[i] - sol2.hole_density[i]) /
                        std::max(sol1.hole_density[i], sol2.hole_density[i]);

        error += pot_error * pot_error + n_error * n_error + p_error * p_error;
    }

    return std::sqrt(error / (3.0 * dof_count));
}

double TransientSolver::compute_optimal_time_step(double current_dt, double error, double tolerance) const {
    if (error <= 0.0) return current_dt * 2.0; // Double if no error

    double factor = 0.9 * std::pow(tolerance / error, 0.2); // Safety factor and optimal scaling
    factor = std::max(0.1, std::min(5.0, factor)); // Limit time step changes

    return current_dt * factor;
}

void TransientSolver::copy_solution_data(const std::map<std::string, std::vector<double>>& solver_results,
                                        TransientSolution& solution) const {
    if (solver_results.find("potential") != solver_results.end()) {
        solution.potential = solver_results.at("potential");
    }
    if (solver_results.find("n") != solver_results.end()) {
        solution.electron_density = solver_results.at("n");
    }
    if (solver_results.find("p") != solver_results.end()) {
        solution.hole_density = solver_results.at("p");
    }
    if (solver_results.find("Jn") != solver_results.end()) {
        const auto& Jn = solver_results.at("Jn");
        // Split current into x and y components (simplified)
        solution.electron_current_x.resize(Jn.size());
        solution.electron_current_y.resize(Jn.size());
        for (size_t i = 0; i < Jn.size(); ++i) {
            solution.electron_current_x[i] = Jn[i] * 0.7; // Simplified split
            solution.electron_current_y[i] = Jn[i] * 0.3;
        }
    }
    if (solver_results.find("Jp") != solver_results.end()) {
        const auto& Jp = solver_results.at("Jp");
        solution.hole_current_x.resize(Jp.size());
        solution.hole_current_y.resize(Jp.size());
        for (size_t i = 0; i < Jp.size(); ++i) {
            solution.hole_current_x[i] = Jp[i] * 0.7; // Simplified split
            solution.hole_current_y[i] = Jp[i] * 0.3;
        }
    }
}

} // namespace transient
} // namespace simulator

// C interface implementation
extern "C" {
    simulator::transient::TransientSolver* create_transient_solver(
        simulator::Device* device, int method, int mesh_type, int order) {
        if (!device) return nullptr;
        try {
            return new simulator::transient::TransientSolver(
                *device,
                static_cast<simulator::Method>(method),
                static_cast<simulator::MeshType>(mesh_type),
                order);
        } catch (...) {
            return nullptr;
        }
    }

    void destroy_transient_solver(simulator::transient::TransientSolver* solver) {
        delete solver;
    }

    int transient_solver_set_time_step(simulator::transient::TransientSolver* solver, double dt) {
        if (!solver) return -1;
        try {
            solver->set_time_step(dt);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int transient_solver_set_final_time(simulator::transient::TransientSolver* solver, double t_final) {
        if (!solver) return -1;
        try {
            solver->set_final_time(t_final);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int transient_solver_set_time_integrator(simulator::transient::TransientSolver* solver, int integrator) {
        if (!solver) return -1;
        try {
            solver->set_time_integrator(static_cast<simulator::transient::TimeIntegrator>(integrator));
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int transient_solver_set_doping(simulator::transient::TransientSolver* solver,
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

    int transient_solver_solve(simulator::transient::TransientSolver* solver,
                              double* static_bc, int bc_size,
                              double* initial_potential, double* initial_n, double* initial_p, int ic_size,
                              double* time_points, double* potential_results, double* n_results, double* p_results,
                              int max_time_points, int* actual_time_points) {
        if (!solver || !static_bc || !initial_potential || !initial_n || !initial_p ||
            !time_points || !potential_results || !n_results || !p_results || !actual_time_points) {
            return -1;
        }

        try {
            // Set up boundary conditions (static for now)
            simulator::transient::TransientBoundaryConditions bc;
            for (int i = 0; i < std::min(bc_size, 4); ++i) {
                bc.set_static_voltage(i, static_bc[i]);
            }

            // Set up initial conditions
            simulator::transient::InitialConditions ic;
            ic.potential.assign(initial_potential, initial_potential + ic_size);
            ic.electron_density.assign(initial_n, initial_n + ic_size);
            ic.hole_density.assign(initial_p, initial_p + ic_size);

            // Solve
            auto solutions = solver->solve(bc, ic);

            // Copy results
            int num_solutions = std::min(static_cast<int>(solutions.size()), max_time_points);
            *actual_time_points = num_solutions;

            for (int t = 0; t < num_solutions; ++t) {
                time_points[t] = solutions[t].time;

                for (int i = 0; i < ic_size; ++i) {
                    potential_results[t * ic_size + i] = solutions[t].potential[i];
                    n_results[t * ic_size + i] = solutions[t].electron_density[i];
                    p_results[t * ic_size + i] = solutions[t].hole_density[i];
                }
            }

            return 0;
        } catch (...) {
            return -1;
        }
    }

    size_t transient_solver_get_dof_count(simulator::transient::TransientSolver* solver) {
        if (!solver) return 0;
        try {
            return solver->get_dof_count();
        } catch (...) {
            return 0;
        }
    }

    double transient_solver_get_current_time(simulator::transient::TransientSolver* solver) {
        if (!solver) return 0.0;
        try {
            return solver->get_current_time();
        } catch (...) {
            return 0.0;
        }
    }

    int transient_solver_is_valid(simulator::transient::TransientSolver* solver) {
        if (!solver) return 0;
        try {
            return solver->is_valid() ? 1 : 0;
        } catch (...) {
            return 0;
        }
    }
}
