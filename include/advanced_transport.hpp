/**
 * Advanced Transport Models for Semiconductor Device Simulation
 * Includes non-equilibrium statistics, energy transport, and hydrodynamics
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#pragma once

#include "device.hpp"
#include "mesh.hpp"
#include "poisson.hpp"
#include "driftdiffusion.hpp"
#include "../src/physics/advanced_physics.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <stdexcept>

namespace simulator {
namespace transport {

/**
 * @brief Advanced transport solver with multiple physics models
 */
class AdvancedTransportSolver {
public:
    AdvancedTransportSolver(const Device& device, Method method, MeshType mesh_type, 
                           SemiDGFEM::Physics::TransportModel transport_model,
                           int order = 3);
    ~AdvancedTransportSolver();

    // Copy and move constructors
    AdvancedTransportSolver(const AdvancedTransportSolver& other);
    AdvancedTransportSolver& operator=(const AdvancedTransportSolver& other);
    AdvancedTransportSolver(AdvancedTransportSolver&& other) noexcept;
    AdvancedTransportSolver& operator=(AdvancedTransportSolver&& other) noexcept;

    // Configuration methods
    void set_physics_config(const SemiDGFEM::Physics::PhysicsConfig& config);
    void set_doping(const std::vector<double>& Nd, const std::vector<double>& Na);
    void set_trap_level(const std::vector<double>& Et);

    // Solver methods
    std::map<std::string, std::vector<double>> solve_transport(
        const std::vector<double>& bc, double Vg = 0.0, int max_steps = 100,
        bool use_amr = false, int poisson_max_iter = 50, double poisson_tol = 1e-6);

    // Utility methods
    bool is_valid() const;
    void validate() const;
    size_t get_dof_count() const;
    double get_convergence_residual() const;
    int get_order() const { return order_; }
    SemiDGFEM::Physics::TransportModel get_transport_model() const { return transport_model_; }

private:
    const Device& device_;
    Method method_;
    MeshType mesh_type_;
    SemiDGFEM::Physics::TransportModel transport_model_;
    int order_;
    
    // Physics configuration
    SemiDGFEM::Physics::PhysicsConfig physics_config_;
    
    // Physics models
    std::unique_ptr<SemiDGFEM::Physics::NonEquilibriumStatistics> non_eq_stats_;
    std::unique_ptr<SemiDGFEM::Physics::EnergyTransportModel> energy_transport_;
    std::unique_ptr<SemiDGFEM::Physics::HydrodynamicModel> hydrodynamic_;
    
    // Base solvers
    std::unique_ptr<Poisson> poisson_;
    std::unique_ptr<DriftDiffusion> drift_diffusion_;
    
    // Doping and material data
    std::vector<double> Nd_, Na_, Et_;
    
    // Convergence tracking
    double convergence_residual_;

    // Helper methods
    void initialize_physics_models();
    void validate_inputs(const std::vector<double>& bc) const;
    void validate_doping() const;
    void validate_order() const;

    // Transport model specific solvers
    std::map<std::string, std::vector<double>> solve_drift_diffusion_transport(
        const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
        int poisson_max_iter, double poisson_tol);
        
    std::map<std::string, std::vector<double>> solve_energy_transport(
        const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
        int poisson_max_iter, double poisson_tol);
        
    std::map<std::string, std::vector<double>> solve_hydrodynamic_transport(
        const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
        int poisson_max_iter, double poisson_tol);
        
    std::map<std::string, std::vector<double>> solve_non_equilibrium_transport(
        const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
        int poisson_max_iter, double poisson_tol);

    // Physics calculations
    void compute_advanced_carrier_densities(const std::vector<double>& V,
                                           const std::vector<double>& quasi_fermi_n,
                                           const std::vector<double>& quasi_fermi_p,
                                           std::vector<double>& n,
                                           std::vector<double>& p) const;
                                           
    void compute_energy_densities(const std::vector<double>& V,
                                 const std::vector<double>& n,
                                 const std::vector<double>& p,
                                 std::vector<double>& energy_n,
                                 std::vector<double>& energy_p) const;
                                 
    void compute_momentum_densities(const std::vector<double>& V,
                                   const std::vector<double>& n,
                                   const std::vector<double>& p,
                                   std::vector<double>& momentum_n,
                                   std::vector<double>& momentum_p) const;

    // Convergence checking
    bool check_convergence(const std::vector<double>& old_solution,
                          const std::vector<double>& new_solution,
                          double tolerance) const;
};

/**
 * @brief Factory class for creating transport solvers
 */
class TransportSolverFactory {
public:
    static std::unique_ptr<AdvancedTransportSolver> create_solver(
        const Device& device, Method method, MeshType mesh_type,
        SemiDGFEM::Physics::TransportModel transport_model,
        int order = 3);
        
    static std::vector<std::string> get_available_transport_models();
    static std::string get_transport_model_description(SemiDGFEM::Physics::TransportModel model);
};

} // namespace transport

// C interface for Cython with enhanced error handling
extern "C" {
    simulator::transport::AdvancedTransportSolver* create_advanced_transport_solver(
        simulator::Device* device, int method, int mesh_type, int transport_model, int order);
    void destroy_advanced_transport_solver(simulator::transport::AdvancedTransportSolver* solver);
    int advanced_transport_solver_is_valid(simulator::transport::AdvancedTransportSolver* solver);
    int advanced_transport_solver_set_doping(simulator::transport::AdvancedTransportSolver* solver, 
                                            double* Nd, double* Na, int size);
    int advanced_transport_solver_set_trap_level(simulator::transport::AdvancedTransportSolver* solver, 
                                                double* Et, int size);
    int advanced_transport_solver_solve(simulator::transport::AdvancedTransportSolver* solver, 
                                       double* bc, int bc_size, double Vg,
                                       int max_steps, int use_amr, int poisson_max_iter, double poisson_tol,
                                       double* V, double* n, double* p, double* Jn, double* Jp, 
                                       double* energy_n, double* energy_p, double* T_n, double* T_p,
                                       int size);
    size_t advanced_transport_solver_get_dof_count(simulator::transport::AdvancedTransportSolver* solver);
    double advanced_transport_solver_get_convergence_residual(simulator::transport::AdvancedTransportSolver* solver);
    int advanced_transport_solver_get_order(simulator::transport::AdvancedTransportSolver* solver);
    int advanced_transport_solver_get_transport_model(simulator::transport::AdvancedTransportSolver* solver);
}

} // namespace simulator
