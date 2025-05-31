#pragma once
#include "device.hpp"
#include "mesh.hpp"
#include "poisson.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <stdexcept>

namespace simulator {

class DriftDiffusion {
public:
    DriftDiffusion(const Device& device, Method method, MeshType mesh_type, int order = 3);
    ~DriftDiffusion();

    // Copy constructor and assignment operator
    DriftDiffusion(const DriftDiffusion& other);
    DriftDiffusion& operator=(const DriftDiffusion& other);

    // Move constructor and assignment operator
    DriftDiffusion(DriftDiffusion&& other) noexcept;
    DriftDiffusion& operator=(DriftDiffusion&& other) noexcept;

    void set_doping(const std::vector<double>& Nd, const std::vector<double>& Na);
    void set_trap_level(const std::vector<double>& Et);

    std::map<std::string, std::vector<double>> solve_drift_diffusion(
        const std::vector<double>& bc, double Vg = 0.0, int max_steps = 100,
        bool use_amr = false, int poisson_max_iter = 50, double poisson_tol = 1e-6);

    // Utility methods
    bool is_valid() const;
    void validate() const;
    size_t get_dof_count() const;
    double get_convergence_residual() const;
    int get_order() const { return order_; }

    // PETSc resource management
    void cleanup_petsc_objects();

private:
    const Device& device_;
    Method method_;
    MeshType mesh_type_;
    int order_;
    std::vector<double> Nd_, Na_, Et_;
    std::unique_ptr<Poisson> poisson_;

    // PETSc objects with proper cleanup
    struct PETScObjects;
    std::unique_ptr<PETScObjects> petsc_objects_;

    // Convergence tracking
    double convergence_residual_;

    // Helper methods
    void initialize_petsc();
    void validate_inputs(const std::vector<double>& bc) const;
    void validate_doping() const;
    void validate_order() const;

    // Solver implementations
    std::map<std::string, std::vector<double>> solve_structured_drift_diffusion(
        const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
        int poisson_max_iter, double poisson_tol);

    std::map<std::string, std::vector<double>> solve_unstructured_drift_diffusion(
        const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
        int poisson_max_iter, double poisson_tol);

    // Physics calculations
    void compute_carrier_densities(const std::vector<double>& V,
                                 std::vector<double>& n,
                                 std::vector<double>& p) const;

    void compute_current_densities(const std::vector<double>& V,
                                 const std::vector<double>& n,
                                 const std::vector<double>& p,
                                 std::vector<double>& Jn,
                                 std::vector<double>& Jp) const;

    // Convergence checking
    bool check_convergence(const std::vector<double>& V_old,
                          const std::vector<double>& V_new,
                          double tolerance) const;
};

} // namespace simulator

// C interface for Cython with enhanced error handling
extern "C" {
    simulator::DriftDiffusion* create_drift_diffusion(simulator::Device* device, int method, int mesh_type, int order);
    void destroy_drift_diffusion(simulator::DriftDiffusion* dd);
    int drift_diffusion_is_valid(simulator::DriftDiffusion* dd);
    int drift_diffusion_set_doping(simulator::DriftDiffusion* dd, double* Nd, double* Na, int size);
    int drift_diffusion_set_trap_level(simulator::DriftDiffusion* dd, double* Et, int size);
    int drift_diffusion_solve(simulator::DriftDiffusion* dd, double* bc, int bc_size, double Vg,
                             int max_steps, int use_amr, int poisson_max_iter, double poisson_tol,
                             double* V, double* n, double* p, double* Jn, double* Jp, int size);
    size_t drift_diffusion_get_dof_count(simulator::DriftDiffusion* dd);
    double drift_diffusion_get_convergence_residual(simulator::DriftDiffusion* dd);
    int drift_diffusion_get_order(simulator::DriftDiffusion* dd);
}