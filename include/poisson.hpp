#pragma once
#include "device.hpp"
#include "mesh.hpp"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

namespace simulator {
enum class Method { FDM, FEM, FVM, SEM, MC, DG };

class Poisson {
public:
    Poisson(const Device& device, Method method, MeshType mesh_type);
    ~Poisson();

    // Copy constructor and assignment operator
    Poisson(const Poisson& other);
    Poisson& operator=(const Poisson& other);

    // Move constructor and assignment operator
    Poisson(Poisson&& other) noexcept;
    Poisson& operator=(Poisson&& other) noexcept;

    void set_charge_density(const std::vector<double>& rho);
    std::vector<double> solve_2d(const std::vector<double>& bc);

    // Enhanced self-consistent solver with better error handling
    std::vector<double> solve_2d_self_consistent(const std::vector<double>& bc,
                                                std::vector<double>& n,
                                                std::vector<double>& p,
                                                const std::vector<double>& Nd,
                                                const std::vector<double>& Na,
                                                int max_iter = 100,
                                                double tol = 1e-6);

    std::vector<double> solve_2d_self_consistent_unstructured(const std::vector<double>& bc,
                                                std::vector<double>& n,
                                                std::vector<double>& p,
                                                const std::vector<double>& Nd,
                                                const std::vector<double>& Na,
                                                int max_iter = 100,
                                                double tol = 1e-6);

    // Utility methods
    bool is_valid() const;
    void validate() const;
    size_t get_dof_count() const;
    double get_residual_norm() const;

    // PETSc resource management
    void cleanup_petsc_objects();

private:
    const Device& device_;
    Method method_;
    MeshType mesh_type_;
    std::vector<double> rho_, V_;

    // PETSc objects with proper cleanup
    struct PETScObjects;
    std::unique_ptr<PETScObjects> petsc_objects_;

    // Helper methods
    void initialize_petsc();
    void validate_inputs(const std::vector<double>& bc) const;

    // DG-specific helper methods
    double interpolate_charge_density(double x, double y) const;
    void add_dg_penalty_terms(int element_index,
                             const std::vector<int>& element_nodes,
                             const std::vector<double>& grid_x,
                             const std::vector<double>& grid_y,
                             std::vector<std::vector<double>>& K_elem,
                             std::vector<double>& f_elem) const;
    void compute_p3_basis_functions(double xi, double eta, double zeta,
                                   std::vector<double>& N,
                                   std::vector<std::array<double, 2>>& grad_N) const;
    void validate_self_consistent_inputs(const std::vector<double>& bc,
                                       const std::vector<double>& n,
                                       const std::vector<double>& p,
                                       const std::vector<double>& Nd,
                                       const std::vector<double>& Na) const;

    // Solver implementations
    std::vector<double> solve_structured_2d(const std::vector<double>& bc);
    std::vector<double> solve_unstructured_2d(const std::vector<double>& bc);

    std::vector<double> solve_structured_2d_self_consistent(const std::vector<double>& bc,
                                                          std::vector<double>& n,
                                                          std::vector<double>& p,
                                                          const std::vector<double>& Nd,
                                                          const std::vector<double>& Na,
                                                          int max_iter, double tol);

    std::vector<double> solve_unstructured_2d_self_consistent(const std::vector<double>& bc,
                                                            std::vector<double>& n,
                                                            std::vector<double>& p,
                                                            const std::vector<double>& Nd,
                                                            const std::vector<double>& Na,
                                                            int max_iter, double tol);
};

// C interface for Cython with enhanced error handling
extern "C" {
    simulator::Poisson* create_poisson(simulator::Device* device, int method, int mesh_type);
    void destroy_poisson(simulator::Poisson* poisson);
    int poisson_is_valid(simulator::Poisson* poisson);
    void poisson_set_charge_density(simulator::Poisson* poisson, double* rho, int size);
    int poisson_solve_2d(simulator::Poisson* poisson, double* bc, int bc_size, double* V, int V_size);
    int poisson_solve_2d_self_consistent(simulator::Poisson* poisson, double* bc, int bc_size,
                                        double* n, double* p, double* Nd, double* Na, int size,
                                        int max_iter, double tol, double* V, int V_size);
    size_t poisson_get_dof_count(simulator::Poisson* poisson);
    double poisson_get_residual_norm(simulator::Poisson* poisson);
}

} // namespace simulator