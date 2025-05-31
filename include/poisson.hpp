#pragma once
#include "device.hpp"
#include "mesh.hpp"
#include <vector>
#include <string>

namespace simulator {
enum class Method { FDM, FEM, FVM, SEM, MC, DG };

class Poisson {
public:
    Poisson(const Device& device, Method method, MeshType mesh_type);
    void set_charge_density(const std::vector<double>& rho);
    std::vector<double> solve_2d(const std::vector<double>& bc);
    // [MODIFICATION]: Added self-consistent solver
    std::vector<double> solve_2d_self_consistent(const std::vector<double>& bc, 
                                                std::vector<double>& n, 
                                                std::vector<double>& p, 
                                                const std::vector<double>& Nd, 
                                                const std::vector<double>& Na, 
                                                int max_iter = 100, 
                                                double tol = 1e-6);
private:
    const Device& device_;
    Method method_;
    MeshType mesh_type_;
    std::vector<double> rho_, V_;
};
} // namespace simulator

// [MODIFICATION]: Expose to Cython
extern "C" {
    simulator::Poisson* create_poisson(simulator::Device* device, int method, int mesh_type);
    void poisson_set_charge_density(simulator::Poisson* poisson, double* rho, int size);
    void poisson_solve_2d(simulator::Poisson* poisson, double* bc, int bc_size, double* V, int V_size);
    void poisson_solve_2d_self_consistent(simulator::Poisson* poisson, double* bc, int bc_size, 
                                          double* n, double* p, double* Nd, double* Na, int size, 
                                          int max_iter, double tol, double* V, int V_size);
}