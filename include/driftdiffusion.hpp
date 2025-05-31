#pragma once
#include "device.hpp"
#include "mesh.hpp"
#include "poisson.hpp"
#include <vector>
#include <string>
#include <map>

namespace simulator {
class DriftDiffusion {
public:
    DriftDiffusion(const Device& device, Method method, MeshType mesh_type, int order = 3);
    void set_doping(const std::vector<double>& Nd, const std::vector<double>& Na);
    void set_trap_level(const std::vector<double>& Et);
    // [MODIFICATION]: Updated to use self-consistent solver
    std::map<std::string, std::vector<double>> solve_drift_diffusion(
        const std::vector<double>& bc, double Vg = 0.0, int max_steps = 100, 
        bool use_amr = false, int poisson_max_iter = 50, double poisson_tol = 1e-6);
private:
    const Device& device_;
    Method method_;
    MeshType mesh_type_;
    int order_;
    std::vector<double> Nd_, Na_, Et_;
    Poisson poisson_;
};
} // namespace simulator

// [MODIFICATION]: Expose to Cython
extern "C" {
    simulator::DriftDiffusion* create_drift_diffusion(simulator::Device* device, int method, int mesh_type, int order);
    void drift_diffusion_set_doping(simulator::DriftDiffusion* dd, double* Nd, double* Na, int size);
    void drift_diffusion_set_trap_level(simulator::DriftDiffusion* dd, double* Et, int size);
    void drift_diffusion_solve(simulator::DriftDiffusion* dd, double* bc, int bc_size, double Vg, 
                               int max_steps, bool use_amr, int poisson_max_iter, double poisson_tol, 
                               double* results, int result_cols, int result_rows);
}