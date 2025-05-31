#pragma once
#include "device.hpp"
#include "mesh.hpp"
#include "poisson.hpp"
#include <vector>
#include <string>

namespace simulator {
class DriftDiffusion {
public:
    DriftDiffusion(const Device& device, Method method, MeshType mesh_type);
    void set_doping(const std::vector<double>& Nd, const std::vector<double>& Na);
    void set_trap_level(const std::vector<double>& Et);
    std::map<std::string, std::vector<double>> solve_drift_diffusion(
        const std::vector<double>& bc, double Vg = 0.0, int max_steps = 100, bool use_amr = false);
private:
    const Device& device_;
    Method method_;
    MeshType mesh_type_;
    std::vector<double> Nd_, Na_, Et_;
    Poisson poisson_;
};
} // namespace simulator