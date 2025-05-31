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
private:
    const Device& device_;
    Method method_;
    MeshType mesh_type_;
    std::vector<double> rho_, V_;
};
} // namespace simulator