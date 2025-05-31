#include "driftdiffusion.hpp"
#include "mesh.hpp"
#include <petscksp.h>
#include <cmath>
#include <algorithm>

namespace simulator {
DriftDiffusion::DriftDiffusion(const Device& device, Method method, MeshType mesh_type, int order)
    : device_(device), method_(method), mesh_type_(mesh_type), order_(order), poisson_(device, method, mesh_type) {}

void DriftDiffusion::set_doping(const std::vector<double>& Nd, const std::vector<double>& Na) {
    Nd_ = Nd;
    Na_ = Na;
}

void DriftDiffusion::set_trap_level(const std::vector<double>& Et) {
    Et_ = Et;
}

std::map<std::string, std::vector<double>> DriftDiffusion::solve_drift_diffusion(
    const std::vector<double>& bc, double Vg, int max_steps, bool use_amr, 
    int poisson_max_iter, double poisson_tol) {
    Mesh mesh(device_, mesh_type_);
    auto grid_x = mesh.get_grid_points_x();
    auto grid_y = mesh.get_grid_points_y();
    auto elements = mesh.get_elements();
    int n_nodes = grid_x.size();
    std::vector<double> V(n_nodes, 0.0), n(n_nodes, 1e10), p(n_nodes, 1e10);
    std::vector<double> Jn(2 * n_nodes, 0.0), Jp(2 * n_nodes, 0.0);

    const double q = 1.602e-19, kT = 0.0259, mu_n = 1000e-4, mu_p = 400e-4;
    const double ni = 1e10 * 1e6;

    // [MODIFICATION]: Use self-consistent Poisson solver
    V = poisson_.solve_2d_self_consistent(bc, n, p, Nd_, Na_, poisson_max_iter, poisson_tol);

    for (int step = 0; step < max_steps; ++step) {
        std::vector<double> Ex(n_nodes, 0.0), Ey(n_nodes, 0.0);
        for (size_t e = 0; e < elements.size(); ++e) {
            int i1 = elements[e][0], i2 = elements[e][1], i3 = elements[e][2];
            double dx12 = grid_x[i2] - grid_x[i1], dy12 = grid_y[i2] - grid_y[i1];
            double dx13 = grid_x[i3] - grid_x[i1], dy13 = grid_y[i3] - grid_y[i1];
            Ex[i1] += -(V[i2] - V[i1]) / dx12 + -(V[i3] - V[i1]) / dx13;
            Ey[i1] += -(V[i2] - V[i1]) / dy12 + -(V[i3] - V[i1]) / dy13;
        }

        for (size_t i = 0; i < n_nodes; ++i) {
            double n_new = ni * std::exp((V[i] - (Et_.size() > i ? Et_[i] : 0.0)) / kT);
            double p_new = ni * std::exp(-((V[i] - (Et_.size() > i ? Et_[i] : 0.0)) / kT));
            n[i] = 0.9 * n[i] + 0.1 * n_new;
            p[i] = 0.9 * p[i] + 0.1 * p_new;

            Jn[2 * i] = q * mu_n * (n[i] * Ex[i] + kT * mu_n * (n[i + 1] - n[i]) / (grid_x[i + 1] - grid_x[i]));
            Jn[2 * i + 1] = q * mu_n * (n[i] * Ey[i] + kT * mu_n * (n[i + (int)std::sqrt(n_nodes)] - n[i]) / (grid_y[i + (int)std::sqrt(n_nodes)] - grid_y[i]));
            Jp[2 * i] = q * mu_p * (p[i] * Ex[i] - kT * mu_p * (p[i + 1] - p[i]) / (grid_x[i + 1] - grid_x[i]));
            Jp[2 * i + 1] = q * mu_p * (p[i] * Ey[i] - kT * mu_p * (p[i + (int)std::sqrt(n_nodes)] - p[i]) / (grid_y[i + (int)std::sqrt(n_nodes)] - grid_y[i]));
        }

        if (use_amr) {
            std::vector<bool> refine_flags(elements.size(), false);
            for (size_t e = 0; e < elements.size(); ++e) {
                auto elem = elements[e];
                double grad_x = std::abs(V[elem[1]] - V[elem[0]]) / (grid_x[elem[1]] - grid_x[elem[0]]);
                double grad_y = std::abs(V[elem[2]] - V[elem[0]]) / (grid_y[elem[2]] - grid_y[elem[0]]);
                if (grad_x > 1e5 || grad_y > 1e5) refine_flags[e] = true;
            }
            mesh.refine(refine_flags);
        }

        // [MODIFICATION]: Update potential with self-consistent solver
        V = poisson_.solve_2d_self_consistent(bc, n, p, Nd_, Na_, poisson_max_iter, poisson_tol);

        double max_change = 0.0;
        for (size_t i = 0; i < n_nodes; ++i) {
            max_change = std::max(max_change, std::abs(V[i] - (n[i] - p[i])));
        }
        if (max_change < 1e-6) break;
    }

    std::map<std::string, std::vector<double>> results;
    results["potential"] = V;
    results["n"] = n;
    results["p"] = p;
    results["Jn"] = Jn;
    results["Jp"] = Jp;
    return results;
}
} // namespace simulator

// [MODIFICATION]: Cython bindings
extern "C" {
    DriftDiffusion* create_drift_diffusion(Device* device, int method, int mesh_type, int order) {
        return new DriftDiffusion(*device, static_cast<Method>(method), static_cast<MeshType>(mesh_type), order);
    }
    void drift_diffusion_set_doping(DriftDiffusion* dd, double* Nd, double* Na, int size) {
        std::vector<double> Nd_vec(Nd, Nd + size), Na_vec(Na, Na + size);
        dd->set_doping(Nd_vec, Na_vec);
    }
    void drift_diffusion_set_trap_level(DriftDiffusion* dd, double* Et, int size) {
        std::vector<double> Et_vec(Et, Et + size);
        dd->set_trap_level(Et_vec);
    }
    void drift_diffusion_solve(DriftDiffusion* dd, double* bc, int bc_size, double Vg, 
                               int max_steps, bool use_amr, int poisson_max_iter, double poisson_tol, 
                               double* results, int result_cols, int result_rows) {
        std::vector<double> bc_vec(bc, bc + bc_size);
        auto result_map = dd->solve_drift_diffusion(bc_vec, Vg, max_steps, use_amr, poisson_max_iter, poisson_tol);
        int idx = 0;
        for (const auto& key : {"potential", "n", "p", "Jn", "Jp"}) {
            auto& vec = result_map[key];
            for (size_t i = 0; i < vec.size() && idx < result_cols * result_rows; ++i) {
                results[idx++] = vec[i];
            }
        }
    }
}