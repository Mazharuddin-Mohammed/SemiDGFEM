#include "poisson.hpp"
#include "mesh.hpp"
#include <petscksp.h>
#include <vector>
#include <cmath>

namespace simulator {
Poisson::Poisson(const Device& device, Method method, MeshType mesh_type)
    : device_(device), method_(method), mesh_type_(mesh_type) {}

void Poisson::set_charge_density(const std::vector<double>& rho) {
    rho_ = rho;
}

std::vector<double> Poisson::solve_2d(const std::vector<double>& bc) {
    if (method_ != Method::DG || mesh_type_ != MeshType::Structured) return V_;
    if (bc.size() != 4) throw std::invalid_argument("2D DG requires 4 boundary conditions");

    Mesh mesh(device_, mesh_type_);
    auto grid_x = mesh.get_grid_points_x();
    auto grid_y = mesh.get_grid_points_y();
    auto elements = mesh.get_elements();
    int n_nodes = grid_x.size();
    int n_elements = elements.size();
    const int order = 3; // P3
    int dofs_per_element = 10;
    int n_dofs = n_elements * dofs_per_element;
    V_.resize(n_dofs, 0.0);

    std::vector<bool> is_boundary(n_nodes, false);
    double Lx = device_.get_extents()[0], Ly = device_.get_extents()[1];
    for (int i = 0; i < n_nodes; ++i) {
        double x = grid_x[i], y = grid_y[i];
        if (x <= 1e-10) { V_[i] = bc[0]; is_boundary[i] = true; }
        else if (x >= Lx - 1e-10) { V_[i] = bc[1]; is_boundary[i] = true; }
        else if (y <= 1e-10) { V_[i] = bc[2]; is_boundary[i] = true; }
        else if (y >= Ly - 1e-10) { V_[i] = bc[3]; is_boundary[i] = true; }
    }

    PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    Mat A;
    Vec x, b;
    KSP ksp;

    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n_dofs, n_dofs);
    MatSetType(A, MATMPIAIJ);
    MatSetUp(A);
    VecCreate(PETSC_COMM_WORLD, &x);
    VecSetSizes(x, PETSC_DECIDE, n_dofs);
    VecSetType(x, VECMPI);
    VecDuplicate(x, &b);

    std::vector<std::vector<double>> quad_points = {
        {1.0/3.0, 1.0/3.0}, {0.6, 0.2}, {0.2, 0.6}, {0.2, 0.2},
        {0.8, 0.1}, {0.1, 0.8}, {0.4, 0.4}
    };
    std::vector<double> quad_weights = {0.225, 0.125, 0.125, 0.125, 0.1, 0.1, 0.1};

    auto phi = [&](double xi, double eta, int j) {
        double zeta = 1.0 - xi - eta;
        if (j == 0) return zeta * (3.0 * zeta - 1.0) * (3.0 * zeta - 2.0) / 2.0;
        if (j == 1) return xi * (3.0 * xi - 1.0) * (3.0 * xi - 2.0) / 2.0;
        if (j == 2) return eta * (3.0 * eta - 1.0) * (3.0 * eta - 2.0) / 2.0;
        if (j == 3) return 9.0 * zeta * xi * (3.0 * zeta - 1.0) / 2.0;
        if (j == 4) return 9.0 * zeta * xi * (3.0 * xi - 1.0) / 2.0;
        if (j == 5) return 9.0 * xi * eta * (3.0 * xi - 1.0) / 2.0;
        if (j == 6) return 9.0 * xi * eta * (3.0 * eta - 1.0) / 2.0;
        if (j == 7) return 9.0 * eta * zeta * (3.0 * eta - 1.0) / 2.0;
        if (j == 8) return 9.0 * eta * zeta * (3.0 * zeta - 1.0) / 2.0;
        return 27.0 * zeta * xi * eta;
    };
    auto dphi_dx = [&](double xi, double eta, int j, double b1, double b2, double b3) {
        double zeta = 1.0 - xi - eta;
        if (j == 0) return (27.0 * zeta * zeta - 18.0 * zeta + 2.0) * b1 / 2.0;
        if (j == 1) return (27.0 * xi * xi - 12.0 * xi + 1.0) * b2 / 2.0;
        if (j == 2) return (27.0 * eta * eta - 12.0 * eta + 1.0) * b3 / 2.0;
        if (j == 3) return 9.0 * ((3.0 * zeta - 1.0) * xi + zeta * (3.0 * zeta - 1.0)) * b1 / 2.0;
        if (j == 4) return 9.0 * (zeta * (6.0 * xi - 1.0) + xi * (3.0 * xi - 1.0)) * b2 / 2.0;
        if (j == 5) return 9.0 * (eta * (6.0 * xi - 1.0) + xi * (3.0 * xi - 1.0)) * b2 / 2.0;
        if (j == 6) return 9.0 * (xi * (6.0 * eta - 1.0) + eta * (3.0 * eta - 1.0)) * b3 / 2.0;
        if (j == 7) return 9.0 * (zeta * (6.0 * eta - 1.0) + eta * (3.0 * eta - 1.0)) * b3 / 2.0;
        if (j == 8) return 9.0 * ((3.0 * zeta - 1.0) * eta + zeta * (3.0 * zeta - 1.0)) * b1 / 2.0;
        return 27.0 * (xi * eta + zeta * eta + zeta * xi) * (b1 + b2 + b3);
    };
    auto dphi_dy = [&](double xi, double eta, int j, double c1, double c2, double c3) {
        double zeta = 1.0 - xi - eta;
        if (j == 0) return (27.0 * zeta * zeta - 18.0 * zeta + 2.0) * c1 / 2.0;
        if (j == 1) return (27.0 * xi * xi - 12.0 * xi + 1.0) * c2 / 2.0;
        if (j == 2) return (27.0 * eta * eta - 12.0 * eta + 1.0) * c3 / 2.0;
        if (j == 3) return 9.0 * ((3.0 * zeta - 1.0) * xi + zeta * (3.0 * zeta - 1.0)) * c1 / 2.0;
        if (j == 4) return 9.0 * (zeta * (6.0 * xi - 1.0) + xi * (3.0 * xi - 1.0)) * c2 / 2.0;
        if (j == 5) return 9.0 * (eta * (6.0 * xi - 1.0) + xi * (3.0 * xi - 1.0)) * c2 / 2.0;
        if (j == 6) return 9.0 * (xi * (6.0 * eta - 1.0) + eta * (3.0 * eta - 1.0)) * c3 / 2.0;
        if (j == 7) return 9.0 * (zeta * (6.0 * eta - 1.0) + eta * (3.0 * eta - 1.0)) * c3 / 2.0;
        if (j == 8) return 9.0 * ((3.0 * zeta - 1.0) * eta + zeta * (3.0 * zeta - 1.0)) * c1 / 2.0;
        return 27.0 * (xi * eta + zeta * eta + zeta * xi) * (c1 + c2 + c3);
    };

    for (int e = 0; e < n_elements; ++e) {
        int i1 = elements[e][0], i2 = elements[e][1], i3 = elements[e][2];
        double x1 = grid_x[i1], y1 = grid_y[i1];
        double x2 = grid_x[i2], y2 = grid_y[i2];
        double x3 = grid_x[i3], y3 = grid_y[i3];
        double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
        if (area < 1e-12) continue;

        double b1 = (y2 - y3) / (2.0 * area), c1 = (x3 - x2) / (2.0 * area);
        double b2 = (y3 - y1) / (2.0 * area), c2 = (x1 - x3) / (2.0 * area);
        double b3 = (y1 - y2) / (2.0 * area), c3 = (x2 - x1) / (2.0 * area);

        double eps = device_.get_epsilon_at((x1 + x2 + x3) / 3.0, (y1 + y2 + y3) / 3.0);
        double K[10][10] = {0}, f[10] = {0};
        int base_idx = e * dofs_per_element;

        for (size_t q = 0; q < quad_points.size(); ++q) {
            double xi = quad_points[q][0], eta = quad_points[q][1];
            double w = quad_weights[q] * area;
            double rho_q = (rho_.size() > i1 ? rho_[i1] : 0.0) + (rho_.size() > i2 ? rho_[i2] : 0.0) + (rho_.size() > i3 ? rho_[i3] : 0.0);
            rho_q /= 3.0;

            for (int i = 0; i < 10; ++i) {
                for (int j = 0; j < 10; ++j) {
                    K[i][j] += w * eps * (dphi_dx(xi, eta, i, b1, b2, b3) * dphi_dx(xi, eta, j, b1, b2, b3) +
                                          dphi_dy(xi, eta, i, c1, c2, c3) * dphi_dy(xi, eta, j, c1, c2, c3));
                }
                f[i] += w * rho_q * phi(xi, eta, i);
            }
        }

        for (int i = 0; i < 10; ++i) {
            int global_i = base_idx + i;
            if (!is_boundary[i1 + i % 3]) {
                for (int j = 0; j < 10; ++j) {
                    MatSetValue(A, global_i, base_idx + j, K[i][j], ADD_VALUES);
                }
                VecSetValue(b, global_i, -f[i], ADD_VALUES);
            }
        }
    }

    for (int i = 0; i < n_nodes; ++i) {
        if (is_boundary[i]) {
            MatSetValue(A, i, i, 1.0, INSERT_VALUES);
            VecSetValue(b, i, V_[i], INSERT_VALUES);
        }
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPCG);
    KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, b, x);

    PetscScalar* array;
    VecGetArray(x, &array);
    for (int i = 0; i < n_dofs; ++i) {
        V_[i] = array[i];
    }
    VecRestoreArray(x, &array);

    MatDestroy(&A);
    VecDestroy(&x);
    VecDestroy(&b);
    KSPDestroy(&ksp);
    PetscFinalize();

    std::vector<double> V_nodes(n_nodes, 0.0);
    for (int e = 0; e < n_elements; ++e) {
        int base_idx = e * dofs_per_element;
        V_nodes[elements[e][0]] = V_[base_idx];
        V_nodes[elements[e][1]] = V_[base_idx + 1];
        V_nodes[elements[e][2]] = V_[base_idx + 2];
    }
    V_ = V_nodes;
    return V_;
}

// [MODIFICATION]: Self-consistent Poisson solver
std::vector<double> Poisson::solve_2d_self_consistent(const std::vector<double>& bc, 
                                                     std::vector<double>& n, 
                                                     std::vector<double>& p, 
                                                     const std::vector<double>& Nd, 
                                                     const std::vector<double>& Na, 
                                                     int max_iter, double tol) {
    if (method_ != Method::DG || mesh_type_ != MeshType::Structured) return V_;
    if (bc.size() != 4) throw std::invalid_argument("2D DG requires 4 boundary conditions");

    Mesh mesh(device_, mesh_type_);
    auto grid_x = mesh.get_grid_points_x();
    auto grid_y = mesh.get_grid_points_y();
    int n_nodes = grid_x.size();
    const double q = 1.602e-19, kT = 0.0259, ni = 1e10 * 1e6;

    V_.resize(n_nodes, 0.0);
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> rho(n_nodes, 0.0);
        for (size_t i = 0; i < n_nodes; ++i) {
            double doping = (Nd.size() > i ? Nd[i] : 0.0) - (Na.size() > i ? Na[i] : 0.0);
            rho[i] = q * (p[i] - n[i] + doping);
        }
        set_charge_density(rho);
        std::vector<double> V_new = solve_2d(bc);

        double max_change = 0.0;
        for (size_t i = 0; i < n_nodes; ++i) {
            max_change = std::max(max_change, std::abs(V_new[i] - V_[i]));
            V_[i] = 0.9 * V_[i] + 0.1 * V_new[i]; // Damping for stability
            n[i] = ni * std::exp(V_[i] / kT);
            p[i] = ni * std::exp(-V_[i] / kT);
        }
        if (max_change < tol) break;
    }
    return V_;
}
} // namespace simulator

// [MODIFICATION]: Cython bindings
extern "C" {
    Poisson* create_poisson(Device* device, int method, int mesh_type) {
        return new Poisson(*device, static_cast<Method>(method), static_cast<MeshType>(mesh_type));
    }
    void poisson_set_charge_density(Poisson* poisson, double* rho, int size) {
        std::vector<double> rho_vec(rho, rho + size);
        poisson->set_charge_density(rho_vec);
    }
    void poisson_solve_2d(Poisson* poisson, double* bc, int bc_size, double* V, int V_size) {
        std::vector<double> bc_vec(bc, bc + bc_size);
        auto result = poisson->solve_2d(bc_vec);
        for (int i = 0; i < std::min(V_size, (int)result.size()); ++i) {
            V[i] = result[i];
        }
    }
    void poisson_solve_2d_self_consistent(Poisson* poisson, double* bc, int bc_size, 
                                          double* n, double* p, double* Nd, double* Na, int size, 
                                          int max_iter, double tol, double* V, int V_size) {
        std::vector<double> bc_vec(bc, bc + bc_size);
        std::vector<double> n_vec(n, n + size), p_vec(p, p + size);
        std::vector<double> Nd_vec(Nd, Nd + size), Na_vec(Na, Na + size);
        auto result = poisson->solve_2d_self_consistent(bc_vec, n_vec, p_vec, Nd_vec, Na_vec, max_iter, tol);
        for (int i = 0; i < std::min(V_size, (int)result.size()); ++i) {
            V[i] = result[i];
        }
        for (int i = 0; i < size; ++i) {
            n[i] = n_vec[i];
            p[i] = p_vec[i];
        }
    }
}