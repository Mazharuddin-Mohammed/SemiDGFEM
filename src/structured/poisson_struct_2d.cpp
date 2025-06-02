#include "poisson.hpp"
#include "mesh.hpp"
#include "dg_assembly.hpp"
#include "dg_basis_functions.hpp"
#include <petscksp.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <memory>
#include <functional>

namespace simulator {

// PETSc objects structure for proper resource management
struct Poisson::PETScObjects {
    Mat A = nullptr;
    Vec x = nullptr;
    Vec b = nullptr;
    KSP ksp = nullptr;
    bool initialized = false;

    ~PETScObjects() {
        cleanup();
    }

    void cleanup() {
        if (ksp) { KSPDestroy(&ksp); ksp = nullptr; }
        if (A) { MatDestroy(&A); A = nullptr; }
        if (x) { VecDestroy(&x); x = nullptr; }
        if (b) { VecDestroy(&b); b = nullptr; }
        initialized = false;
    }
};

Poisson::Poisson(const Device& device, Method method, MeshType mesh_type)
    : device_(device), method_(method), mesh_type_(mesh_type),
      petsc_objects_(std::make_unique<PETScObjects>()) {

    if (!device.is_valid()) {
        throw std::invalid_argument("Invalid device provided to Poisson constructor");
    }

    if (method != Method::DG) {
        throw std::invalid_argument("Only DG method is currently supported");
    }

    try {
        initialize_petsc();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize Poisson solver: " + std::string(e.what()));
    }
}

Poisson::~Poisson() {
    cleanup_petsc_objects();
}

Poisson::Poisson(const Poisson& other)
    : device_(other.device_), method_(other.method_), mesh_type_(other.mesh_type_),
      rho_(other.rho_), V_(other.V_), petsc_objects_(std::make_unique<PETScObjects>()) {
    initialize_petsc();
}

Poisson& Poisson::operator=(const Poisson& other) {
    if (this != &other) {
        cleanup_petsc_objects();
        // Note: device_ is a const reference and cannot be reassigned
        method_ = other.method_;
        mesh_type_ = other.mesh_type_;
        rho_ = other.rho_;
        V_ = other.V_;
        petsc_objects_ = std::make_unique<PETScObjects>();
        initialize_petsc();
    }
    return *this;
}

Poisson::Poisson(Poisson&& other) noexcept
    : device_(other.device_), method_(other.method_), mesh_type_(other.mesh_type_),
      rho_(std::move(other.rho_)), V_(std::move(other.V_)),
      petsc_objects_(std::move(other.petsc_objects_)) {
}

Poisson& Poisson::operator=(Poisson&& other) noexcept {
    if (this != &other) {
        cleanup_petsc_objects();
        // Note: device_ is a const reference and cannot be reassigned
        method_ = other.method_;
        mesh_type_ = other.mesh_type_;
        rho_ = std::move(other.rho_);
        V_ = std::move(other.V_);
        petsc_objects_ = std::move(other.petsc_objects_);
    }
    return *this;
}

void Poisson::set_charge_density(const std::vector<double>& rho) {
    if (rho.empty()) {
        throw std::invalid_argument("Charge density vector cannot be empty");
    }
    rho_ = rho;
}

void Poisson::initialize_petsc() {
    static bool petsc_initialized = false;
    if (!petsc_initialized) {
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
        petsc_initialized = true;
    }
    petsc_objects_->initialized = true;
}

void Poisson::cleanup_petsc_objects() {
    if (petsc_objects_) {
        petsc_objects_->cleanup();
    }
}

bool Poisson::is_valid() const {
    return device_.is_valid() && petsc_objects_ && petsc_objects_->initialized;
}

void Poisson::validate() const {
    if (!is_valid()) {
        throw std::runtime_error("Poisson solver is in invalid state");
    }
}

void Poisson::validate_inputs(const std::vector<double>& bc) const {
    if (bc.size() != 4) {
        throw std::invalid_argument("2D Poisson solver requires exactly 4 boundary conditions");
    }

    for (size_t i = 0; i < bc.size(); ++i) {
        if (!std::isfinite(bc[i])) {
            throw std::invalid_argument("Boundary condition " + std::to_string(i) + " is not finite");
        }
    }
}

std::vector<double> Poisson::solve_2d(const std::vector<double>& bc) {
    validate();
    validate_inputs(bc);

    if (method_ != Method::DG) {
        throw std::invalid_argument("Only DG method is supported");
    }

    if (mesh_type_ == MeshType::Structured) {
        return solve_structured_2d(bc);
    } else {
        return solve_unstructured_2d(bc);
    }
}

std::vector<double> Poisson::solve_structured_2d(const std::vector<double>& bc) {
    try {
        Mesh mesh(device_, mesh_type_);
        auto grid_x = mesh.get_grid_points_x();
        auto grid_y = mesh.get_grid_points_y();
        auto elements = mesh.get_elements();

        if (grid_x.empty() || grid_y.empty() || elements.empty()) {
            throw std::runtime_error("Invalid mesh data");
        }

        int n_nodes = static_cast<int>(grid_x.size());
        int n_elements = static_cast<int>(elements.size());
        const int order = 3; // P3 elements
        int dofs_per_element = 10; // P3 has 10 DOFs per triangular element
        int n_dofs = n_elements * dofs_per_element;

        if (n_dofs <= 0) {
            throw std::runtime_error("Invalid number of degrees of freedom");
        }

        V_.resize(n_dofs, 0.0);

        // Identify boundary nodes and set boundary conditions
        std::vector<bool> is_boundary(n_nodes, false);
        auto extents = device_.get_extents();
        double Lx = extents[0], Ly = extents[1];

        // Improved boundary tolerance based on device dimensions
        const double boundary_tol = std::max(1e-8, std::min(Lx, Ly) * 1e-6);

        for (int i = 0; i < n_nodes; ++i) {
            if (i >= static_cast<int>(grid_x.size()) || i >= static_cast<int>(grid_y.size())) {
                throw std::runtime_error("Node index out of bounds");
            }

            double x = grid_x[i], y = grid_y[i];

            // Improved boundary detection with relative tolerance
            bool on_left = (x <= boundary_tol);
            bool on_right = (x >= Lx - boundary_tol);
            bool on_bottom = (y <= boundary_tol);
            bool on_top = (y >= Ly - boundary_tol);

            if (on_left) {
                V_[i] = bc[0]; is_boundary[i] = true;
            } else if (on_right) {
                V_[i] = bc[1]; is_boundary[i] = true;
            } else if (on_bottom) {
                V_[i] = bc[2]; is_boundary[i] = true;
            } else if (on_top) {
                V_[i] = bc[3]; is_boundary[i] = true;
            }
        }

        // Clean up any existing PETSc objects
        petsc_objects_->cleanup();

        // Create PETSc objects with error checking
        PetscErrorCode ierr;
        ierr = MatCreate(PETSC_COMM_WORLD, &petsc_objects_->A);
        if (ierr != 0) throw std::runtime_error("Failed to create PETSc matrix");
        ierr = MatSetSizes(petsc_objects_->A, PETSC_DECIDE, PETSC_DECIDE, n_dofs, n_dofs);
        if (ierr != 0) throw std::runtime_error("Failed to set matrix sizes");
        ierr = MatSetType(petsc_objects_->A, MATMPIAIJ);
        if (ierr != 0) throw std::runtime_error("Failed to set matrix type");
        ierr = MatSetUp(petsc_objects_->A);
        if (ierr != 0) throw std::runtime_error("Failed to set up matrix");

        ierr = VecCreate(PETSC_COMM_WORLD, &petsc_objects_->x);
        if (ierr != 0) throw std::runtime_error("Failed to create solution vector");
        ierr = VecSetSizes(petsc_objects_->x, PETSC_DECIDE, n_dofs);
        if (ierr != 0) throw std::runtime_error("Failed to set vector sizes");
        ierr = VecSetType(petsc_objects_->x, VECMPI);
        if (ierr != 0) throw std::runtime_error("Failed to set vector type");
        ierr = VecDuplicate(petsc_objects_->x, &petsc_objects_->b);
        if (ierr != 0) throw std::runtime_error("Failed to duplicate vector");

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
                    MatSetValue(petsc_objects_->A, global_i, base_idx + j, K[i][j], ADD_VALUES);
                }
                VecSetValue(petsc_objects_->b, global_i, -f[i], ADD_VALUES);
            }
        }
    }

    for (int i = 0; i < n_nodes; ++i) {
        if (is_boundary[i]) {
            MatSetValue(petsc_objects_->A, i, i, 1.0, INSERT_VALUES);
            VecSetValue(petsc_objects_->b, i, V_[i], INSERT_VALUES);
        }
    }

    MatAssemblyBegin(petsc_objects_->A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(petsc_objects_->A, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(petsc_objects_->b);
    VecAssemblyEnd(petsc_objects_->b);

    KSPCreate(PETSC_COMM_WORLD, &petsc_objects_->ksp);
    KSPSetOperators(petsc_objects_->ksp, petsc_objects_->A, petsc_objects_->A);
    KSPSetType(petsc_objects_->ksp, KSPCG);
    KSPSetTolerances(petsc_objects_->ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetFromOptions(petsc_objects_->ksp);
    KSPSolve(petsc_objects_->ksp, petsc_objects_->b, petsc_objects_->x);

    PetscScalar* array;
    VecGetArray(petsc_objects_->x, &array);
    for (int i = 0; i < n_dofs; ++i) {
        V_[i] = array[i];
    }
    VecRestoreArray(petsc_objects_->x, &array);

    // Note: Don't call PetscFinalize here as it should be called once at program end
    // The cleanup will be handled by the PETScObjects destructor

    std::vector<double> V_nodes(n_nodes, 0.0);
    for (int e = 0; e < n_elements; ++e) {
        int base_idx = e * dofs_per_element;
        V_nodes[elements[e][0]] = V_[base_idx];
        V_nodes[elements[e][1]] = V_[base_idx + 1];
        V_nodes[elements[e][2]] = V_[base_idx + 2];
    }
    V_ = V_nodes;
    return V_;

    } catch (const std::exception& e) {
        std::cerr << "Error in structured Poisson solver: " << e.what() << std::endl;
        return V_;
    }
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

// Helper method implementations
double Poisson::interpolate_charge_density(double x, double y) const {
    if (rho_.empty()) return 0.0;

    // Simple nearest-neighbor interpolation
    // In a full implementation, this would use proper interpolation
    // based on the mesh structure

    // For now, return average charge density
    double sum = 0.0;
    for (double rho_val : rho_) {
        sum += rho_val;
    }
    return sum / rho_.size();
}

void Poisson::add_dg_penalty_terms(int element_index,
                                  const std::vector<int>& element_nodes,
                                  const std::vector<double>& grid_x,
                                  const std::vector<double>& grid_y,
                                  std::vector<std::vector<double>>& K_elem,
                                  std::vector<double>& f_elem) const {
    // Simplified DG penalty implementation
    // In a full implementation, this would:
    // 1. Identify element faces and neighbors
    // 2. Compute penalty parameters based on element size
    // 3. Add consistency, symmetry, and penalty terms

    const double penalty_param = 50.0; // Penalty parameter

    // For now, add a simple penalty to diagonal terms
    for (size_t i = 0; i < K_elem.size(); ++i) {
        K_elem[i][i] += penalty_param * 1e-6; // Small penalty for stability
    }
}

void Poisson::compute_p3_basis_functions(double xi, double eta, double zeta,
                                        std::vector<double>& N,
                                        std::vector<std::array<double, 2>>& grad_N) const {
    if (N.size() != 10 || grad_N.size() != 10) {
        throw std::invalid_argument("P3 basis requires exactly 10 DOF arrays");
    }

    // P3 Lagrange basis functions on reference triangle
    // Vertex functions
    N[0] = 0.5 * zeta * (3.0 * zeta - 1.0) * (3.0 * zeta - 2.0);
    N[1] = 0.5 * xi * (3.0 * xi - 1.0) * (3.0 * xi - 2.0);
    N[2] = 0.5 * eta * (3.0 * eta - 1.0) * (3.0 * eta - 2.0);

    // Edge functions
    N[3] = 4.5 * zeta * xi * (3.0 * zeta - 1.0);
    N[4] = 4.5 * zeta * xi * (3.0 * xi - 1.0);
    N[5] = 4.5 * xi * eta * (3.0 * xi - 1.0);
    N[6] = 4.5 * xi * eta * (3.0 * eta - 1.0);
    N[7] = 4.5 * eta * zeta * (3.0 * eta - 1.0);
    N[8] = 4.5 * eta * zeta * (3.0 * zeta - 1.0);

    // Interior function
    N[9] = 27.0 * zeta * xi * eta;

    // Gradients (simplified - would need proper implementation)
    for (int i = 0; i < 10; ++i) {
        grad_N[i][0] = 0.0; // d/dxi
        grad_N[i][1] = 0.0; // d/deta
    }

    // Vertex gradients
    grad_N[0][0] = -0.5 * (27.0 * zeta * zeta - 18.0 * zeta + 2.0);
    grad_N[0][1] = -0.5 * (27.0 * zeta * zeta - 18.0 * zeta + 2.0);

    grad_N[1][0] = 0.5 * (27.0 * xi * xi - 12.0 * xi + 1.0);
    grad_N[1][1] = 0.0;

    grad_N[2][0] = 0.0;
    grad_N[2][1] = 0.5 * (27.0 * eta * eta - 12.0 * eta + 1.0);

    // Edge gradients (simplified)
    grad_N[3][0] = 4.5 * (zeta * (3.0 * zeta - 1.0) - xi * (6.0 * zeta - 1.0));
    grad_N[3][1] = -4.5 * xi * (6.0 * zeta - 1.0);

    // Additional edge and interior gradients would be computed similarly
    // This is a simplified implementation for demonstration
}

} // namespace simulator

// [MODIFICATION]: Cython bindings
extern "C" {
    simulator::Poisson* create_poisson(simulator::Device* device, int method, int mesh_type) {
        return new simulator::Poisson(*device, static_cast<simulator::Method>(method), static_cast<simulator::MeshType>(mesh_type));
    }
    void destroy_poisson(simulator::Poisson* poisson) {
        delete poisson;
    }
    void poisson_set_charge_density(simulator::Poisson* poisson, double* rho, int size) {
        std::vector<double> rho_vec(rho, rho + size);
        poisson->set_charge_density(rho_vec);
    }
    void poisson_solve_2d(simulator::Poisson* poisson, double* bc, int bc_size, double* V, int V_size) {
        std::vector<double> bc_vec(bc, bc + bc_size);
        auto result = poisson->solve_2d(bc_vec);
        for (int i = 0; i < std::min(V_size, (int)result.size()); ++i) {
            V[i] = result[i];
        }
    }
    void poisson_solve_2d_self_consistent(simulator::Poisson* poisson, double* bc, int bc_size,
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