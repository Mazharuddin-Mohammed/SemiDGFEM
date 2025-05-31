#include "dg_assembly.hpp"
#include "dg_basis_functions.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace simulator {
namespace dg {

DGAssembly::DGAssembly(int polynomial_order, double penalty_constant)
    : poly_order_(polynomial_order), penalty_const_(penalty_constant) {
    
    if (polynomial_order < 1 || polynomial_order > 3) {
        throw std::invalid_argument("Polynomial order must be 1, 2, or 3");
    }
    
    // Set number of DOFs per element
    if (polynomial_order == 1) {
        dofs_per_element_ = 3;  // P1: 3 vertices
    } else if (polynomial_order == 2) {
        dofs_per_element_ = 6;  // P2: 3 vertices + 3 edges
    } else if (polynomial_order == 3) {
        dofs_per_element_ = 10; // P3: 3 vertices + 6 edges + 1 interior
    }
    
    // Get quadrature rule
    quad_points_ = get_triangle_quadrature(polynomial_order + 1);
    line_quad_points_ = face_integration::get_line_quadrature(polynomial_order + 1);
}

void DGAssembly::assemble_element_matrix(
    const std::array<std::array<double, 2>, 3>& vertices,
    const std::function<double(double, double)>& coefficient_func,
    std::vector<std::vector<double>>& K_elem,
    std::vector<double>& f_elem,
    const std::function<double(double, double)>& source_func) {
    
    // Initialize matrices
    K_elem.assign(dofs_per_element_, std::vector<double>(dofs_per_element_, 0.0));
    f_elem.assign(dofs_per_element_, 0.0);
    
    // Compute geometric quantities
    auto inv_jacobian = compute_inverse_jacobian(vertices[0], vertices[1], vertices[2]);
    double area = compute_element_area(vertices[0], vertices[1], vertices[2]);
    
    if (area < 1e-12) {
        throw std::runtime_error("Degenerate element with near-zero area");
    }
    
    // Prepare basis function storage
    std::vector<double> N(dofs_per_element_);
    std::vector<std::array<double, 2>> grad_N_ref(dofs_per_element_);
    std::vector<std::array<double, 2>> grad_N_phys(dofs_per_element_);
    
    // Numerical integration over element
    for (const auto& qp : quad_points_) {
        double xi = qp.coords[0];
        double eta = qp.coords[1];
        double weight = qp.weight;
        
        // Validate quadrature point
        if (!validate_reference_coordinates(xi, eta)) {
            continue;
        }
        
        // Compute basis functions and gradients
        if (poly_order_ == 2) {
            compute_p2_basis_functions(xi, eta, N, grad_N_ref);
        } else if (poly_order_ == 3) {
            compute_p3_basis_functions(xi, eta, N, grad_N_ref);
        } else {
            throw std::runtime_error("P1 basis functions not implemented yet");
        }
        
        // Transform gradients to physical coordinates
        transform_gradients_to_physical(grad_N_ref, grad_N_phys, inv_jacobian);
        
        // Physical coordinates at quadrature point
        double x_phys = vertices[0][0] + xi * (vertices[1][0] - vertices[0][0]) + 
                       eta * (vertices[2][0] - vertices[0][0]);
        double y_phys = vertices[0][1] + xi * (vertices[1][1] - vertices[0][1]) + 
                       eta * (vertices[2][1] - vertices[0][1]);
        
        // Material coefficient at quadrature point
        double coeff = coefficient_func(x_phys, y_phys);
        
        // Source term at quadrature point
        double source = source_func ? source_func(x_phys, y_phys) : 0.0;
        
        // Integration weight
        double int_weight = weight * area;
        
        // Assemble stiffness matrix: ∫ coeff * ∇φᵢ · ∇φⱼ dΩ
        for (int i = 0; i < dofs_per_element_; ++i) {
            for (int j = 0; j < dofs_per_element_; ++j) {
                K_elem[i][j] += coeff * (grad_N_phys[i][0] * grad_N_phys[j][0] + 
                                        grad_N_phys[i][1] * grad_N_phys[j][1]) * int_weight;
            }
            
            // Assemble load vector: ∫ source * φᵢ dΩ
            f_elem[i] += source * N[i] * int_weight;
        }
    }
}

void DGAssembly::add_interface_penalty(
    const std::array<std::array<double, 2>, 2>& edge_vertices,
    const std::array<int, 2>& element_indices,
    const std::array<std::vector<int>, 2>& local_dof_maps,
    double avg_element_size,
    const std::function<double(double, double)>& coefficient_func,
    std::vector<std::vector<double>>& global_matrix) {
    
    // Compute penalty parameter
    double penalty = PenaltyParameter::compute(poly_order_, avg_element_size, penalty_const_);
    
    // Map line quadrature to triangle edge
    auto edge_quad_points = face_integration::map_to_triangle_edge(line_quad_points_, edge_vertices);
    
    // Edge length for integration
    double edge_length = std::sqrt(
        std::pow(edge_vertices[1][0] - edge_vertices[0][0], 2) +
        std::pow(edge_vertices[1][1] - edge_vertices[0][1], 2)
    );
    
    // Prepare basis function storage for both elements
    std::vector<double> N_left(dofs_per_element_), N_right(dofs_per_element_);
    std::vector<std::array<double, 2>> grad_N_left(dofs_per_element_), grad_N_right(dofs_per_element_);
    
    // Integration along edge
    for (const auto& qp : edge_quad_points) {
        double x_phys = qp.coords[0];
        double y_phys = qp.coords[1];
        double weight = qp.weight;
        
        // Material coefficient at interface
        double coeff = coefficient_func(x_phys, y_phys);
        
        // Map physical point back to reference coordinates for each element
        // This requires inverse mapping - simplified implementation
        double xi_left = 0.5, eta_left = 0.5;   // Placeholder
        double xi_right = 0.5, eta_right = 0.5; // Placeholder
        
        // Compute basis functions on both sides
        if (poly_order_ == 2) {
            compute_p2_basis_functions(xi_left, eta_left, N_left, grad_N_left);
            compute_p2_basis_functions(xi_right, eta_right, N_right, grad_N_right);
        } else if (poly_order_ == 3) {
            compute_p3_basis_functions(xi_left, eta_left, N_left, grad_N_left);
            compute_p3_basis_functions(xi_right, eta_right, N_right, grad_N_right);
        }
        
        double int_weight = weight * edge_length;
        
        // Add penalty terms: σ/h ∫ [u][v] ds
        for (int i = 0; i < dofs_per_element_; ++i) {
            for (int j = 0; j < dofs_per_element_; ++j) {
                int global_i_left = element_indices[0] * dofs_per_element_ + local_dof_maps[0][i];
                int global_j_left = element_indices[0] * dofs_per_element_ + local_dof_maps[0][j];
                int global_i_right = element_indices[1] * dofs_per_element_ + local_dof_maps[1][i];
                int global_j_right = element_indices[1] * dofs_per_element_ + local_dof_maps[1][j];
                
                double penalty_term = penalty * coeff * int_weight;
                
                // Penalty contributions
                global_matrix[global_i_left][global_j_left] += penalty_term * N_left[i] * N_left[j];
                global_matrix[global_i_left][global_j_right] -= penalty_term * N_left[i] * N_right[j];
                global_matrix[global_i_right][global_j_left] -= penalty_term * N_right[i] * N_left[j];
                global_matrix[global_i_right][global_j_right] += penalty_term * N_right[i] * N_right[j];
            }
        }
    }
}

void DGAssembly::add_boundary_penalty(
    const std::array<std::array<double, 2>, 2>& edge_vertices,
    int element_index,
    const std::vector<int>& local_dof_map,
    double element_size,
    double boundary_value,
    const std::function<double(double, double)>& coefficient_func,
    std::vector<std::vector<double>>& global_matrix,
    std::vector<double>& global_rhs) {
    
    // Compute penalty parameter
    double penalty = PenaltyParameter::compute(poly_order_, element_size, penalty_const_);
    
    // Map line quadrature to boundary edge
    auto edge_quad_points = face_integration::map_to_triangle_edge(line_quad_points_, edge_vertices);
    
    // Edge length
    double edge_length = std::sqrt(
        std::pow(edge_vertices[1][0] - edge_vertices[0][0], 2) +
        std::pow(edge_vertices[1][1] - edge_vertices[0][1], 2)
    );
    
    // Basis function storage
    std::vector<double> N(dofs_per_element_);
    std::vector<std::array<double, 2>> grad_N(dofs_per_element_);
    
    // Integration along boundary edge
    for (const auto& qp : edge_quad_points) {
        double x_phys = qp.coords[0];
        double y_phys = qp.coords[1];
        double weight = qp.weight;
        
        // Material coefficient
        double coeff = coefficient_func(x_phys, y_phys);
        
        // Map to reference coordinates (simplified)
        double xi = 0.5, eta = 0.5; // Placeholder
        
        // Compute basis functions
        if (poly_order_ == 2) {
            compute_p2_basis_functions(xi, eta, N, grad_N);
        } else if (poly_order_ == 3) {
            compute_p3_basis_functions(xi, eta, N, grad_N);
        }
        
        double int_weight = weight * edge_length;
        
        // Add boundary penalty terms
        for (int i = 0; i < dofs_per_element_; ++i) {
            int global_i = element_index * dofs_per_element_ + local_dof_map[i];
            
            for (int j = 0; j < dofs_per_element_; ++j) {
                int global_j = element_index * dofs_per_element_ + local_dof_map[j];
                
                // Penalty matrix: σ/h ∫ φᵢ φⱼ ds
                global_matrix[global_i][global_j] += penalty * coeff * N[i] * N[j] * int_weight;
            }
            
            // Penalty RHS: σ/h ∫ g φᵢ ds
            global_rhs[global_i] += penalty * coeff * boundary_value * N[i] * int_weight;
        }
    }
}

int DGAssembly::get_dofs_per_element() const {
    return dofs_per_element_;
}

int DGAssembly::get_polynomial_order() const {
    return poly_order_;
}

} // namespace dg
} // namespace simulator
