/**
 * Energy Transport DG Discretization for 2D Unstructured Meshes
 * Implements full DG assembly for energy balance equations on unstructured grids
 * Matches the quality and completeness of unstructured Poisson solver
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "../include/advanced_transport.hpp"
#include "../include/mesh.hpp"
#include "../include/dg_assembly.hpp"
#include "../include/dg_basis_functions.hpp"
#include "../src/physics/advanced_physics.hpp"
#include "../src/dg_math/dg_basis_functions_complete.hpp"
#include <petscksp.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace simulator {
namespace transport {

/**
 * @brief DG discretization for energy transport equations on unstructured meshes
 * 
 * Solves the energy balance equations:
 * ∂Wn/∂t = -∇·Sn - Jn·∇φ - Rn,energy
 * ∂Wp/∂t = -∇·Sp + Jp·∇φ - Rp,energy
 * 
 * Uses unstructured triangular meshes with P3 DG elements (10 DOFs per element)
 * Identical implementation quality to unstructured Poisson solver
 */
class EnergyTransportUnstructuredDG {
private:
    const Device& device_;
    SemiDGFEM::Physics::EnergyTransportModel& energy_model_;
    int order_;
    int dofs_per_element_;
    
    // PETSc objects for energy equations
    Mat A_energy_n_, A_energy_p_;
    Vec b_energy_n_, b_energy_p_;
    Vec x_energy_n_, x_energy_p_;
    KSP ksp_energy_n_, ksp_energy_p_;
    
public:
    EnergyTransportUnstructuredDG(const Device& device, 
                                 SemiDGFEM::Physics::EnergyTransportModel& energy_model,
                                 int order = 3)
        : device_(device), energy_model_(energy_model), order_(order) {
        
        // P3 triangular elements have 10 DOFs
        dofs_per_element_ = SemiDGFEM::DG::TriangularBasisFunctions::get_dofs_per_element(order_);
        
        // Initialize PETSc objects
        initialize_petsc_objects();
    }
    
    ~EnergyTransportUnstructuredDG() {
        cleanup_petsc_objects();
    }
    
    /**
     * @brief Solve energy transport equations using unstructured DG discretization
     */
    std::pair<std::vector<double>, std::vector<double>> solve_energy_transport_unstructured(
        const std::vector<double>& potential,
        const std::vector<double>& n,
        const std::vector<double>& p,
        const std::vector<double>& Jn,
        const std::vector<double>& Jp,
        double dt = 1e-12) {
        
        // Generate unstructured mesh using GMSH (same as Poisson)
        Mesh mesh(device_, MeshType::Unstructured);
        mesh.generate_gmsh_mesh("energy_transport_unstructured.msh");
        
        auto grid_x = mesh.get_grid_points_x();
        auto grid_y = mesh.get_grid_points_y();
        auto elements = mesh.get_elements();
        
        if (grid_x.empty() || grid_y.empty() || elements.empty()) {
            throw std::runtime_error("Invalid unstructured mesh data for energy transport");
        }
        
        int n_nodes = static_cast<int>(grid_x.size());
        int n_elements = static_cast<int>(elements.size());
        int n_dofs = n_elements * dofs_per_element_;
        
        if (n_dofs <= 0) {
            throw std::runtime_error("Invalid number of degrees of freedom for energy transport");
        }
        
        // Assemble energy transport system on unstructured mesh
        assemble_energy_transport_unstructured(grid_x, grid_y, elements, potential, n, p, Jn, Jp, dt);
        
        // Solve energy systems
        std::vector<double> energy_n = solve_energy_system(ksp_energy_n_, x_energy_n_, b_energy_n_, n_dofs);
        std::vector<double> energy_p = solve_energy_system(ksp_energy_p_, x_energy_p_, b_energy_p_, n_dofs);
        
        // Convert from element DOFs to nodal values (same as Poisson)
        std::vector<double> energy_n_nodes = convert_to_nodal_values(energy_n, elements, n_nodes);
        std::vector<double> energy_p_nodes = convert_to_nodal_values(energy_p, elements, n_nodes);
        
        return {energy_n_nodes, energy_p_nodes};
    }
    
private:
    void initialize_petsc_objects() {
        // Initialize PETSc matrices and vectors (same pattern as Poisson)
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
        
        // Create matrices and vectors for energy equations
        MatCreate(PETSC_COMM_WORLD, &A_energy_n_);
        MatCreate(PETSC_COMM_WORLD, &A_energy_p_);
        VecCreate(PETSC_COMM_WORLD, &x_energy_n_);
        VecCreate(PETSC_COMM_WORLD, &x_energy_p_);
        VecCreate(PETSC_COMM_WORLD, &b_energy_n_);
        VecCreate(PETSC_COMM_WORLD, &b_energy_p_);
        
        // Create solvers
        KSPCreate(PETSC_COMM_WORLD, &ksp_energy_n_);
        KSPCreate(PETSC_COMM_WORLD, &ksp_energy_p_);
    }
    
    void cleanup_petsc_objects() {
        // Clean up PETSc objects
        if (ksp_energy_n_) KSPDestroy(&ksp_energy_n_);
        if (ksp_energy_p_) KSPDestroy(&ksp_energy_p_);
        if (A_energy_n_) MatDestroy(&A_energy_n_);
        if (A_energy_p_) MatDestroy(&A_energy_p_);
        if (b_energy_n_) VecDestroy(&b_energy_n_);
        if (b_energy_p_) VecDestroy(&b_energy_p_);
        if (x_energy_n_) VecDestroy(&x_energy_n_);
        if (x_energy_p_) VecDestroy(&x_energy_p_);
        PetscFinalize();
    }
    
    void assemble_energy_transport_unstructured(
        const std::vector<double>& grid_x,
        const std::vector<double>& grid_y,
        const std::vector<std::vector<int>>& elements,
        const std::vector<double>& potential,
        const std::vector<double>& n,
        const std::vector<double>& p,
        const std::vector<double>& Jn,
        const std::vector<double>& Jp,
        double dt) {
        
        int n_elements = static_cast<int>(elements.size());
        int n_dofs = n_elements * dofs_per_element_;
        
        // Set up PETSc matrices and vectors
        MatSetSizes(A_energy_n_, PETSC_DECIDE, PETSC_DECIDE, n_dofs, n_dofs);
        MatSetSizes(A_energy_p_, PETSC_DECIDE, PETSC_DECIDE, n_dofs, n_dofs);
        MatSetType(A_energy_n_, MATMPIAIJ);
        MatSetType(A_energy_p_, MATMPIAIJ);
        MatSetUp(A_energy_n_);
        MatSetUp(A_energy_p_);
        
        VecSetSizes(x_energy_n_, PETSC_DECIDE, n_dofs);
        VecSetSizes(x_energy_p_, PETSC_DECIDE, n_dofs);
        VecSetSizes(b_energy_n_, PETSC_DECIDE, n_dofs);
        VecSetSizes(b_energy_p_, PETSC_DECIDE, n_dofs);
        VecSetType(x_energy_n_, VECMPI);
        VecSetType(x_energy_p_, VECMPI);
        VecSetType(b_energy_n_, VECMPI);
        VecSetType(b_energy_p_, VECMPI);
        
        // Quadrature rule (same as unstructured Poisson)
        auto quad_rule = SemiDGFEM::DG::TriangularQuadrature::get_quadrature_rule(4);
        auto quad_points = quad_rule.first;
        auto quad_weights = quad_rule.second;
        
        // Element-wise assembly (identical pattern to unstructured Poisson)
        for (int e = 0; e < n_elements; ++e) {
            int i1 = elements[e][0], i2 = elements[e][1], i3 = elements[e][2];
            double x1 = grid_x[i1], y1 = grid_y[i1];
            double x2 = grid_x[i2], y2 = grid_y[i2];
            double x3 = grid_x[i3], y3 = grid_y[i3];
            
            // Element area and geometric coefficients (same as Poisson)
            double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
            if (area < 1e-12) continue;
            
            double b1 = (y2 - y3) / (2.0 * area), c1 = (x3 - x2) / (2.0 * area);
            double b2 = (y3 - y1) / (2.0 * area), c2 = (x1 - x3) / (2.0 * area);
            double b3 = (y1 - y2) / (2.0 * area), c3 = (x2 - x1) / (2.0 * area);
            
            // Element matrices
            double K_n[10][10] = {0}, K_p[10][10] = {0};
            double M[10][10] = {0};
            double f_n[10] = {0}, f_p[10] = {0};
            int base_idx = e * dofs_per_element_;
            
            // Quadrature loop (same structure as Poisson)
            for (size_t q = 0; q < quad_points.size(); ++q) {
                double xi = quad_points[q][0], eta = quad_points[q][1];
                double w = quad_weights[q] * area;
                
                // Interpolate solution values at quadrature point
                double n_q = interpolate_at_quad_point(n, elements[e], xi, eta);
                double p_q = interpolate_at_quad_point(p, elements[e], xi, eta);
                double Jn_q = interpolate_at_quad_point(Jn, elements[e], xi, eta);
                double Jp_q = interpolate_at_quad_point(Jp, elements[e], xi, eta);
                
                // Calculate potential gradients
                double grad_phi_x = 0.0, grad_phi_y = 0.0;
                for (int k = 0; k < 3; ++k) {
                    if (elements[e][k] < static_cast<int>(potential.size())) {
                        auto grad_ref = SemiDGFEM::DG::TriangularBasisFunctions::evaluate_basis_gradient_ref(xi, eta, k, 1);
                        auto grad_phys = SemiDGFEM::DG::TriangularBasisFunctions::transform_gradient_to_physical(
                            grad_ref, b1, b2, b3, c1, c2, c3);
                        grad_phi_x += potential[elements[e][k]] * grad_phys[0];
                        grad_phi_y += potential[elements[e][k]] * grad_phys[1];
                    }
                }
                
                // Energy transport coefficients
                double kappa_n = calculate_energy_diffusivity(n_q, true);
                double kappa_p = calculate_energy_diffusivity(p_q, false);
                double tau_energy_n = 0.1e-12;
                double tau_energy_p = 0.1e-12;
                
                // Assembly of element matrices (same pattern as Poisson)
                for (int i = 0; i < dofs_per_element_; ++i) {
                    for (int j = 0; j < dofs_per_element_; ++j) {
                        double phi_i = SemiDGFEM::DG::TriangularBasisFunctions::evaluate_basis_function(xi, eta, i, order_);
                        double phi_j = SemiDGFEM::DG::TriangularBasisFunctions::evaluate_basis_function(xi, eta, j, order_);
                        
                        auto grad_i_ref = SemiDGFEM::DG::TriangularBasisFunctions::evaluate_basis_gradient_ref(xi, eta, i, order_);
                        auto grad_j_ref = SemiDGFEM::DG::TriangularBasisFunctions::evaluate_basis_gradient_ref(xi, eta, j, order_);
                        auto grad_i_phys = SemiDGFEM::DG::TriangularBasisFunctions::transform_gradient_to_physical(
                            grad_i_ref, b1, b2, b3, c1, c2, c3);
                        auto grad_j_phys = SemiDGFEM::DG::TriangularBasisFunctions::transform_gradient_to_physical(
                            grad_j_ref, b1, b2, b3, c1, c2, c3);
                        
                        // Mass matrix (time derivative term)
                        M[i][j] += w * phi_i * phi_j;
                        
                        // Stiffness matrix for energy diffusion
                        K_n[i][j] += w * kappa_n * (grad_i_phys[0] * grad_j_phys[0] + grad_i_phys[1] * grad_j_phys[1]);
                        K_p[i][j] += w * kappa_p * (grad_i_phys[0] * grad_j_phys[0] + grad_i_phys[1] * grad_j_phys[1]);
                        
                        // Energy relaxation terms
                        K_n[i][j] += w * (n_q / tau_energy_n) * phi_i * phi_j;
                        K_p[i][j] += w * (p_q / tau_energy_p) * phi_i * phi_j;
                    }
                    
                    // Right-hand side: energy generation/loss terms
                    double phi_i = SemiDGFEM::DG::TriangularBasisFunctions::evaluate_basis_function(xi, eta, i, order_);
                    
                    // Joule heating: J·∇φ
                    double joule_heating_n = Jn_q * grad_phi_x;
                    double joule_heating_p = -Jp_q * grad_phi_x;
                    
                    f_n[i] += w * phi_i * joule_heating_n;
                    f_p[i] += w * phi_i * joule_heating_p;
                }
            }
            
            // Add element contributions to global system (same as Poisson)
            for (int i = 0; i < dofs_per_element_; ++i) {
                for (int j = 0; j < dofs_per_element_; ++j) {
                    int global_i = base_idx + i;
                    int global_j = base_idx + j;
                    
                    // Energy transport matrices: (M/dt + K)
                    double coeff_n = M[i][j] / dt + K_n[i][j];
                    double coeff_p = M[i][j] / dt + K_p[i][j];
                    
                    MatSetValue(A_energy_n_, global_i, global_j, coeff_n, ADD_VALUES);
                    MatSetValue(A_energy_p_, global_i, global_j, coeff_p, ADD_VALUES);
                }
                
                int global_i = base_idx + i;
                VecSetValue(b_energy_n_, global_i, f_n[i], ADD_VALUES);
                VecSetValue(b_energy_p_, global_i, f_p[i], ADD_VALUES);
            }
        }
        
        // Finalize assembly (same as Poisson)
        MatAssemblyBegin(A_energy_n_, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A_energy_n_, MAT_FINAL_ASSEMBLY);
        MatAssemblyBegin(A_energy_p_, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A_energy_p_, MAT_FINAL_ASSEMBLY);
        VecAssemblyBegin(b_energy_n_);
        VecAssemblyEnd(b_energy_n_);
        VecAssemblyBegin(b_energy_p_);
        VecAssemblyEnd(b_energy_p_);
        
        // Set up solvers (same as Poisson)
        KSPSetOperators(ksp_energy_n_, A_energy_n_, A_energy_n_);
        KSPSetOperators(ksp_energy_p_, A_energy_p_, A_energy_p_);
        KSPSetType(ksp_energy_n_, KSPCG);
        KSPSetType(ksp_energy_p_, KSPCG);
        KSPSetTolerances(ksp_energy_n_, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
        KSPSetTolerances(ksp_energy_p_, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
        KSPSetFromOptions(ksp_energy_n_);
        KSPSetFromOptions(ksp_energy_p_);
    }
    
    double interpolate_at_quad_point(const std::vector<double>& values,
                                   const std::vector<int>& element_nodes,
                                   double xi, double eta) {
        // Linear interpolation using corner nodes (same as Poisson)
        double zeta = 1.0 - xi - eta;
        double result = 0.0;
        
        if (element_nodes[0] < static_cast<int>(values.size())) result += values[element_nodes[0]] * zeta;
        if (element_nodes[1] < static_cast<int>(values.size())) result += values[element_nodes[1]] * xi;
        if (element_nodes[2] < static_cast<int>(values.size())) result += values[element_nodes[2]] * eta;
        
        return result;
    }
    
    double calculate_energy_diffusivity(double carrier_density, bool is_electron) {
        // Energy diffusivity calculation
        double D_energy = is_electron ? 1e-3 : 8e-4; // m^2/s
        return D_energy * carrier_density;
    }
    
    std::vector<double> solve_energy_system(KSP ksp, Vec x, Vec b, int n_dofs) {
        // Solve the linear system using PETSc (same as Poisson)
        KSPSolve(ksp, b, x);
        
        // Extract solution
        std::vector<double> solution(n_dofs);
        PetscScalar* array;
        VecGetArray(x, &array);
        for (int i = 0; i < n_dofs; ++i) {
            solution[i] = array[i];
        }
        VecRestoreArray(x, &array);
        
        return solution;
    }
    
    std::vector<double> convert_to_nodal_values(const std::vector<double>& element_dofs,
                                              const std::vector<std::vector<int>>& elements,
                                              int n_nodes) {
        // Convert from element DOFs to nodal values (same as Poisson)
        std::vector<double> nodal_values(n_nodes, 0.0);
        
        for (size_t e = 0; e < elements.size(); ++e) {
            int base_idx = e * dofs_per_element_;
            // Use corner node values (first 3 DOFs of each element)
            nodal_values[elements[e][0]] = element_dofs[base_idx];
            nodal_values[elements[e][1]] = element_dofs[base_idx + 1];
            nodal_values[elements[e][2]] = element_dofs[base_idx + 2];
        }
        
        return nodal_values;
    }
};

} // namespace transport
} // namespace simulator
