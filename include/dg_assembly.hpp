#pragma once

#include "dg_basis_functions.hpp"
#include <vector>
#include <array>
#include <functional>

namespace simulator {
namespace dg {

/**
 * @brief Discontinuous Galerkin assembly class for 2D triangular elements
 * 
 * This class handles the assembly of DG discretizations for elliptic PDEs
 * of the form: -∇·(ε∇u) = f in Ω, with appropriate boundary conditions.
 * 
 * The DG formulation includes:
 * - Volume integrals for element contributions
 * - Interface penalty terms for element coupling
 * - Boundary penalty terms for boundary conditions
 */
class DGAssembly {
public:
    /**
     * @brief Constructor
     * 
     * @param polynomial_order Order of polynomial basis (1, 2, or 3)
     * @param penalty_constant Penalty parameter constant (default: 50.0)
     */
    explicit DGAssembly(int polynomial_order, double penalty_constant = 50.0);
    
    /**
     * @brief Assemble element stiffness matrix and load vector
     * 
     * Computes the element contributions:
     * K_ij = ∫_T ε ∇φᵢ · ∇φⱼ dΩ
     * f_i = ∫_T f φᵢ dΩ
     * 
     * @param vertices Physical coordinates of triangle vertices [3][2]
     * @param coefficient_func Material coefficient function ε(x,y)
     * @param K_elem Output: element stiffness matrix [dofs_per_element][dofs_per_element]
     * @param f_elem Output: element load vector [dofs_per_element]
     * @param source_func Source function f(x,y) (optional)
     */
    void assemble_element_matrix(
        const std::array<std::array<double, 2>, 3>& vertices,
        const std::function<double(double, double)>& coefficient_func,
        std::vector<std::vector<double>>& K_elem,
        std::vector<double>& f_elem,
        const std::function<double(double, double)>& source_func = nullptr);
    
    /**
     * @brief Add interface penalty terms between adjacent elements
     * 
     * Adds the DG penalty terms for interior faces:
     * - Consistency terms: ∫_e {ε∇u}·n [v] ds
     * - Symmetry terms: ∫_e {ε∇v}·n [u] ds  
     * - Penalty terms: ∫_e (σ/h) [u][v] ds
     * 
     * where [·] denotes jump, {·} denotes average, σ is penalty parameter
     * 
     * @param edge_vertices Physical coordinates of edge endpoints [2][2]
     * @param element_indices Global indices of adjacent elements [2]
     * @param local_dof_maps Local DOF numbering for each element [2][dofs_per_element]
     * @param avg_element_size Average size of adjacent elements
     * @param coefficient_func Material coefficient function ε(x,y)
     * @param global_matrix Global system matrix to modify
     */
    void add_interface_penalty(
        const std::array<std::array<double, 2>, 2>& edge_vertices,
        const std::array<int, 2>& element_indices,
        const std::array<std::vector<int>, 2>& local_dof_maps,
        double avg_element_size,
        const std::function<double(double, double)>& coefficient_func,
        std::vector<std::vector<double>>& global_matrix);
    
    /**
     * @brief Add boundary penalty terms for Dirichlet boundary conditions
     * 
     * Adds the DG penalty terms for boundary faces:
     * - Consistency terms: ∫_∂Ω ε∇u·n v ds
     * - Symmetry terms: ∫_∂Ω ε∇v·n u ds
     * - Penalty terms: ∫_∂Ω (σ/h) u v ds
     * - Boundary data: ∫_∂Ω (σ/h) g v ds
     * 
     * @param edge_vertices Physical coordinates of boundary edge [2][2]
     * @param element_index Global index of boundary element
     * @param local_dof_map Local DOF numbering [dofs_per_element]
     * @param element_size Characteristic size of boundary element
     * @param boundary_value Dirichlet boundary value g
     * @param coefficient_func Material coefficient function ε(x,y)
     * @param global_matrix Global system matrix to modify
     * @param global_rhs Global right-hand side vector to modify
     */
    void add_boundary_penalty(
        const std::array<std::array<double, 2>, 2>& edge_vertices,
        int element_index,
        const std::vector<int>& local_dof_map,
        double element_size,
        double boundary_value,
        const std::function<double(double, double)>& coefficient_func,
        std::vector<std::vector<double>>& global_matrix,
        std::vector<double>& global_rhs);
    
    /**
     * @brief Get number of degrees of freedom per element
     * @return DOFs per element (3 for P1, 6 for P2, 10 for P3)
     */
    int get_dofs_per_element() const;
    
    /**
     * @brief Get polynomial order
     * @return Polynomial order (1, 2, or 3)
     */
    int get_polynomial_order() const;

private:
    int poly_order_;                    ///< Polynomial order
    int dofs_per_element_;             ///< DOFs per triangular element
    double penalty_const_;             ///< Penalty parameter constant
    
    /// Quadrature points for volume integration
    std::vector<struct QuadraturePoint> quad_points_;
    
    /// Quadrature points for line integration (faces)
    std::vector<face_integration::LineQuadraturePoint> line_quad_points_;
};

/**
 * @brief Utility functions for DG assembly
 */
namespace assembly_utils {

/**
 * @brief Compute element connectivity for DG mesh
 * 
 * @param elements Element connectivity [num_elements][3]
 * @param num_elements Number of elements
 * @return Face-to-element mapping for interface assembly
 */
struct FaceInfo {
    std::array<int, 2> elements;     ///< Adjacent element indices (-1 for boundary)
    std::array<int, 2> local_faces;  ///< Local face numbers in each element
    std::array<std::array<double, 2>, 2> vertices; ///< Face vertex coordinates
    bool is_boundary;                ///< True if boundary face
};

std::vector<FaceInfo> compute_face_connectivity(
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices);

/**
 * @brief Compute characteristic element sizes
 * 
 * @param elements Element connectivity
 * @param vertices Vertex coordinates
 * @return Vector of element sizes (diameter or area-based)
 */
std::vector<double> compute_element_sizes(
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices);

/**
 * @brief Create local-to-global DOF mapping for DG discretization
 * 
 * In DG methods, each element has its own DOFs (no sharing between elements)
 * 
 * @param num_elements Number of elements
 * @param dofs_per_element DOFs per element
 * @return Local-to-global DOF mapping [num_elements][dofs_per_element]
 */
std::vector<std::vector<int>> create_dg_dof_mapping(int num_elements, int dofs_per_element);

} // namespace assembly_utils

} // namespace dg
} // namespace simulator
