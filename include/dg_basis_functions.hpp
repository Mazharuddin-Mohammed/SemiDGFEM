#pragma once

#include <vector>
#include <array>

namespace simulator {
namespace dg {

/**
 * @brief Quadrature point structure for numerical integration
 */
struct QuadraturePoint {
    std::array<double, 2> coords;  // Reference coordinates (xi, eta)
    double weight;                 // Quadrature weight
};

/**
 * @brief Compute P3 Lagrange basis functions and their gradients on reference triangle
 * 
 * P3 elements have 10 degrees of freedom:
 * - 3 vertex nodes (0, 1, 2)
 * - 6 edge nodes (3, 4, 5, 6, 7, 8) - 2 per edge
 * - 1 interior node (9)
 * 
 * @param xi Reference coordinate xi
 * @param eta Reference coordinate eta
 * @param N Output: basis function values (size 10)
 * @param grad_N Output: basis function gradients in reference coordinates (size 10)
 */
void compute_p3_basis_functions(double xi, double eta, 
                               std::vector<double>& N,
                               std::vector<std::array<double, 2>>& grad_N);

/**
 * @brief Compute P2 Lagrange basis functions and their gradients on reference triangle
 * 
 * P2 elements have 6 degrees of freedom:
 * - 3 vertex nodes (0, 1, 2)
 * - 3 edge nodes (3, 4, 5) - 1 per edge
 * 
 * @param xi Reference coordinate xi
 * @param eta Reference coordinate eta
 * @param N Output: basis function values (size 6)
 * @param grad_N Output: basis function gradients in reference coordinates (size 6)
 */
void compute_p2_basis_functions(double xi, double eta,
                               std::vector<double>& N,
                               std::vector<std::array<double, 2>>& grad_N);

/**
 * @brief Compute P1 Linear basis functions and their gradients on reference triangle
 *
 * P1 elements have 3 degrees of freedom:
 * - 3 vertex nodes (0, 1, 2)
 *
 * @param xi Reference coordinate xi
 * @param eta Reference coordinate eta
 * @param N Output: basis function values (size 3)
 * @param grad_N Output: basis function gradients in reference coordinates (size 3)
 */
void compute_p1_basis_functions(double xi, double eta,
                               std::vector<double>& N,
                               std::vector<std::array<double, 2>>& grad_N);

/**
 * @brief Get Gauss quadrature points and weights for triangular elements
 * 
 * @param order Polynomial order to integrate exactly
 * @return Vector of quadrature points with coordinates and weights
 */
std::vector<QuadraturePoint> get_triangle_quadrature(int order);

/**
 * @brief Transform gradients from reference to physical coordinates
 * 
 * @param grad_ref Gradients in reference coordinates
 * @param grad_phys Output: gradients in physical coordinates
 * @param inv_jacobian Inverse Jacobian matrix [2x2]
 */
void transform_gradients_to_physical(const std::vector<std::array<double, 2>>& grad_ref,
                                   std::vector<std::array<double, 2>>& grad_phys,
                                   const std::array<std::array<double, 2>, 2>& inv_jacobian);

/**
 * @brief Compute inverse Jacobian matrix for coordinate transformation
 * 
 * @param x1 Physical coordinates of vertex 1
 * @param x2 Physical coordinates of vertex 2
 * @param x3 Physical coordinates of vertex 3
 * @return Inverse Jacobian matrix [2x2]
 */
std::array<std::array<double, 2>, 2> compute_inverse_jacobian(
    const std::array<double, 2>& x1, const std::array<double, 2>& x2, const std::array<double, 2>& x3);

/**
 * @brief Compute area of triangular element
 * 
 * @param x1 Physical coordinates of vertex 1
 * @param x2 Physical coordinates of vertex 2
 * @param x3 Physical coordinates of vertex 3
 * @return Element area
 */
double compute_element_area(const std::array<double, 2>& x1, 
                          const std::array<double, 2>& x2, 
                          const std::array<double, 2>& x3);

/**
 * @brief Validate reference coordinates are within reference triangle
 * 
 * @param xi Reference coordinate xi
 * @param eta Reference coordinate eta
 * @return True if coordinates are valid
 */
bool validate_reference_coordinates(double xi, double eta);

/**
 * @brief DG penalty parameter calculation
 * 
 * For interior penalty DG methods, the penalty parameter σ should be chosen
 * large enough to ensure stability. Typical choice: σ = C * p² / h
 * where C is a constant, p is polynomial degree, h is element size.
 */
class PenaltyParameter {
public:
    /**
     * @brief Compute penalty parameter for DG method
     * 
     * @param polynomial_degree Polynomial degree of basis functions
     * @param element_size Characteristic element size
     * @param penalty_constant Penalty constant (typically 10-100)
     * @return Penalty parameter value
     */
    static double compute(int polynomial_degree, double element_size, double penalty_constant = 50.0) {
        return penalty_constant * polynomial_degree * polynomial_degree / element_size;
    }
};

/**
 * @brief Face integration utilities for DG methods
 */
namespace face_integration {

/**
 * @brief Gauss quadrature points for line segments (1D)
 */
struct LineQuadraturePoint {
    double coord;    // Reference coordinate [-1, 1]
    double weight;   // Quadrature weight
};

/**
 * @brief Get 1D Gauss quadrature for line integration
 * 
 * @param order Polynomial order to integrate exactly
 * @return Vector of 1D quadrature points
 */
std::vector<LineQuadraturePoint> get_line_quadrature(int order);

/**
 * @brief Map 1D quadrature points to triangle edge
 * 
 * @param line_points 1D quadrature points on [-1, 1]
 * @param edge_vertices Physical coordinates of edge endpoints
 * @return 2D quadrature points on triangle edge
 */
std::vector<QuadraturePoint> map_to_triangle_edge(
    const std::vector<LineQuadraturePoint>& line_points,
    const std::array<std::array<double, 2>, 2>& edge_vertices);

std::vector<LineQuadraturePoint> get_line_quadrature(int order);

} // namespace face_integration

} // namespace dg
} // namespace simulator
