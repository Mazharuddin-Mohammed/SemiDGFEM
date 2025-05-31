#include "dg_basis_functions.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace simulator {
namespace dg {

// P3 Lagrange basis functions on reference triangle
// Reference triangle: (0,0), (1,0), (0,1)
// 10 DOFs: 3 vertices + 6 edge nodes + 1 interior node

void compute_p3_basis_functions(double xi, double eta, 
                               std::vector<double>& N,
                               std::vector<std::array<double, 2>>& grad_N) {
    if (N.size() != 10 || grad_N.size() != 10) {
        throw std::invalid_argument("P3 basis requires exactly 10 DOF arrays");
    }
    
    double zeta = 1.0 - xi - eta;
    
    // Validate reference coordinates
    if (xi < -1e-12 || eta < -1e-12 || zeta < -1e-12) {
        throw std::invalid_argument("Invalid reference coordinates for P3 basis");
    }
    
    // Vertex basis functions (corner nodes)
    // N0: vertex (0,0) -> zeta = 1
    N[0] = 0.5 * zeta * (3.0 * zeta - 1.0) * (3.0 * zeta - 2.0);
    grad_N[0][0] = -0.5 * (27.0 * zeta * zeta - 18.0 * zeta + 2.0);
    grad_N[0][1] = -0.5 * (27.0 * zeta * zeta - 18.0 * zeta + 2.0);
    
    // N1: vertex (1,0) -> xi = 1
    N[1] = 0.5 * xi * (3.0 * xi - 1.0) * (3.0 * xi - 2.0);
    grad_N[1][0] = 0.5 * (27.0 * xi * xi - 12.0 * xi + 1.0);
    grad_N[1][1] = 0.0;
    
    // N2: vertex (0,1) -> eta = 1
    N[2] = 0.5 * eta * (3.0 * eta - 1.0) * (3.0 * eta - 2.0);
    grad_N[2][0] = 0.0;
    grad_N[2][1] = 0.5 * (27.0 * eta * eta - 12.0 * eta + 1.0);
    
    // Edge basis functions
    // N3, N4: edge from vertex 0 to vertex 1 (zeta-xi edge)
    N[3] = 4.5 * zeta * xi * (3.0 * zeta - 1.0);
    grad_N[3][0] = 4.5 * (zeta * (3.0 * zeta - 1.0) - xi * (6.0 * zeta - 1.0));
    grad_N[3][1] = -4.5 * xi * (6.0 * zeta - 1.0);
    
    N[4] = 4.5 * zeta * xi * (3.0 * xi - 1.0);
    grad_N[4][0] = 4.5 * (zeta * (6.0 * xi - 1.0) + xi * (3.0 * xi - 1.0));
    grad_N[4][1] = -4.5 * xi * (3.0 * xi - 1.0);
    
    // N5, N6: edge from vertex 1 to vertex 2 (xi-eta edge)
    N[5] = 4.5 * xi * eta * (3.0 * xi - 1.0);
    grad_N[5][0] = 4.5 * eta * (6.0 * xi - 1.0);
    grad_N[5][1] = 4.5 * xi * (3.0 * xi - 1.0);
    
    N[6] = 4.5 * xi * eta * (3.0 * eta - 1.0);
    grad_N[6][0] = 4.5 * eta * (3.0 * eta - 1.0);
    grad_N[6][1] = 4.5 * xi * (6.0 * eta - 1.0);
    
    // N7, N8: edge from vertex 2 to vertex 0 (eta-zeta edge)
    N[7] = 4.5 * eta * zeta * (3.0 * eta - 1.0);
    grad_N[7][0] = -4.5 * eta * (3.0 * eta - 1.0);
    grad_N[7][1] = 4.5 * (zeta * (6.0 * eta - 1.0) + eta * (3.0 * eta - 1.0));
    
    N[8] = 4.5 * eta * zeta * (3.0 * zeta - 1.0);
    grad_N[8][0] = -4.5 * eta * (6.0 * zeta - 1.0);
    grad_N[8][1] = 4.5 * (zeta * (3.0 * zeta - 1.0) - eta * (6.0 * zeta - 1.0));
    
    // N9: interior bubble function
    N[9] = 27.0 * zeta * xi * eta;
    grad_N[9][0] = 27.0 * (zeta * eta - xi * eta);
    grad_N[9][1] = 27.0 * (zeta * xi - eta * xi);
}

void compute_p2_basis_functions(double xi, double eta,
                               std::vector<double>& N,
                               std::vector<std::array<double, 2>>& grad_N) {
    if (N.size() != 6 || grad_N.size() != 6) {
        throw std::invalid_argument("P2 basis requires exactly 6 DOF arrays");
    }
    
    double zeta = 1.0 - xi - eta;
    
    // Vertex basis functions
    N[0] = zeta * (2.0 * zeta - 1.0);
    grad_N[0][0] = -(4.0 * zeta - 1.0);
    grad_N[0][1] = -(4.0 * zeta - 1.0);
    
    N[1] = xi * (2.0 * xi - 1.0);
    grad_N[1][0] = 4.0 * xi - 1.0;
    grad_N[1][1] = 0.0;
    
    N[2] = eta * (2.0 * eta - 1.0);
    grad_N[2][0] = 0.0;
    grad_N[2][1] = 4.0 * eta - 1.0;
    
    // Edge basis functions
    N[3] = 4.0 * zeta * xi;
    grad_N[3][0] = 4.0 * (zeta - xi);
    grad_N[3][1] = -4.0 * xi;
    
    N[4] = 4.0 * xi * eta;
    grad_N[4][0] = 4.0 * eta;
    grad_N[4][1] = 4.0 * xi;
    
    N[5] = 4.0 * eta * zeta;
    grad_N[5][0] = -4.0 * eta;
    grad_N[5][1] = 4.0 * (zeta - eta);
}

// Gauss quadrature rules for triangular elements
std::vector<QuadraturePoint> get_triangle_quadrature(int order) {
    std::vector<QuadraturePoint> points;
    
    if (order <= 1) {
        // 1-point rule (exact for P1)
        points.push_back({{1.0/3.0, 1.0/3.0}, 0.5});
    } else if (order <= 2) {
        // 3-point rule (exact for P2)
        points.push_back({{1.0/6.0, 1.0/6.0}, 1.0/6.0});
        points.push_back({{2.0/3.0, 1.0/6.0}, 1.0/6.0});
        points.push_back({{1.0/6.0, 2.0/3.0}, 1.0/6.0});
    } else if (order <= 3) {
        // 4-point rule (exact for P3)
        points.push_back({{1.0/3.0, 1.0/3.0}, -27.0/96.0});
        points.push_back({{0.6, 0.2}, 25.0/96.0});
        points.push_back({{0.2, 0.6}, 25.0/96.0});
        points.push_back({{0.2, 0.2}, 25.0/96.0});
    } else if (order <= 5) {
        // 7-point rule (exact for P5)
        points.push_back({{1.0/3.0, 1.0/3.0}, 0.225});
        points.push_back({{0.797426985353087, 0.101286507323456}, 0.125939180544827});
        points.push_back({{0.101286507323456, 0.797426985353087}, 0.125939180544827});
        points.push_back({{0.101286507323456, 0.101286507323456}, 0.125939180544827});
        points.push_back({{0.059715871789770, 0.470142064105115}, 0.132394152788506});
        points.push_back({{0.470142064105115, 0.059715871789770}, 0.132394152788506});
        points.push_back({{0.470142064105115, 0.470142064105115}, 0.132394152788506});
    } else {
        throw std::invalid_argument("Quadrature order > 5 not implemented");
    }
    
    return points;
}

void transform_gradients_to_physical(const std::vector<std::array<double, 2>>& grad_ref,
                                   std::vector<std::array<double, 2>>& grad_phys,
                                   const std::array<std::array<double, 2>, 2>& inv_jacobian) {
    if (grad_ref.size() != grad_phys.size()) {
        throw std::invalid_argument("Reference and physical gradient arrays must have same size");
    }
    
    for (size_t i = 0; i < grad_ref.size(); ++i) {
        grad_phys[i][0] = inv_jacobian[0][0] * grad_ref[i][0] + inv_jacobian[0][1] * grad_ref[i][1];
        grad_phys[i][1] = inv_jacobian[1][0] * grad_ref[i][0] + inv_jacobian[1][1] * grad_ref[i][1];
    }
}

std::array<std::array<double, 2>, 2> compute_inverse_jacobian(
    const std::array<double, 2>& x1, const std::array<double, 2>& x2, const std::array<double, 2>& x3) {
    
    // Jacobian matrix: J = [x2-x1, x3-x1; y2-y1, y3-y1]
    double J11 = x2[0] - x1[0], J12 = x3[0] - x1[0];
    double J21 = x2[1] - x1[1], J22 = x3[1] - x1[1];
    
    double det_J = J11 * J22 - J12 * J21;
    if (std::abs(det_J) < 1e-12) {
        throw std::runtime_error("Degenerate element: Jacobian determinant near zero");
    }
    
    // Inverse Jacobian
    std::array<std::array<double, 2>, 2> inv_J;
    inv_J[0][0] = J22 / det_J;
    inv_J[0][1] = -J12 / det_J;
    inv_J[1][0] = -J21 / det_J;
    inv_J[1][1] = J11 / det_J;
    
    return inv_J;
}

double compute_element_area(const std::array<double, 2>& x1, 
                          const std::array<double, 2>& x2, 
                          const std::array<double, 2>& x3) {
    return 0.5 * std::abs((x2[0] - x1[0]) * (x3[1] - x1[1]) - (x3[0] - x1[0]) * (x2[1] - x1[1]));
}

bool validate_reference_coordinates(double xi, double eta) {
    const double tol = 1e-12;
    double zeta = 1.0 - xi - eta;
    return (xi >= -tol && eta >= -tol && zeta >= -tol);
}

} // namespace dg
} // namespace simulator
