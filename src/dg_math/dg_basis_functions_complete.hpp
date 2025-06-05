/**
 * Complete DG Basis Functions Implementation
 * Provides full P1, P2, P3 triangular basis functions and gradients
 * Used by all advanced transport models for consistent DG discretization
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#ifndef DG_BASIS_FUNCTIONS_COMPLETE_HPP
#define DG_BASIS_FUNCTIONS_COMPLETE_HPP

#include <vector>
#include <cmath>
#include <stdexcept>

namespace SemiDGFEM {
namespace DG {

/**
 * @brief Complete implementation of DG basis functions for triangular elements
 * 
 * Supports P1 (3 DOFs), P2 (6 DOFs), P3 (10 DOFs) triangular elements
 * with proper basis function evaluation and gradient computation
 */
class TriangularBasisFunctions {
public:
    /**
     * @brief Evaluate basis function at reference coordinates
     * @param xi First barycentric coordinate
     * @param eta Second barycentric coordinate  
     * @param j Basis function index
     * @param order Polynomial order (1, 2, or 3)
     * @return Value of basis function j at (xi, eta)
     */
    static double evaluate_basis_function(double xi, double eta, int j, int order) {
        double zeta = 1.0 - xi - eta;
        
        switch (order) {
            case 1: return evaluate_p1_basis(xi, eta, zeta, j);
            case 2: return evaluate_p2_basis(xi, eta, zeta, j);
            case 3: return evaluate_p3_basis(xi, eta, zeta, j);
            default: 
                throw std::invalid_argument("Unsupported polynomial order: " + std::to_string(order));
        }
    }
    
    /**
     * @brief Evaluate basis function gradient in reference coordinates
     * @param xi First barycentric coordinate
     * @param eta Second barycentric coordinate
     * @param j Basis function index
     * @param order Polynomial order
     * @return Gradient vector [d/dxi, d/deta]
     */
    static std::vector<double> evaluate_basis_gradient_ref(double xi, double eta, int j, int order) {
        double zeta = 1.0 - xi - eta;
        
        switch (order) {
            case 1: return evaluate_p1_gradient_ref(xi, eta, zeta, j);
            case 2: return evaluate_p2_gradient_ref(xi, eta, zeta, j);
            case 3: return evaluate_p3_gradient_ref(xi, eta, zeta, j);
            default:
                throw std::invalid_argument("Unsupported polynomial order: " + std::to_string(order));
        }
    }
    
    /**
     * @brief Transform gradient from reference to physical coordinates
     * @param grad_ref Gradient in reference coordinates [d/dxi, d/deta]
     * @param b1, b2, b3 Geometric transformation coefficients for x-direction
     * @param c1, c2, c3 Geometric transformation coefficients for y-direction
     * @return Gradient in physical coordinates [d/dx, d/dy]
     */
    static std::vector<double> transform_gradient_to_physical(
        const std::vector<double>& grad_ref,
        double b1, double b2, double b3,
        double c1, double c2, double c3) {
        
        double dxi_dx = b2, dxi_dy = c2;
        double deta_dx = b3, deta_dy = c3;
        
        // Chain rule: ∇φ = (∂φ/∂xi)(∂xi/∂x) + (∂φ/∂eta)(∂eta/∂x)
        double dphi_dx = grad_ref[0] * dxi_dx + grad_ref[1] * deta_dx;
        double dphi_dy = grad_ref[0] * dxi_dy + grad_ref[1] * deta_dy;
        
        return {dphi_dx, dphi_dy};
    }
    
    /**
     * @brief Get number of DOFs for given polynomial order
     */
    static int get_dofs_per_element(int order) {
        return (order + 1) * (order + 2) / 2;
    }

private:
    // P1 basis functions (3 DOFs)
    static double evaluate_p1_basis(double xi, double eta, double zeta, int j) {
        switch (j) {
            case 0: return zeta;  // N1 = 1 - xi - eta
            case 1: return xi;    // N2 = xi
            case 2: return eta;   // N3 = eta
            default: return 0.0;
        }
    }
    
    static std::vector<double> evaluate_p1_gradient_ref(double xi, double eta, double zeta, int j) {
        switch (j) {
            case 0: return {-1.0, -1.0};  // ∇N1 = [-1, -1]
            case 1: return {1.0, 0.0};    // ∇N2 = [1, 0]
            case 2: return {0.0, 1.0};    // ∇N3 = [0, 1]
            default: return {0.0, 0.0};
        }
    }
    
    // P2 basis functions (6 DOFs)
    static double evaluate_p2_basis(double xi, double eta, double zeta, int j) {
        switch (j) {
            case 0: return zeta * (2.0 * zeta - 1.0);           // Corner node 1
            case 1: return xi * (2.0 * xi - 1.0);               // Corner node 2
            case 2: return eta * (2.0 * eta - 1.0);             // Corner node 3
            case 3: return 4.0 * zeta * xi;                     // Edge node 1-2
            case 4: return 4.0 * xi * eta;                      // Edge node 2-3
            case 5: return 4.0 * eta * zeta;                    // Edge node 3-1
            default: return 0.0;
        }
    }
    
    static std::vector<double> evaluate_p2_gradient_ref(double xi, double eta, double zeta, int j) {
        switch (j) {
            case 0: return {-(4.0 * zeta - 1.0), -(4.0 * zeta - 1.0)};
            case 1: return {4.0 * xi - 1.0, 0.0};
            case 2: return {0.0, 4.0 * eta - 1.0};
            case 3: return {4.0 * (zeta - xi), -4.0 * xi};
            case 4: return {4.0 * eta, 4.0 * xi};
            case 5: return {-4.0 * eta, 4.0 * (zeta - eta)};
            default: return {0.0, 0.0};
        }
    }
    
    // P3 basis functions (10 DOFs)
    static double evaluate_p3_basis(double xi, double eta, double zeta, int j) {
        switch (j) {
            // Corner nodes
            case 0: return zeta * (3.0 * zeta - 1.0) * (3.0 * zeta - 2.0) / 2.0;
            case 1: return xi * (3.0 * xi - 1.0) * (3.0 * xi - 2.0) / 2.0;
            case 2: return eta * (3.0 * eta - 1.0) * (3.0 * eta - 2.0) / 2.0;
            
            // Edge nodes (2 per edge)
            case 3: return 9.0 * zeta * xi * (3.0 * zeta - 1.0) / 2.0;
            case 4: return 9.0 * zeta * xi * (3.0 * xi - 1.0) / 2.0;
            case 5: return 9.0 * xi * eta * (3.0 * xi - 1.0) / 2.0;
            case 6: return 9.0 * xi * eta * (3.0 * eta - 1.0) / 2.0;
            case 7: return 9.0 * eta * zeta * (3.0 * eta - 1.0) / 2.0;
            case 8: return 9.0 * eta * zeta * (3.0 * zeta - 1.0) / 2.0;
            
            // Interior node
            case 9: return 27.0 * zeta * xi * eta;
            
            default: return 0.0;
        }
    }
    
    static std::vector<double> evaluate_p3_gradient_ref(double xi, double eta, double zeta, int j) {
        switch (j) {
            // Corner nodes gradients
            case 0: {
                double dN0_dzeta = (27.0 * zeta * zeta - 18.0 * zeta + 2.0) / 2.0;
                return {-dN0_dzeta, -dN0_dzeta};
            }
            case 1: {
                double dN1_dxi = (27.0 * xi * xi - 18.0 * xi + 2.0) / 2.0;
                return {dN1_dxi, 0.0};
            }
            case 2: {
                double dN2_deta = (27.0 * eta * eta - 18.0 * eta + 2.0) / 2.0;
                return {0.0, dN2_deta};
            }
            
            // Edge nodes gradients
            case 3: {
                double dN3_dxi = 9.0 * (zeta * (3.0 * zeta - 1.0) - xi * (6.0 * zeta - 1.0)) / 2.0;
                double dN3_deta = 9.0 * xi * (6.0 * zeta - 1.0 - 3.0 * zeta) / 2.0;
                return {dN3_dxi, dN3_deta};
            }
            case 4: {
                double dN4_dxi = 9.0 * (zeta * (6.0 * xi - 1.0) + xi * (3.0 * xi - 1.0)) / 2.0;
                double dN4_deta = -9.0 * xi * (3.0 * xi - 1.0) / 2.0;
                return {dN4_dxi, dN4_deta};
            }
            case 5: {
                double dN5_dxi = 9.0 * eta * (6.0 * xi - 1.0) / 2.0;
                double dN5_deta = 9.0 * xi * (3.0 * xi - 1.0) / 2.0;
                return {dN5_dxi, dN5_deta};
            }
            case 6: {
                double dN6_dxi = 9.0 * eta * (3.0 * eta - 1.0) / 2.0;
                double dN6_deta = 9.0 * xi * (6.0 * eta - 1.0) / 2.0;
                return {dN6_dxi, dN6_deta};
            }
            case 7: {
                double dN7_dxi = -9.0 * eta * (3.0 * eta - 1.0) / 2.0;
                double dN7_deta = 9.0 * (zeta * (6.0 * eta - 1.0) + eta * (3.0 * eta - 1.0)) / 2.0;
                return {dN7_dxi, dN7_deta};
            }
            case 8: {
                double dN8_dxi = 9.0 * eta * (1.0 - 6.0 * zeta + 3.0 * zeta) / 2.0;
                double dN8_deta = 9.0 * (zeta * (3.0 * zeta - 1.0) - eta * (6.0 * zeta - 1.0)) / 2.0;
                return {dN8_dxi, dN8_deta};
            }
            
            // Interior node gradient
            case 9: {
                double dN9_dxi = 27.0 * eta * (zeta - xi);
                double dN9_deta = 27.0 * xi * (zeta - eta);
                return {dN9_dxi, dN9_deta};
            }
            
            default: return {0.0, 0.0};
        }
    }
};

/**
 * @brief Quadrature rules for triangular elements
 */
class TriangularQuadrature {
public:
    /**
     * @brief Get quadrature points and weights for triangular elements
     * @param order Desired accuracy order
     * @return Pair of {points, weights} where points[i] = {xi, eta}
     */
    static std::pair<std::vector<std::vector<double>>, std::vector<double>> 
    get_quadrature_rule(int order) {
        
        if (order <= 1) {
            // 1-point rule (exact for P1)
            return {{{1.0/3.0, 1.0/3.0}}, {1.0}};
        }
        else if (order <= 2) {
            // 3-point rule (exact for P2)
            return {
                {{1.0/6.0, 1.0/6.0}, {2.0/3.0, 1.0/6.0}, {1.0/6.0, 2.0/3.0}},
                {1.0/3.0, 1.0/3.0, 1.0/3.0}
            };
        }
        else if (order <= 4) {
            // 7-point rule (exact for P4, good for P3)
            return {
                {
                    {1.0/3.0, 1.0/3.0},
                    {0.797426985353087, 0.101286507323456},
                    {0.101286507323456, 0.797426985353087},
                    {0.101286507323456, 0.101286507323456},
                    {0.470142064105115, 0.470142064105115},
                    {0.470142064105115, 0.059715871789770},
                    {0.059715871789770, 0.470142064105115}
                },
                {
                    0.225000000000000,
                    0.125939180544827,
                    0.125939180544827,
                    0.125939180544827,
                    0.132394152788506,
                    0.132394152788506,
                    0.132394152788506
                }
            };
        }
        else {
            // 12-point rule (exact for P6)
            return {
                {
                    {0.873821971016996, 0.063089014491502},
                    {0.063089014491502, 0.873821971016996},
                    {0.063089014491502, 0.063089014491502},
                    {0.501426509658179, 0.249286745170910},
                    {0.249286745170910, 0.501426509658179},
                    {0.249286745170910, 0.249286745170910},
                    {0.636502499121399, 0.310352451033785},
                    {0.636502499121399, 0.053145049844816},
                    {0.310352451033785, 0.636502499121399},
                    {0.310352451033785, 0.053145049844816},
                    {0.053145049844816, 0.636502499121399},
                    {0.053145049844816, 0.310352451033785}
                },
                {
                    0.050844906370207,
                    0.050844906370207,
                    0.050844906370207,
                    0.116786275726379,
                    0.116786275726379,
                    0.116786275726379,
                    0.082851075618374,
                    0.082851075618374,
                    0.082851075618374,
                    0.082851075618374,
                    0.082851075618374,
                    0.082851075618374
                }
            };
        }
    }
};

} // namespace DG
} // namespace SemiDGFEM

#endif // DG_BASIS_FUNCTIONS_COMPLETE_HPP
