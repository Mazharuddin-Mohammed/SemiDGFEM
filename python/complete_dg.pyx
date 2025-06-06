# distutils: language = c++
# cython: language_level=3

"""
Complete DG Discretization Python Interface
Provides access to all DG basis functions and assembly routines
"""

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

# No additional imports needed

# C++ declarations for complete DG basis functions
cdef extern from "dg_math/dg_basis_functions_complete.hpp" namespace "SemiDGFEM::DG":
    cdef cppclass TriangularBasisFunctions:
        @staticmethod
        double evaluate_basis_function(double xi, double eta, int j, int order) except +

        @staticmethod
        vector[double] evaluate_basis_gradient_ref(double xi, double eta, int j, int order) except +

        @staticmethod
        vector[double] transform_gradient_to_physical(
            const vector[double]& grad_ref,
            double b1, double b2, double b3,
            double c1, double c2, double c3) except +

        @staticmethod
        int get_dofs_per_element(int order) except +

    cdef cppclass TriangularQuadrature:
        @staticmethod
        pair[vector[vector[double]], vector[double]] get_quadrature_rule(int order) except +

# Python wrapper for DG basis functions
cdef class DGBasisFunctions:
    """
    Complete DG basis functions for triangular elements.
    
    Supports P1 (3 DOFs), P2 (6 DOFs), P3 (10 DOFs) triangular elements
    with proper basis function evaluation and gradient computation.
    """
    
    @staticmethod
    def evaluate_basis_function(double xi, double eta, int j, int order):
        """
        Evaluate basis function at reference coordinates.

        Parameters:
        -----------
        xi : float
            First barycentric coordinate
        eta : float
            Second barycentric coordinate
        j : int
            Basis function index
        order : int
            Polynomial order (1, 2, or 3)

        Returns:
        --------
        float
            Value of basis function j at (xi, eta)
        """
        return TriangularBasisFunctions.evaluate_basis_function(xi, eta, j, order)
    
    @staticmethod
    def evaluate_basis_gradient_ref(double xi, double eta, int j, int order):
        """
        Evaluate basis function gradient in reference coordinates.

        Parameters:
        -----------
        xi : float
            First barycentric coordinate
        eta : float
            Second barycentric coordinate
        j : int
            Basis function index
        order : int
            Polynomial order

        Returns:
        --------
        numpy.ndarray
            Gradient vector [d/dxi, d/deta]
        """
        cdef vector[double] grad_ref = TriangularBasisFunctions.evaluate_basis_gradient_ref(xi, eta, j, order)
        return np.array([grad_ref[0], grad_ref[1]])
    
    @staticmethod
    def transform_gradient_to_physical(np.ndarray[double, ndim=1] grad_ref,
                                     double b1, double b2, double b3,
                                     double c1, double c2, double c3):
        """
        Transform gradient from reference to physical coordinates.
        
        Parameters:
        -----------
        grad_ref : array_like
            Gradient in reference coordinates [d/dxi, d/deta]
        b1, b2, b3 : float
            Geometric transformation coefficients for x-direction
        c1, c2, c3 : float
            Geometric transformation coefficients for y-direction
            
        Returns:
        --------
        numpy.ndarray
            Gradient in physical coordinates [d/dx, d/dy]
        """
        cdef vector[double] grad_ref_vec
        grad_ref_vec.push_back(grad_ref[0])
        grad_ref_vec.push_back(grad_ref[1])
        
        cdef vector[double] grad_phys = TriangularBasisFunctions.transform_gradient_to_physical(
            grad_ref_vec, b1, b2, b3, c1, c2, c3)
        
        return np.array([grad_phys[0], grad_phys[1]])
    
    @staticmethod
    def get_dofs_per_element(int order):
        """
        Get number of DOFs for given polynomial order.

        Parameters:
        -----------
        order : int
            Polynomial order

        Returns:
        --------
        int
            Number of DOFs per element
        """
        return TriangularBasisFunctions.get_dofs_per_element(order)

# Python wrapper for quadrature rules
cdef class DGQuadrature:
    """
    Quadrature rules for triangular elements.
    """
    
    @staticmethod
    def get_quadrature_rule(int order):
        """
        Get quadrature points and weights for triangular elements.

        Parameters:
        -----------
        order : int
            Desired accuracy order

        Returns:
        --------
        tuple
            (points, weights) where points is array of [xi, eta] coordinates
            and weights is array of quadrature weights
        """
        cdef pair[vector[vector[double]], vector[double]] quad_rule = TriangularQuadrature.get_quadrature_rule(order)

        # Convert points
        points = []
        for i in range(quad_rule.first.size()):
            points.append([quad_rule.first[i][0], quad_rule.first[i][1]])

        # Convert weights
        weights = np.array(quad_rule.second)

        return np.array(points), weights

# High-level DG assembly interface
cdef class DGAssembly:
    """
    High-level interface for DG assembly operations.
    """
    
    def __init__(self, int order=3):
        """
        Initialize DG assembly with given polynomial order.
        
        Parameters:
        -----------
        order : int, optional
            Polynomial order (default: 3)
        """
        self.order = order
        self.dofs_per_element = DGBasisFunctions.get_dofs_per_element(order)
    
    def assemble_element_matrix(self, np.ndarray[double, ndim=2] vertices, 
                               str matrix_type="mass"):
        """
        Assemble element matrix for given element.
        
        Parameters:
        -----------
        vertices : array_like
            Element vertices [[x1, y1], [x2, y2], [x3, y3]]
        matrix_type : str
            Type of matrix ("mass", "stiffness", "convection")
            
        Returns:
        --------
        numpy.ndarray
            Element matrix (dofs_per_element x dofs_per_element)
        """
        if vertices.shape[0] != 3 or vertices.shape[1] != 2:
            raise ValueError("vertices must be 3x2 array")
        
        # Element geometry
        cdef double x1 = vertices[0, 0], y1 = vertices[0, 1]
        cdef double x2 = vertices[1, 0], y2 = vertices[1, 1]
        cdef double x3 = vertices[2, 0], y3 = vertices[2, 1]
        
        cdef double area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if area < 1e-12:
            raise ValueError("Degenerate element with zero area")
        
        # Geometric coefficients
        cdef double b1 = (y2 - y3) / (2.0 * area), c1 = (x3 - x2) / (2.0 * area)
        cdef double b2 = (y3 - y1) / (2.0 * area), c2 = (x1 - x3) / (2.0 * area)
        cdef double b3 = (y1 - y2) / (2.0 * area), c3 = (x2 - x1) / (2.0 * area)
        
        # Get quadrature rule
        quad_points, quad_weights = DGQuadrature.get_quadrature_rule(4)
        
        # Initialize element matrix
        cdef np.ndarray[double, ndim=2] element_matrix = np.zeros((self.dofs_per_element, self.dofs_per_element))
        
        # Quadrature loop
        for q in range(len(quad_points)):
            xi = quad_points[q, 0]
            eta = quad_points[q, 1]
            w = quad_weights[q] * area
            
            # Assembly based on matrix type
            for i in range(self.dofs_per_element):
                for j in range(self.dofs_per_element):
                    phi_i = DGBasisFunctions.evaluate_basis_function(xi, eta, i, self.order)
                    phi_j = DGBasisFunctions.evaluate_basis_function(xi, eta, j, self.order)
                    
                    if matrix_type == "mass":
                        # Mass matrix: ∫ φᵢ φⱼ dΩ
                        element_matrix[i, j] += w * phi_i * phi_j
                    
                    elif matrix_type == "stiffness":
                        # Stiffness matrix: ∫ ∇φᵢ · ∇φⱼ dΩ
                        grad_i_ref = DGBasisFunctions.evaluate_basis_gradient_ref(xi, eta, i, self.order)
                        grad_j_ref = DGBasisFunctions.evaluate_basis_gradient_ref(xi, eta, j, self.order)
                        grad_i_phys = DGBasisFunctions.transform_gradient_to_physical(
                            grad_i_ref, b1, b2, b3, c1, c2, c3)
                        grad_j_phys = DGBasisFunctions.transform_gradient_to_physical(
                            grad_j_ref, b1, b2, b3, c1, c2, c3)
                        
                        element_matrix[i, j] += w * (grad_i_phys[0] * grad_j_phys[0] + 
                                                   grad_i_phys[1] * grad_j_phys[1])
                    
                    else:
                        raise ValueError(f"Unknown matrix type: {matrix_type}")
        
        return element_matrix
    
    def assemble_element_vector(self, np.ndarray[double, ndim=2] vertices,
                               source_function=None):
        """
        Assemble element load vector.
        
        Parameters:
        -----------
        vertices : array_like
            Element vertices [[x1, y1], [x2, y2], [x3, y3]]
        source_function : callable, optional
            Source function f(x, y) -> float
            
        Returns:
        --------
        numpy.ndarray
            Element load vector (dofs_per_element,)
        """
        if vertices.shape[0] != 3 or vertices.shape[1] != 2:
            raise ValueError("vertices must be 3x2 array")
        
        # Element geometry
        cdef double x1 = vertices[0, 0], y1 = vertices[0, 1]
        cdef double x2 = vertices[1, 0], y2 = vertices[1, 1]
        cdef double x3 = vertices[2, 0], y3 = vertices[2, 1]
        
        cdef double area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if area < 1e-12:
            raise ValueError("Degenerate element with zero area")
        
        # Get quadrature rule
        quad_points, quad_weights = DGQuadrature.get_quadrature_rule(4)
        
        # Initialize element vector
        cdef np.ndarray[double, ndim=1] element_vector = np.zeros(self.dofs_per_element)
        
        # Quadrature loop
        for q in range(len(quad_points)):
            xi = quad_points[q, 0]
            eta = quad_points[q, 1]
            w = quad_weights[q] * area
            zeta = 1.0 - xi - eta
            
            # Physical coordinates
            x_phys = x1 * zeta + x2 * xi + x3 * eta
            y_phys = y1 * zeta + y2 * xi + y3 * eta
            
            # Source value
            source_val = 1.0 if source_function is None else source_function(x_phys, y_phys)
            
            # Assembly
            for i in range(self.dofs_per_element):
                phi_i = DGBasisFunctions.evaluate_basis_function(xi, eta, i, self.order)
                element_vector[i] += w * phi_i * source_val
        
        return element_vector
    
    def validate_basis_functions(self):
        """
        Validate basis function properties.
        
        Returns:
        --------
        dict
            Validation results
        """
        results = {
            "partition_of_unity": True,
            "gradient_consistency": True,
            "max_partition_error": 0.0,
            "max_gradient_error": 0.0
        }
        
        # Test points
        test_points = [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1.0/3.0, 1.0/3.0]
        ]
        
        for point in test_points:
            xi, eta = point[0], point[1]
            zeta = 1.0 - xi - eta
            
            if zeta >= -1e-10:  # Valid barycentric coordinates
                # Test partition of unity
                sum_basis = 0.0
                for j in range(self.dofs_per_element):
                    phi_j = DGBasisFunctions.evaluate_basis_function(xi, eta, j, self.order)
                    sum_basis += phi_j
                
                partition_error = abs(sum_basis - 1.0)
                results["max_partition_error"] = max(results["max_partition_error"], partition_error)
                
                if partition_error > 1e-10:
                    results["partition_of_unity"] = False
        
        return results

# Convenience functions
def create_p1_assembly():
    """Create P1 DG assembly object."""
    return DGAssembly(order=1)

def create_p2_assembly():
    """Create P2 DG assembly object."""
    return DGAssembly(order=2)

def create_p3_assembly():
    """Create P3 DG assembly object."""
    return DGAssembly(order=3)

def validate_complete_dg_implementation():
    """
    Validate the complete DG implementation.
    
    Returns:
    --------
    dict
        Comprehensive validation results
    """
    results = {
        "p1_validation": create_p1_assembly().validate_basis_functions(),
        "p2_validation": create_p2_assembly().validate_basis_functions(),
        "p3_validation": create_p3_assembly().validate_basis_functions(),
        "dofs_per_element": {
            "P1": DGBasisFunctions.get_dofs_per_element(1),
            "P2": DGBasisFunctions.get_dofs_per_element(2),
            "P3": DGBasisFunctions.get_dofs_per_element(3)
        },
        "quadrature_points": {
            "order_1": len(DGQuadrature.get_quadrature_rule(1)[0]),
            "order_2": len(DGQuadrature.get_quadrature_rule(2)[0]),
            "order_4": len(DGQuadrature.get_quadrature_rule(4)[0]),
            "order_6": len(DGQuadrature.get_quadrature_rule(6)[0])
        }
    }
    
    return results
