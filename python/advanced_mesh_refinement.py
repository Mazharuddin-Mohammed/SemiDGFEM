"""
Advanced Adaptive Mesh Refinement for SemiDGFEM Simulator

This module provides advanced adaptive mesh refinement capabilities including:
- Multiple error estimation strategies
- Anisotropic refinement
- Feature detection
- Boundary layer handling
- Load balancing for parallel execution

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from enum import Enum
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedErrorEstimatorType(Enum):
    """Advanced error estimator types"""
    GRADIENT_RECOVERY = "gradient_recovery"
    HIERARCHICAL_BASIS = "hierarchical_basis"
    DUAL_WEIGHTED_RESIDUAL = "dual_weighted_residual"
    FEATURE_DETECTION = "feature_detection"
    PHYSICS_BASED = "physics_based"
    HYBRID = "hybrid"

class RefinementDirection(Enum):
    """Anisotropic refinement directions"""
    ISOTROPIC = "isotropic"
    X_DIRECTION = "x_direction"
    Y_DIRECTION = "y_direction"
    ADAPTIVE = "adaptive"

@dataclass
class AdvancedRefinementInfo:
    """Element refinement information"""
    refine: bool = False
    coarsen: bool = False
    direction: RefinementDirection = RefinementDirection.ISOTROPIC
    anisotropy_ratio: float = 1.0
    target_level: int = 0
    error_indicator: float = 0.0
    gradient_direction: np.ndarray = None
    is_boundary_layer: bool = False
    feature_strength: float = 0.0
    
    def __post_init__(self):
        if self.gradient_direction is None:
            self.gradient_direction = np.array([0.0, 0.0])

@dataclass
class MeshQualityMetrics:
    """Mesh quality metrics"""
    min_angle: float = 0.0
    max_angle: float = 0.0
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = 0.0
    min_element_size: float = 0.0
    max_element_size: float = 0.0
    average_quality: float = 0.0
    num_poor_quality_elements: int = 0
    mesh_regularity: float = 0.0

@dataclass
class RefinementStatistics:
    """Refinement statistics"""
    total_elements_refined: int = 0
    total_elements_coarsened: int = 0
    anisotropic_refinements: int = 0
    average_error_reduction: float = 0.0
    mesh_quality_improvement: float = 0.0
    computational_cost_increase: float = 0.0

class AdvancedAMRController:
    """Advanced adaptive mesh refinement controller"""
    
    def __init__(self):
        self.estimator_type = AdvancedErrorEstimatorType.GRADIENT_RECOVERY
        self.refine_fraction = 0.3
        self.coarsen_fraction = 0.1
        self.error_tolerance = 1e-3
        self.anisotropic_enabled = False
        self.max_anisotropy_ratio = 10.0
        self.boundary_layer_threshold = 0.1
        self.feature_threshold = 0.5
        self.gradient_threshold = 1e-3
        self.physics_type = "semiconductor"
        self.solution_variables = []
        self.stats = RefinementStatistics()
        
    def set_error_estimator(self, estimator_type: AdvancedErrorEstimatorType):
        """Set error estimator type"""
        self.estimator_type = estimator_type
        logger.info(f"Error estimator set to: {estimator_type.value}")
        
    def set_refinement_parameters(self, refine_fraction: float, 
                                coarsen_fraction: float, error_tolerance: float):
        """Set refinement parameters"""
        if not (0.0 <= refine_fraction <= 1.0) or not (0.0 <= coarsen_fraction <= 1.0):
            raise ValueError("Refinement fractions must be between 0 and 1")
            
        self.refine_fraction = refine_fraction
        self.coarsen_fraction = coarsen_fraction
        self.error_tolerance = error_tolerance
        
    def set_anisotropic_parameters(self, enable: bool, max_anisotropy_ratio: float = 10.0,
                                 boundary_layer_threshold: float = 0.1):
        """Set anisotropic refinement parameters"""
        self.anisotropic_enabled = enable
        self.max_anisotropy_ratio = max(1.0, max_anisotropy_ratio)
        self.boundary_layer_threshold = boundary_layer_threshold
        
    def set_feature_detection_parameters(self, feature_threshold: float = 0.5,
                                       gradient_threshold: float = 1e-3):
        """Set feature detection parameters"""
        self.feature_threshold = feature_threshold
        self.gradient_threshold = gradient_threshold
        
    def set_physics_parameters(self, physics_type: str = "semiconductor",
                             solution_variables: List[str] = None):
        """Set physics-specific parameters"""
        self.physics_type = physics_type
        self.solution_variables = solution_variables or []
        
    def estimate_error(self, solution: np.ndarray, elements: np.ndarray, 
                      vertices: np.ndarray, variable_names: List[str] = None) -> np.ndarray:
        """Estimate error for each element"""
        if solution.size == 0 or elements.size == 0 or vertices.size == 0:
            return np.zeros(len(elements))
            
        if self.estimator_type == AdvancedErrorEstimatorType.GRADIENT_RECOVERY:
            return self._compute_gradient_recovery_error(solution, elements, vertices)
        elif self.estimator_type == AdvancedErrorEstimatorType.HIERARCHICAL_BASIS:
            return self._compute_hierarchical_error(solution, elements, vertices)
        elif self.estimator_type == AdvancedErrorEstimatorType.DUAL_WEIGHTED_RESIDUAL:
            return self._compute_dual_weighted_residual_error(solution, elements, vertices)
        elif self.estimator_type == AdvancedErrorEstimatorType.FEATURE_DETECTION:
            return self.detect_features(solution, elements, vertices)
        elif self.estimator_type == AdvancedErrorEstimatorType.PHYSICS_BASED:
            return self._compute_physics_based_error(solution, elements, vertices)
        elif self.estimator_type == AdvancedErrorEstimatorType.HYBRID:
            # Combine multiple estimators
            gradient_errors = self._compute_gradient_recovery_error(solution, elements, vertices)
            feature_errors = self.detect_features(solution, elements, vertices)
            physics_errors = self._compute_physics_based_error(solution, elements, vertices)
            
            return 0.4 * gradient_errors + 0.3 * feature_errors + 0.3 * physics_errors
        else:
            return self._compute_gradient_recovery_error(solution, elements, vertices)
            
    def determine_refinement(self, error_indicators: np.ndarray, elements: np.ndarray,
                           vertices: np.ndarray, solution: np.ndarray = None,
                           current_levels: np.ndarray = None) -> List[AdvancedRefinementInfo]:
        """Determine refinement strategy for each element"""
        refinement_info = []
        
        if len(error_indicators) == 0:
            return refinement_info
            
        # Sort elements by error
        sorted_indices = np.argsort(error_indicators)[::-1]
        
        # Compute thresholds
        max_error = np.max(error_indicators)
        mean_error = np.mean(error_indicators)
        
        refine_threshold = max(self.error_tolerance, 0.5 * max_error)
        coarsen_threshold = 0.1 * mean_error
        
        # Detect boundary layers if enabled
        boundary_layer_elements = []
        if self.anisotropic_enabled:
            # Simple boundary detection - assume first 10% of vertices are boundary
            boundary_count = max(1, len(vertices) // 10)
            boundary_nodes = list(range(boundary_count))
            boundary_layer_elements = self.detect_boundary_layers(elements, vertices, boundary_nodes)
            
        # Determine refinement for each element
        for e in range(len(elements)):
            info = AdvancedRefinementInfo()
            info.error_indicator = error_indicators[e]
            
            # Refinement decision
            if error_indicators[e] > refine_threshold:
                info.refine = True
                
                # Anisotropic analysis
                if self.anisotropic_enabled and solution is not None:
                    info.direction = self._analyze_anisotropy(solution, elements[e], vertices)
                    
                    if info.direction != RefinementDirection.ISOTROPIC:
                        info.anisotropy_ratio = self._compute_anisotropy_ratio(
                            info.gradient_direction, elements[e], vertices)
                        self.stats.anisotropic_refinements += 1
                        
                    # Check for boundary layer
                    if e < len(boundary_layer_elements) and boundary_layer_elements[e]:
                        info.is_boundary_layer = True
                        info.direction = RefinementDirection.Y_DIRECTION
                        info.anisotropy_ratio = min(self.max_anisotropy_ratio, 5.0)
                        
                self.stats.total_elements_refined += 1
                
            # Coarsening decision
            elif error_indicators[e] < coarsen_threshold:
                info.coarsen = True
                self.stats.total_elements_coarsened += 1
                
            # Feature detection
            if (solution is not None and
                (self.estimator_type == AdvancedErrorEstimatorType.FEATURE_DETECTION or
                 self.estimator_type == AdvancedErrorEstimatorType.HYBRID)):
                gradient = self._compute_element_gradient(
                    np.array(solution) if hasattr(solution, '__len__') else solution,
                    elements[e], vertices)
                gradient_magnitude = np.linalg.norm(gradient)
                info.feature_strength = gradient_magnitude

                if gradient_magnitude > self.feature_threshold:
                    info.refine = True
                    info.direction = RefinementDirection.ADAPTIVE
                    
            refinement_info.append(info)
            
        return refinement_info
        
    def analyze_mesh_quality(self, elements: np.ndarray, vertices: np.ndarray) -> MeshQualityMetrics:
        """Analyze mesh quality metrics"""
        metrics = MeshQualityMetrics()
        
        if len(elements) == 0 or len(vertices) == 0:
            return metrics
            
        angles = []
        aspect_ratios = []
        element_sizes = []
        qualities = []
        
        for element in elements:
            if len(element) != 3:
                continue  # Only triangular elements
                
            # Get vertices
            v1, v2, v3 = vertices[element[0]], vertices[element[1]], vertices[element[2]]
            
            # Compute edge lengths
            a = np.linalg.norm(v2 - v3)
            b = np.linalg.norm(v1 - v3)
            c = np.linalg.norm(v1 - v2)
            
            # Compute angles using law of cosines
            angle1 = np.arccos(np.clip((b**2 + c**2 - a**2) / (2*b*c), -1, 1))
            angle2 = np.arccos(np.clip((a**2 + c**2 - b**2) / (2*a*c), -1, 1))
            angle3 = np.pi - angle1 - angle2
            
            angles.extend([np.degrees(angle1), np.degrees(angle2), np.degrees(angle3)])
            
            # Compute aspect ratio
            max_edge = max(a, b, c)
            min_edge = min(a, b, c)
            aspect_ratio = max_edge / min_edge if min_edge > 0 else float('inf')
            aspect_ratios.append(aspect_ratio)
            
            # Compute element size (area) - handle 2D vectors
            edge1 = v2 - v1
            edge2 = v3 - v1
            area = 0.5 * abs(edge1[0]*edge2[1] - edge1[1]*edge2[0])
            element_sizes.append(area)
            
            # Compute quality (ratio of inscribed to circumscribed circle radii)
            perimeter = a + b + c
            if perimeter > 0 and area > 0:
                inradius = area / (0.5 * perimeter)
                circumradius = (a * b * c) / (4.0 * area)
                quality = inradius / circumradius if circumradius > 0 else 0.0
                qualities.append(quality)
                
        # Compute metrics
        if angles:
            metrics.min_angle = min(angles)
            metrics.max_angle = max(angles)
            
        if aspect_ratios:
            metrics.min_aspect_ratio = min(aspect_ratios)
            metrics.max_aspect_ratio = max(aspect_ratios)
            metrics.num_poor_quality_elements = sum(1 for ar in aspect_ratios if ar > 5.0)
            
        if element_sizes:
            metrics.min_element_size = min(element_sizes)
            metrics.max_element_size = max(element_sizes)
            
        if qualities:
            metrics.average_quality = np.mean(qualities)
            # Mesh regularity (inverse of quality standard deviation)
            quality_std = np.std(qualities)
            metrics.mesh_regularity = 1.0 / (1.0 + quality_std)
            
        return metrics

    def detect_features(self, solution: np.ndarray, elements: np.ndarray,
                       vertices: np.ndarray) -> np.ndarray:
        """Detect features in the solution"""
        feature_indicators = np.zeros(len(elements))

        for e, element in enumerate(elements):
            gradient = self._compute_element_gradient(solution, element, vertices)
            gradient_magnitude = np.linalg.norm(gradient)

            # Compute second derivatives (Hessian) for curvature detection
            curvature = 0.0
            if e > 0 and e < len(elements) - 1:
                prev_gradient = self._compute_element_gradient(solution, elements[e-1], vertices)
                next_gradient = self._compute_element_gradient(solution, elements[e+1], vertices)

                grad_change = next_gradient - prev_gradient
                curvature = np.linalg.norm(grad_change)

            # Feature strength combines gradient magnitude and curvature
            feature_indicators[e] = gradient_magnitude + 0.5 * curvature

        return feature_indicators

    def detect_boundary_layers(self, elements: np.ndarray, vertices: np.ndarray,
                             boundary_nodes: List[int]) -> List[bool]:
        """Detect boundary layer elements"""
        is_boundary_layer = [False] * len(elements)
        boundary_set = set(boundary_nodes)

        for e, element in enumerate(elements):
            # Check if element has boundary nodes
            boundary_node_count = sum(1 for node in element if node in boundary_set)

            # Element is in boundary layer if it has at least one boundary node
            if boundary_node_count > 0:
                # Additional check: element size should be small in normal direction
                element_size = self._compute_element_size(element, vertices)
                if element_size < self.boundary_layer_threshold:
                    is_boundary_layer[e] = True

        return is_boundary_layer

    def get_refinement_statistics(self) -> RefinementStatistics:
        """Get refinement statistics"""
        return self.stats

    def reset_statistics(self):
        """Reset refinement statistics"""
        self.stats = RefinementStatistics()

    # Private methods for error estimation
    def _compute_gradient_recovery_error(self, solution: np.ndarray, elements: np.ndarray,
                                       vertices: np.ndarray) -> np.ndarray:
        """Compute Zienkiewicz-Zhu gradient recovery error"""
        errors = np.zeros(len(elements))

        # Zienkiewicz-Zhu gradient recovery
        recovered_gradients = np.zeros((len(vertices), 2))
        node_weights = np.zeros(len(vertices))

        # Compute element gradients and project to nodes
        for e, element in enumerate(elements):
            element_gradient = self._compute_element_gradient(solution, element, vertices)
            element_area = self._compute_element_size(element, vertices)

            # Distribute gradient to element nodes
            for node in element:
                recovered_gradients[node] += element_gradient * element_area
                node_weights[node] += element_area

        # Normalize by weights
        for n in range(len(vertices)):
            if node_weights[n] > 0:
                recovered_gradients[n] /= node_weights[n]

        # Compute error as difference between element and recovered gradients
        for e, element in enumerate(elements):
            element_gradient = self._compute_element_gradient(solution, element, vertices)

            # Average recovered gradient over element
            avg_recovered = np.mean([recovered_gradients[node] for node in element], axis=0)

            # Error norm
            error_vector = element_gradient - avg_recovered
            errors[e] = np.linalg.norm(error_vector)

        return errors

    def _compute_hierarchical_error(self, solution: np.ndarray, elements: np.ndarray,
                                  vertices: np.ndarray) -> np.ndarray:
        """Compute hierarchical basis error estimation"""
        errors = np.zeros(len(elements))

        # Simplified hierarchical error estimation
        # In practice, this would use higher-order basis functions
        for e, element in enumerate(elements):
            # Compute solution variation within element
            element_solution = solution[element]
            solution_variation = np.std(element_solution)

            # Scale by element size
            element_size = self._compute_element_size(element, vertices)
            characteristic_length = np.sqrt(element_size)

            errors[e] = solution_variation * characteristic_length

        return errors

    def _compute_dual_weighted_residual_error(self, solution: np.ndarray, elements: np.ndarray,
                                            vertices: np.ndarray) -> np.ndarray:
        """Compute dual-weighted residual error (goal-oriented)"""
        errors = np.zeros(len(elements))

        # Simplified dual-weighted residual estimation
        # In practice, this would solve an adjoint problem
        for e, element in enumerate(elements):
            gradient = self._compute_element_gradient(solution, element, vertices)
            gradient_magnitude = np.linalg.norm(gradient)

            # Weight by element size and solution magnitude
            element_size = self._compute_element_size(element, vertices)
            solution_magnitude = np.mean(np.abs(solution[element]))

            errors[e] = gradient_magnitude * element_size * solution_magnitude

        return errors

    def _compute_physics_based_error(self, solution: np.ndarray, elements: np.ndarray,
                                   vertices: np.ndarray) -> np.ndarray:
        """Compute physics-based error indicators"""
        errors = np.zeros(len(elements))

        if self.physics_type == "semiconductor":
            # Semiconductor-specific error indicators
            for e, element in enumerate(elements):
                gradient = self._compute_element_gradient(solution, element, vertices)
                gradient_magnitude = np.linalg.norm(gradient)

                # High gradients indicate depletion regions, junctions, etc.
                element_size = self._compute_element_size(element, vertices)
                characteristic_length = np.sqrt(element_size)

                # Error based on gradient and characteristic length
                errors[e] = gradient_magnitude * characteristic_length

                # Additional physics-based criteria
                if len(solution) >= len(element):
                    # Check for rapid solution changes (e.g., potential drops)
                    element_solution = solution[element]
                    solution_variation = np.max(element_solution) - np.min(element_solution)
                    errors[e] += 0.5 * solution_variation / characteristic_length

        return errors

    # Utility methods
    def _compute_element_gradient(self, solution: np.ndarray, element: np.ndarray,
                                vertices: np.ndarray) -> np.ndarray:
        """Compute element gradient using linear basis functions"""
        if len(element) != 3 or np.max(element) >= len(solution):
            return np.array([0.0, 0.0])

        # Get element vertices and solution values
        v1, v2, v3 = vertices[element[0]], vertices[element[1]], vertices[element[2]]
        u1, u2, u3 = solution[element[0]], solution[element[1]], solution[element[2]]

        # Compute gradient using linear basis functions
        det = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])

        if abs(det) < 1e-12:
            return np.array([0.0, 0.0])

        grad_x = ((v3[1] - v1[1]) * (u2 - u1) - (v2[1] - v1[1]) * (u3 - u1)) / det
        grad_y = ((v2[0] - v1[0]) * (u3 - u1) - (v3[0] - v1[0]) * (u2 - u1)) / det

        return np.array([grad_x, grad_y])

    def _compute_element_size(self, element: np.ndarray, vertices: np.ndarray) -> float:
        """Compute element size (area for triangles)"""
        if len(element) != 3:
            return 0.0

        v1, v2, v3 = vertices[element[0]], vertices[element[1]], vertices[element[2]]

        # Triangle area using cross product (handle 2D vectors)
        edge1 = v2 - v1
        edge2 = v3 - v1
        # For 2D vectors, cross product is: edge1[0]*edge2[1] - edge1[1]*edge2[0]
        area = 0.5 * abs(edge1[0]*edge2[1] - edge1[1]*edge2[0])
        return area

    def _analyze_anisotropy(self, solution: np.ndarray, element: np.ndarray,
                          vertices: np.ndarray) -> RefinementDirection:
        """Analyze anisotropy for refinement direction"""
        gradient = self._compute_element_gradient(solution, element, vertices)
        grad_magnitude = np.linalg.norm(gradient)

        if grad_magnitude < self.gradient_threshold:
            return RefinementDirection.ISOTROPIC

        # Determine dominant gradient direction
        angle = np.arctan2(gradient[1], gradient[0])
        abs_angle = abs(angle)

        if abs_angle < np.pi/4 or abs_angle > 3*np.pi/4:
            return RefinementDirection.X_DIRECTION
        else:
            return RefinementDirection.Y_DIRECTION

    def _compute_anisotropy_ratio(self, gradient_direction: np.ndarray, element: np.ndarray,
                                vertices: np.ndarray) -> float:
        """Compute anisotropy ratio for refinement"""
        grad_magnitude = np.linalg.norm(gradient_direction)

        if grad_magnitude < self.gradient_threshold:
            return 1.0

        # Simple anisotropy ratio based on gradient magnitude
        ratio = min(self.max_anisotropy_ratio, 1.0 + grad_magnitude * 10.0)
        return ratio


@dataclass
class RefinementResult:
    """Result of mesh refinement operation"""
    new_elements: np.ndarray = None
    new_vertices: np.ndarray = None
    element_levels: np.ndarray = None
    parent_mapping: np.ndarray = None
    vertex_mapping: np.ndarray = None
    success: bool = False
    error_message: str = ""

    def __post_init__(self):
        if self.new_elements is None:
            self.new_elements = np.array([])
        if self.new_vertices is None:
            self.new_vertices = np.array([])
        if self.element_levels is None:
            self.element_levels = np.array([])
        if self.parent_mapping is None:
            self.parent_mapping = np.array([])
        if self.vertex_mapping is None:
            self.vertex_mapping = np.array([])


class MeshRefinementExecutor:
    """Execute mesh refinement operations"""

    def __init__(self):
        self.tolerance = 1e-12

    def execute_refinement(self, elements: np.ndarray, vertices: np.ndarray,
                         refinement_info: List[AdvancedRefinementInfo]) -> RefinementResult:
        """Execute mesh refinement"""
        result = RefinementResult()

        try:
            # Initialize with original mesh
            result.new_elements = elements.copy()
            result.new_vertices = vertices.copy().tolist()  # Convert to list for easier manipulation
            result.element_levels = np.zeros(len(elements), dtype=int)
            result.parent_mapping = np.arange(len(elements))
            result.vertex_mapping = np.arange(len(vertices))

            # Process refinement requests
            elements_to_add = []
            elements_to_remove = []

            for e, info in enumerate(refinement_info):
                if e >= len(elements) or not info.refine:
                    continue

                element = elements[e]

                if info.direction == RefinementDirection.ISOTROPIC:
                    new_elements = self._refine_element_isotropic(element, result.new_vertices)
                else:
                    new_elements = self._refine_element_anisotropic(
                        element, result.new_vertices, info.direction, info.anisotropy_ratio)

                # Add new elements
                for new_elem in new_elements:
                    elements_to_add.append(new_elem)

                # Mark original element for removal
                elements_to_remove.append(e)

            # Remove refined elements (in reverse order to maintain indices)
            elements_to_remove.sort(reverse=True)
            result.new_elements = np.delete(result.new_elements, elements_to_remove, axis=0)
            result.element_levels = np.delete(result.element_levels, elements_to_remove)
            result.parent_mapping = np.delete(result.parent_mapping, elements_to_remove)

            # Add new elements
            if elements_to_add:
                result.new_elements = np.vstack([result.new_elements, np.array(elements_to_add)])

                # Update mappings
                for removed_idx in elements_to_remove:
                    new_levels = np.full(4, result.element_levels[removed_idx] + 1)  # 4 new elements per refined
                    new_parents = np.full(4, removed_idx)
                    result.element_levels = np.concatenate([result.element_levels, new_levels])
                    result.parent_mapping = np.concatenate([result.parent_mapping, new_parents])

            # Update vertex mapping
            original_vertex_count = len(vertices)
            new_vertex_mapping = np.full(len(result.new_vertices), -1)
            new_vertex_mapping[:original_vertex_count] = np.arange(original_vertex_count)
            result.vertex_mapping = new_vertex_mapping

            # Convert vertices back to numpy array and ensure mesh conformity
            result.new_vertices = np.array(result.new_vertices)
            self._ensure_mesh_conformity(result.new_elements, result.new_vertices)

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = f"Refinement failed: {str(e)}"
            logger.error(result.error_message)

        return result

    def transfer_solution(self, old_solution: np.ndarray, refinement_result: RefinementResult,
                        transfer_method: str = "interpolation") -> np.ndarray:
        """Transfer solution from old mesh to new mesh"""
        if not refinement_result.success:
            return old_solution

        new_solution = np.zeros(len(refinement_result.new_vertices))

        if transfer_method == "interpolation":
            # Linear interpolation for new vertices
            for i, original_vertex in enumerate(refinement_result.vertex_mapping):
                if original_vertex >= 0 and original_vertex < len(old_solution):
                    # Original vertex - copy solution
                    new_solution[i] = old_solution[original_vertex]
                else:
                    # New vertex - interpolate from neighbors
                    # Simple average of surrounding vertices
                    surrounding_values = []

                    for element in refinement_result.new_elements:
                        if i in element:
                            for vertex_idx in element:
                                if (vertex_idx != i and vertex_idx < len(refinement_result.vertex_mapping) and
                                    refinement_result.vertex_mapping[vertex_idx] >= 0 and
                                    refinement_result.vertex_mapping[vertex_idx] < len(old_solution)):
                                    surrounding_values.append(old_solution[refinement_result.vertex_mapping[vertex_idx]])

                    if surrounding_values:
                        new_solution[i] = np.mean(surrounding_values)
        else:
            # Default: copy available values
            copy_length = min(len(new_solution), len(old_solution))
            new_solution[:copy_length] = old_solution[:copy_length]

        return new_solution

    def _refine_element_isotropic(self, element: np.ndarray, vertices: list) -> List[np.ndarray]:
        """Refine element isotropically (4-way split for triangles)"""
        if len(element) != 3:
            raise ValueError("Only triangular elements supported")

        # Get element vertices (convert to numpy arrays)
        v1 = np.array(vertices[element[0]])
        v2 = np.array(vertices[element[1]])
        v3 = np.array(vertices[element[2]])

        # Create midpoint vertices
        m12 = (v1 + v2) / 2.0
        m23 = (v2 + v3) / 2.0
        m31 = (v3 + v1) / 2.0

        # Add new vertices to mesh
        idx_m12 = self._add_vertex_if_new(m12, vertices)
        idx_m23 = self._add_vertex_if_new(m23, vertices)
        idx_m31 = self._add_vertex_if_new(m31, vertices)

        # Create four new triangular elements
        new_elements = [
            np.array([element[0], idx_m12, idx_m31]),  # Corner triangle 1
            np.array([element[1], idx_m23, idx_m12]),  # Corner triangle 2
            np.array([element[2], idx_m31, idx_m23]),  # Corner triangle 3
            np.array([idx_m12, idx_m23, idx_m31])      # Central triangle
        ]

        return new_elements

    def _refine_element_anisotropic(self, element: np.ndarray, vertices: list,
                                  direction: RefinementDirection, anisotropy_ratio: float) -> List[np.ndarray]:
        """Refine element anisotropically"""
        if len(element) != 3:
            raise ValueError("Only triangular elements supported")

        # Get element vertices (convert to numpy arrays)
        v1 = np.array(vertices[element[0]])
        v2 = np.array(vertices[element[1]])
        v3 = np.array(vertices[element[2]])

        new_elements = []

        if direction == RefinementDirection.X_DIRECTION:
            # Refine primarily in X direction
            # Find the edge most aligned with Y direction (to split in X)
            edge12_y = abs(v2[1] - v1[1])
            edge23_y = abs(v3[1] - v2[1])
            edge31_y = abs(v1[1] - v3[1])

            if edge12_y >= edge23_y and edge12_y >= edge31_y:
                # Split edge 1-2
                mid = (v1 + v2) / 2.0
                idx_mid = self._add_vertex_if_new(mid, vertices)
                new_elements = [
                    np.array([element[0], idx_mid, element[2]]),
                    np.array([idx_mid, element[1], element[2]])
                ]
            elif edge23_y >= edge31_y:
                # Split edge 2-3
                mid = (v2 + v3) / 2.0
                idx_mid = self._add_vertex_if_new(mid, vertices)
                new_elements = [
                    np.array([element[0], element[1], idx_mid]),
                    np.array([element[0], idx_mid, element[2]])
                ]
            else:
                # Split edge 3-1
                mid = (v3 + v1) / 2.0
                idx_mid = self._add_vertex_if_new(mid, vertices)
                new_elements = [
                    np.array([element[0], element[1], idx_mid]),
                    np.array([idx_mid, element[1], element[2]])
                ]

        elif direction == RefinementDirection.Y_DIRECTION:
            # Refine primarily in Y direction
            # Find the edge most aligned with X direction (to split in Y)
            edge12_x = abs(v2[0] - v1[0])
            edge23_x = abs(v3[0] - v2[0])
            edge31_x = abs(v1[0] - v3[0])

            if edge12_x >= edge23_x and edge12_x >= edge31_x:
                # Split edge 1-2
                mid = (v1 + v2) / 2.0
                idx_mid = self._add_vertex_if_new(mid, vertices)
                new_elements = [
                    np.array([element[0], idx_mid, element[2]]),
                    np.array([idx_mid, element[1], element[2]])
                ]
            elif edge23_x >= edge31_x:
                # Split edge 2-3
                mid = (v2 + v3) / 2.0
                idx_mid = self._add_vertex_if_new(mid, vertices)
                new_elements = [
                    np.array([element[0], element[1], idx_mid]),
                    np.array([element[0], idx_mid, element[2]])
                ]
            else:
                # Split edge 3-1
                mid = (v3 + v1) / 2.0
                idx_mid = self._add_vertex_if_new(mid, vertices)
                new_elements = [
                    np.array([element[0], element[1], idx_mid]),
                    np.array([idx_mid, element[1], element[2]])
                ]
        else:
            # Adaptive direction - use isotropic refinement as fallback
            return self._refine_element_isotropic(element, vertices)

        return new_elements

    def _add_vertex_if_new(self, vertex: np.ndarray, vertices: list) -> int:
        """Add vertex if it doesn't already exist"""
        # Convert to list if numpy array
        if isinstance(vertices, np.ndarray):
            vertices = vertices.tolist()

        # Check if vertex already exists (within tolerance)
        for i, existing_vertex in enumerate(vertices):
            existing_vertex = np.array(existing_vertex)
            if np.linalg.norm(vertex - existing_vertex) < self.tolerance:
                return i

        # Add new vertex
        vertices.append(vertex.tolist())
        return len(vertices) - 1

    def _ensure_mesh_conformity(self, elements: np.ndarray, vertices: np.ndarray):
        """Ensure mesh conformity"""
        # Simple conformity check - ensure all vertex indices are valid
        max_vertex_idx = len(vertices) - 1
        for element in elements:
            for vertex_idx in element:
                if vertex_idx < 0 or vertex_idx > max_vertex_idx:
                    raise ValueError(f"Invalid vertex index {vertex_idx} in refined mesh")

        # Additional conformity checks could be added here
        # such as hanging node resolution, edge consistency, etc.


class AdvancedMeshRefinementVisualizer:
    """Visualization tools for advanced mesh refinement"""

    def __init__(self):
        self.figure_size = (12, 8)
        self.dpi = 100

    def plot_mesh_with_error(self, vertices: np.ndarray, elements: np.ndarray,
                           error_indicators: np.ndarray, title: str = "Mesh with Error Indicators"):
        """Plot mesh with error indicators"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size, dpi=self.dpi)

        # Plot mesh
        for element in elements:
            triangle = vertices[element]
            triangle = np.vstack([triangle, triangle[0]])  # Close the triangle
            ax1.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=0.5)

        ax1.scatter(vertices[:, 0], vertices[:, 1], c='red', s=10)
        ax1.set_title("Mesh Structure")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # Plot error indicators
        element_centers = np.array([np.mean(vertices[element], axis=0) for element in elements])
        scatter = ax2.scatter(element_centers[:, 0], element_centers[:, 1],
                            c=error_indicators, cmap='viridis', s=50)

        # Draw mesh outline
        for element in elements:
            triangle = vertices[element]
            triangle = np.vstack([triangle, triangle[0]])
            ax2.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=0.3, alpha=0.5)

        ax2.set_title("Error Indicators")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')

        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label='Error Magnitude')

        plt.suptitle(title)
        plt.tight_layout()
        return fig

    def plot_refinement_comparison(self, original_vertices: np.ndarray, original_elements: np.ndarray,
                                 refined_vertices: np.ndarray, refined_elements: np.ndarray,
                                 refinement_info: List[AdvancedRefinementInfo]):
        """Plot comparison between original and refined mesh"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size, dpi=self.dpi)

        # Plot original mesh
        for element in original_elements:
            triangle = original_vertices[element]
            triangle = np.vstack([triangle, triangle[0]])
            ax1.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=0.8)

        # Highlight elements to be refined
        for e, info in enumerate(refinement_info):
            if e < len(original_elements) and info.refine:
                element = original_elements[e]
                triangle = original_vertices[element]
                ax1.fill(triangle[:, 0], triangle[:, 1], 'red', alpha=0.3)

        ax1.scatter(original_vertices[:, 0], original_vertices[:, 1], c='blue', s=15)
        ax1.set_title(f"Original Mesh ({len(original_elements)} elements)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # Plot refined mesh
        for element in refined_elements:
            triangle = refined_vertices[element]
            triangle = np.vstack([triangle, triangle[0]])
            ax2.plot(triangle[:, 0], triangle[:, 1], 'g-', linewidth=0.5)

        ax2.scatter(refined_vertices[:, 0], refined_vertices[:, 1], c='green', s=10)
        ax2.set_title(f"Refined Mesh ({len(refined_elements)} elements)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')

        plt.suptitle("Mesh Refinement Comparison")
        plt.tight_layout()
        return fig

    def plot_mesh_quality_metrics(self, metrics: MeshQualityMetrics):
        """Plot mesh quality metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)

        # Angle distribution
        angles = [metrics.min_angle, (metrics.min_angle + metrics.max_angle) / 2, metrics.max_angle]
        ax1.bar(['Min', 'Avg', 'Max'], angles, color=['red', 'yellow', 'green'])
        ax1.set_title("Angle Distribution")
        ax1.set_ylabel("Angle (degrees)")
        ax1.grid(True, alpha=0.3)

        # Aspect ratio
        aspect_ratios = [metrics.min_aspect_ratio, metrics.max_aspect_ratio]
        ax2.bar(['Min', 'Max'], aspect_ratios, color=['green', 'red'])
        ax2.set_title("Aspect Ratio")
        ax2.set_ylabel("Ratio")
        ax2.grid(True, alpha=0.3)

        # Element size
        element_sizes = [metrics.min_element_size, metrics.max_element_size]
        ax3.bar(['Min', 'Max'], element_sizes, color=['blue', 'orange'])
        ax3.set_title("Element Size")
        ax3.set_ylabel("Size")
        ax3.grid(True, alpha=0.3)

        # Quality summary
        quality_metrics = [metrics.average_quality, metrics.mesh_regularity]
        ax4.bar(['Avg Quality', 'Regularity'], quality_metrics, color=['purple', 'cyan'])
        ax4.set_title("Quality Summary")
        ax4.set_ylabel("Quality Metric")
        ax4.grid(True, alpha=0.3)

        plt.suptitle("Mesh Quality Analysis")
        plt.tight_layout()
        return fig


def create_test_mesh(nx: int = 10, ny: int = 10, Lx: float = 1.0, Ly: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple test mesh for demonstration"""
    # Generate grid points
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    X, Y = np.meshgrid(x, y)

    vertices = np.column_stack([X.ravel(), Y.ravel()])

    # Generate triangular elements
    elements = []
    for i in range(nx):
        for j in range(ny):
            # Node indices
            n1 = i * (ny + 1) + j
            n2 = n1 + 1
            n3 = (i + 1) * (ny + 1) + j
            n4 = n3 + 1

            # Two triangles per quad
            elements.append([n1, n3, n2])
            elements.append([n2, n3, n4])

    return vertices, np.array(elements)


def create_test_solution(vertices: np.ndarray, solution_type: str = "gaussian") -> np.ndarray:
    """Create a test solution for demonstration"""
    if solution_type == "gaussian":
        # Gaussian peak
        center_x, center_y = 0.5, 0.5
        sigma = 0.2
        solution = np.exp(-((vertices[:, 0] - center_x)**2 + (vertices[:, 1] - center_y)**2) / (2 * sigma**2))
    elif solution_type == "linear":
        # Linear gradient
        solution = vertices[:, 0] + vertices[:, 1]
    elif solution_type == "step":
        # Step function
        solution = np.where(vertices[:, 0] > 0.5, 1.0, 0.0)
    else:
        # Default: sine wave
        solution = np.sin(2 * np.pi * vertices[:, 0]) * np.cos(2 * np.pi * vertices[:, 1])

    return solution
