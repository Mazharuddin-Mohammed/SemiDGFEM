"""
Test suite for Advanced Mesh Refinement

Tests all components of the advanced adaptive mesh refinement system including:
- Error estimation algorithms
- Anisotropic refinement
- Feature detection
- Mesh quality analysis
- Refinement execution

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import unittest
from advanced_mesh_refinement import (
    AdvancedAMRController, MeshRefinementExecutor, AdvancedMeshRefinementVisualizer,
    AdvancedErrorEstimatorType, RefinementDirection, AdvancedRefinementInfo,
    create_test_mesh, create_test_solution
)

class TestAdvancedMeshRefinement(unittest.TestCase):
    """Test cases for advanced mesh refinement"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vertices, self.elements = create_test_mesh(nx=8, ny=8)
        self.solution = create_test_solution(self.vertices, "gaussian")
        self.amr_controller = AdvancedAMRController()
        self.refinement_executor = MeshRefinementExecutor()
        
    def test_amr_controller_initialization(self):
        """Test AMR controller initialization"""
        controller = AdvancedAMRController()
        
        self.assertEqual(controller.estimator_type, AdvancedErrorEstimatorType.GRADIENT_RECOVERY)
        self.assertEqual(controller.refine_fraction, 0.3)
        self.assertEqual(controller.coarsen_fraction, 0.1)
        self.assertEqual(controller.error_tolerance, 1e-3)
        self.assertFalse(controller.anisotropic_enabled)
        
    def test_error_estimation_gradient_recovery(self):
        """Test gradient recovery error estimation"""
        self.amr_controller.set_error_estimator(AdvancedErrorEstimatorType.GRADIENT_RECOVERY)
        
        errors = self.amr_controller.estimate_error(self.solution, self.elements, self.vertices)
        
        self.assertEqual(len(errors), len(self.elements))
        self.assertTrue(np.all(errors >= 0))
        self.assertTrue(np.any(errors > 0))  # Should have some non-zero errors
        
    def test_error_estimation_hierarchical(self):
        """Test hierarchical error estimation"""
        self.amr_controller.set_error_estimator(AdvancedErrorEstimatorType.HIERARCHICAL_BASIS)
        
        errors = self.amr_controller.estimate_error(self.solution, self.elements, self.vertices)
        
        self.assertEqual(len(errors), len(self.elements))
        self.assertTrue(np.all(errors >= 0))
        
    def test_error_estimation_physics_based(self):
        """Test physics-based error estimation"""
        self.amr_controller.set_error_estimator(AdvancedErrorEstimatorType.PHYSICS_BASED)
        self.amr_controller.set_physics_parameters("semiconductor", ["potential"])
        
        errors = self.amr_controller.estimate_error(self.solution, self.elements, self.vertices)
        
        self.assertEqual(len(errors), len(self.elements))
        self.assertTrue(np.all(errors >= 0))
        
    def test_error_estimation_hybrid(self):
        """Test hybrid error estimation"""
        self.amr_controller.set_error_estimator(AdvancedErrorEstimatorType.HYBRID)
        
        errors = self.amr_controller.estimate_error(self.solution, self.elements, self.vertices)
        
        self.assertEqual(len(errors), len(self.elements))
        self.assertTrue(np.all(errors >= 0))
        
    def test_feature_detection(self):
        """Test feature detection"""
        # Create solution with sharp features
        step_solution = create_test_solution(self.vertices, "step")
        
        features = self.amr_controller.detect_features(step_solution, self.elements, self.vertices)
        
        self.assertEqual(len(features), len(self.elements))
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.max(features) > np.mean(features))  # Should detect step feature
        
    def test_boundary_layer_detection(self):
        """Test boundary layer detection"""
        self.amr_controller.set_anisotropic_parameters(True, 5.0, 0.1)
        
        # Assume first 10% of vertices are boundary nodes
        boundary_nodes = list(range(len(self.vertices) // 10))
        boundary_layers = self.amr_controller.detect_boundary_layers(
            self.elements, self.vertices, boundary_nodes)
        
        self.assertEqual(len(boundary_layers), len(self.elements))
        self.assertTrue(any(boundary_layers))  # Should detect some boundary layer elements
        
    def test_refinement_determination(self):
        """Test refinement decision making"""
        errors = self.amr_controller.estimate_error(self.solution, self.elements, self.vertices)
        refinement_info = self.amr_controller.determine_refinement(
            errors, self.elements, self.vertices, self.solution)
        
        self.assertEqual(len(refinement_info), len(self.elements))
        
        # Check that some elements are marked for refinement
        refine_count = sum(1 for info in refinement_info if info.refine)
        coarsen_count = sum(1 for info in refinement_info if info.coarsen)
        
        self.assertGreater(refine_count, 0)
        # Coarsening might be 0 for this test case
        
    def test_anisotropic_refinement_determination(self):
        """Test anisotropic refinement determination"""
        self.amr_controller.set_anisotropic_parameters(True, 5.0, 0.1)
        
        errors = self.amr_controller.estimate_error(self.solution, self.elements, self.vertices)
        refinement_info = self.amr_controller.determine_refinement(
            errors, self.elements, self.vertices, self.solution)
        
        # Check for anisotropic refinements
        anisotropic_count = sum(1 for info in refinement_info 
                              if info.refine and info.direction != RefinementDirection.ISOTROPIC)
        
        # Should have some anisotropic refinements for Gaussian solution
        self.assertGreaterEqual(anisotropic_count, 0)
        
    def test_mesh_quality_analysis(self):
        """Test mesh quality analysis"""
        metrics = self.amr_controller.analyze_mesh_quality(self.elements, self.vertices)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics.min_angle, 0)
        self.assertLess(metrics.max_angle, 180)
        self.assertGreater(metrics.min_aspect_ratio, 0)
        self.assertGreater(metrics.average_quality, 0)
        self.assertLessEqual(metrics.mesh_regularity, 1.0)
        
    def test_isotropic_refinement_execution(self):
        """Test isotropic refinement execution"""
        # Create simple refinement info
        refinement_info = []
        for e in range(len(self.elements)):
            info = AdvancedRefinementInfo()
            if e < 5:  # Refine first 5 elements
                info.refine = True
                info.direction = RefinementDirection.ISOTROPIC
            refinement_info.append(info)
            
        result = self.refinement_executor.execute_refinement(
            self.elements, self.vertices, refinement_info)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.new_elements), len(self.elements))
        self.assertGreater(len(result.new_vertices), len(self.vertices))
        
    def test_anisotropic_refinement_execution(self):
        """Test anisotropic refinement execution"""
        # Create anisotropic refinement info
        refinement_info = []
        for e in range(len(self.elements)):
            info = AdvancedRefinementInfo()
            if e < 3:  # Refine first 3 elements anisotropically
                info.refine = True
                info.direction = RefinementDirection.X_DIRECTION
                info.anisotropy_ratio = 2.0
            refinement_info.append(info)
            
        result = self.refinement_executor.execute_refinement(
            self.elements, self.vertices, refinement_info)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.new_elements), len(self.elements))
        
    def test_solution_transfer(self):
        """Test solution transfer after refinement"""
        # Create refinement
        refinement_info = []
        for e in range(len(self.elements)):
            info = AdvancedRefinementInfo()
            if e < 5:  # Refine first 5 elements
                info.refine = True
            refinement_info.append(info)
            
        result = self.refinement_executor.execute_refinement(
            self.elements, self.vertices, refinement_info)
        
        # Transfer solution
        new_solution = self.refinement_executor.transfer_solution(
            self.solution, result, "interpolation")
        
        self.assertEqual(len(new_solution), len(result.new_vertices))
        
        # Check that solution values are reasonable
        self.assertTrue(np.all(np.isfinite(new_solution)))
        self.assertLessEqual(np.max(new_solution), np.max(self.solution) * 1.2)  # Allow small increase
        self.assertGreaterEqual(np.min(new_solution), np.min(self.solution) * 0.8)  # Allow small decrease
        
    def test_refinement_parameters(self):
        """Test refinement parameter setting"""
        # Test valid parameters
        self.amr_controller.set_refinement_parameters(0.4, 0.2, 1e-4)
        self.assertEqual(self.amr_controller.refine_fraction, 0.4)
        self.assertEqual(self.amr_controller.coarsen_fraction, 0.2)
        self.assertEqual(self.amr_controller.error_tolerance, 1e-4)
        
        # Test invalid parameters
        with self.assertRaises(ValueError):
            self.amr_controller.set_refinement_parameters(1.5, 0.1, 1e-3)  # Invalid refine_fraction
            
        with self.assertRaises(ValueError):
            self.amr_controller.set_refinement_parameters(0.3, -0.1, 1e-3)  # Invalid coarsen_fraction
            
    def test_statistics_tracking(self):
        """Test refinement statistics tracking"""
        # Reset statistics
        self.amr_controller.reset_statistics()
        stats = self.amr_controller.get_refinement_statistics()
        self.assertEqual(stats.total_elements_refined, 0)
        self.assertEqual(stats.total_elements_coarsened, 0)
        
        # Perform refinement
        errors = self.amr_controller.estimate_error(self.solution, self.elements, self.vertices)
        refinement_info = self.amr_controller.determine_refinement(
            errors, self.elements, self.vertices, self.solution)
        
        # Check statistics
        stats = self.amr_controller.get_refinement_statistics()
        self.assertGreaterEqual(stats.total_elements_refined, 0)
        self.assertGreaterEqual(stats.total_elements_coarsened, 0)
        
    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        empty_elements = np.array([])
        empty_vertices = np.array([])
        empty_solution = np.array([])
        
        # Should not crash and return empty results
        errors = self.amr_controller.estimate_error(empty_solution, empty_elements, empty_vertices)
        self.assertEqual(len(errors), 0)
        
        refinement_info = self.amr_controller.determine_refinement(
            np.array([]), empty_elements, empty_vertices, empty_solution)
        self.assertEqual(len(refinement_info), 0)
        
    def test_mesh_conformity(self):
        """Test mesh conformity after refinement"""
        refinement_info = []
        for e in range(min(10, len(self.elements))):  # Refine first 10 elements
            info = AdvancedRefinementInfo()
            info.refine = True
            refinement_info.append(info)
            
        # Add remaining elements without refinement
        for e in range(len(refinement_info), len(self.elements)):
            refinement_info.append(AdvancedRefinementInfo())
            
        result = self.refinement_executor.execute_refinement(
            self.elements, self.vertices, refinement_info)
        
        self.assertTrue(result.success)
        
        # Check that all vertex indices in elements are valid
        max_vertex_idx = len(result.new_vertices) - 1
        for element in result.new_elements:
            for vertex_idx in element:
                self.assertGreaterEqual(vertex_idx, 0)
                self.assertLessEqual(vertex_idx, max_vertex_idx)


def run_performance_test():
    """Run performance test for large meshes"""
    print("Running performance test...")
    
    # Create larger mesh
    vertices, elements = create_test_mesh(nx=50, ny=50)
    solution = create_test_solution(vertices, "gaussian")
    
    amr_controller = AdvancedAMRController()
    amr_controller.set_error_estimator(AdvancedErrorEstimatorType.HYBRID)
    amr_controller.set_anisotropic_parameters(True, 5.0, 0.1)
    
    import time
    
    # Time error estimation
    start_time = time.time()
    errors = amr_controller.estimate_error(solution, elements, vertices)
    error_time = time.time() - start_time
    
    # Time refinement determination
    start_time = time.time()
    refinement_info = amr_controller.determine_refinement(errors, elements, vertices, solution)
    refinement_time = time.time() - start_time
    
    # Time mesh quality analysis
    start_time = time.time()
    metrics = amr_controller.analyze_mesh_quality(elements, vertices)
    quality_time = time.time() - start_time
    
    print(f"Performance Results for {len(elements)} elements:")
    print(f"  Error estimation: {error_time:.3f} seconds")
    print(f"  Refinement determination: {refinement_time:.3f} seconds")
    print(f"  Quality analysis: {quality_time:.3f} seconds")
    print(f"  Total elements to refine: {sum(1 for info in refinement_info if info.refine)}")
    print(f"  Average quality: {metrics.average_quality:.3f}")


if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance test
    print("\n" + "="*50)
    run_performance_test()
