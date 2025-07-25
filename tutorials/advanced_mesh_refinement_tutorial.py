"""
Advanced Mesh Refinement Tutorial

This tutorial demonstrates the advanced adaptive mesh refinement capabilities
of the SemiDGFEM simulator including:

1. Multiple error estimation strategies
2. Anisotropic refinement
3. Feature detection and boundary layer handling
4. Mesh quality analysis
5. Visualization tools

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import matplotlib.pyplot as plt
from advanced_mesh_refinement import (
    AdvancedAMRController, MeshRefinementExecutor, AdvancedMeshRefinementVisualizer,
    AdvancedErrorEstimatorType, RefinementDirection, AdvancedRefinementInfo,
    create_test_mesh, create_test_solution
)

def tutorial_1_basic_error_estimation():
    """Tutorial 1: Basic error estimation strategies"""
    print("="*60)
    print("Tutorial 1: Basic Error Estimation Strategies")
    print("="*60)
    
    # Create test mesh and solution
    vertices, elements = create_test_mesh(nx=12, ny=12)
    solution = create_test_solution(vertices, "gaussian")
    
    print(f"Created mesh with {len(vertices)} vertices and {len(elements)} elements")
    
    # Initialize AMR controller
    amr_controller = AdvancedAMRController()
    
    # Test different error estimators
    estimator_types = [
        AdvancedErrorEstimatorType.GRADIENT_RECOVERY,
        AdvancedErrorEstimatorType.HIERARCHICAL_BASIS,
        AdvancedErrorEstimatorType.PHYSICS_BASED,
        AdvancedErrorEstimatorType.HYBRID
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, estimator_type in enumerate(estimator_types):
        amr_controller.set_error_estimator(estimator_type)
        errors = amr_controller.estimate_error(solution, elements, vertices)
        
        print(f"\n{estimator_type.value.replace('_', ' ').title()}:")
        print(f"  Max error: {np.max(errors):.6f}")
        print(f"  Mean error: {np.mean(errors):.6f}")
        print(f"  Error std: {np.std(errors):.6f}")
        
        # Plot error distribution
        element_centers = np.array([np.mean(vertices[element], axis=0) for element in elements])
        scatter = axes[i].scatter(element_centers[:, 0], element_centers[:, 1], 
                                c=errors, cmap='viridis', s=30)
        
        # Draw mesh outline
        for element in elements:
            triangle = vertices[element]
            triangle = np.vstack([triangle, triangle[0]])
            axes[i].plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=0.2, alpha=0.3)
            
        axes[i].set_title(f"{estimator_type.value.replace('_', ' ').title()}")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].axis('equal')
        plt.colorbar(scatter, ax=axes[i], label='Error')
        
    plt.suptitle("Comparison of Error Estimation Strategies")
    plt.tight_layout()
    plt.savefig("tutorial_1_error_estimation.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return vertices, elements, solution, amr_controller

def tutorial_2_anisotropic_refinement():
    """Tutorial 2: Anisotropic refinement"""
    print("\n" + "="*60)
    print("Tutorial 2: Anisotropic Refinement")
    print("="*60)
    
    # Create mesh with anisotropic features
    vertices, elements = create_test_mesh(nx=10, ny=10)
    
    # Create solution with strong directional gradient
    x, y = vertices[:, 0], vertices[:, 1]
    solution = np.exp(-((x - 0.5)**2 / 0.01 + (y - 0.5)**2 / 0.1))  # Anisotropic Gaussian
    
    print(f"Created anisotropic solution on {len(elements)} element mesh")
    
    # Configure AMR for anisotropic refinement
    amr_controller = AdvancedAMRController()
    amr_controller.set_error_estimator(AdvancedErrorEstimatorType.GRADIENT_RECOVERY)
    amr_controller.set_anisotropic_parameters(True, max_anisotropy_ratio=5.0)
    amr_controller.set_refinement_parameters(0.2, 0.05, 1e-4)
    
    # Estimate errors and determine refinement
    errors = amr_controller.estimate_error(solution, elements, vertices)
    refinement_info = amr_controller.determine_refinement(errors, elements, vertices)
    
    # Analyze refinement decisions
    isotropic_count = sum(1 for info in refinement_info 
                         if info.refine and info.direction == RefinementDirection.ISOTROPIC)
    x_direction_count = sum(1 for info in refinement_info 
                           if info.refine and info.direction == RefinementDirection.X_DIRECTION)
    y_direction_count = sum(1 for info in refinement_info 
                           if info.refine and info.direction == RefinementDirection.Y_DIRECTION)
    
    print(f"\nRefinement Analysis:")
    print(f"  Total elements to refine: {sum(1 for info in refinement_info if info.refine)}")
    print(f"  Isotropic refinements: {isotropic_count}")
    print(f"  X-direction refinements: {x_direction_count}")
    print(f"  Y-direction refinements: {y_direction_count}")
    
    # Execute refinement
    executor = MeshRefinementExecutor()
    result = executor.execute_refinement(elements, vertices, refinement_info)
    
    if result.success:
        print(f"  Refinement successful!")
        print(f"  New mesh: {len(result.new_vertices)} vertices, {len(result.new_elements)} elements")
        
        # Visualize results
        visualizer = AdvancedMeshRefinementVisualizer()
        fig = visualizer.plot_refinement_comparison(
            vertices, elements, result.new_vertices, result.new_elements, refinement_info)
        plt.savefig("tutorial_2_anisotropic_refinement.png", dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print(f"  Refinement failed: {result.error_message}")
        
    return result

def tutorial_3_feature_detection():
    """Tutorial 3: Feature detection and adaptive refinement"""
    print("\n" + "="*60)
    print("Tutorial 3: Feature Detection and Adaptive Refinement")
    print("="*60)
    
    # Create mesh with sharp features
    vertices, elements = create_test_mesh(nx=15, ny=15)
    
    # Create solution with multiple features
    x, y = vertices[:, 0], vertices[:, 1]
    
    # Combine multiple features
    gaussian_peak = np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / 0.02)
    step_function = np.where((x > 0.6) & (y > 0.6), 1.0, 0.0)
    sine_wave = 0.3 * np.sin(10 * np.pi * x) * np.sin(10 * np.pi * y)
    
    solution = gaussian_peak + step_function + sine_wave
    
    print(f"Created multi-feature solution with:")
    print(f"  - Gaussian peak at (0.3, 0.3)")
    print(f"  - Step function at x,y > 0.6")
    print(f"  - High-frequency sine waves")
    
    # Configure AMR for feature detection
    amr_controller = AdvancedAMRController()
    amr_controller.set_error_estimator(AdvancedErrorEstimatorType.FEATURE_DETECTION)
    amr_controller.set_feature_detection_parameters(feature_threshold=0.3, gradient_threshold=1e-2)
    amr_controller.set_refinement_parameters(0.25, 0.1, 1e-3)
    
    # Detect features
    feature_indicators = amr_controller.detect_features(solution, elements, vertices)
    errors = amr_controller.estimate_error(solution, elements, vertices)
    
    print(f"\nFeature Detection Results:")
    print(f"  Max feature strength: {np.max(feature_indicators):.6f}")
    print(f"  Mean feature strength: {np.mean(feature_indicators):.6f}")
    print(f"  Elements with strong features: {np.sum(feature_indicators > 0.3)}")
    
    # Determine refinement
    refinement_info = amr_controller.determine_refinement(errors, elements, vertices)
    
    # Visualize features and refinement
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original solution
    element_centers = np.array([np.mean(vertices[element], axis=0) for element in elements])
    element_solutions = np.array([np.mean(solution[element]) for element in elements])
    
    scatter1 = ax1.scatter(element_centers[:, 0], element_centers[:, 1], 
                          c=element_solutions, cmap='viridis', s=40)
    ax1.set_title("Original Solution")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.axis('equal')
    plt.colorbar(scatter1, ax=ax1, label='Solution Value')
    
    # Plot feature indicators
    scatter2 = ax2.scatter(element_centers[:, 0], element_centers[:, 1], 
                          c=feature_indicators, cmap='plasma', s=40)
    ax2.set_title("Feature Indicators")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.axis('equal')
    plt.colorbar(scatter2, ax=ax2, label='Feature Strength')
    
    # Plot refinement decisions
    refinement_flags = np.array([1 if info.refine else 0 for info in refinement_info])
    scatter3 = ax3.scatter(element_centers[:, 0], element_centers[:, 1], 
                          c=refinement_flags, cmap='RdYlBu', s=40)
    ax3.set_title("Refinement Decisions")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.axis('equal')
    plt.colorbar(scatter3, ax=ax3, label='Refine (1) / Keep (0)')
    
    plt.tight_layout()
    plt.savefig("tutorial_3_feature_detection.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return solution, feature_indicators, refinement_info

def tutorial_4_mesh_quality_analysis():
    """Tutorial 4: Mesh quality analysis and improvement"""
    print("\n" + "="*60)
    print("Tutorial 4: Mesh Quality Analysis and Improvement")
    print("="*60)
    
    # Create initial mesh
    vertices, elements = create_test_mesh(nx=8, ny=8)
    solution = create_test_solution(vertices, "gaussian")
    
    # Analyze initial mesh quality
    amr_controller = AdvancedAMRController()
    initial_metrics = amr_controller.analyze_mesh_quality(elements, vertices)
    
    print(f"Initial Mesh Quality:")
    print(f"  Elements: {len(elements)}")
    print(f"  Min angle: {initial_metrics.min_angle:.1f}°")
    print(f"  Max angle: {initial_metrics.max_angle:.1f}°")
    print(f"  Max aspect ratio: {initial_metrics.max_aspect_ratio:.2f}")
    print(f"  Average quality: {initial_metrics.average_quality:.3f}")
    print(f"  Mesh regularity: {initial_metrics.mesh_regularity:.3f}")
    print(f"  Poor quality elements: {initial_metrics.num_poor_quality_elements}")
    
    # Perform adaptive refinement
    amr_controller.set_error_estimator(AdvancedErrorEstimatorType.HYBRID)
    amr_controller.set_anisotropic_parameters(True, 3.0, 0.1)
    amr_controller.set_refinement_parameters(0.3, 0.1, 1e-3)
    
    errors = amr_controller.estimate_error(solution, elements, vertices)
    refinement_info = amr_controller.determine_refinement(errors, elements, vertices)
    
    # Execute refinement
    executor = MeshRefinementExecutor()
    result = executor.execute_refinement(elements, vertices, refinement_info)
    
    if result.success:
        # Analyze refined mesh quality
        refined_metrics = amr_controller.analyze_mesh_quality(result.new_elements, result.new_vertices)
        
        print(f"\nRefined Mesh Quality:")
        print(f"  Elements: {len(result.new_elements)}")
        print(f"  Min angle: {refined_metrics.min_angle:.1f}°")
        print(f"  Max angle: {refined_metrics.max_angle:.1f}°")
        print(f"  Max aspect ratio: {refined_metrics.max_aspect_ratio:.2f}")
        print(f"  Average quality: {refined_metrics.average_quality:.3f}")
        print(f"  Mesh regularity: {refined_metrics.mesh_regularity:.3f}")
        print(f"  Poor quality elements: {refined_metrics.num_poor_quality_elements}")
        
        # Quality improvement analysis
        quality_improvement = refined_metrics.average_quality - initial_metrics.average_quality
        regularity_improvement = refined_metrics.mesh_regularity - initial_metrics.mesh_regularity
        
        print(f"\nQuality Improvement:")
        print(f"  Average quality change: {quality_improvement:+.3f}")
        print(f"  Regularity change: {regularity_improvement:+.3f}")
        print(f"  Element count increase: {len(result.new_elements) - len(elements)}")
        
        # Visualize quality metrics
        visualizer = AdvancedMeshRefinementVisualizer()
        
        fig1 = visualizer.plot_mesh_quality_metrics(initial_metrics)
        fig1.suptitle("Initial Mesh Quality Metrics")
        plt.savefig("tutorial_4_initial_quality.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        fig2 = visualizer.plot_mesh_quality_metrics(refined_metrics)
        fig2.suptitle("Refined Mesh Quality Metrics")
        plt.savefig("tutorial_4_refined_quality.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        return initial_metrics, refined_metrics, result
    else:
        print(f"Refinement failed: {result.error_message}")
        return initial_metrics, None, None

def tutorial_5_complete_workflow():
    """Tutorial 5: Complete adaptive mesh refinement workflow"""
    print("\n" + "="*60)
    print("Tutorial 5: Complete Adaptive Mesh Refinement Workflow")
    print("="*60)
    
    # Create initial mesh
    vertices, elements = create_test_mesh(nx=12, ny=12)
    
    # Create complex solution (semiconductor device simulation)
    x, y = vertices[:, 0], vertices[:, 1]
    
    # Simulate potential distribution in a semiconductor device
    # Junction at x = 0.5, depletion regions, contact regions
    potential = np.zeros_like(x)
    
    # P-region (x < 0.5)
    p_region = x < 0.5
    potential[p_region] = -0.5 * np.exp(-(x[p_region] - 0.2)**2 / 0.01)
    
    # N-region (x > 0.5)
    n_region = x > 0.5
    potential[n_region] = 0.5 * np.exp(-(x[n_region] - 0.8)**2 / 0.01)
    
    # Junction region (sharp transition)
    junction_region = np.abs(x - 0.5) < 0.05
    potential[junction_region] = 10 * (x[junction_region] - 0.5)
    
    print(f"Created semiconductor device simulation:")
    print(f"  Initial mesh: {len(vertices)} vertices, {len(elements)} elements")
    print(f"  Potential range: [{np.min(potential):.3f}, {np.max(potential):.3f}] V")
    
    # Configure comprehensive AMR
    amr_controller = AdvancedAMRController()
    amr_controller.set_error_estimator(AdvancedErrorEstimatorType.HYBRID)
    amr_controller.set_physics_parameters("semiconductor", ["potential"])
    amr_controller.set_anisotropic_parameters(True, 5.0, 0.05)
    amr_controller.set_feature_detection_parameters(0.2, 1e-2)
    amr_controller.set_refinement_parameters(0.25, 0.1, 1e-4)
    
    # Perform multiple refinement cycles
    current_vertices = vertices.copy()
    current_elements = elements.copy()
    current_solution = potential.copy()
    
    refinement_history = []
    
    for cycle in range(3):
        print(f"\nRefinement Cycle {cycle + 1}:")
        
        # Error estimation
        errors = amr_controller.estimate_error(current_solution, current_elements, current_vertices)
        
        # Feature detection
        features = amr_controller.detect_features(current_solution, current_elements, current_vertices)
        
        # Quality analysis
        quality_metrics = amr_controller.analyze_mesh_quality(current_elements, current_vertices)
        
        print(f"  Current mesh: {len(current_vertices)} vertices, {len(current_elements)} elements")
        print(f"  Max error: {np.max(errors):.6f}")
        print(f"  Max feature strength: {np.max(features):.6f}")
        print(f"  Average quality: {quality_metrics.average_quality:.3f}")
        
        # Determine refinement
        refinement_info = amr_controller.determine_refinement(
            errors, current_elements, current_vertices)
        
        refine_count = sum(1 for info in refinement_info if info.refine)
        print(f"  Elements to refine: {refine_count}")
        
        if refine_count == 0:
            print("  No refinement needed, stopping.")
            break
            
        # Execute refinement
        executor = MeshRefinementExecutor()
        result = executor.execute_refinement(current_elements, current_vertices, refinement_info)
        
        if result.success:
            # Transfer solution
            current_solution = executor.transfer_solution(current_solution, result)
            current_vertices = result.new_vertices
            current_elements = result.new_elements
            
            refinement_history.append({
                'cycle': cycle + 1,
                'vertices': len(current_vertices),
                'elements': len(current_elements),
                'max_error': np.max(errors),
                'quality': quality_metrics.average_quality
            })
            
            print(f"  Refinement successful!")
        else:
            print(f"  Refinement failed: {result.error_message}")
            break
    
    # Final analysis
    final_errors = amr_controller.estimate_error(current_solution, current_elements, current_vertices)
    final_quality = amr_controller.analyze_mesh_quality(current_elements, current_vertices)
    stats = amr_controller.get_refinement_statistics()
    
    print(f"\nFinal Results:")
    print(f"  Final mesh: {len(current_vertices)} vertices, {len(current_elements)} elements")
    print(f"  Mesh size increase: {len(current_elements) / len(elements):.1f}x")
    print(f"  Final max error: {np.max(final_errors):.6f}")
    print(f"  Error reduction: {np.max(errors) / np.max(final_errors):.1f}x")
    print(f"  Final quality: {final_quality.average_quality:.3f}")
    print(f"  Total refinements: {stats.total_elements_refined}")
    print(f"  Anisotropic refinements: {stats.anisotropic_refinements}")
    
    # Visualize final result
    visualizer = AdvancedMeshRefinementVisualizer()
    fig = visualizer.plot_mesh_with_error(current_vertices, current_elements, final_errors,
                                        "Final Adaptive Mesh with Error Distribution")
    plt.savefig("tutorial_5_final_mesh.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return refinement_history, current_vertices, current_elements, current_solution

def main():
    """Run all tutorials"""
    print("Advanced Mesh Refinement Tutorial")
    print("SemiDGFEM Simulator - Next Generation Improvements")
    print("Author: Dr. Mazharuddin Mohammed")
    
    try:
        # Tutorial 1: Error estimation
        vertices, elements, solution, amr_controller = tutorial_1_basic_error_estimation()
        
        # Tutorial 2: Anisotropic refinement
        result = tutorial_2_anisotropic_refinement()
        
        # Tutorial 3: Feature detection
        solution, features, refinement_info = tutorial_3_feature_detection()
        
        # Tutorial 4: Quality analysis
        initial_quality, refined_quality, refinement_result = tutorial_4_mesh_quality_analysis()
        
        # Tutorial 5: Complete workflow
        history, final_vertices, final_elements, final_solution = tutorial_5_complete_workflow()
        
        print("\n" + "="*60)
        print("All tutorials completed successfully!")
        print("Generated visualization files:")
        print("  - tutorial_1_error_estimation.png")
        print("  - tutorial_2_anisotropic_refinement.png")
        print("  - tutorial_3_feature_detection.png")
        print("  - tutorial_4_initial_quality.png")
        print("  - tutorial_4_refined_quality.png")
        print("  - tutorial_5_final_mesh.png")
        print("="*60)
        
    except Exception as e:
        print(f"Tutorial failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
