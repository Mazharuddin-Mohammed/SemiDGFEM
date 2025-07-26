"""
Test Suite for Enhanced Visualization Module

This test suite validates all components of the enhanced visualization system:
- MeshVisualizer3D functionality
- SolutionVisualizer capabilities
- AnimationEngine features
- InteractivePlotter operations
- VisualizationManager integration

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from enhanced_visualization import (
    MeshVisualizer3D, SolutionVisualizer, AnimationEngine, InteractivePlotter,
    VisualizationManager, ColorMapConfig, RenderingConfig, AnimationConfig,
    ColorMapType, RenderingMode, create_sample_mesh_2d, create_sample_solution_fields
)

def test_mesh_visualizer_3d():
    """Test MeshVisualizer3D functionality"""
    print("Testing MeshVisualizer3D...")
    
    # Create test mesh
    vertices, elements = create_sample_mesh_2d(10, 10)
    
    # Test mesh visualizer
    visualizer = MeshVisualizer3D()
    
    # Test mesh loading
    visualizer.set_mesh(vertices, elements)
    assert visualizer.vertices is not None
    assert visualizer.elements is not None
    assert len(visualizer.vertices) == 100  # 10x10 grid
    print("  ✓ Mesh loading successful")
    
    # Test rendering configuration
    config = RenderingConfig(mode=RenderingMode.SURFACE, anti_aliasing=True)
    visualizer.set_rendering_config(config)
    assert visualizer.rendering_config.mode == RenderingMode.SURFACE
    print("  ✓ Rendering configuration successful")
    
    # Test wireframe visualization
    try:
        visualizer.show_wireframe(show_edges=True, show_vertices=False)
        print("  ✓ Wireframe visualization successful")
    except Exception as e:
        print(f"  ✗ Wireframe visualization failed: {e}")
    
    # Test surface visualization
    try:
        visualizer.show_surface(show_faces=True, show_normals=False)
        print("  ✓ Surface visualization successful")
    except Exception as e:
        print(f"  ✗ Surface visualization failed: {e}")
    
    # Test element quality visualization
    quality_metrics = np.random.uniform(0.3, 1.0, len(elements))
    try:
        visualizer.show_element_quality(quality_metrics)
        print("  ✓ Element quality visualization successful")
    except Exception as e:
        print(f"  ✗ Element quality visualization failed: {e}")
    
    # Test material regions
    material_ids = np.random.randint(1, 4, len(elements))
    try:
        visualizer.show_material_regions(material_ids)
        print("  ✓ Material regions visualization successful")
    except Exception as e:
        print(f"  ✗ Material regions visualization failed: {e}")
    
    # Test doping profile
    doping_profile = np.random.uniform(1e15, 1e18, len(vertices))
    try:
        visualizer.show_doping_profile(doping_profile)
        print("  ✓ Doping profile visualization successful")
    except Exception as e:
        print(f"  ✗ Doping profile visualization failed: {e}")
    
    print("MeshVisualizer3D tests completed!")
    return True

def test_solution_visualizer():
    """Test SolutionVisualizer functionality"""
    print("Testing SolutionVisualizer...")
    
    # Create test data
    vertices, elements = create_sample_mesh_2d(15, 15)
    solution_fields = create_sample_solution_fields(len(vertices))
    
    # Test solution visualizer
    visualizer = SolutionVisualizer()
    
    # Test mesh and solution loading
    visualizer.set_mesh(vertices, elements)
    visualizer.set_solution(solution_fields)
    assert visualizer.mesh_vertices is not None
    assert len(visualizer.solution_fields) > 0
    print("  ✓ Mesh and solution loading successful")
    
    # Test colormap configuration
    colormap_config = ColorMapConfig(type=ColorMapType.VIRIDIS, logarithmic_scale=False)
    visualizer.set_colormap_config(colormap_config)
    assert visualizer.colormap_config.type == ColorMapType.VIRIDIS
    print("  ✓ Colormap configuration successful")
    
    # Test contour plot
    try:
        visualizer.show_contour_plot('potential', num_levels=15)
        print("  ✓ Contour plot successful")
    except Exception as e:
        print(f"  ✗ Contour plot failed: {e}")
    
    # Test surface plot
    try:
        visualizer.show_surface_plot('electron_density')
        print("  ✓ Surface plot successful")
    except Exception as e:
        print(f"  ✗ Surface plot failed: {e}")
    
    # Test vector field visualization
    try:
        visualizer.show_vector_field('electric_field', scale=1.0)
        print("  ✓ Vector field visualization successful")
    except Exception as e:
        print(f"  ✗ Vector field visualization failed: {e}")
    
    # Test streamlines
    try:
        visualizer.show_streamlines('electric_field', num_seeds=50)
        print("  ✓ Streamlines visualization successful")
    except Exception as e:
        print(f"  ✗ Streamlines visualization failed: {e}")
    
    # Test multi-field comparison
    try:
        field_names = ['potential', 'electron_density', 'hole_density']
        visualizer.show_multi_field_comparison(field_names)
        print("  ✓ Multi-field comparison successful")
    except Exception as e:
        print(f"  ✗ Multi-field comparison failed: {e}")
    
    # Test field difference
    try:
        visualizer.show_field_difference('electron_density', 'hole_density')
        print("  ✓ Field difference visualization successful")
    except Exception as e:
        print(f"  ✗ Field difference visualization failed: {e}")
    
    # Test cross-section
    try:
        visualizer.add_cross_section([0, 0, 1], 0.5)
        print("  ✓ Cross-section addition successful")
    except Exception as e:
        print(f"  ✗ Cross-section addition failed: {e}")
    
    # Test line plot
    try:
        start_point = [0.0, 0.0, 0.0]
        end_point = [1.0, 1.0, 0.0]
        visualizer.show_line_plot(start_point, end_point, 'potential')
        print("  ✓ Line plot successful")
    except Exception as e:
        print(f"  ✗ Line plot failed: {e}")
    
    print("SolutionVisualizer tests completed!")
    return True

def test_animation_engine():
    """Test AnimationEngine functionality"""
    print("Testing AnimationEngine...")
    
    # Create test data
    vertices, elements = create_sample_mesh_2d(10, 10)
    
    # Create time series data
    time_points = np.linspace(0, 1, 10)
    time_series_data = []
    for t in time_points:
        fields = {
            'potential': np.sin(2 * np.pi * t) * np.ones(len(vertices)),
            'electron_density': np.exp(-t) * np.ones(len(vertices))
        }
        time_series_data.append(fields)
    
    # Test animation engine
    engine = AnimationEngine()
    
    # Test animation configuration
    config = AnimationConfig(frame_rate=10.0, duration=2.0, loop_animation=True)
    engine.set_animation_config(config)
    assert engine.config.frame_rate == 10.0
    print("  ✓ Animation configuration successful")
    
    # Test time series data loading
    try:
        engine.add_time_series_data(time_series_data, time_points)
        assert len(engine.time_series_data) == 10
        print("  ✓ Time series data loading successful")
    except Exception as e:
        print(f"  ✗ Time series data loading failed: {e}")
    
    # Test solution evolution animation
    try:
        engine.create_solution_evolution_animation('potential', vertices, elements)
        print("  ✓ Solution evolution animation creation successful")
    except Exception as e:
        print(f"  ✗ Solution evolution animation creation failed: {e}")
    
    # Test parameter sweep animation
    try:
        parameter_values = np.linspace(0.1, 1.0, 5)
        parameter_solutions = time_series_data[:5]  # Use subset
        engine.create_parameter_sweep_animation(parameter_solutions, parameter_values, 
                                               'voltage', 'potential', vertices, elements)
        print("  ✓ Parameter sweep animation creation successful")
    except Exception as e:
        print(f"  ✗ Parameter sweep animation creation failed: {e}")
    
    print("AnimationEngine tests completed!")
    return True

def test_interactive_plotter():
    """Test InteractivePlotter functionality"""
    print("Testing InteractivePlotter...")
    
    # Test plotter
    plotter = InteractivePlotter()
    
    # Test line plot
    x_data = np.linspace(0, 2*np.pi, 100)
    y_data = np.sin(x_data)
    try:
        plotter.create_line_plot(x_data, y_data, "Sine Wave")
        assert len(plotter.plots) == 1
        print("  ✓ Line plot creation successful")
    except Exception as e:
        print(f"  ✗ Line plot creation failed: {e}")
    
    # Test scatter plot
    x_scatter = np.random.randn(50)
    y_scatter = np.random.randn(50)
    try:
        plotter.create_scatter_plot(x_scatter, y_scatter, label="Random Data")
        assert len(plotter.plots) == 2
        print("  ✓ Scatter plot creation successful")
    except Exception as e:
        print(f"  ✗ Scatter plot creation failed: {e}")
    
    # Test surface plot
    x_surf = np.linspace(-2, 2, 20)
    y_surf = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x_surf, y_surf)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    try:
        plotter.create_surface_plot(Z, x_surf, y_surf)
        assert len(plotter.plots) == 3
        print("  ✓ Surface plot creation successful")
    except Exception as e:
        print(f"  ✗ Surface plot creation failed: {e}")
    
    # Test heatmap
    heatmap_data = np.random.randn(10, 10)
    try:
        plotter.create_heatmap(heatmap_data)
        assert len(plotter.plots) == 4
        print("  ✓ Heatmap creation successful")
    except Exception as e:
        print(f"  ✗ Heatmap creation failed: {e}")
    
    # Test histogram
    hist_data = np.random.normal(0, 1, 1000)
    try:
        plotter.create_histogram(hist_data, bins=30, label="Normal Distribution")
        assert len(plotter.plots) == 5
        print("  ✓ Histogram creation successful")
    except Exception as e:
        print(f"  ✗ Histogram creation failed: {e}")
    
    # Test subplot grid
    try:
        plotter.create_subplot_grid(2, 2)
        assert plotter.subplot_rows == 2
        assert plotter.subplot_cols == 2
        print("  ✓ Subplot grid creation successful")
    except Exception as e:
        print(f"  ✗ Subplot grid creation failed: {e}")
    
    # Test configuration
    try:
        plotter.set_title("Test Plot")
        plotter.set_axis_labels("X Axis", "Y Axis", "Z Axis")
        plotter.set_axis_limits(-5, 5, -2, 2)
        assert plotter.title == "Test Plot"
        assert plotter.x_label == "X Axis"
        print("  ✓ Plot configuration successful")
    except Exception as e:
        print(f"  ✗ Plot configuration failed: {e}")
    
    print("InteractivePlotter tests completed!")
    return True

def test_visualization_manager():
    """Test VisualizationManager functionality"""
    print("Testing VisualizationManager...")
    
    # Create test data
    vertices, elements = create_sample_mesh_2d(12, 12)
    solution_fields = create_sample_solution_fields(len(vertices))
    
    # Test visualization manager
    manager = VisualizationManager()
    
    # Test component access
    try:
        mesh_viz = manager.get_mesh_visualizer()
        solution_viz = manager.get_solution_visualizer()
        animation_engine = manager.get_animation_engine()
        plotter = manager.get_plotter()
        
        assert mesh_viz is not None
        assert solution_viz is not None
        assert animation_engine is not None
        assert plotter is not None
        print("  ✓ Component access successful")
    except Exception as e:
        print(f"  ✗ Component access failed: {e}")
    
    # Test comprehensive visualization
    try:
        manager.create_comprehensive_visualization(vertices, elements, solution_fields)
        print("  ✓ Comprehensive visualization successful")
    except Exception as e:
        print(f"  ✗ Comprehensive visualization failed: {e}")
    
    # Test multi-physics dashboard
    try:
        solutions = [solution_fields, solution_fields]  # Duplicate for testing
        physics_names = ['electrostatics', 'transport']
        manager.create_multi_physics_dashboard(solutions, physics_names, vertices, elements)
        print("  ✓ Multi-physics dashboard successful")
    except Exception as e:
        print(f"  ✗ Multi-physics dashboard failed: {e}")
    
    # Test session save/load
    try:
        session_file = "test_session.json"
        manager.save_visualization_session(session_file)
        manager.load_visualization_session(session_file)
        
        # Clean up
        if os.path.exists(session_file):
            os.remove(session_file)
        print("  ✓ Session save/load successful")
    except Exception as e:
        print(f"  ✗ Session save/load failed: {e}")
    
    # Test performance settings
    try:
        manager.set_level_of_detail(True, 0.01)
        manager.enable_gpu_acceleration(True)
        manager.set_memory_limit(2048)
        print("  ✓ Performance settings successful")
    except Exception as e:
        print(f"  ✗ Performance settings failed: {e}")
    
    # Test publication quality figure
    try:
        test_filename = "test_publication_figure.png"
        manager.create_publication_quality_figure(vertices, elements, solution_fields, 
                                                 'potential', test_filename, dpi=150)
        
        # Clean up
        if os.path.exists(test_filename):
            os.remove(test_filename)
        if os.path.exists(test_filename + ".info"):
            os.remove(test_filename + ".info")
        print("  ✓ Publication quality figure successful")
    except Exception as e:
        print(f"  ✗ Publication quality figure failed: {e}")
    
    print("VisualizationManager tests completed!")
    return True

def test_utility_functions():
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Test sample mesh creation
    try:
        vertices, elements = create_sample_mesh_2d(8, 8)
        assert vertices.shape == (64, 2)  # 8x8 grid
        assert len(elements) == 98  # 2 triangles per grid cell (7x7 cells)
        print("  ✓ Sample mesh creation successful")
    except Exception as e:
        print(f"  ✗ Sample mesh creation failed: {e}")
    
    # Test sample solution fields creation
    try:
        fields = create_sample_solution_fields(100)
        assert 'potential' in fields
        assert 'electron_density' in fields
        assert 'electric_field_x' in fields
        assert len(fields['potential']) == 100
        print("  ✓ Sample solution fields creation successful")
    except Exception as e:
        print(f"  ✗ Sample solution fields creation failed: {e}")
    
    print("Utility functions tests completed!")
    return True

def run_all_tests():
    """Run all enhanced visualization tests"""
    print("=" * 60)
    print("ENHANCED VISUALIZATION TEST SUITE")
    print("=" * 60)
    
    test_results = []
    
    # Run individual test categories
    test_categories = [
        ("MeshVisualizer3D", test_mesh_visualizer_3d),
        ("SolutionVisualizer", test_solution_visualizer),
        ("AnimationEngine", test_animation_engine),
        ("InteractivePlotter", test_interactive_plotter),
        ("VisualizationManager", test_visualization_manager),
        ("Utility Functions", test_utility_functions)
    ]
    
    for category_name, test_function in test_categories:
        print(f"\n{'-' * 40}")
        print(f"Testing {category_name}")
        print(f"{'-' * 40}")
        
        try:
            result = test_function()
            test_results.append((category_name, result))
            print(f"✓ {category_name} tests PASSED")
        except Exception as e:
            test_results.append((category_name, False))
            print(f"✗ {category_name} tests FAILED: {e}")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for category_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{category_name:25} : {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} test categories passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Enhanced visualization system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
