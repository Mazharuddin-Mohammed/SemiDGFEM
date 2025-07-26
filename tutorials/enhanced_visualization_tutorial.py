"""
Enhanced Visualization Tutorial

This tutorial demonstrates the advanced visualization capabilities of the SemiDGFEM simulator:
- 3D mesh visualization with quality analysis
- Solution field visualization with multiple techniques
- Interactive plotting and dashboard creation
- Animation of time-dependent solutions
- Publication-quality figure generation

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import matplotlib.pyplot as plt
from enhanced_visualization import (
    VisualizationManager, MeshVisualizer3D, SolutionVisualizer, 
    AnimationEngine, InteractivePlotter, ColorMapConfig, RenderingConfig, 
    AnimationConfig, ColorMapType, RenderingMode,
    create_sample_mesh_2d, create_sample_solution_fields
)

def tutorial_1_mesh_visualization():
    """Tutorial 1: Advanced 3D Mesh Visualization"""
    print("=" * 60)
    print("TUTORIAL 1: Advanced 3D Mesh Visualization")
    print("=" * 60)
    
    # Create a more complex mesh for demonstration
    print("Creating sample mesh...")
    vertices, elements = create_sample_mesh_2d(25, 25)
    print(f"  - Mesh created: {len(vertices)} vertices, {len(elements)} elements")
    
    # Initialize mesh visualizer
    mesh_viz = MeshVisualizer3D()
    mesh_viz.set_mesh(vertices, elements)
    
    # Configure rendering
    rendering_config = RenderingConfig(
        mode=RenderingMode.SURFACE,
        anti_aliasing=True,
        shadows_enabled=True
    )
    mesh_viz.set_rendering_config(rendering_config)
    
    print("\n1.1 Basic Mesh Visualization")
    print("-" * 30)
    
    # Show wireframe
    print("Displaying wireframe...")
    mesh_viz.show_wireframe(show_edges=True, show_vertices=False)
    mesh_viz.render_to_file("tutorial_mesh_wireframe.png", dpi=150)
    
    # Show surface
    print("Displaying surface...")
    mesh_viz.show_surface(show_faces=True, show_normals=False)
    mesh_viz.render_to_file("tutorial_mesh_surface.png", dpi=150)
    
    print("\n1.2 Element Quality Analysis")
    print("-" * 30)
    
    # Generate realistic quality metrics
    # Simulate some poor quality elements near boundaries
    quality_metrics = np.ones(len(elements))
    for i, element in enumerate(elements):
        # Get element vertices
        elem_vertices = vertices[element]
        center = np.mean(elem_vertices, axis=0)
        
        # Reduce quality near boundaries
        distance_to_boundary = min(center[0], center[1], 1-center[0], 1-center[1])
        if distance_to_boundary < 0.1:
            quality_metrics[i] = 0.3 + 0.4 * np.random.random()
        else:
            quality_metrics[i] = 0.7 + 0.3 * np.random.random()
    
    print("Analyzing element quality...")
    mesh_viz.show_element_quality(quality_metrics)
    mesh_viz.render_to_file("tutorial_mesh_quality.png", dpi=150)
    
    print("\n1.3 Material Regions")
    print("-" * 30)
    
    # Create material regions (simulate a device structure)
    material_ids = np.ones(len(elements), dtype=int)
    for i, element in enumerate(elements):
        elem_vertices = vertices[element]
        center = np.mean(elem_vertices, axis=0)
        
        # Create different material regions
        if center[0] < 0.3:
            material_ids[i] = 1  # Source
        elif center[0] > 0.7:
            material_ids[i] = 3  # Drain
        elif 0.4 < center[0] < 0.6 and 0.4 < center[1] < 0.6:
            material_ids[i] = 4  # Gate oxide
        else:
            material_ids[i] = 2  # Channel
    
    print("Displaying material regions...")
    mesh_viz.show_material_regions(material_ids)
    mesh_viz.render_to_file("tutorial_mesh_materials.png", dpi=150)
    
    print("\n1.4 Doping Profile")
    print("-" * 30)
    
    # Create realistic doping profile
    doping_concentration = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        x, y = vertex[0], vertex[1]
        
        # Create a MOSFET-like doping profile
        if x < 0.3:  # Source region
            doping_concentration[i] = 1e19  # Heavily doped
        elif x > 0.7:  # Drain region
            doping_concentration[i] = 1e19  # Heavily doped
        else:  # Channel region
            # Gaussian profile
            doping_concentration[i] = 1e15 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1)
    
    print("Displaying doping profile...")
    mesh_viz.show_doping_profile(doping_concentration)
    mesh_viz.render_to_file("tutorial_mesh_doping.png", dpi=150)
    
    print("\nTutorial 1 completed! Generated files:")
    print("  - tutorial_mesh_wireframe.png")
    print("  - tutorial_mesh_surface.png")
    print("  - tutorial_mesh_quality.png")
    print("  - tutorial_mesh_materials.png")
    print("  - tutorial_mesh_doping.png")

def tutorial_2_solution_visualization():
    """Tutorial 2: Advanced Solution Field Visualization"""
    print("\n" + "=" * 60)
    print("TUTORIAL 2: Advanced Solution Field Visualization")
    print("=" * 60)
    
    # Create mesh and solution data
    print("Creating solution data...")
    vertices, elements = create_sample_mesh_2d(30, 30)
    
    # Create more realistic solution fields for a MOSFET
    solution_fields = {}
    
    for i, vertex in enumerate(vertices):
        x, y = vertex[0], vertex[1]
        
        # Potential field (parabolic in channel)
        if 0.3 <= x <= 0.7:  # Channel region
            solution_fields.setdefault('potential', []).append(
                0.5 * (x - 0.3) * (0.7 - x) + 0.1 * np.sin(2 * np.pi * y)
            )
        else:
            solution_fields.setdefault('potential', []).append(0.0)
        
        # Electron density (exponential dependence on potential)
        potential = solution_fields['potential'][-1]
        solution_fields.setdefault('electron_density', []).append(
            1e16 * np.exp(potential / 0.026)  # kT = 26 meV at room temperature
        )
        
        # Hole density (complementary to electron density)
        solution_fields.setdefault('hole_density', []).append(
            1e15 * np.exp(-potential / 0.026)
        )
        
        # Electric field (gradient of potential)
        if 0.3 < x < 0.7:
            ex = -(0.5 * (0.7 - 2*x + 0.3))  # Derivative of potential
            ey = -0.1 * 2 * np.pi * np.cos(2 * np.pi * y)
        else:
            ex, ey = 0.0, 0.0
        
        solution_fields.setdefault('electric_field_x', []).append(ex)
        solution_fields.setdefault('electric_field_y', []).append(ey)
        
        # Current density (proportional to electric field and carrier density)
        jx = 1e-6 * solution_fields['electron_density'][-1] * ex
        jy = 1e-6 * solution_fields['electron_density'][-1] * ey
        
        solution_fields.setdefault('current_density_x', []).append(jx)
        solution_fields.setdefault('current_density_y', []).append(jy)
    
    # Convert to numpy arrays
    for field_name in solution_fields:
        solution_fields[field_name] = np.array(solution_fields[field_name])
    
    print(f"  - Created {len(solution_fields)} solution fields")
    
    # Initialize solution visualizer
    solution_viz = SolutionVisualizer()
    solution_viz.set_mesh(vertices, elements)
    solution_viz.set_solution(solution_fields)
    
    # Configure colormap
    colormap_config = ColorMapConfig(
        type=ColorMapType.VIRIDIS,
        logarithmic_scale=False,
        reverse_colormap=False
    )
    solution_viz.set_colormap_config(colormap_config)
    
    print("\n2.1 Contour Plots")
    print("-" * 30)
    
    # Create contour plots for different fields
    fields_to_plot = ['potential', 'electron_density', 'hole_density']
    for field_name in fields_to_plot:
        print(f"Creating contour plot for {field_name}...")
        solution_viz.show_contour_plot(field_name, num_levels=20)
        solution_viz.render_to_file(f"tutorial_contour_{field_name}.png", dpi=150)
    
    print("\n2.2 Surface Plots")
    print("-" * 30)
    
    # Create 3D surface plots
    print("Creating 3D surface plot for potential...")
    solution_viz.show_surface_plot('potential')
    solution_viz.render_to_file("tutorial_surface_potential.png", dpi=150)
    
    print("\n2.3 Vector Field Visualization")
    print("-" * 30)
    
    # Visualize electric field
    print("Visualizing electric field...")
    solution_viz.show_vector_field('electric_field', scale=1.0)
    solution_viz.render_to_file("tutorial_vector_electric_field.png", dpi=150)
    
    # Visualize current density
    print("Visualizing current density...")
    solution_viz.show_vector_field('current_density', scale=1e6)
    solution_viz.render_to_file("tutorial_vector_current_density.png", dpi=150)
    
    print("\n2.4 Streamlines")
    print("-" * 30)
    
    # Create streamlines for electric field
    print("Creating streamlines for electric field...")
    solution_viz.show_streamlines('electric_field', num_seeds=100)
    solution_viz.render_to_file("tutorial_streamlines_electric_field.png", dpi=150)
    
    print("\n2.5 Multi-Field Comparison")
    print("-" * 30)
    
    # Compare multiple fields
    print("Creating multi-field comparison...")
    comparison_fields = ['potential', 'electron_density', 'hole_density']
    solution_viz.show_multi_field_comparison(comparison_fields)
    solution_viz.render_to_file("tutorial_multi_field_comparison.png", dpi=150)
    
    print("\n2.6 Field Difference Analysis")
    print("-" * 30)
    
    # Analyze difference between electron and hole densities
    print("Analyzing electron-hole density difference...")
    solution_viz.show_field_difference('electron_density', 'hole_density')
    solution_viz.render_to_file("tutorial_field_difference.png", dpi=150)
    
    print("\n2.7 Line Plots")
    print("-" * 30)
    
    # Create line plots along device cross-sections
    print("Creating line plot along channel...")
    start_point = [0.3, 0.5, 0.0]
    end_point = [0.7, 0.5, 0.0]
    solution_viz.show_line_plot(start_point, end_point, 'potential')
    solution_viz.render_to_file("tutorial_line_plot_potential.png", dpi=150)
    
    print("\nTutorial 2 completed! Generated files:")
    generated_files = [
        "tutorial_contour_potential.png",
        "tutorial_contour_electron_density.png", 
        "tutorial_contour_hole_density.png",
        "tutorial_surface_potential.png",
        "tutorial_vector_electric_field.png",
        "tutorial_vector_current_density.png",
        "tutorial_streamlines_electric_field.png",
        "tutorial_multi_field_comparison.png",
        "tutorial_field_difference.png",
        "tutorial_line_plot_potential.png"
    ]
    for filename in generated_files:
        print(f"  - {filename}")

def tutorial_3_animation_and_interactive_plotting():
    """Tutorial 3: Animation and Interactive Plotting"""
    print("\n" + "=" * 60)
    print("TUTORIAL 3: Animation and Interactive Plotting")
    print("=" * 60)
    
    # Create mesh
    vertices, elements = create_sample_mesh_2d(20, 20)
    
    print("\n3.1 Time-Dependent Animation")
    print("-" * 30)
    
    # Create time series data (simulate transient response)
    time_points = np.linspace(0, 1, 20)
    time_series_data = []
    
    print("Generating time series data...")
    for t in time_points:
        fields = {}
        for i, vertex in enumerate(vertices):
            x, y = vertex[0], vertex[1]
            
            # Time-dependent potential (wave propagation)
            potential = np.sin(2 * np.pi * (x - 0.5 * t)) * np.exp(-t) * np.exp(-((y-0.5)**2)/0.1)
            fields.setdefault('potential', []).append(potential)
            
            # Time-dependent electron density
            electron_density = 1e16 * (1 + 0.5 * potential) * np.exp(-2*t)
            fields.setdefault('electron_density', []).append(electron_density)
        
        # Convert to numpy arrays
        for field_name in fields:
            fields[field_name] = np.array(fields[field_name])
        
        time_series_data.append(fields)
    
    # Create animation
    animation_engine = AnimationEngine()
    animation_config = AnimationConfig(
        frame_rate=5.0,
        duration=4.0,
        loop_animation=True,
        save_frames=False
    )
    animation_engine.set_animation_config(animation_config)
    animation_engine.add_time_series_data(time_series_data, time_points)
    
    print("Creating solution evolution animation...")
    animation_engine.create_solution_evolution_animation('potential', vertices, elements)
    
    # Export animation
    try:
        print("Exporting animation as GIF...")
        animation_engine.export_gif("tutorial_animation_potential.gif", quality=80)
        print("  - Animation exported successfully")
    except Exception as e:
        print(f"  - Animation export failed: {e}")
        print("  - This is normal if Pillow is not installed")
    
    print("\n3.2 Parameter Sweep Animation")
    print("-" * 30)
    
    # Create parameter sweep data (voltage sweep)
    voltage_values = np.linspace(0.1, 1.0, 10)
    parameter_solutions = []
    
    print("Generating parameter sweep data...")
    for voltage in voltage_values:
        fields = {}
        for i, vertex in enumerate(vertices):
            x, y = vertex[0], vertex[1]
            
            # Voltage-dependent potential
            if 0.3 <= x <= 0.7:  # Channel region
                potential = voltage * (x - 0.3) / 0.4
            else:
                potential = 0.0
            
            fields.setdefault('potential', []).append(potential)
            
            # Voltage-dependent current
            current = voltage**2 * np.exp(-((x-0.5)**2 + (y-0.5)**2)/0.1)
            fields.setdefault('current_density', []).append(current)
        
        # Convert to numpy arrays
        for field_name in fields:
            fields[field_name] = np.array(fields[field_name])
        
        parameter_solutions.append(fields)
    
    print("Creating parameter sweep animation...")
    animation_engine.create_parameter_sweep_animation(
        parameter_solutions, voltage_values, 'Gate Voltage [V]', 
        'potential', vertices, elements
    )
    
    print("\n3.3 Interactive Plotting")
    print("-" * 30)
    
    # Create interactive plotter
    plotter = InteractivePlotter()
    
    # Create various plot types
    print("Creating interactive plots...")
    
    # Line plot - I-V characteristics
    voltages = np.linspace(0, 1, 50)
    currents = voltages**2 * (1 + 0.1 * np.sin(10 * voltages))  # Nonlinear I-V
    plotter.create_line_plot(voltages, currents, "I-V Characteristic")
    
    # Scatter plot - Device parameters
    threshold_voltages = np.random.normal(0.5, 0.05, 100)
    transconductances = 1.0 + 0.3 * threshold_voltages + 0.1 * np.random.randn(100)
    plotter.create_scatter_plot(threshold_voltages, transconductances, label="Device Variation")
    
    # Surface plot - 2D potential distribution
    x_2d = np.linspace(0, 1, 20)
    y_2d = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x_2d, y_2d)
    Z = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y) * np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.2)
    plotter.create_surface_plot(Z, x_2d, y_2d)
    
    # Histogram - Threshold voltage distribution
    plotter.create_histogram(threshold_voltages, bins=20, label="Vth Distribution")
    
    # Configure plots
    plotter.set_title("SemiDGFEM Device Analysis")
    plotter.set_axis_labels("Voltage [V]", "Current [A]", "Potential [V]")
    
    # Create subplot grid for dashboard
    plotter.create_subplot_grid(2, 2)
    
    print("Rendering interactive plots...")
    plotter.show_plot()
    plotter.save_plot("tutorial_interactive_plots.png", dpi=150)
    
    print("\nTutorial 3 completed! Generated files:")
    print("  - tutorial_animation_potential.gif (if Pillow available)")
    print("  - tutorial_interactive_plots.png")

def tutorial_4_comprehensive_dashboard():
    """Tutorial 4: Comprehensive Visualization Dashboard"""
    print("\n" + "=" * 60)
    print("TUTORIAL 4: Comprehensive Visualization Dashboard")
    print("=" * 60)
    
    # Create comprehensive device simulation data
    print("Creating comprehensive device data...")
    vertices, elements = create_sample_mesh_2d(25, 25)
    
    # Create realistic MOSFET solution fields
    solution_fields = {}
    
    for i, vertex in enumerate(vertices):
        x, y = vertex[0], vertex[1]
        
        # Create device regions
        is_source = x < 0.25
        is_drain = x > 0.75
        is_channel = 0.25 <= x <= 0.75 and 0.4 <= y <= 0.6
        is_gate_oxide = 0.3 <= x <= 0.7 and 0.6 < y <= 0.8
        
        # Potential distribution
        if is_source:
            potential = 0.0
        elif is_drain:
            potential = 1.0
        elif is_channel:
            potential = (x - 0.25) / 0.5  # Linear in channel
        elif is_gate_oxide:
            potential = 0.5 + 0.3 * np.sin(np.pi * (x - 0.3) / 0.4)
        else:
            potential = 0.1 * np.random.random()
        
        solution_fields.setdefault('potential', []).append(potential)
        
        # Electron density
        if is_channel:
            electron_density = 1e17 * np.exp(potential / 0.026)
        elif is_source or is_drain:
            electron_density = 1e19
        else:
            electron_density = 1e15
        
        solution_fields.setdefault('electron_density', []).append(electron_density)
        
        # Electric field
        if is_channel:
            ex = -1.0 / 0.5  # Constant field in channel
            ey = 0.0
        elif is_gate_oxide:
            ex = 0.0
            ey = -5.0  # Strong vertical field
        else:
            ex, ey = 0.0, 0.0
        
        solution_fields.setdefault('electric_field_x', []).append(ex)
        solution_fields.setdefault('electric_field_y', []).append(ey)
        
        # Temperature (self-heating effect)
        power_density = electron_density * (ex**2 + ey**2) * 1e-20
        temperature = 300 + 50 * power_density  # Kelvin
        solution_fields.setdefault('temperature', []).append(temperature)
    
    # Convert to numpy arrays
    for field_name in solution_fields:
        solution_fields[field_name] = np.array(solution_fields[field_name])
    
    print(f"  - Created comprehensive solution with {len(solution_fields)} fields")
    
    # Create visualization manager
    viz_manager = VisualizationManager()
    
    print("\n4.1 Comprehensive Visualization")
    print("-" * 30)
    
    # Create comprehensive visualization
    print("Creating comprehensive visualization...")
    viz_manager.create_comprehensive_visualization(vertices, elements, solution_fields)
    
    print("\n4.2 Multi-Physics Dashboard")
    print("-" * 30)
    
    # Create multi-physics solutions
    physics_solutions = [
        {'electrostatics': solution_fields['potential']},
        {'transport': solution_fields['electron_density']},
        {'thermal': solution_fields['temperature']},
        {'electromagnetic': solution_fields['electric_field_x']}
    ]
    physics_names = ['electrostatics', 'transport', 'thermal', 'electromagnetic']
    
    print("Creating multi-physics dashboard...")
    viz_manager.create_multi_physics_dashboard(physics_solutions, physics_names, vertices, elements)
    
    print("\n4.3 Interactive Dashboard")
    print("-" * 30)
    
    # Create interactive dashboard
    print("Creating interactive dashboard...")
    viz_manager.create_interactive_dashboard(vertices, elements, solution_fields)
    
    print("\n4.4 Publication Quality Figures")
    print("-" * 30)
    
    # Create publication quality figures
    publication_fields = ['potential', 'electron_density', 'temperature']
    for field_name in publication_fields:
        filename = f"tutorial_publication_{field_name}.png"
        print(f"Creating publication figure for {field_name}...")
        viz_manager.create_publication_quality_figure(
            vertices, elements, solution_fields, field_name, filename, dpi=300
        )
    
    print("\n4.5 Session Management")
    print("-" * 30)
    
    # Save and load visualization session
    session_file = "tutorial_visualization_session.json"
    print("Saving visualization session...")
    viz_manager.save_visualization_session(session_file)
    
    print("Loading visualization session...")
    viz_manager.load_visualization_session(session_file)
    
    print("\n4.6 Performance Optimization")
    print("-" * 30)
    
    # Configure performance settings
    print("Configuring performance settings...")
    viz_manager.set_level_of_detail(True, 0.01)
    viz_manager.enable_gpu_acceleration(True)
    viz_manager.set_memory_limit(4096)
    
    print("\nTutorial 4 completed! Generated files:")
    generated_files = [
        "tutorial_publication_potential.png",
        "tutorial_publication_electron_density.png",
        "tutorial_publication_temperature.png",
        "tutorial_visualization_session.json"
    ]
    for filename in generated_files:
        print(f"  - {filename}")

def run_all_tutorials():
    """Run all enhanced visualization tutorials"""
    print("🚀 ENHANCED VISUALIZATION TUTORIALS")
    print("=" * 60)
    print("This tutorial demonstrates the advanced visualization capabilities")
    print("of the SemiDGFEM simulator including 3D mesh visualization,")
    print("solution field analysis, animation, and interactive plotting.")
    print("=" * 60)
    
    try:
        # Run all tutorials
        tutorial_1_mesh_visualization()
        tutorial_2_solution_visualization()
        tutorial_3_animation_and_interactive_plotting()
        tutorial_4_comprehensive_dashboard()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TUTORIALS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated visualization files demonstrate:")
        print("✓ 3D mesh visualization with quality analysis")
        print("✓ Advanced solution field visualization techniques")
        print("✓ Time-dependent animations and parameter sweeps")
        print("✓ Interactive plotting and dashboard creation")
        print("✓ Publication-quality figure generation")
        print("✓ Comprehensive multi-physics visualization")
        print("\nThe enhanced visualization system provides powerful tools")
        print("for analyzing and presenting semiconductor device simulation results!")
        
    except Exception as e:
        print(f"\n❌ Tutorial failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tutorials()
