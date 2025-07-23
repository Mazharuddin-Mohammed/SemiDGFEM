#!/usr/bin/env python3
"""
Tutorial 3: Advanced Self-Consistent Simulation
===============================================

This tutorial demonstrates advanced self-consistent simulation capabilities
including material properties, comprehensive visualization, and analysis tools.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def tutorial_1_material_database():
    """Tutorial 3.1: Material Properties Database"""
    print("=" * 60)
    print("Tutorial 3.1: Material Properties Database")
    print("=" * 60)
    
    try:
        import enhanced_simulator as es
        
        # Create material database
        materials = es.create_material_database()
        print("✓ Material database created")
        
        # Display available materials
        available_materials = materials.get_available_materials()
        print(f"✓ Available materials: {available_materials}")
        
        # Test different materials at different temperatures
        temperatures = [77, 300, 400, 500]  # K
        materials_to_test = [
            (es.MaterialType.SILICON, "Silicon"),
            (es.MaterialType.GALLIUM_ARSENIDE, "GaAs"),
            (es.MaterialType.GERMANIUM, "Germanium"),
            (es.MaterialType.SILICON_CARBIDE, "SiC")
        ]
        
        print("\nMaterial Properties vs Temperature:")
        print("-" * 80)
        print(f"{'Material':<15} {'T(K)':<6} {'Eg(eV)':<8} {'ni(m^-3)':<12} {'μn(m²/V·s)':<12} {'μp(m²/V·s)':<12}")
        print("-" * 80)
        
        for mat_type, name in materials_to_test:
            for T in temperatures:
                try:
                    Eg = materials.get_bandgap(mat_type, T)
                    ni = materials.get_intrinsic_concentration(mat_type, T)
                    mu_n = materials.get_electron_mobility(mat_type, T, 1e16)
                    mu_p = materials.get_hole_mobility(mat_type, T, 1e16)
                    
                    print(f"{name:<15} {T:<6} {Eg:<8.3f} {ni:<12.2e} {mu_n:<12.4f} {mu_p:<12.4f}")
                except Exception as e:
                    print(f"{name:<15} {T:<6} Error: {e}")
        
        # Test recombination calculations
        print("\nRecombination Mechanisms:")
        print("-" * 40)
        
        n = 1e18  # m^-3
        p = 1e17  # m^-3
        ni = 1.45e16  # m^-3 (Si at 300K)
        
        R_srh = materials.calculate_srh_recombination(n, p, ni)
        R_rad = materials.calculate_radiative_recombination(n, p, ni)
        R_auger = materials.calculate_auger_recombination(n, p, ni)
        
        print(f"SRH recombination: {R_srh:.2e} m^-3/s")
        print(f"Radiative recombination: {R_rad:.2e} m^-3/s")
        print(f"Auger recombination: {R_auger:.2e} m^-3/s")
        
        return True
        
    except Exception as e:
        print(f"✗ Tutorial 3.1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def tutorial_2_self_consistent_solver():
    """Tutorial 3.2: Self-Consistent Solver with Material Properties"""
    print("\n" + "=" * 60)
    print("Tutorial 3.2: Self-Consistent Solver with Material Properties")
    print("=" * 60)
    
    try:
        import enhanced_simulator as es
        
        # Create self-consistent solver for different materials
        materials_to_test = [
            (es.MaterialType.SILICON, "Silicon"),
            (es.MaterialType.GALLIUM_ARSENIDE, "GaAs")
        ]
        
        results_comparison = {}
        
        for mat_type, name in materials_to_test:
            print(f"\nSolving for {name}...")
            
            # Create solver
            solver = es.create_self_consistent_solver(
                2e-6, 1e-6, mat_type, "DG", "Structured", 3)
            
            # Set convergence criteria
            solver.set_convergence_criteria(1e-6, 1e-3, 1e-3, 50)
            
            # Set doping profile (p-n junction)
            dof_count = solver.get_dof_count()
            Nd = np.zeros(dof_count)
            Na = np.zeros(dof_count)
            
            # Create abrupt p-n junction
            Na[:dof_count//2] = 1e17 * 1e6  # 1e17 cm^-3 = 1e23 m^-3
            Nd[dof_count//2:] = 1e17 * 1e6
            
            solver.set_doping(Nd, Na)
            
            # Solve for different bias conditions
            bias_voltages = [0.0, 0.3, 0.6, 0.9]
            material_results = []
            
            for V_bias in bias_voltages:
                print(f"  Solving at {V_bias:.1f}V bias...")
                
                # Boundary conditions
                bc = [0.0, V_bias, 0.0, 0.0]
                
                try:
                    results = solver.solve_steady_state(bc)
                    material_results.append(results)
                    
                    print(f"    ✓ Converged: V_range=[{np.min(results['potential']):.3f}, {np.max(results['potential']):.3f}]V")
                    
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    material_results.append(None)
            
            results_comparison[name] = material_results
        
        print(f"\n✓ Self-consistent solutions completed for {len(results_comparison)} materials")
        return results_comparison
        
    except Exception as e:
        print(f"✗ Tutorial 3.2 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def tutorial_3_advanced_visualization():
    """Tutorial 3.3: Advanced Visualization and Analysis"""
    print("\n" + "=" * 60)
    print("Tutorial 3.3: Advanced Visualization and Analysis")
    print("=" * 60)
    
    try:
        import enhanced_simulator as es
        import advanced_visualization as av
        
        # Create solver and solve
        solver = es.create_self_consistent_solver(2e-6, 1e-6, es.MaterialType.SILICON)
        solver.set_convergence_criteria(1e-6, 1e-3, 1e-3, 30)
        
        # Set doping
        dof_count = solver.get_dof_count()
        Nd = np.zeros(dof_count)
        Na = np.zeros(dof_count)
        Na[:dof_count//2] = 1e16 * 1e6
        Nd[dof_count//2:] = 1e16 * 1e6
        solver.set_doping(Nd, Na)
        
        # Solve
        bc = [0.0, 0.7, 0.0, 0.0]  # 0.7V forward bias
        results = solver.solve_steady_state(bc)
        
        print("✓ Simulation completed, creating visualizations...")
        
        # Create visualizer
        visualizer = av.create_device_visualizer(2e-6, 1e-6)
        
        # Create comprehensive plots
        figures = {}
        
        # Plot potential
        if 'potential' in results:
            fig, ax = visualizer.plot_potential_2d(results['potential'])
            figures['potential'] = fig
            print("  ✓ Potential plot created")
        
        # Plot carrier densities
        if 'n' in results and 'p' in results:
            fig, axes = visualizer.plot_carrier_densities(results['n'], results['p'])
            figures['carriers'] = fig
            print("  ✓ Carrier density plots created")
        
        # Plot 1D profiles
        profile_data = {}
        for key in ['potential', 'n', 'p']:
            if key in results:
                profile_data[key] = results[key]
        
        if profile_data:
            fig, axes = visualizer.plot_1d_profiles(profile_data)
            figures['profiles'] = fig
            print("  ✓ 1D profile plots created")
        
        # Analyze device metrics
        metrics = av.AnalysisTools.calculate_device_metrics(results, 2e-6, 1e-6)
        
        print("\nDevice Metrics:")
        print("-" * 30)
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'density' in key.lower():
                    print(f"{key}: {value:.2e} m^-3")
                elif 'potential' in key.lower():
                    print(f"{key}: {value:.3f} V")
                elif 'current' in key.lower():
                    print(f"{key}: {value:.2e} A/m^2")
                elif 'area' in key.lower():
                    print(f"{key}: {value:.2e} m^2")
                else:
                    print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        
        # Save plots if possible
        try:
            plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            for name, fig in figures.items():
                save_path = os.path.join(plot_dir, f'tutorial_3_{name}.png')
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  ✓ {name} plot saved to {save_path}")
            
        except Exception as e:
            print(f"  ⚠️  Could not save plots: {e}")
        
        # Clean up
        plt.close('all')
        
        print(f"\n✓ Created {len(figures)} visualization plots")
        return True
        
    except Exception as e:
        print(f"✗ Tutorial 3.3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def tutorial_4_comparative_analysis():
    """Tutorial 3.4: Comparative Analysis of Different Materials"""
    print("\n" + "=" * 60)
    print("Tutorial 3.4: Comparative Analysis of Different Materials")
    print("=" * 60)
    
    try:
        import enhanced_simulator as es
        import advanced_visualization as av
        
        # Materials to compare
        materials = [
            (es.MaterialType.SILICON, "Silicon"),
            (es.MaterialType.GALLIUM_ARSENIDE, "GaAs"),
            (es.MaterialType.GERMANIUM, "Germanium")
        ]
        
        # Simulation parameters
        device_width, device_height = 1e-6, 0.5e-6
        bias_voltage = 0.5  # V
        
        comparison_results = {}
        
        for mat_type, name in materials:
            print(f"\nAnalyzing {name}...")
            
            try:
                # Create solver
                solver = es.create_self_consistent_solver(
                    device_width, device_height, mat_type)
                solver.set_convergence_criteria(1e-6, 1e-3, 1e-3, 25)
                
                # Set doping
                dof_count = solver.get_dof_count()
                Nd = np.zeros(dof_count)
                Na = np.zeros(dof_count)
                Na[:dof_count//3] = 5e16 * 1e6
                Nd[2*dof_count//3:] = 5e16 * 1e6
                solver.set_doping(Nd, Na)
                
                # Solve
                bc = [0.0, bias_voltage, 0.0, 0.0]
                results = solver.solve_steady_state(bc)
                
                # Calculate metrics
                metrics = av.AnalysisTools.calculate_device_metrics(
                    results, device_width, device_height)
                
                comparison_results[name] = {
                    'results': results,
                    'metrics': metrics
                }
                
                print(f"  ✓ {name} simulation completed")
                
            except Exception as e:
                print(f"  ✗ {name} simulation failed: {e}")
                comparison_results[name] = None
        
        # Create comparison plots
        print("\nCreating comparative analysis...")
        
        # Compare key metrics
        print("\nComparative Metrics:")
        print("-" * 60)
        print(f"{'Material':<15} {'Max V (V)':<10} {'Max n (m^-3)':<12} {'Max p (m^-3)':<12}")
        print("-" * 60)
        
        for name, data in comparison_results.items():
            if data:
                metrics = data['metrics']
                max_V = metrics.get('max_potential', 0)
                max_n = metrics.get('max_electron_density', 0)
                max_p = metrics.get('max_hole_density', 0)
                
                print(f"{name:<15} {max_V:<10.3f} {max_n:<12.2e} {max_p:<12.2e}")
        
        # Plot comparison
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            for i, (name, data) in enumerate(comparison_results.items()):
                if data and i < 4:
                    results = data['results']
                    
                    # Plot potential profile
                    if 'potential' in results:
                        ax = axes[i//2, i%2]
                        x_pos = np.linspace(0, device_width*1e6, len(results['potential']))
                        ax.plot(x_pos, results['potential'], 'o-', linewidth=2, label=name)
                        ax.set_xlabel('Position (μm)')
                        ax.set_ylabel('Potential (V)')
                        ax.set_title(f'{name} - Potential Profile')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
            
            plt.suptitle('Material Comparison - Potential Profiles', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save comparison plot
            try:
                plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                save_path = os.path.join(plot_dir, 'tutorial_3_material_comparison.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  ✓ Comparison plot saved to {save_path}")
            except:
                pass
            
            plt.close()
            
        except Exception as e:
            print(f"  ⚠️  Could not create comparison plots: {e}")
        
        print(f"\n✓ Comparative analysis completed for {len([d for d in comparison_results.values() if d])} materials")
        return comparison_results
        
    except Exception as e:
        print(f"✗ Tutorial 3.4 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_all_tutorials():
    """Run all advanced self-consistent tutorials"""
    print("SemiDGFEM Advanced Self-Consistent Simulation Tutorial")
    print("=" * 60)
    print("This tutorial demonstrates advanced self-consistent simulation")
    print("capabilities including comprehensive material properties,")
    print("advanced visualization, and comparative analysis.\n")
    
    tutorials = [
        tutorial_1_material_database,
        tutorial_2_self_consistent_solver,
        tutorial_3_advanced_visualization,
        tutorial_4_comparative_analysis
    ]
    
    results = []
    tutorial_results = {}
    
    for i, tutorial in enumerate(tutorials, 1):
        try:
            result = tutorial()
            success = result is not None and result is not False
            results.append(success)
            tutorial_results[f"tutorial_3_{i}"] = result
            
            if success:
                print(f"✓ Tutorial 3.{i} completed successfully")
            else:
                print(f"✗ Tutorial 3.{i} failed")
        except Exception as e:
            print(f"✗ Tutorial 3.{i} crashed: {e}")
            results.append(False)
            tutorial_results[f"tutorial_3_{i}"] = None
    
    # Summary
    print("\n" + "=" * 60)
    print("ADVANCED TUTORIAL SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tutorials passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All advanced tutorials completed successfully!")
        print("You have mastered advanced self-consistent simulation with SemiDGFEM.")
        print("The simulator now includes:")
        print("  • Comprehensive material properties database")
        print("  • Self-consistent coupling of all physics")
        print("  • Advanced visualization and analysis tools")
        print("  • Multi-material comparative analysis")
    else:
        print("⚠️  Some tutorials failed. Please check the error messages above.")
        print("Advanced simulation requires all components to work together.")
    
    return passed == total, tutorial_results

if __name__ == "__main__":
    # Set up environment
    import os
    import sys
    
    # Add library path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_root, "build")
    
    if os.path.exists(build_dir):
        os.environ['LD_LIBRARY_PATH'] = build_dir + ":" + os.environ.get('LD_LIBRARY_PATH', '')
    
    # Run tutorials
    success, results = run_all_tutorials()
    sys.exit(0 if success else 1)
