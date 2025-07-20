#!/usr/bin/env python3
"""
Tutorial 2: Transient Simulation
=================================

This tutorial demonstrates transient simulation capabilities including
time-dependent boundary conditions, different time integration methods,
and dynamic analysis of semiconductor devices.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def tutorial_1_basic_transient():
    """Tutorial 2.1: Basic Transient Simulation"""
    print("=" * 60)
    print("Tutorial 2.1: Basic Transient Simulation")
    print("=" * 60)
    
    try:
        # Import the transient solver
        import transient_solver_simple as ts
        print("✓ Transient solver module imported")
        
        # Create transient solver
        device_width = 2e-6   # 2 μm
        device_height = 1e-6  # 1 μm
        
        solver = ts.create_transient_solver(device_width, device_height, "DG", "Structured", 3)
        print(f"✓ Transient solver created: {device_width*1e6:.1f}μm × {device_height*1e6:.1f}μm")
        
        # Configure time parameters
        solver.set_time_step(1e-12)    # 1 ps time step
        solver.set_final_time(1e-10)   # 100 ps simulation
        solver.set_time_integrator("backward_euler")
        print("✓ Time parameters configured")
        
        # Set up doping (p-n junction)
        dof_count = solver.get_dof_count()
        Nd = np.zeros(dof_count)
        Na = np.zeros(dof_count)
        
        Na[:dof_count//2] = 1e16 * 1e6  # p-region
        Nd[dof_count//2:] = 1e16 * 1e6  # n-region
        
        solver.set_doping(Nd, Na)
        print(f"✓ Doping profile set (DOF: {dof_count})")
        
        # Initial conditions (equilibrium)
        initial_conditions = {
            'potential': np.zeros(dof_count),
            'n': np.full(dof_count, 1e10),  # Low initial carrier density
            'p': np.full(dof_count, 1e10)
        }
        
        # Boundary conditions (step voltage)
        bc = [0.0, 0.5, 0.0, 0.0]  # 0.5V step at t=0
        
        # Solve transient
        print("Solving transient simulation...")
        results = solver.solve(bc, initial_conditions, max_time_points=50)
        print("✓ Transient simulation completed")
        
        # Display results
        time_points = results['time']
        print(f"✓ Time points: {len(time_points)}")
        print(f"✓ Time range: {time_points[0]*1e12:.1f} - {time_points[-1]*1e12:.1f} ps")
        
        # Check final state
        final_potential = results['potential'][-1]
        final_n = results['n'][-1]
        final_p = results['p'][-1]
        
        print(f"✓ Final potential range: [{np.nanmin(final_potential):.3f}, {np.nanmax(final_potential):.3f}] V")
        print(f"✓ Final electron density: [{np.min(final_n):.2e}, {np.max(final_n):.2e}] m^-3")
        print(f"✓ Final hole density: [{np.min(final_p):.2e}, {np.max(final_p):.2e}] m^-3")
        
        return results
        
    except Exception as e:
        print(f"✗ Tutorial 2.1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def tutorial_2_time_integrators():
    """Tutorial 2.2: Comparison of Time Integrators"""
    print("\n" + "=" * 60)
    print("Tutorial 2.2: Comparison of Time Integrators")
    print("=" * 60)
    
    try:
        import transient_solver_simple as ts
        
        # Get available integrators
        integrators = ts.get_available_integrators()
        print(f"✓ Available integrators: {integrators}")
        
        # Test different integrators
        device_width, device_height = 1e-6, 0.5e-6
        results_comparison = {}
        
        for integrator in ["forward_euler", "backward_euler", "crank_nicolson"]:
            print(f"\nTesting {integrator}...")
            
            try:
                solver = ts.create_transient_solver(device_width, device_height)
                solver.set_time_step(5e-13)  # 0.5 ps
                solver.set_final_time(5e-11)  # 50 ps
                solver.set_time_integrator(integrator)
                
                # Simple doping
                dof_count = solver.get_dof_count()
                Nd = np.zeros(dof_count)
                Na = np.zeros(dof_count)
                Na[:dof_count//3] = 1e17 * 1e6
                Nd[2*dof_count//3:] = 1e17 * 1e6
                solver.set_doping(Nd, Na)
                
                # Initial conditions
                initial_conditions = {
                    'potential': np.zeros(dof_count),
                    'n': np.full(dof_count, 1e10),
                    'p': np.full(dof_count, 1e10)
                }
                
                # Small step voltage
                bc = [0.0, 0.3, 0.0, 0.0]
                
                results = solver.solve(bc, initial_conditions, max_time_points=20)
                results_comparison[integrator] = results
                
                print(f"  ✓ {integrator}: {len(results['time'])} time points")
                
            except Exception as e:
                print(f"  ✗ {integrator} failed: {e}")
                results_comparison[integrator] = None
        
        print(f"\n✓ Integrator comparison completed")
        return results_comparison
        
    except Exception as e:
        print(f"✗ Tutorial 2.2 failed: {e}")
        return None

def tutorial_3_dynamic_boundary_conditions():
    """Tutorial 2.3: Dynamic Boundary Conditions"""
    print("\n" + "=" * 60)
    print("Tutorial 2.3: Dynamic Boundary Conditions")
    print("=" * 60)
    
    try:
        import transient_solver_simple as ts
        
        # Create solver
        solver = ts.create_transient_solver(2e-6, 1e-6)
        solver.set_time_step(2e-12)  # 2 ps
        solver.set_final_time(2e-10)  # 200 ps
        solver.set_time_integrator("backward_euler")
        
        # Set doping
        dof_count = solver.get_dof_count()
        Nd = np.zeros(dof_count)
        Na = np.zeros(dof_count)
        Na[:dof_count//2] = 5e16 * 1e6
        Nd[dof_count//2:] = 5e16 * 1e6
        solver.set_doping(Nd, Na)
        
        # Initial conditions
        initial_conditions = {
            'potential': np.zeros(dof_count),
            'n': np.full(dof_count, 1e10),
            'p': np.full(dof_count, 1e10)
        }
        
        # Simulate different boundary condition scenarios
        scenarios = {
            'step': [0.0, 0.6, 0.0, 0.0],           # Step voltage
            'ramp': [0.0, 0.4, 0.0, 0.0],           # Ramp (simplified as constant)
            'pulse': [0.0, 0.8, 0.0, 0.0],          # Pulse (simplified as constant)
        }
        
        scenario_results = {}
        
        for scenario_name, bc in scenarios.items():
            print(f"\nSimulating {scenario_name} boundary condition...")
            
            try:
                # Create fresh solver for each scenario
                solver_scenario = ts.create_transient_solver(2e-6, 1e-6)
                solver_scenario.set_time_step(2e-12)
                solver_scenario.set_final_time(1e-10)  # Shorter for demo
                solver_scenario.set_time_integrator("backward_euler")
                solver_scenario.set_doping(Nd, Na)
                
                results = solver_scenario.solve(bc, initial_conditions, max_time_points=25)
                scenario_results[scenario_name] = results
                
                print(f"  ✓ {scenario_name}: {len(results['time'])} time points")
                
            except Exception as e:
                print(f"  ✗ {scenario_name} failed: {e}")
                scenario_results[scenario_name] = None
        
        print("✓ Dynamic boundary conditions simulation completed")
        return scenario_results
        
    except Exception as e:
        print(f"✗ Tutorial 2.3 failed: {e}")
        return None

def tutorial_4_transient_visualization():
    """Tutorial 2.4: Transient Results Visualization"""
    print("\n" + "=" * 60)
    print("Tutorial 2.4: Transient Results Visualization")
    print("=" * 60)
    
    try:
        import transient_solver_simple as ts
        
        # Run a transient simulation
        solver = ts.create_transient_solver(2e-6, 1e-6)
        solver.set_time_step(1e-12)
        solver.set_final_time(1e-10)
        solver.set_time_integrator("backward_euler")
        
        dof_count = solver.get_dof_count()
        Nd = np.zeros(dof_count)
        Na = np.zeros(dof_count)
        Na[:dof_count//2] = 1e16 * 1e6
        Nd[dof_count//2:] = 1e16 * 1e6
        solver.set_doping(Nd, Na)
        
        initial_conditions = {
            'potential': np.zeros(dof_count),
            'n': np.full(dof_count, 1e10),
            'p': np.full(dof_count, 1e10)
        }
        
        bc = [0.0, 0.7, 0.0, 0.0]
        results = solver.solve(bc, initial_conditions, max_time_points=30)
        
        if results and len(results['time']) > 1:
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            time_ps = results['time'] * 1e12  # Convert to ps
            
            # Plot 1: Potential evolution at center point
            center_idx = dof_count // 2
            potential_center = [pot[center_idx] for pot in results['potential']]
            
            axes[0, 0].plot(time_ps, potential_center, 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Time (ps)')
            axes[0, 0].set_ylabel('Potential (V)')
            axes[0, 0].set_title('Potential Evolution (Center Point)')
            axes[0, 0].grid(True)
            
            # Plot 2: Carrier density evolution
            n_center = [n[center_idx] for n in results['n']]
            p_center = [p[center_idx] for p in results['p']]
            
            axes[0, 1].semilogy(time_ps, n_center, 'r-', label='Electrons', linewidth=2)
            axes[0, 1].semilogy(time_ps, p_center, 'b-', label='Holes', linewidth=2)
            axes[0, 1].set_xlabel('Time (ps)')
            axes[0, 1].set_ylabel('Density (m^-3)')
            axes[0, 1].set_title('Carrier Density Evolution')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Plot 3: Spatial potential profile at different times
            time_indices = [0, len(results['time'])//3, 2*len(results['time'])//3, -1]
            colors = ['blue', 'green', 'orange', 'red']
            
            for i, (t_idx, color) in enumerate(zip(time_indices, colors)):
                if t_idx < len(results['potential']):
                    axes[1, 0].plot(results['potential'][t_idx], color=color, 
                                   label=f't = {time_ps[t_idx]:.1f} ps', linewidth=2)
            
            axes[1, 0].set_xlabel('Grid Point')
            axes[1, 0].set_ylabel('Potential (V)')
            axes[1, 0].set_title('Spatial Potential Profiles')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Plot 4: Energy vs time (simplified)
            total_energy = []
            for i in range(len(results['time'])):
                # Simple energy estimate: ∫ ε|∇V|²/2 dV (simplified)
                V = results['potential'][i]
                energy = np.sum(np.gradient(V)**2)  # Simplified
                total_energy.append(energy)
            
            axes[1, 1].plot(time_ps, total_energy, 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Time (ps)')
            axes[1, 1].set_ylabel('Energy (arb. units)')
            axes[1, 1].set_title('System Energy Evolution')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(os.path.dirname(__file__), 'tutorial_2_transient_results.png')
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"✓ Transient results plotted and saved to {plot_file}")
            
            # Show plot if in interactive mode
            try:
                plt.show()
            except:
                pass
        
        return True
        
    except Exception as e:
        print(f"✗ Tutorial 2.4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tutorials():
    """Run all transient simulation tutorials"""
    print("SemiDGFEM Transient Simulation Tutorial")
    print("=" * 60)
    print("This tutorial covers transient simulation capabilities including")
    print("time integration methods, dynamic boundary conditions, and")
    print("time-dependent analysis of semiconductor devices.\n")
    
    tutorials = [
        tutorial_1_basic_transient,
        tutorial_2_time_integrators,
        tutorial_3_dynamic_boundary_conditions,
        tutorial_4_transient_visualization
    ]
    
    results = []
    tutorial_results = {}
    
    for i, tutorial in enumerate(tutorials, 1):
        try:
            result = tutorial()
            success = result is not None
            results.append(success)
            tutorial_results[f"tutorial_2_{i}"] = result
            
            if success:
                print(f"✓ Tutorial 2.{i} completed successfully")
            else:
                print(f"✗ Tutorial 2.{i} failed")
        except Exception as e:
            print(f"✗ Tutorial 2.{i} crashed: {e}")
            results.append(False)
            tutorial_results[f"tutorial_2_{i}"] = None
    
    # Summary
    print("\n" + "=" * 60)
    print("TRANSIENT TUTORIAL SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tutorials passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All transient tutorials completed successfully!")
        print("You have mastered transient simulation with SemiDGFEM.")
    else:
        print("⚠️  Some tutorials failed. Please check the error messages above.")
        print("Transient simulation requires careful numerical setup.")
    
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
