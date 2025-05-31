#!/usr/bin/env python3
"""
SemiDGFEM Tutorial: P-N Junction Simulation
==========================================

This tutorial demonstrates how to simulate a basic p-n junction diode
using the SemiDGFEM semiconductor device simulator.

Topics covered:
- Device geometry setup
- Doping profile definition
- Boundary condition specification
- Forward and reverse bias simulation
- Results visualization and analysis
- Performance optimization

Author: SemiDGFEM Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import SemiDGFEM modules
try:
    from simulator import Simulator, Device, Method, MeshType
    from simulator.gpu import GPUContext, GPUBackend
    from simulator.visualization import plot_potential, plot_carrier_density, plot_current_density
    from simulator.utils import save_results, compute_iv_curve
except ImportError as e:
    print(f"Error importing SemiDGFEM: {e}")
    print("Please ensure SemiDGFEM is properly installed.")
    exit(1)

def create_pn_junction_device():
    """
    Create a p-n junction device with specified geometry and doping.
    
    Device structure:
    - Total length: 2 μm
    - Total width: 1 μm  
    - p-region: 0 to 1 μm (left side)
    - n-region: 1 to 2 μm (right side)
    """
    print("Creating p-n junction device...")
    
    # Device dimensions
    Lx = 2e-6  # 2 μm length
    Ly = 1e-6  # 1 μm width
    
    # Create device
    device = Device(Lx=Lx, Ly=Ly)
    
    print(f"Device created: {Lx*1e6:.1f} μm × {Ly*1e6:.1f} μm")
    return device

def setup_doping_profile(simulator, junction_position=1e-6):
    """
    Set up the doping profile for the p-n junction.
    
    Args:
        simulator: SemiDGFEM simulator instance
        junction_position: Position of p-n junction in meters
    """
    print("Setting up doping profile...")
    
    # Get mesh information
    n_dofs = simulator.get_dof_count()
    print(f"Number of DOFs: {n_dofs}")
    
    # For this tutorial, we'll use a simplified approach
    # In practice, you would get the actual mesh coordinates
    
    # Create doping arrays
    Nd = np.zeros(n_dofs)  # Donor concentration (cm⁻³)
    Na = np.zeros(n_dofs)  # Acceptor concentration (cm⁻³)
    
    # Simplified doping profile
    # Assume uniform distribution across DOFs
    # First half: p-type (Na = 1e16 cm⁻³)
    # Second half: n-type (Nd = 1e16 cm⁻³)
    
    mid_point = n_dofs // 2
    
    # p-region (left side)
    Na[:mid_point] = 1e16 * 1e6  # Convert cm⁻³ to m⁻³
    Nd[:mid_point] = 0
    
    # n-region (right side)  
    Nd[mid_point:] = 1e16 * 1e6  # Convert cm⁻³ to m⁻³
    Na[mid_point:] = 0
    
    # Set doping in simulator
    simulator.set_doping(Nd, Na)
    
    print(f"Doping set: p-region Na = 1e16 cm⁻³, n-region Nd = 1e16 cm⁻³")
    return Nd, Na

def run_equilibrium_simulation(simulator):
    """
    Run equilibrium simulation (zero bias).
    
    Args:
        simulator: SemiDGFEM simulator instance
        
    Returns:
        dict: Simulation results
    """
    print("\nRunning equilibrium simulation...")
    
    # Boundary conditions: all contacts at 0V
    bc = [0.0, 0.0, 0.0, 0.0]  # [left, right, bottom, top]
    
    start_time = time.time()
    
    # Run simulation
    results = simulator.solve_drift_diffusion(
        bc=bc,
        Vg=0.0,           # No gate voltage
        max_steps=50,     # Maximum iterations
        use_amr=True,     # Enable adaptive mesh refinement
        poisson_max_iter=100,
        poisson_tol=1e-8
    )
    
    simulation_time = time.time() - start_time
    
    print(f"Equilibrium simulation completed in {simulation_time:.2f} seconds")
    print(f"Convergence residual: {simulator.get_convergence_residual():.2e}")
    
    return results

def run_forward_bias_simulation(simulator, voltage=0.7):
    """
    Run forward bias simulation.
    
    Args:
        simulator: SemiDGFEM simulator instance
        voltage: Applied voltage in volts
        
    Returns:
        dict: Simulation results
    """
    print(f"\nRunning forward bias simulation (V = {voltage:.1f} V)...")
    
    # Boundary conditions: voltage applied to right contact
    bc = [0.0, voltage, 0.0, 0.0]  # [left, right, bottom, top]
    
    start_time = time.time()
    
    # Run simulation
    results = simulator.solve_drift_diffusion(
        bc=bc,
        Vg=0.0,
        max_steps=100,    # More iterations for bias conditions
        use_amr=True,
        poisson_max_iter=100,
        poisson_tol=1e-8
    )
    
    simulation_time = time.time() - start_time
    
    print(f"Forward bias simulation completed in {simulation_time:.2f} seconds")
    print(f"Convergence residual: {simulator.get_convergence_residual():.2e}")
    
    return results

def run_reverse_bias_simulation(simulator, voltage=-5.0):
    """
    Run reverse bias simulation.
    
    Args:
        simulator: SemiDGFEM simulator instance
        voltage: Applied voltage in volts (negative)
        
    Returns:
        dict: Simulation results
    """
    print(f"\nRunning reverse bias simulation (V = {voltage:.1f} V)...")
    
    # Boundary conditions: negative voltage applied to right contact
    bc = [0.0, voltage, 0.0, 0.0]  # [left, right, bottom, top]
    
    start_time = time.time()
    
    # Run simulation
    results = simulator.solve_drift_diffusion(
        bc=bc,
        Vg=0.0,
        max_steps=100,
        use_amr=True,
        poisson_max_iter=100,
        poisson_tol=1e-8
    )
    
    simulation_time = time.time() - start_time
    
    print(f"Reverse bias simulation completed in {simulation_time:.2f} seconds")
    print(f"Convergence residual: {simulator.get_convergence_residual():.2e}")
    
    return results

def analyze_results(results, title="Simulation Results"):
    """
    Analyze and print key results from simulation.
    
    Args:
        results: Dictionary containing simulation results
        title: Title for the analysis
    """
    print(f"\n{title}:")
    print("=" * len(title))
    
    # Extract key quantities
    potential = results['potential']
    n_density = results['n']
    p_density = results['p']
    Jn = results['Jn']
    Jp = results['Jp']
    
    # Calculate statistics
    print(f"Potential range: {np.min(potential):.3f} to {np.max(potential):.3f} V")
    print(f"Electron density range: {np.min(n_density):.2e} to {np.max(n_density):.2e} m⁻³")
    print(f"Hole density range: {np.min(p_density):.2e} to {np.max(p_density):.2e} m⁻³")
    
    # Calculate total current (simplified)
    total_current_n = np.sum(Jn)
    total_current_p = np.sum(Jp)
    total_current = total_current_n + total_current_p
    
    print(f"Total electron current: {total_current_n:.2e} A")
    print(f"Total hole current: {total_current_p:.2e} A")
    print(f"Total current: {total_current:.2e} A")

def visualize_results(results_equilibrium, results_forward, results_reverse, output_dir="output"):
    """
    Create visualizations of the simulation results.
    
    Args:
        results_equilibrium: Equilibrium simulation results
        results_forward: Forward bias simulation results  
        results_reverse: Reverse bias simulation results
        output_dir: Directory to save plots
    """
    print("\nCreating visualizations...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up the plot style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # Plot 1: Potential comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Equilibrium potential
    axes[0].plot(results_equilibrium['potential'], 'b-', linewidth=2)
    axes[0].set_title('Equilibrium Potential')
    axes[0].set_xlabel('Position Index')
    axes[0].set_ylabel('Potential (V)')
    axes[0].grid(True, alpha=0.3)
    
    # Forward bias potential
    axes[1].plot(results_forward['potential'], 'r-', linewidth=2)
    axes[1].set_title('Forward Bias Potential (0.7V)')
    axes[1].set_xlabel('Position Index')
    axes[1].set_ylabel('Potential (V)')
    axes[1].grid(True, alpha=0.3)
    
    # Reverse bias potential
    axes[2].plot(results_reverse['potential'], 'g-', linewidth=2)
    axes[2].set_title('Reverse Bias Potential (-5V)')
    axes[2].set_xlabel('Position Index')
    axes[2].set_ylabel('Potential (V)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/potential_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Carrier densities
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Electron densities
    axes[0,0].semilogy(results_equilibrium['n'], 'b-', linewidth=2)
    axes[0,0].set_title('Equilibrium Electron Density')
    axes[0,0].set_ylabel('n (m⁻³)')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].semilogy(results_forward['n'], 'r-', linewidth=2)
    axes[0,1].set_title('Forward Bias Electron Density')
    axes[0,1].set_ylabel('n (m⁻³)')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].semilogy(results_reverse['n'], 'g-', linewidth=2)
    axes[0,2].set_title('Reverse Bias Electron Density')
    axes[0,2].set_ylabel('n (m⁻³)')
    axes[0,2].grid(True, alpha=0.3)
    
    # Hole densities
    axes[1,0].semilogy(results_equilibrium['p'], 'b-', linewidth=2)
    axes[1,0].set_title('Equilibrium Hole Density')
    axes[1,0].set_xlabel('Position Index')
    axes[1,0].set_ylabel('p (m⁻³)')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].semilogy(results_forward['p'], 'r-', linewidth=2)
    axes[1,1].set_title('Forward Bias Hole Density')
    axes[1,1].set_xlabel('Position Index')
    axes[1,1].set_ylabel('p (m⁻³)')
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].semilogy(results_reverse['p'], 'g-', linewidth=2)
    axes[1,2].set_title('Reverse Bias Hole Density')
    axes[1,2].set_xlabel('Position Index')
    axes[1,2].set_ylabel('p (m⁻³)')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/carrier_densities.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to {output_dir}/")

def demonstrate_gpu_acceleration():
    """
    Demonstrate GPU acceleration capabilities.
    """
    print("\n" + "="*50)
    print("GPU ACCELERATION DEMONSTRATION")
    print("="*50)
    
    # Check GPU availability
    gpu_ctx = GPUContext.instance()
    gpu_available = gpu_ctx.initialize(GPUBackend.AUTO)
    
    if gpu_available:
        device_info = gpu_ctx.get_device_info()
        print(f"GPU detected: {device_info.name}")
        print(f"Global memory: {device_info.global_memory / 1e9:.1f} GB")
        print(f"Compute capability: {device_info.compute_capability_major}.{device_info.compute_capability_minor}")
        
        # TODO: Add GPU performance comparison
        print("GPU acceleration is available but not demonstrated in this basic tutorial.")
        print("See advanced examples for GPU performance comparisons.")
    else:
        print("No GPU detected or GPU support not available.")
        print("Running on CPU only.")

def main():
    """
    Main tutorial function.
    """
    print("SemiDGFEM P-N Junction Tutorial")
    print("=" * 40)
    
    try:
        # Step 1: Create device
        device = create_pn_junction_device()
        
        # Step 2: Create simulator
        print("\nCreating simulator with P3 DG method...")
        simulator = Simulator(
            device=device,
            method=Method.DG,
            mesh_type=MeshType.Structured,
            order=3  # P3 elements
        )
        
        # Enable optimizations
        simulator.enable_simd(True)
        simulator.set_num_threads(-1)  # Use all available cores
        
        print(f"Simulator created with {simulator.get_dof_count()} DOFs")
        
        # Step 3: Set up doping
        Nd, Na = setup_doping_profile(simulator)
        
        # Step 4: Run simulations
        results_eq = run_equilibrium_simulation(simulator)
        results_fwd = run_forward_bias_simulation(simulator, voltage=0.7)
        results_rev = run_reverse_bias_simulation(simulator, voltage=-5.0)
        
        # Step 5: Analyze results
        analyze_results(results_eq, "Equilibrium Results")
        analyze_results(results_fwd, "Forward Bias Results")
        analyze_results(results_rev, "Reverse Bias Results")
        
        # Step 6: Visualize results
        visualize_results(results_eq, results_fwd, results_rev)
        
        # Step 7: Save results
        print("\nSaving results...")
        save_results(results_eq, "pn_junction_equilibrium.h5")
        save_results(results_fwd, "pn_junction_forward.h5")
        save_results(results_rev, "pn_junction_reverse.h5")
        
        # Step 8: Demonstrate GPU capabilities
        demonstrate_gpu_acceleration()
        
        print("\n" + "="*50)
        print("TUTORIAL COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nNext steps:")
        print("1. Explore the generated plots in the output/ directory")
        print("2. Modify doping concentrations and re-run")
        print("3. Try different bias voltages")
        print("4. Experiment with mesh refinement parameters")
        print("5. Check out advanced examples for more complex devices")
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        print("Please check your installation and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
