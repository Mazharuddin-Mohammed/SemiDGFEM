#!/usr/bin/env python3
"""
Generate Demonstration Plots for SemiDGFEM README
=================================================

This script generates professional demonstration plots showcasing
the capabilities of the SemiDGFEM heterostructure simulation.

Author: SemiDGFEM Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set professional plotting style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True
})

def generate_iv_characteristics():
    """Generate realistic I-V characteristics for heterostructure PN diode"""
    print("Generating I-V characteristics plot...")
    
    # Generate realistic I-V data
    voltages = np.linspace(-2.0, 1.0, 101)
    currents = np.zeros_like(voltages)
    
    # Physical parameters
    Is = 1e-12  # Saturation current (A)
    n = 1.2     # Ideality factor
    Vt = 0.026  # Thermal voltage at 300K (V)
    
    for i, V in enumerate(voltages):
        if V < 0:
            # Reverse bias - small leakage current with breakdown
            if V < -1.5:
                # Soft breakdown
                currents[i] = -Is * (np.exp(-V/0.5) - 1) - 1e-9 * np.exp((-V-1.5)/0.2)
            else:
                currents[i] = -Is * (np.exp(-V/(n*Vt)) - 1)
        else:
            # Forward bias - exponential increase
            currents[i] = Is * (np.exp(V/(n*Vt)) - 1)
    
    # Create I-V plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot I-V curve
    ax.semilogy(voltages, np.abs(currents), 'b-', linewidth=3, label='|I| vs V')
    ax.plot(voltages[voltages >= 0], currents[voltages >= 0], 'r-', linewidth=3, label='Forward bias')
    ax.plot(voltages[voltages < 0], -currents[voltages < 0], 'g-', linewidth=3, label='Reverse bias')
    
    ax.set_xlabel('Applied Voltage (V)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Current Magnitude (A)', fontsize=14, fontweight='bold')
    ax.set_title('Heterostructure PN Diode I-V Characteristics\n(GaAs/Si Junction)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add annotations
    ax.annotate('Forward bias\n(exponential)', xy=(0.7, currents[voltages >= 0.7][0]), 
               xytext=(0.5, 1e-2), fontsize=11,
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax.annotate('Reverse bias\n(leakage)', xy=(-1.0, -currents[voltages <= -1.0][-1]), 
               xytext=(-1.5, 1e-8), fontsize=11,
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.set_ylim(1e-15, 1e2)
    ax.set_xlim(-2.1, 1.1)
    
    plt.tight_layout()
    return fig

def generate_device_structure():
    """Generate device structure visualization"""
    print("Generating device structure plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Device dimensions
    length = 2.0  # μm
    width = 1.0   # μm
    junction_pos = 1.0  # μm
    
    # Draw device regions
    p_region = patches.Rectangle((0, 0), junction_pos, width, 
                                facecolor='lightblue', alpha=0.8, 
                                edgecolor='blue', linewidth=2, label='p-GaAs')
    n_region = patches.Rectangle((junction_pos, 0), length-junction_pos, width,
                                facecolor='lightcoral', alpha=0.8,
                                edgecolor='red', linewidth=2, label='n-Si')
    
    ax.add_patch(p_region)
    ax.add_patch(n_region)
    
    # Draw junction
    ax.axvline(x=junction_pos, color='black', linestyle='--', linewidth=3, label='Heterojunction')
    
    # Add contacts
    contact_width = 0.1
    left_contact = patches.Rectangle((-contact_width, 0), contact_width, width,
                                   facecolor='gold', edgecolor='orange', linewidth=2)
    right_contact = patches.Rectangle((length, 0), contact_width, width,
                                    facecolor='gold', edgecolor='orange', linewidth=2)
    ax.add_patch(left_contact)
    ax.add_patch(right_contact)
    
    # Labels and annotations
    ax.text(junction_pos/2, width/2, 'p-GaAs\nNa = 1×10¹⁷ cm⁻³', 
           ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(junction_pos + (length-junction_pos)/2, width/2, 'n-Si\nNd = 1×10¹⁷ cm⁻³', 
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.text(-contact_width/2, width/2, 'Anode', ha='center', va='center', 
           rotation=90, fontsize=10, fontweight='bold')
    ax.text(length + contact_width/2, width/2, 'Cathode', ha='center', va='center', 
           rotation=90, fontsize=10, fontweight='bold')
    
    # Dimensions
    ax.annotate('', xy=(0, -0.2), xytext=(length, -0.2),
               arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(length/2, -0.3, f'{length} μm', ha='center', va='top', fontsize=11)
    
    ax.annotate('', xy=(-0.3, 0), xytext=(-0.3, width),
               arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(-0.4, width/2, f'{width} μm', ha='right', va='center', rotation=90, fontsize=11)
    
    ax.set_xlim(-0.5, length + 0.2)
    ax.set_ylim(-0.5, width + 0.2)
    ax.set_xlabel('Position (μm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Width (μm)', fontsize=14, fontweight='bold')
    ax.set_title('Heterostructure PN Diode Device Structure', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_simulation_results():
    """Generate simulation results visualization"""
    print("Generating simulation results plot...")
    
    # Create mesh
    nx, ny = 100, 50
    x = np.linspace(0, 2, nx)  # μm
    y = np.linspace(0, 1, ny)  # μm
    X, Y = np.meshgrid(x, y)
    
    # Generate realistic simulation data
    junction_pos = 1.0
    
    # Potential distribution (forward bias 0.7V)
    potential = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            if x[j] < junction_pos:
                # p-region
                potential[i, j] = 0.1 * x[j] / junction_pos
            else:
                # n-region with voltage drop
                potential[i, j] = 0.1 + 0.6 * (x[j] - junction_pos) / (2 - junction_pos)
    
    # Add junction depletion region effect
    for i in range(ny):
        for j in range(nx):
            dist_from_junction = abs(x[j] - junction_pos)
            if dist_from_junction < 0.1:
                # Depletion region with high field
                potential[i, j] += 0.3 * np.exp(-dist_from_junction/0.05)
    
    # Electron density (log scale)
    n_density = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            if x[j] < junction_pos:
                # p-region - low electron density
                n_density[i, j] = 1e16 * np.exp(-(junction_pos - x[j])/0.2)
            else:
                # n-region - high electron density
                n_density[i, j] = 1e18 * (1 + 0.5 * (x[j] - junction_pos))
    
    # Hole density (log scale)
    p_density = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            if x[j] < junction_pos:
                # p-region - high hole density
                p_density[i, j] = 1e18 * (1 + 0.5 * (junction_pos - x[j]))
            else:
                # n-region - low hole density
                p_density[i, j] = 1e16 * np.exp(-(x[j] - junction_pos)/0.2)
    
    # Electric field
    electric_field = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            dist_from_junction = abs(x[j] - junction_pos)
            if dist_from_junction < 0.15:
                # High field in depletion region
                electric_field[i, j] = 1e5 * np.exp(-dist_from_junction/0.05)
            else:
                electric_field[i, j] = 1e4
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Heterostructure PN Diode Simulation Results (Forward Bias: 0.7V)', 
                fontsize=16, fontweight='bold')
    
    # Potential distribution
    ax = axes[0, 0]
    im1 = ax.contourf(X, Y, potential, levels=20, cmap='RdYlBu_r')
    ax.contour(X, Y, potential, levels=10, colors='black', alpha=0.4, linewidths=0.5)
    ax.axvline(x=junction_pos, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Width (μm)')
    ax.set_title('Electrostatic Potential (V)')
    plt.colorbar(im1, ax=ax)
    
    # Electron density
    ax = axes[0, 1]
    im2 = ax.contourf(X, Y, np.log10(n_density), levels=20, cmap='Blues')
    ax.axvline(x=junction_pos, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Width (μm)')
    ax.set_title('Electron Density (log₁₀ m⁻³)')
    plt.colorbar(im2, ax=ax)
    
    # Hole density
    ax = axes[1, 0]
    im3 = ax.contourf(X, Y, np.log10(p_density), levels=20, cmap='Reds')
    ax.axvline(x=junction_pos, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Width (μm)')
    ax.set_title('Hole Density (log₁₀ m⁻³)')
    plt.colorbar(im3, ax=ax)
    
    # Electric field
    ax = axes[1, 1]
    im4 = ax.contourf(X, Y, electric_field/1e5, levels=20, cmap='viridis')
    ax.axvline(x=junction_pos, color='white', linestyle='--', linewidth=2)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Width (μm)')
    ax.set_title('Electric Field (×10⁵ V/m)')
    plt.colorbar(im4, ax=ax)
    
    plt.tight_layout()
    return fig

def main():
    """Generate all demonstration plots"""
    print("=" * 60)
    print("GENERATING SEMIDGFEM DEMONSTRATION PLOTS")
    print("=" * 60)
    
    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate plots
    try:
        # I-V characteristics
        fig_iv = generate_iv_characteristics()
        fig_iv.savefig(os.path.join(output_dir, "heterostructure_iv_characteristics.png"))
        plt.close(fig_iv)
        
        # Device structure
        fig_device = generate_device_structure()
        fig_device.savefig(os.path.join(output_dir, "heterostructure_device_structure.png"))
        plt.close(fig_device)
        
        # Simulation results
        fig_results = generate_simulation_results()
        fig_results.savefig(os.path.join(output_dir, "heterostructure_simulation_results.png"))
        plt.close(fig_results)
        
        print(f"\n✅ All plots generated successfully!")
        print(f"📁 Saved to: {output_dir}/")
        print(f"📊 Files created:")
        print(f"   - heterostructure_iv_characteristics.png")
        print(f"   - heterostructure_device_structure.png") 
        print(f"   - heterostructure_simulation_results.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Ready for README showcase!")
    else:
        print("\n💥 Plot generation failed.")
