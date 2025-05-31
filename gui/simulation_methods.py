#!/usr/bin/env python3
"""
SemiDGFEM GUI - Simulation Methods
Contains all simulation-related methods for the GUI
"""

import numpy as np
import matplotlib.pyplot as plt
from tkinter import messagebox
import sys
import os

# Add parent directory for simulator import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import simulator
    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False

class SimulationMethods:
    """Mixin class containing simulation methods for the GUI"""
    
    def create_device(self):
        """Create device with current parameters"""
        
        try:
            self.log("Creating device...")
            self.status_var.set("Creating device...")
            
            # Get parameters
            width = float(self.width_var.get()) * 1e-6  # Convert μm to m
            height = float(self.height_var.get()) * 1e-6
            nx = int(self.nx_var.get())
            ny = int(self.ny_var.get())
            method = self.method_var.get()
            
            if SIMULATOR_AVAILABLE:
                # Create real simulator
                self.sim = simulator.Simulator(
                    extents=[width, height],
                    num_points_x=nx,
                    num_points_y=ny,
                    method=method,
                    mesh_type="Structured"
                )
                
                # Set up doping
                self.setup_doping()
                
                self.log(f"✓ Device created: {width*1e6:.1f}μm × {height*1e6:.1f}μm")
                self.log(f"✓ Grid: {nx} × {ny}, Method: {method}")
                
            else:
                # Create mock simulator for demo
                self.sim = self.create_mock_simulator(nx, ny)
                self.log("✓ Demo device created (simulator not available)")
            
            self.status_var.set("Device ready")
            
        except Exception as e:
            error_msg = f"Failed to create device: {str(e)}"
            self.log(f"✗ {error_msg}")
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Error")
    
    def setup_doping(self):
        """Setup doping profile"""
        
        if not self.sim:
            return
        
        try:
            # Get doping parameters
            nd_val = float(self.nd_var.get())
            na_val = float(self.na_var.get())
            
            # Create simple P-N junction
            total_points = self.sim.num_points_x * self.sim.num_points_y
            junction_point = self.sim.num_points_x // 2
            
            Nd = np.zeros(total_points)
            Na = np.zeros(total_points)
            
            for i in range(total_points):
                x_idx = i % self.sim.num_points_x
                if x_idx < junction_point:
                    Na[i] = na_val  # P-type (left)
                else:
                    Nd[i] = nd_val  # N-type (right)
            
            if SIMULATOR_AVAILABLE:
                self.sim.set_doping(Nd, Na)
            
            self.log(f"✓ Doping set: P-type={na_val:.1e}, N-type={nd_val:.1e}")
            
        except Exception as e:
            self.log(f"⚠ Doping setup warning: {str(e)}")
    
    def create_mock_simulator(self, nx, ny):
        """Create mock simulator for demo mode"""
        
        class MockSimulator:
            def __init__(self, nx, ny):
                self.num_points_x = nx
                self.num_points_y = ny
                
            def solve_poisson(self, bc):
                # Generate realistic-looking potential
                x = np.linspace(0, 1, self.num_points_x)
                y = np.linspace(0, 1, self.num_points_y)
                X, Y = np.meshgrid(x, y)
                
                # Linear potential with some variation
                V = bc[0] + (bc[1] - bc[0]) * X + 0.1 * np.sin(np.pi * X) * np.sin(np.pi * Y)
                return V.flatten()
                
            def solve_drift_diffusion(self, bc, **kwargs):
                V = self.solve_poisson(bc)
                n = self.num_points_x * self.num_points_y
                
                # Mock carrier densities
                x = np.linspace(0, 1, self.num_points_x)
                y = np.linspace(0, 1, self.num_points_y)
                X, Y = np.meshgrid(x, y)
                
                # Electron density (higher on right side)
                n_carriers = 1e16 * (1 + X) * np.exp(-V.reshape(self.num_points_y, self.num_points_x) / 0.026)
                
                # Hole density (higher on left side)
                p_carriers = 1e16 * (2 - X) * np.exp(V.reshape(self.num_points_y, self.num_points_x) / 0.026)
                
                return {
                    'potential': V,
                    'n': n_carriers.flatten(),
                    'p': p_carriers.flatten(),
                    'Jn': np.gradient(n_carriers.flatten()) * 1e-3,
                    'Jp': np.gradient(p_carriers.flatten()) * 1e-3
                }
        
        return MockSimulator(nx, ny)
    
    def run_simulation(self):
        """Run complete simulation"""
        
        if not self.sim:
            messagebox.showwarning("Warning", "Please create device first")
            return
        
        try:
            self.log("Running simulation...")
            self.status_var.set("Running simulation...")
            
            # Get boundary conditions
            bc = [
                float(self.bc_left_var.get()),
                float(self.bc_right_var.get()),
                0.0,  # Bottom
                0.0   # Top
            ]
            
            # Run drift-diffusion simulation
            if hasattr(self.sim, 'solve_drift_diffusion'):
                self.results = self.sim.solve_drift_diffusion(
                    bc=bc,
                    max_steps=30,
                    use_amr=False,
                    poisson_tol=1e-6
                )
                self.log("✓ Drift-diffusion simulation completed")
            else:
                self.log("⚠ Using Poisson-only simulation")
                V = self.sim.solve_poisson(bc)
                self.results = {'potential': V}
            
            # Update plots and data
            self.update_plots()
            self.update_data_display()
            
            self.status_var.set("Simulation completed")
            
        except Exception as e:
            error_msg = f"Simulation failed: {str(e)}"
            self.log(f"✗ {error_msg}")
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Error")
    
    def run_poisson(self):
        """Run Poisson equation only"""
        
        if not self.sim:
            messagebox.showwarning("Warning", "Please create device first")
            return
        
        try:
            self.log("Running Poisson solver...")
            self.status_var.set("Running Poisson...")
            
            bc = [
                float(self.bc_left_var.get()),
                float(self.bc_right_var.get()),
                0.0, 0.0
            ]
            
            V = self.sim.solve_poisson(bc)
            self.results = {'potential': V}
            
            self.update_plots()
            self.update_data_display()
            
            self.log("✓ Poisson equation solved")
            self.status_var.set("Poisson completed")
            
        except Exception as e:
            error_msg = f"Poisson solver failed: {str(e)}"
            self.log(f"✗ {error_msg}")
            messagebox.showerror("Error", error_msg)
    
    def run_drift_diffusion(self):
        """Run drift-diffusion equations"""
        self.run_simulation()  # Same as full simulation for now
    
    def run_full_analysis(self):
        """Run comprehensive analysis"""
        
        if not self.sim:
            messagebox.showwarning("Warning", "Please create device first")
            return
        
        try:
            self.log("Running full analysis...")
            self.status_var.set("Running full analysis...")
            
            # Run multiple simulations with different parameters
            voltages = [0.1, 0.3, 0.5, 0.7]
            analysis_results = {}
            
            for V in voltages:
                bc = [0.0, V, 0.0, 0.0]
                
                if hasattr(self.sim, 'solve_drift_diffusion'):
                    result = self.sim.solve_drift_diffusion(bc=bc, max_steps=20)
                else:
                    result = {'potential': self.sim.solve_poisson(bc)}
                
                analysis_results[f'V_{V}V'] = result
                self.log(f"  ✓ Analysis at {V}V completed")
            
            # Store results and update display
            self.results = analysis_results[f'V_{voltages[-1]}V']  # Show last result
            self.analysis_results = analysis_results
            
            self.update_plots()
            self.update_data_display()
            
            self.log("✓ Full analysis completed")
            self.status_var.set("Analysis completed")
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.log(f"✗ {error_msg}")
            messagebox.showerror("Error", error_msg)
    
    def update_plots(self):
        """Update all plots with current results"""
        
        if not self.results:
            return
        
        try:
            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Get grid dimensions
            nx = self.sim.num_points_x
            ny = self.sim.num_points_y
            
            # Create coordinate arrays
            width = float(self.width_var.get())
            height = float(self.height_var.get())
            extent = [0, width, 0, height]
            
            # Plot potential
            if 'potential' in self.results:
                V = self.results['potential'].reshape(ny, nx)
                im1 = self.axes[0, 0].imshow(V, extent=extent, aspect='auto', origin='lower')
                self.axes[0, 0].set_title('Electrostatic Potential (V)')
                self.axes[0, 0].set_xlabel('x (μm)')
                self.axes[0, 0].set_ylabel('y (μm)')
                plt.colorbar(im1, ax=self.axes[0, 0])
            
            # Plot electron density
            if 'n' in self.results:
                n = self.results['n'].reshape(ny, nx)
                im2 = self.axes[0, 1].imshow(np.log10(np.maximum(n, 1e10)), 
                                           extent=extent, aspect='auto', origin='lower')
                self.axes[0, 1].set_title('Electron Density (log₁₀ /m³)')
                self.axes[0, 1].set_xlabel('x (μm)')
                self.axes[0, 1].set_ylabel('y (μm)')
                plt.colorbar(im2, ax=self.axes[0, 1])
            
            # Plot hole density
            if 'p' in self.results:
                p = self.results['p'].reshape(ny, nx)
                im3 = self.axes[1, 0].imshow(np.log10(np.maximum(p, 1e10)), 
                                           extent=extent, aspect='auto', origin='lower')
                self.axes[1, 0].set_title('Hole Density (log₁₀ /m³)')
                self.axes[1, 0].set_xlabel('x (μm)')
                self.axes[1, 0].set_ylabel('y (μm)')
                plt.colorbar(im3, ax=self.axes[1, 0])
            
            # Plot current density magnitude
            if 'Jn' in self.results and 'Jp' in self.results:
                Jn = self.results['Jn'].reshape(ny, nx)
                Jp = self.results['Jp'].reshape(ny, nx)
                J_mag = np.sqrt(Jn**2 + Jp**2)
                im4 = self.axes[1, 1].imshow(np.log10(np.maximum(J_mag, 1e-10)), 
                                           extent=extent, aspect='auto', origin='lower')
                self.axes[1, 1].set_title('Current Density (log₁₀ A/m²)')
                self.axes[1, 1].set_xlabel('x (μm)')
                self.axes[1, 1].set_ylabel('y (μm)')
                plt.colorbar(im4, ax=self.axes[1, 1])
            
            # Refresh canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
            self.log("✓ Plots updated")
            
        except Exception as e:
            self.log(f"⚠ Plot update warning: {str(e)}")
    
    def update_data_display(self):
        """Update data display with simulation results"""
        
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if not self.results:
            return
        
        try:
            # Add simulation parameters
            self.data_tree.insert("", "end", values=("Device Width", f"{self.width_var.get()}", "μm"))
            self.data_tree.insert("", "end", values=("Device Height", f"{self.height_var.get()}", "μm"))
            self.data_tree.insert("", "end", values=("Grid Points", f"{self.nx_var.get()} × {self.ny_var.get()}", ""))
            self.data_tree.insert("", "end", values=("Method", self.method_var.get(), ""))
            
            # Add separator
            self.data_tree.insert("", "end", values=("", "", ""))
            
            # Add results statistics
            if 'potential' in self.results:
                V = self.results['potential']
                self.data_tree.insert("", "end", values=("Min Potential", f"{np.min(V):.3f}", "V"))
                self.data_tree.insert("", "end", values=("Max Potential", f"{np.max(V):.3f}", "V"))
                self.data_tree.insert("", "end", values=("Potential Range", f"{np.max(V) - np.min(V):.3f}", "V"))
            
            if 'n' in self.results:
                n = self.results['n']
                self.data_tree.insert("", "end", values=("Max Electron Density", f"{np.max(n):.2e}", "/m³"))
                self.data_tree.insert("", "end", values=("Min Electron Density", f"{np.min(n):.2e}", "/m³"))
            
            if 'p' in self.results:
                p = self.results['p']
                self.data_tree.insert("", "end", values=("Max Hole Density", f"{np.max(p):.2e}", "/m³"))
                self.data_tree.insert("", "end", values=("Min Hole Density", f"{np.min(p):.2e}", "/m³"))
            
            self.log("✓ Data display updated")
            
        except Exception as e:
            self.log(f"⚠ Data update warning: {str(e)}")
    
    def clear_results(self):
        """Clear all results and plots"""
        
        self.results = {}
        
        # Clear plots
        for ax in self.axes.flat:
            ax.clear()
            ax.set_xlabel("x (μm)")
            ax.set_ylabel("y (μm)")
            ax.grid(True, alpha=0.3)
        
        self.axes[0, 0].set_title("Electrostatic Potential")
        self.axes[0, 1].set_title("Electron Density")
        self.axes[1, 0].set_title("Hole Density")
        self.axes[1, 1].set_title("Current Density")
        
        self.canvas.draw()
        
        # Clear data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        self.log("✓ Results cleared")
        self.status_var.set("Ready")
