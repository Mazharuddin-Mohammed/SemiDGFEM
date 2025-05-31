#!/usr/bin/env python3
"""
SemiDGFEM GUI - Menu Methods
Contains all menu-related methods for the GUI
"""

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import json
import os

class MenuMethods:
    """Mixin class containing menu methods for the GUI"""
    
    def new_simulation(self):
        """Start a new simulation"""
        
        result = messagebox.askyesno("New Simulation", 
                                   "This will clear current results. Continue?")
        if result:
            self.clear_results()
            self.sim = None
            self.log("✓ New simulation started")
            self.status_var.set("Ready for new simulation")
    
    def load_parameters(self):
        """Load simulation parameters from file"""
        
        filename = filedialog.askopenfilename(
            title="Load Parameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    params = json.load(f)
                
                # Update GUI parameters
                if 'width' in params:
                    self.width_var.set(str(params['width']))
                if 'height' in params:
                    self.height_var.set(str(params['height']))
                if 'nx' in params:
                    self.nx_var.set(str(params['nx']))
                if 'ny' in params:
                    self.ny_var.set(str(params['ny']))
                if 'method' in params:
                    self.method_var.set(params['method'])
                if 'nd' in params:
                    self.nd_var.set(str(params['nd']))
                if 'na' in params:
                    self.na_var.set(str(params['na']))
                if 'bc_left' in params:
                    self.bc_left_var.set(str(params['bc_left']))
                if 'bc_right' in params:
                    self.bc_right_var.set(str(params['bc_right']))
                
                self.log(f"✓ Parameters loaded from {os.path.basename(filename)}")
                messagebox.showinfo("Success", "Parameters loaded successfully")
                
            except Exception as e:
                error_msg = f"Failed to load parameters: {str(e)}"
                self.log(f"✗ {error_msg}")
                messagebox.showerror("Error", error_msg)
    
    def save_results(self):
        """Save simulation results to file"""
        
        if not self.results:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Prepare data for saving
                save_data = {
                    'parameters': {
                        'width': self.width_var.get(),
                        'height': self.height_var.get(),
                        'nx': self.nx_var.get(),
                        'ny': self.ny_var.get(),
                        'method': self.method_var.get(),
                        'nd': self.nd_var.get(),
                        'na': self.na_var.get(),
                        'bc_left': self.bc_left_var.get(),
                        'bc_right': self.bc_right_var.get()
                    },
                    'results': {}
                }
                
                # Convert numpy arrays to lists for JSON serialization
                for key, value in self.results.items():
                    if hasattr(value, 'tolist'):
                        save_data['results'][key] = value.tolist()
                    else:
                        save_data['results'][key] = value
                
                with open(filename, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                self.log(f"✓ Results saved to {os.path.basename(filename)}")
                messagebox.showinfo("Success", "Results saved successfully")
                
            except Exception as e:
                error_msg = f"Failed to save results: {str(e)}"
                self.log(f"✗ {error_msg}")
                messagebox.showerror("Error", error_msg)
    
    def show_profiler(self):
        """Show performance profiler window"""
        
        profiler_window = tk.Toplevel(self.root)
        profiler_window.title("Performance Profiler")
        profiler_window.geometry("600x400")
        
        # Create text widget for profiler output
        text_frame = tk.Frame(profiler_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        profiler_text = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=profiler_text.yview)
        profiler_text.configure(yscrollcommand=scrollbar.set)
        
        profiler_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add profiler content
        profiler_content = """
PERFORMANCE PROFILER
===================

This feature will show detailed performance analysis of your simulations.

Key Metrics:
• Execution time per function
• Memory usage
• Hotspot identification
• Optimization recommendations

To enable profiling:
1. Run a simulation
2. Performance data will be collected automatically
3. View detailed analysis here

Current Status: Ready for profiling
        """
        
        profiler_text.insert(tk.END, profiler_content)
        
        # Add buttons
        button_frame = tk.Frame(profiler_window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(button_frame, text="Run Benchmark", 
                 command=lambda: self.run_benchmark(profiler_text)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear", 
                 command=lambda: profiler_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Close", 
                 command=profiler_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def run_benchmark(self, text_widget):
        """Run performance benchmark"""
        
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, "Running performance benchmark...\n\n")
        text_widget.update()
        
        import time
        import numpy as np
        
        # Simple benchmark
        results = []
        
        # Vector operations benchmark
        text_widget.insert(tk.END, "Testing vector operations...\n")
        text_widget.update()
        
        sizes = [1000, 10000, 100000]
        for size in sizes:
            a = np.random.random(size)
            b = np.random.random(size)
            
            start_time = time.time()
            for _ in range(100):
                c = a + b
            end_time = time.time()
            
            duration = (end_time - start_time) * 1000  # ms
            throughput = (size * 3 * 100 * 8) / (duration * 1e-3) / 1e9  # GB/s
            
            result_text = f"  Size {size:6d}: {duration:6.2f} ms, {throughput:5.2f} GB/s\n"
            text_widget.insert(tk.END, result_text)
            text_widget.update()
        
        # Matrix operations benchmark
        text_widget.insert(tk.END, "\nTesting matrix operations...\n")
        text_widget.update()
        
        sizes = [100, 200, 500]
        for size in sizes:
            A = np.random.random((size, size))
            x = np.random.random(size)
            
            start_time = time.time()
            y = A @ x
            end_time = time.time()
            
            duration = (end_time - start_time) * 1000  # ms
            flops = 2 * size * size  # multiply-add operations
            gflops = flops / (duration * 1e-3) / 1e9
            
            result_text = f"  {size:3d}x{size:3d}: {duration:6.2f} ms, {gflops:5.2f} GFLOPS\n"
            text_widget.insert(tk.END, result_text)
            text_widget.update()
        
        text_widget.insert(tk.END, "\n✓ Benchmark completed!\n")
    
    def show_device_builder(self):
        """Show device builder window"""
        
        builder_window = tk.Toplevel(self.root)
        builder_window.title("Device Builder")
        builder_window.geometry("800x600")
        
        # Create notebook for different device types
        notebook = tk.ttk.Notebook(builder_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # P-N Junction tab
        pn_frame = tk.Frame(notebook)
        notebook.add(pn_frame, text="P-N Junction")
        
        tk.Label(pn_frame, text="P-N Junction Builder", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Parameters frame
        params_frame = tk.Frame(pn_frame)
        params_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(params_frame, text="Junction Position (μm):").grid(row=0, column=0, sticky=tk.W, pady=5)
        junction_pos = tk.Entry(params_frame, width=10)
        junction_pos.insert(0, "1.0")
        junction_pos.grid(row=0, column=1, pady=5)
        
        tk.Label(params_frame, text="P-type Doping (/m³):").grid(row=1, column=0, sticky=tk.W, pady=5)
        p_doping = tk.Entry(params_frame, width=10)
        p_doping.insert(0, "1e16")
        p_doping.grid(row=1, column=1, pady=5)
        
        tk.Label(params_frame, text="N-type Doping (/m³):").grid(row=2, column=0, sticky=tk.W, pady=5)
        n_doping = tk.Entry(params_frame, width=10)
        n_doping.insert(0, "1e16")
        n_doping.grid(row=2, column=1, pady=5)
        
        def apply_pn_junction():
            try:
                # Update main GUI parameters
                self.na_var.set(p_doping.get())
                self.nd_var.set(n_doping.get())
                
                self.log(f"✓ P-N junction parameters applied")
                messagebox.showinfo("Success", "P-N junction parameters applied to main window")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply parameters: {str(e)}")
        
        tk.Button(params_frame, text="Apply to Main Window", 
                 command=apply_pn_junction).grid(row=3, column=0, columnspan=2, pady=10)
        
        # MOSFET tab
        mosfet_frame = tk.Frame(notebook)
        notebook.add(mosfet_frame, text="MOSFET")
        
        tk.Label(mosfet_frame, text="MOSFET Builder", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(mosfet_frame, text="Advanced MOSFET device builder\n(Coming soon...)", 
                justify=tk.CENTER).pack(pady=20)
        
        # Solar Cell tab
        solar_frame = tk.Frame(notebook)
        notebook.add(solar_frame, text="Solar Cell")
        
        tk.Label(solar_frame, text="Solar Cell Builder", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(solar_frame, text="Solar cell device builder\n(Coming soon...)", 
                justify=tk.CENTER).pack(pady=20)
    
    def show_help(self):
        """Show help window"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("SemiDGFEM User Guide")
        help_window.geometry("700x500")
        
        # Create text widget with scrollbar
        text_frame = tk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        help_text = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=help_text.yview)
        help_text.configure(yscrollcommand=scrollbar.set)
        
        help_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add help content
        help_content = """
SEMIDGFEM USER GUIDE
===================

Welcome to SemiDGFEM, a comprehensive semiconductor device simulator!

GETTING STARTED
---------------

1. Device Parameters:
   • Set device dimensions (width, height in μm)
   • Choose grid resolution (nx, ny)
   • Select numerical method (DG recommended)

2. Doping Configuration:
   • Set N-type and P-type doping concentrations
   • Default creates a P-N junction at center

3. Boundary Conditions:
   • Left/Right: Contact voltages
   • Bottom/Top: Usually grounded (0V)

4. Running Simulations:
   • Click "Create Device" first
   • Then "Run Simulation" for full analysis
   • Or use menu for specific solvers

FEATURES
--------

• Poisson Equation Solver
  - Electrostatic potential calculation
  - Built-in charge distribution

• Drift-Diffusion Solver
  - Carrier transport simulation
  - Current density calculation

• Advanced Methods
  - Discontinuous Galerkin (DG)
  - Adaptive mesh refinement
  - GPU acceleration support

VISUALIZATION
-------------

The results are displayed in four plots:
1. Electrostatic Potential (V)
2. Electron Density (log scale)
3. Hole Density (log scale)
4. Current Density Magnitude

TIPS
----

• Start with small grid sizes for testing
• Use DG method for best accuracy
• Check the log tab for detailed information
• Save parameters and results for later use

TROUBLESHOOTING
---------------

• If simulation fails, try:
  - Smaller grid size
  - Different boundary conditions
  - Check doping values

• For performance issues:
  - Use structured mesh
  - Reduce grid resolution
  - Enable GPU acceleration if available

KEYBOARD SHORTCUTS
------------------

Ctrl+N: New simulation
Ctrl+O: Load parameters
Ctrl+S: Save results
F1: Show this help

For more information, visit the documentation or contact support.
        """
        
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)  # Make read-only
        
        # Add close button
        tk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)
    
    def show_about(self):
        """Show about dialog"""
        
        about_text = """
SemiDGFEM v1.0.0

Semiconductor Device Simulator using
Discontinuous Galerkin Finite Element Methods

Features:
• 2D device simulation
• Poisson and drift-diffusion equations
• Advanced numerical methods
• GPU acceleration support
• Adaptive mesh refinement

Developed with Python, C++, and CUDA
Built on PETSc, Boost, and OpenMP

© 2024 SemiDGFEM Project
Open Source Software
        """
        
        messagebox.showinfo("About SemiDGFEM", about_text)
