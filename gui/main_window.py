#!/usr/bin/env python3
"""
SemiDGFEM GUI - Main Window
Simple and clean graphical interface for semiconductor device simulation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os

# Add parent directory for simulator import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import simulator
    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False

# Import mixin classes
from simulation_methods import SimulationMethods
from menu_methods import MenuMethods

class SemiDGFEMGUI(SimulationMethods, MenuMethods):
    """Main GUI class for SemiDGFEM simulator"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("SemiDGFEM - Semiconductor Device Simulator")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.sim = None
        self.results = {}
        
        # Create GUI components
        self.create_menu()
        self.create_main_layout()
        self.create_status_bar()
        
        # Show welcome message
        self.show_welcome()
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Simulation", command=self.new_simulation)
        file_menu.add_command(label="Load Parameters", command=self.load_parameters)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Simulation menu
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Run Poisson", command=self.run_poisson)
        sim_menu.add_command(label="Run Drift-Diffusion", command=self.run_drift_diffusion)
        sim_menu.add_command(label="Run Full Analysis", command=self.run_full_analysis)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Performance Profiler", command=self.show_profiler)
        tools_menu.add_command(label="Device Builder", command=self.show_device_builder)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_layout(self):
        """Create main layout with panels"""
        
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        self.create_control_panel(main_paned)
        
        # Right panel - Results
        self.create_results_panel(main_paned)
    
    def create_control_panel(self, parent):
        """Create control panel with simulation parameters"""
        
        # Control frame
        control_frame = ttk.Frame(parent)
        parent.add(control_frame, weight=1)
        
        # Device parameters section
        device_group = ttk.LabelFrame(control_frame, text="Device Parameters", padding=10)
        device_group.pack(fill=tk.X, padx=5, pady=5)
        
        # Grid dimensions
        ttk.Label(device_group, text="Width (μm):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.width_var = tk.StringVar(value="2.0")
        ttk.Entry(device_group, textvariable=self.width_var, width=10).grid(row=0, column=1, pady=2)
        
        ttk.Label(device_group, text="Height (μm):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.height_var = tk.StringVar(value="1.0")
        ttk.Entry(device_group, textvariable=self.height_var, width=10).grid(row=1, column=1, pady=2)
        
        # Mesh parameters
        mesh_group = ttk.LabelFrame(control_frame, text="Mesh Parameters", padding=10)
        mesh_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(mesh_group, text="Grid X:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.nx_var = tk.StringVar(value="50")
        ttk.Entry(mesh_group, textvariable=self.nx_var, width=10).grid(row=0, column=1, pady=2)
        
        ttk.Label(mesh_group, text="Grid Y:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ny_var = tk.StringVar(value="25")
        ttk.Entry(mesh_group, textvariable=self.ny_var, width=10).grid(row=1, column=1, pady=2)
        
        ttk.Label(mesh_group, text="Method:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.method_var = tk.StringVar(value="DG")
        method_combo = ttk.Combobox(mesh_group, textvariable=self.method_var, 
                                   values=["DG", "FEM", "FDM"], width=8)
        method_combo.grid(row=2, column=1, pady=2)
        
        # Doping parameters
        doping_group = ttk.LabelFrame(control_frame, text="Doping Parameters", padding=10)
        doping_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(doping_group, text="N-type (1/m³):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.nd_var = tk.StringVar(value="1e16")
        ttk.Entry(doping_group, textvariable=self.nd_var, width=10).grid(row=0, column=1, pady=2)
        
        ttk.Label(doping_group, text="P-type (1/m³):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.na_var = tk.StringVar(value="1e16")
        ttk.Entry(doping_group, textvariable=self.na_var, width=10).grid(row=1, column=1, pady=2)
        
        # Boundary conditions
        bc_group = ttk.LabelFrame(control_frame, text="Boundary Conditions", padding=10)
        bc_group.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(bc_group, text="Left (V):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.bc_left_var = tk.StringVar(value="0.0")
        ttk.Entry(bc_group, textvariable=self.bc_left_var, width=10).grid(row=0, column=1, pady=2)
        
        ttk.Label(bc_group, text="Right (V):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.bc_right_var = tk.StringVar(value="0.7")
        ttk.Entry(bc_group, textvariable=self.bc_right_var, width=10).grid(row=1, column=1, pady=2)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Create Device", 
                  command=self.create_device).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Run Simulation", 
                  command=self.run_simulation).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Clear Results", 
                  command=self.clear_results).pack(fill=tk.X, pady=2)
    
    def create_results_panel(self, parent):
        """Create results panel with plots and data"""
        
        # Results frame
        results_frame = ttk.Frame(parent)
        parent.add(results_frame, weight=2)
        
        # Create notebook for different result views
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Plot tab
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="Plots")
        
        # Data tab
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data")
        
        # Log tab
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="Log")
        
        # Initialize plot area
        self.setup_plot_area()
        self.setup_data_area()
        self.setup_log_area()
    
    def setup_plot_area(self):
        """Setup matplotlib plot area"""
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plots
        self.axes[0, 0].set_title("Electrostatic Potential")
        self.axes[0, 1].set_title("Electron Density")
        self.axes[1, 0].set_title("Hole Density")
        self.axes[1, 1].set_title("Current Density")
        
        for ax in self.axes.flat:
            ax.set_xlabel("x (μm)")
            ax.set_ylabel("y (μm)")
            ax.grid(True, alpha=0.3)
    
    def setup_data_area(self):
        """Setup data display area"""
        
        # Create treeview for data display
        columns = ("Parameter", "Value", "Unit")
        self.data_tree = ttk.Treeview(self.data_frame, columns=columns, show="headings")
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.data_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_log_area(self):
        """Setup log display area"""
        
        # Create text widget for log
        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(self.log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Pack widgets
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add initial log message
        self.log("SemiDGFEM GUI initialized")
        if SIMULATOR_AVAILABLE:
            self.log("✓ Simulator module loaded successfully")
        else:
            self.log("⚠ Simulator module not available - using demo mode")
    
    def create_status_bar(self):
        """Create status bar"""
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def log(self, message):
        """Add message to log"""
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def show_welcome(self):
        """Show welcome message"""
        
        welcome_text = """
Welcome to SemiDGFEM!

This is a graphical interface for the SemiDGFEM semiconductor device simulator.

Quick Start:
1. Set device parameters in the left panel
2. Click 'Create Device' to initialize
3. Click 'Run Simulation' to solve
4. View results in the plots and data tabs

Features:
• Poisson and drift-diffusion equations
• Discontinuous Galerkin methods
• Adaptive mesh refinement
• GPU acceleration support
• Performance profiling

Get started by creating your first device!
        """
        
        # Clear all axes and show welcome text
        for ax in self.axes.flat:
            ax.clear()
            ax.text(0.5, 0.5, welcome_text, transform=ax.transAxes,
                   ha='center', va='center', fontsize=10, wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
        
        self.canvas.draw()
        self.log("Welcome message displayed")

def main():
    """Main function to run the GUI"""
    
    root = tk.Tk()
    app = SemiDGFEMGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
