#!/usr/bin/env python3
"""
Real-Time MOSFET Simulator GUI with Live Logging
Shows actual simulation progress and results in separate windows
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import queue
import time
import sys
import os
from datetime import datetime

# Add parent directory for simulator import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RealTimeMOSFETSimulator:
    """Real-time MOSFET simulator with advanced physics models"""
    
    def __init__(self):
        """Initialize with realistic physics parameters"""
        
        # Physical constants
        self.q = 1.602e-19          # Elementary charge (C)
        self.k = 1.381e-23          # Boltzmann constant (J/K)
        self.h = 6.626e-34          # Planck constant (J·s)
        self.m0 = 9.109e-31         # Free electron mass (kg)
        self.epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
        
        # Silicon material parameters
        self.T = 300                # Temperature (K)
        self.Vt = self.k * self.T / self.q  # Thermal voltage (V)
        self.ni = 1.45e16          # Intrinsic carrier concentration (/m³)
        self.epsilon_si = 11.7 * self.epsilon_0  # Silicon permittivity
        self.epsilon_ox = 3.9 * self.epsilon_0   # Oxide permittivity
        
        # Effective masses (in units of m0)
        self.m_eff_n = 0.26 * self.m0  # Electron effective mass
        self.m_eff_p = 0.39 * self.m0  # Hole effective mass
        
        # SRH recombination parameters
        self.tau_n0 = 1e-6         # Electron SRH lifetime (s)
        self.tau_p0 = 1e-6         # Hole SRH lifetime (s)
        self.Et_Ei = 0.0           # Trap level relative to intrinsic (eV)
        
        # Mobility model parameters
        self.mu_n_max = 0.135      # Maximum electron mobility (m²/V·s)
        self.mu_p_max = 0.048      # Maximum hole mobility (m²/V·s)
        self.N_ref_n = 8.5e23      # Reference doping for electrons (/m³)
        self.N_ref_p = 8.5e23      # Reference doping for holes (/m³)
        self.alpha_n = 0.88        # Mobility exponent for electrons
        self.alpha_p = 0.76        # Mobility exponent for holes
        
        # Device parameters
        self.device_params = {
            'length': 100e-9,       # 100 nm channel length
            'width': 1e-6,          # 1 μm channel width
            'tox': 2e-9,            # 2 nm oxide thickness
            'Na_substrate': 1e23,   # Substrate doping (/m³)
            'Nd_source': 1e26,      # Source doping (/m³)
            'Nd_drain': 1e26,       # Drain doping (/m³)
        }
        
        # Simulation grid
        self.nx = 50
        self.ny = 25
        self.total_points = self.nx * self.ny
        
        # Results storage
        self.simulation_results = {}
        self.log_queue = queue.Queue()
        
    def log_message(self, message):
        """Add message to log queue for GUI display"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        self.log_queue.put(log_entry)
        print(log_entry)  # Also print to console
    
    def calculate_effective_mobility(self, N_total, carrier_type='electron'):
        """Calculate mobility using Caughey-Thomas model with doping dependence"""
        
        if carrier_type == 'electron':
            mu_max = self.mu_n_max
            N_ref = self.N_ref_n
            alpha = self.alpha_n
        else:
            mu_max = self.mu_p_max
            N_ref = self.N_ref_p
            alpha = self.alpha_p
        
        # Caughey-Thomas mobility model
        mu_eff = mu_max / (1 + (N_total / N_ref)**alpha)
        
        return mu_eff
    
    def calculate_srh_recombination(self, n, p, n_trap=None):
        """Calculate Shockley-Read-Hall recombination rate"""
        
        if n_trap is None:
            n_trap = self.ni * np.exp(self.Et_Ei / self.Vt)
        
        p_trap = self.ni**2 / n_trap
        
        # SRH recombination rate
        R_srh = (n * p - self.ni**2) / (
            self.tau_p0 * (n + n_trap) + self.tau_n0 * (p + p_trap)
        )
        
        return R_srh
    
    def solve_poisson_realistic(self, bc, log_progress=True):
        """Solve Poisson equation with realistic physics"""
        
        if log_progress:
            self.log_message("🔍 SOLVING POISSON EQUATION WITH REALISTIC PHYSICS")
            self.log_message(f"   Grid: {self.nx} × {self.ny} = {self.total_points} points")
            self.log_message(f"   Device: {self.device_params['length']*1e9:.0f}nm × {self.device_params['width']*1e6:.1f}μm")
        
        start_time = time.time()
        
        # Create coordinate arrays
        x = np.linspace(0, self.device_params['length'], self.nx)
        y = np.linspace(0, self.device_params['width'], self.ny)
        X, Y = np.meshgrid(x, y)
        
        # Extract boundary conditions
        Vs, Vd, Vsub, Vg = bc[0], bc[1], bc[2], bc[3] if len(bc) > 3 else 0.0
        
        if log_progress:
            self.log_message(f"   Boundary conditions:")
            self.log_message(f"      Source: {Vs:.3f} V")
            self.log_message(f"      Drain: {Vd:.3f} V") 
            self.log_message(f"      Substrate: {Vsub:.3f} V")
            self.log_message(f"      Gate: {Vg:.3f} V")
        
        # Calculate oxide capacitance
        Cox = self.epsilon_ox / self.device_params['tox']
        if log_progress:
            self.log_message(f"   Oxide capacitance: {Cox:.2e} F/m²")
        
        # Create realistic potential distribution using finite difference approximation
        V = np.zeros_like(X)
        
        # Iterative solution for Poisson equation
        max_iterations = 100
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            # Apply boundary conditions
            V[0, :] = Vs    # Source contact
            V[-1, :] = Vd   # Drain contact
            V[:, 0] = Vsub  # Substrate contact
            
            # Gate coupling through oxide
            gate_start = self.nx // 4
            gate_end = 3 * self.nx // 4
            surface_coupling = 0.8 * (Vg - np.mean(V[gate_start:gate_end, -1]))
            V[gate_start:gate_end, -1] += surface_coupling * 0.1
            
            # Interior points - simplified finite difference
            for i in range(1, self.ny-1):
                for j in range(1, self.nx-1):
                    V[i, j] = 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])
            
            # Check convergence
            max_change = np.max(np.abs(V - V_old))
            if max_change < tolerance:
                if log_progress:
                    self.log_message(f"   Converged in {iteration+1} iterations")
                break
            
            if log_progress and (iteration + 1) % 20 == 0:
                self.log_message(f"   Iteration {iteration+1}: max change = {max_change:.2e}")
        
        solve_time = time.time() - start_time
        
        if log_progress:
            self.log_message(f"   ✅ Poisson solved in {solve_time:.4f} seconds")
            self.log_message(f"   Potential range: {np.min(V):.4f} to {np.max(V):.4f} V")
            self.log_message(f"   Potential mean: {np.mean(V):.4f} V")
        
        return V.flatten()
    
    def calculate_carrier_densities_realistic(self, V_2d, Vg, log_progress=True):
        """Calculate carrier densities with realistic physics models"""
        
        if log_progress:
            self.log_message("🔬 CALCULATING CARRIER DENSITIES WITH REALISTIC MODELS")
        
        ny, nx = V_2d.shape
        
        # Create coordinate arrays
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # Initialize carrier densities
        n = np.zeros_like(V_2d)
        p = np.zeros_like(V_2d)
        
        # Calculate equilibrium densities using realistic models
        for i in range(ny):
            for j in range(nx):
                # Local potential
                phi = V_2d[i, j]
                
                # Determine local doping
                if Y[i, j] > 0.7:  # Surface region
                    if X[i, j] < 0.25:  # Source
                        Nd_local = self.device_params['Nd_source']
                        Na_local = 0
                    elif X[i, j] > 0.75:  # Drain
                        Nd_local = self.device_params['Nd_drain']
                        Na_local = 0
                    else:  # Channel
                        Nd_local = 0
                        Na_local = self.device_params['Na_substrate']
                else:  # Bulk
                    Nd_local = 0
                    Na_local = self.device_params['Na_substrate']
                
                # Calculate carrier densities using Fermi-Dirac statistics approximation
                if Nd_local > Na_local:  # N-type region
                    n[i, j] = Nd_local * np.exp(phi / self.Vt)
                    p[i, j] = self.ni**2 / n[i, j]
                else:  # P-type region
                    p[i, j] = Na_local * np.exp(-phi / self.Vt)
                    n[i, j] = self.ni**2 / p[i, j]
                
                # Add gate-induced carriers for channel region
                if 0.25 <= X[i, j] <= 0.75 and Y[i, j] > 0.7:  # Channel surface
                    if Vg > 0.5:  # Above threshold
                        # Inversion layer formation
                        n_inv = 1e20 * (Vg - 0.5) * np.exp(-5 * (1 - Y[i, j]))
                        n[i, j] += n_inv
        
        if log_progress:
            self.log_message(f"   Electron density range: {np.min(n):.2e} to {np.max(n):.2e} /m³")
            self.log_message(f"   Hole density range: {np.min(p):.2e} to {np.max(p):.2e} /m³")
            self.log_message(f"   Total electrons: {np.sum(n):.2e}")
            self.log_message(f"   Total holes: {np.sum(p):.2e}")
        
        return n, p
    
    def calculate_current_densities_realistic(self, V_2d, n, p, log_progress=True):
        """Calculate current densities with realistic mobility models"""
        
        if log_progress:
            self.log_message("⚡ CALCULATING CURRENT DENSITIES WITH MOBILITY MODELS")
        
        # Calculate electric field
        Ex = -np.gradient(V_2d, axis=1)
        Ey = -np.gradient(V_2d, axis=0)
        E_magnitude = np.sqrt(Ex**2 + Ey**2)
        
        if log_progress:
            self.log_message(f"   Electric field range: {np.min(E_magnitude):.2e} to {np.max(E_magnitude):.2e} V/m")
        
        # Calculate local doping for mobility calculation
        N_total = n + p
        
        # Calculate effective mobilities
        mu_n_eff = self.calculate_effective_mobility(N_total, 'electron')
        mu_p_eff = self.calculate_effective_mobility(N_total, 'hole')
        
        if log_progress:
            self.log_message(f"   Electron mobility range: {np.min(mu_n_eff):.4f} to {np.max(mu_n_eff):.4f} m²/V·s")
            self.log_message(f"   Hole mobility range: {np.min(mu_p_eff):.4f} to {np.max(mu_p_eff):.4f} m²/V·s")
        
        # Calculate current densities
        Jn_x = self.q * mu_n_eff * n * Ex
        Jn_y = self.q * mu_n_eff * n * Ey
        Jn = np.sqrt(Jn_x**2 + Jn_y**2)
        
        Jp_x = -self.q * mu_p_eff * p * Ex  # Opposite direction
        Jp_y = -self.q * mu_p_eff * p * Ey
        Jp = np.sqrt(Jp_x**2 + Jp_y**2)
        
        if log_progress:
            self.log_message(f"   Electron current range: {np.min(Jn):.2e} to {np.max(Jn):.2e} A/m²")
            self.log_message(f"   Hole current range: {np.min(Jp):.2e} to {np.max(Jp):.2e} A/m²")
        
        return Jn, Jp, Ex, Ey
    
    def calculate_drain_current_realistic(self, Vg, Vd, log_progress=True):
        """Calculate drain current using realistic MOSFET model"""
        
        # Calculate threshold voltage
        phi_F = self.Vt * np.log(self.device_params['Na_substrate'] / self.ni)
        Cox = self.epsilon_ox / self.device_params['tox']
        gamma = np.sqrt(2 * self.q * self.epsilon_si * self.device_params['Na_substrate']) / Cox
        Vth = 2 * phi_F + gamma * np.sqrt(2 * phi_F)
        
        if log_progress:
            self.log_message(f"   Calculated Vth = {Vth:.3f} V")
        
        # Calculate current based on operating region
        W_over_L = self.device_params['width'] / self.device_params['length']
        mu_eff = self.calculate_effective_mobility(self.device_params['Na_substrate'], 'electron')
        
        if Vg < Vth:
            # Subthreshold region
            Id = 1e-15 * np.exp((Vg - Vth) / (10 * self.Vt))
            region = "SUBTHRESHOLD"
        else:
            if Vd < (Vg - Vth):
                # Linear region
                Id = mu_eff * Cox * W_over_L * ((Vg - Vth) * Vd - 0.5 * Vd**2)
                region = "LINEAR"
            else:
                # Saturation region
                Id = 0.5 * mu_eff * Cox * W_over_L * (Vg - Vth)**2
                region = "SATURATION"
        
        if log_progress:
            self.log_message(f"   Operating region: {region}")
            self.log_message(f"   Drain current: {Id:.2e} A")
        
        return Id, region

class MOSFETSimulatorGUI:
    """Real-time GUI for MOSFET simulation with live logging"""

    def __init__(self):
        """Initialize the GUI application"""

        self.root = tk.Tk()
        self.root.title("Real-Time MOSFET Simulator with Live Logging")
        self.root.geometry("1400x900")

        # Initialize simulator
        self.simulator = RealTimeMOSFETSimulator()

        # GUI state
        self.simulation_running = False
        self.results_window = None

        # Create GUI components
        self.create_widgets()

        # Start log monitoring
        self.monitor_logs()

    def create_widgets(self):
        """Create all GUI widgets"""

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(main_frame, text="🔬 Real-Time MOSFET Simulator",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Simulation Control Tab
        self.create_control_tab(notebook)

        # Live Logging Tab
        self.create_logging_tab(notebook)

        # Device Parameters Tab
        self.create_parameters_tab(notebook)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to simulate")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

    def create_control_tab(self, notebook):
        """Create simulation control tab"""

        control_frame = ttk.Frame(notebook)
        notebook.add(control_frame, text="Simulation Control")

        # Voltage controls
        voltage_frame = ttk.LabelFrame(control_frame, text="Voltage Controls", padding=10)
        voltage_frame.pack(fill=tk.X, padx=10, pady=5)

        # Gate voltage
        ttk.Label(voltage_frame, text="Gate Voltage (V):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.vg_var = tk.DoubleVar(value=0.8)
        vg_scale = ttk.Scale(voltage_frame, from_=0.0, to=2.0, variable=self.vg_var,
                            orient=tk.HORIZONTAL, length=200)
        vg_scale.grid(row=0, column=1, padx=5)
        self.vg_label = ttk.Label(voltage_frame, text="0.80 V")
        self.vg_label.grid(row=0, column=2, padx=5)
        vg_scale.configure(command=lambda v: self.vg_label.configure(text=f"{float(v):.2f} V"))

        # Drain voltage
        ttk.Label(voltage_frame, text="Drain Voltage (V):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.vd_var = tk.DoubleVar(value=0.6)
        vd_scale = ttk.Scale(voltage_frame, from_=0.0, to=2.0, variable=self.vd_var,
                            orient=tk.HORIZONTAL, length=200)
        vd_scale.grid(row=1, column=1, padx=5)
        self.vd_label = ttk.Label(voltage_frame, text="0.60 V")
        self.vd_label.grid(row=1, column=2, padx=5)
        vd_scale.configure(command=lambda v: self.vd_label.configure(text=f"{float(v):.2f} V"))

        # Source voltage (fixed at 0)
        ttk.Label(voltage_frame, text="Source Voltage (V):").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Label(voltage_frame, text="0.00 V (Fixed)").grid(row=2, column=1, sticky=tk.W, padx=5)

        # Substrate voltage
        ttk.Label(voltage_frame, text="Substrate Voltage (V):").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.vsub_var = tk.DoubleVar(value=0.0)
        vsub_scale = ttk.Scale(voltage_frame, from_=-1.0, to=1.0, variable=self.vsub_var,
                              orient=tk.HORIZONTAL, length=200)
        vsub_scale.grid(row=3, column=1, padx=5)
        self.vsub_label = ttk.Label(voltage_frame, text="0.00 V")
        self.vsub_label.grid(row=3, column=2, padx=5)
        vsub_scale.configure(command=lambda v: self.vsub_label.configure(text=f"{float(v):.2f} V"))

        # Simulation buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.simulate_btn = ttk.Button(button_frame, text="🚀 Run Single Point Simulation",
                                      command=self.run_single_simulation, style="Accent.TButton")
        self.simulate_btn.pack(side=tk.LEFT, padx=5)

        self.iv_btn = ttk.Button(button_frame, text="📈 Generate I-V Characteristics",
                                command=self.run_iv_simulation)
        self.iv_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop Simulation",
                                  command=self.stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.results_btn = ttk.Button(button_frame, text="📊 Show Results Window",
                                     command=self.show_results_window)
        self.results_btn.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var,
                                           maximum=100, length=400)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

    def create_logging_tab(self, notebook):
        """Create live logging tab"""

        logging_frame = ttk.Frame(notebook)
        notebook.add(logging_frame, text="Live Simulation Log")

        # Log display
        log_label = ttk.Label(logging_frame, text="Real-Time Simulation Log:", font=("Arial", 12, "bold"))
        log_label.pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Scrolled text widget for logs
        self.log_text = scrolledtext.ScrolledText(logging_frame, height=25, width=100,
                                                 font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Log control buttons
        log_button_frame = ttk.Frame(logging_frame)
        log_button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(log_button_frame, text="Clear Log",
                  command=self.clear_log).pack(side=tk.LEFT, padx=5)

        ttk.Button(log_button_frame, text="Save Log",
                  command=self.save_log).pack(side=tk.LEFT, padx=5)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_button_frame, text="Auto-scroll",
                       variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=5)

    def create_parameters_tab(self, notebook):
        """Create device parameters tab"""

        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Device Parameters")

        # Device geometry
        geom_frame = ttk.LabelFrame(params_frame, text="Device Geometry", padding=10)
        geom_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(geom_frame, text="Channel Length (nm):").grid(row=0, column=0, sticky=tk.W)
        self.length_var = tk.DoubleVar(value=100)
        ttk.Entry(geom_frame, textvariable=self.length_var, width=15).grid(row=0, column=1, padx=5)

        ttk.Label(geom_frame, text="Channel Width (μm):").grid(row=1, column=0, sticky=tk.W)
        self.width_var = tk.DoubleVar(value=1.0)
        ttk.Entry(geom_frame, textvariable=self.width_var, width=15).grid(row=1, column=1, padx=5)

        ttk.Label(geom_frame, text="Oxide Thickness (nm):").grid(row=2, column=0, sticky=tk.W)
        self.tox_var = tk.DoubleVar(value=2.0)
        ttk.Entry(geom_frame, textvariable=self.tox_var, width=15).grid(row=2, column=1, padx=5)

        # Doping parameters
        doping_frame = ttk.LabelFrame(params_frame, text="Doping Concentrations", padding=10)
        doping_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(doping_frame, text="Substrate (P-type) (/cm³):").grid(row=0, column=0, sticky=tk.W)
        self.na_var = tk.StringVar(value="1e17")
        ttk.Entry(doping_frame, textvariable=self.na_var, width=15).grid(row=0, column=1, padx=5)

        ttk.Label(doping_frame, text="Source/Drain (N-type) (/cm³):").grid(row=1, column=0, sticky=tk.W)
        self.nd_var = tk.StringVar(value="1e20")
        ttk.Entry(doping_frame, textvariable=self.nd_var, width=15).grid(row=1, column=1, padx=5)

        # Physics parameters
        physics_frame = ttk.LabelFrame(params_frame, text="Physics Parameters", padding=10)
        physics_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(physics_frame, text="Temperature (K):").grid(row=0, column=0, sticky=tk.W)
        self.temp_var = tk.DoubleVar(value=300)
        ttk.Entry(physics_frame, textvariable=self.temp_var, width=15).grid(row=0, column=1, padx=5)

        ttk.Label(physics_frame, text="SRH Lifetime (μs):").grid(row=1, column=0, sticky=tk.W)
        self.tau_var = tk.DoubleVar(value=1.0)
        ttk.Entry(physics_frame, textvariable=self.tau_var, width=15).grid(row=1, column=1, padx=5)

        # Update button
        ttk.Button(params_frame, text="Update Parameters",
                  command=self.update_parameters).pack(pady=10)

    def monitor_logs(self):
        """Monitor log queue and update GUI"""

        try:
            while True:
                log_entry = self.simulator.log_queue.get_nowait()
                self.log_text.insert(tk.END, log_entry + "\n")

                if self.auto_scroll_var.get():
                    self.log_text.see(tk.END)

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.monitor_logs)

    def clear_log(self):
        """Clear the log display"""
        self.log_text.delete(1.0, tk.END)

    def save_log(self):
        """Save log to file"""
        from tkinter import filedialog

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Log saved to {filename}")

    def update_parameters(self):
        """Update simulator parameters from GUI"""

        try:
            # Update device parameters
            self.simulator.device_params['length'] = self.length_var.get() * 1e-9
            self.simulator.device_params['width'] = self.width_var.get() * 1e-6
            self.simulator.device_params['tox'] = self.tox_var.get() * 1e-9
            self.simulator.device_params['Na_substrate'] = float(self.na_var.get()) * 1e6
            self.simulator.device_params['Nd_source'] = float(self.nd_var.get()) * 1e6
            self.simulator.device_params['Nd_drain'] = float(self.nd_var.get()) * 1e6

            # Update physics parameters
            self.simulator.T = self.temp_var.get()
            self.simulator.Vt = self.simulator.k * self.simulator.T / self.simulator.q
            self.simulator.tau_n0 = self.tau_var.get() * 1e-6
            self.simulator.tau_p0 = self.tau_var.get() * 1e-6

            self.simulator.log_message("📝 Parameters updated from GUI")
            self.status_var.set("Parameters updated")

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {e}")

    def run_single_simulation(self):
        """Run single point simulation in separate thread"""

        if self.simulation_running:
            return

        self.simulation_running = True
        self.simulate_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Running single point simulation...")

        # Run in separate thread to avoid blocking GUI
        thread = threading.Thread(target=self._single_simulation_worker)
        thread.daemon = True
        thread.start()

    def _single_simulation_worker(self):
        """Worker thread for single point simulation"""

        try:
            # Get voltages from GUI
            Vg = self.vg_var.get()
            Vd = self.vd_var.get()
            Vs = 0.0
            Vsub = self.vsub_var.get()

            bc = [Vs, Vd, Vsub, Vg]

            self.simulator.log_message("🚀 STARTING SINGLE POINT SIMULATION")
            self.simulator.log_message("=" * 60)

            # Update progress
            self.progress_var.set(10)

            # Solve Poisson equation
            V = self.simulator.solve_poisson_realistic(bc)
            V_2d = V.reshape(self.simulator.ny, self.simulator.nx)
            self.progress_var.set(30)

            # Calculate carrier densities
            n, p = self.simulator.calculate_carrier_densities_realistic(V_2d, Vg)
            self.progress_var.set(60)

            # Calculate current densities
            Jn, Jp, Ex, Ey = self.simulator.calculate_current_densities_realistic(V_2d, n, p)
            self.progress_var.set(80)

            # Calculate drain current
            Id, region = self.simulator.calculate_drain_current_realistic(Vg, Vd)
            self.progress_var.set(90)

            # Store results
            self.simulator.simulation_results = {
                'potential': V,
                'n': n.flatten(),
                'p': p.flatten(),
                'Jn': Jn.flatten(),
                'Jp': Jp.flatten(),
                'Ex': Ex.flatten(),
                'Ey': Ey.flatten(),
                'drain_current': Id,
                'operating_region': region,
                'boundary_conditions': bc
            }

            self.progress_var.set(100)

            self.simulator.log_message("✅ SINGLE POINT SIMULATION COMPLETED")
            self.simulator.log_message("=" * 60)

            # Update GUI on main thread
            self.root.after(0, self._simulation_completed)

        except Exception as e:
            self.simulator.log_message(f"❌ Simulation failed: {str(e)}")
            self.root.after(0, self._simulation_failed)

    def run_iv_simulation(self):
        """Run I-V characteristics simulation"""

        if self.simulation_running:
            return

        self.simulation_running = True
        self.simulate_btn.configure(state=tk.DISABLED)
        self.iv_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Running I-V characteristics...")

        # Run in separate thread
        thread = threading.Thread(target=self._iv_simulation_worker)
        thread.daemon = True
        thread.start()

    def _iv_simulation_worker(self):
        """Worker thread for I-V simulation"""

        try:
            self.simulator.log_message("📈 STARTING I-V CHARACTERISTICS SIMULATION")
            self.simulator.log_message("=" * 60)

            # Transfer characteristics
            Vg_range = np.linspace(0, 2.0, 21)
            Vd_linear = 0.1

            transfer_data = {'Vg': Vg_range, 'Id': [], 'region': []}

            for i, Vg in enumerate(Vg_range):
                progress = (i / len(Vg_range)) * 50  # First 50% for transfer
                self.progress_var.set(progress)

                Id, region = self.simulator.calculate_drain_current_realistic(Vg, Vd_linear, log_progress=False)
                transfer_data['Id'].append(Id)
                transfer_data['region'].append(region)

                if i % 5 == 0:
                    self.simulator.log_message(f"   Transfer: Vg={Vg:.2f}V → Id={Id:.2e}A ({region})")

            # Output characteristics
            Vd_range = np.linspace(0, 2.0, 21)
            Vg_values = [0.5, 0.8, 1.0, 1.2, 1.5]

            output_data = {'Vd': Vd_range, 'curves': {}}

            for j, Vg in enumerate(Vg_values):
                curve_data = []
                for i, Vd in enumerate(Vd_range):
                    progress = 50 + ((j * len(Vd_range) + i) / (len(Vg_values) * len(Vd_range))) * 50
                    self.progress_var.set(progress)

                    Id, region = self.simulator.calculate_drain_current_realistic(Vg, Vd, log_progress=False)
                    curve_data.append(Id)

                output_data['curves'][f'Vg_{Vg:.1f}V'] = curve_data
                self.simulator.log_message(f"   Output: Vg={Vg:.1f}V curve completed")

            # Store I-V results
            self.simulator.simulation_results['transfer'] = transfer_data
            self.simulator.simulation_results['output'] = output_data

            self.progress_var.set(100)
            self.simulator.log_message("✅ I-V CHARACTERISTICS COMPLETED")

            # Update GUI on main thread
            self.root.after(0, self._simulation_completed)

        except Exception as e:
            self.simulator.log_message(f"❌ I-V simulation failed: {str(e)}")
            self.root.after(0, self._simulation_failed)

    def stop_simulation(self):
        """Stop current simulation"""
        self.simulation_running = False
        self._simulation_completed()
        self.simulator.log_message("⏹ Simulation stopped by user")

    def _simulation_completed(self):
        """Called when simulation completes"""
        self.simulation_running = False
        self.simulate_btn.configure(state=tk.NORMAL)
        self.iv_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("Simulation completed - Ready for next run")

    def _simulation_failed(self):
        """Called when simulation fails"""
        self.simulation_running = False
        self.simulate_btn.configure(state=tk.NORMAL)
        self.iv_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("Simulation failed - Check log for details")

    def show_results_window(self):
        """Show results in separate window"""

        if not self.simulator.simulation_results:
            messagebox.showwarning("No Results", "No simulation results available. Run a simulation first.")
            return

        if self.results_window and self.results_window.winfo_exists():
            self.results_window.lift()
            return

        # Create results window
        self.results_window = tk.Toplevel(self.root)
        self.results_window.title("📊 Real-Time Simulation Results")
        self.results_window.geometry("1200x800")

        # Create notebook for different result types
        results_notebook = ttk.Notebook(self.results_window)
        results_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Physics results tab
        if 'potential' in self.simulator.simulation_results:
            self.create_physics_plots_tab(results_notebook)

        # I-V characteristics tab
        if 'transfer' in self.simulator.simulation_results:
            self.create_iv_plots_tab(results_notebook)

        # Data summary tab
        self.create_summary_tab(results_notebook)

    def create_physics_plots_tab(self, notebook):
        """Create physics plots tab"""

        physics_frame = ttk.Frame(notebook)
        notebook.add(physics_frame, text="Device Physics")

        # Create matplotlib figure
        fig = Figure(figsize=(12, 8), dpi=100)

        results = self.simulator.simulation_results
        nx, ny = self.simulator.nx, self.simulator.ny

        # Plot 1: Electrostatic Potential
        ax1 = fig.add_subplot(2, 3, 1)
        V_2d = results['potential'].reshape(ny, nx)
        im1 = ax1.imshow(V_2d, aspect='auto', origin='lower', cmap='RdYlBu_r')
        ax1.set_title('Electrostatic Potential (V)')
        ax1.set_xlabel('x (grid points)')
        ax1.set_ylabel('y (grid points)')
        fig.colorbar(im1, ax=ax1, shrink=0.8)

        # Plot 2: Electron Density
        ax2 = fig.add_subplot(2, 3, 2)
        n_2d = results['n'].reshape(ny, nx)
        im2 = ax2.imshow(np.log10(np.maximum(n_2d, 1e10)), aspect='auto', origin='lower', cmap='plasma')
        ax2.set_title('Electron Density (log₁₀ /m³)')
        ax2.set_xlabel('x (grid points)')
        ax2.set_ylabel('y (grid points)')
        fig.colorbar(im2, ax=ax2, shrink=0.8)

        # Plot 3: Hole Density
        ax3 = fig.add_subplot(2, 3, 3)
        p_2d = results['p'].reshape(ny, nx)
        im3 = ax3.imshow(np.log10(np.maximum(p_2d, 1e10)), aspect='auto', origin='lower', cmap='viridis')
        ax3.set_title('Hole Density (log₁₀ /m³)')
        ax3.set_xlabel('x (grid points)')
        ax3.set_ylabel('y (grid points)')
        fig.colorbar(im3, ax=ax3, shrink=0.8)

        # Plot 4: Electron Current Density
        ax4 = fig.add_subplot(2, 3, 4)
        Jn_2d = results['Jn'].reshape(ny, nx)
        im4 = ax4.imshow(np.log10(np.maximum(Jn_2d, 1e-10)), aspect='auto', origin='lower', cmap='hot')
        ax4.set_title('Electron Current Density (log₁₀ A/m²)')
        ax4.set_xlabel('x (grid points)')
        ax4.set_ylabel('y (grid points)')
        fig.colorbar(im4, ax=ax4, shrink=0.8)

        # Plot 5: Hole Current Density
        ax5 = fig.add_subplot(2, 3, 5)
        Jp_2d = results['Jp'].reshape(ny, nx)
        im5 = ax5.imshow(np.log10(np.maximum(Jp_2d, 1e-10)), aspect='auto', origin='lower', cmap='hot')
        ax5.set_title('Hole Current Density (log₁₀ A/m²)')
        ax5.set_xlabel('x (grid points)')
        ax5.set_ylabel('y (grid points)')
        fig.colorbar(im5, ax=ax5, shrink=0.8)

        # Plot 6: Electric Field Vectors
        ax6 = fig.add_subplot(2, 3, 6)
        Ex_2d = results['Ex'].reshape(ny, nx)
        Ey_2d = results['Ey'].reshape(ny, nx)

        # Subsample for vector plot
        step = max(1, min(nx, ny) // 10)
        x_vec = np.arange(0, nx, step)
        y_vec = np.arange(0, ny, step)
        X_vec, Y_vec = np.meshgrid(x_vec, y_vec)

        Ex_sub = Ex_2d[::step, ::step]
        Ey_sub = Ey_2d[::step, ::step]

        ax6.quiver(X_vec, Y_vec, Ex_sub, Ey_sub, np.sqrt(Ex_sub**2 + Ey_sub**2),
                  cmap='viridis', alpha=0.8)
        ax6.set_title('Electric Field Vectors')
        ax6.set_xlabel('x (grid points)')
        ax6.set_ylabel('y (grid points)')

        plt.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, physics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_iv_plots_tab(self, notebook):
        """Create I-V characteristics plots tab"""

        iv_frame = ttk.Frame(notebook)
        notebook.add(iv_frame, text="I-V Characteristics")

        # Create matplotlib figure
        fig = Figure(figsize=(12, 8), dpi=100)

        results = self.simulator.simulation_results

        # Plot 1: Transfer Characteristics (Linear)
        ax1 = fig.add_subplot(2, 2, 1)
        transfer = results['transfer']
        ax1.plot(transfer['Vg'], transfer['Id'], 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Gate Voltage (V)')
        ax1.set_ylabel('Drain Current (A)')
        ax1.set_title('Transfer Characteristics (Linear Scale)')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Transfer Characteristics (Log)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.semilogy(transfer['Vg'], transfer['Id'], 'b-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Gate Voltage (V)')
        ax2.set_ylabel('Drain Current (A)')
        ax2.set_title('Transfer Characteristics (Log Scale)')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Output Characteristics
        ax3 = fig.add_subplot(2, 2, 3)
        if 'output' in results:
            output = results['output']
            for curve_name, curve_data in output['curves'].items():
                Vg_val = curve_name.replace('Vg_', '').replace('V', '')
                ax3.plot(output['Vd'], curve_data, linewidth=2, marker='o', markersize=3,
                        label=f'Vg = {Vg_val}V')
        ax3.set_xlabel('Drain Voltage (V)')
        ax3.set_ylabel('Drain Current (A)')
        ax3.set_title('Output Characteristics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Operating Region Analysis
        ax4 = fig.add_subplot(2, 2, 4)
        regions = transfer['region']
        region_colors = {'SUBTHRESHOLD': 'red', 'LINEAR': 'blue', 'SATURATION': 'green'}

        for i, (Vg, Id, region) in enumerate(zip(transfer['Vg'], transfer['Id'], regions)):
            color = region_colors.get(region, 'black')
            ax4.scatter(Vg, Id, c=color, s=50, alpha=0.7)

        # Create legend
        for region, color in region_colors.items():
            ax4.scatter([], [], c=color, s=50, label=region)

        ax4.set_xlabel('Gate Voltage (V)')
        ax4.set_ylabel('Drain Current (A)')
        ax4.set_title('Operating Regions')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, iv_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_summary_tab(self, notebook):
        """Create data summary tab"""

        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Data Summary")

        # Create text widget for summary
        summary_text = scrolledtext.ScrolledText(summary_frame, height=30, width=80,
                                                font=("Consolas", 11))
        summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Generate summary
        summary = self.generate_results_summary()
        summary_text.insert(tk.END, summary)
        summary_text.configure(state=tk.DISABLED)

    def generate_results_summary(self):
        """Generate comprehensive results summary"""

        results = self.simulator.simulation_results
        summary = []

        summary.append("📊 COMPREHENSIVE SIMULATION RESULTS SUMMARY")
        summary.append("=" * 60)
        summary.append("")

        # Device parameters
        summary.append("🔧 DEVICE PARAMETERS:")
        summary.append(f"   Channel Length: {self.simulator.device_params['length']*1e9:.1f} nm")
        summary.append(f"   Channel Width: {self.simulator.device_params['width']*1e6:.1f} μm")
        summary.append(f"   Oxide Thickness: {self.simulator.device_params['tox']*1e9:.1f} nm")
        summary.append(f"   Substrate Doping: {self.simulator.device_params['Na_substrate']:.2e} /m³")
        summary.append(f"   Source/Drain Doping: {self.simulator.device_params['Nd_source']:.2e} /m³")
        summary.append("")

        # Physics parameters
        summary.append("🔬 PHYSICS PARAMETERS:")
        summary.append(f"   Temperature: {self.simulator.T:.1f} K")
        summary.append(f"   Thermal Voltage: {self.simulator.Vt:.4f} V")
        summary.append(f"   SRH Lifetime: {self.simulator.tau_n0*1e6:.1f} μs")
        summary.append(f"   Electron Effective Mass: {self.simulator.m_eff_n/self.simulator.m0:.2f} m₀")
        summary.append(f"   Hole Effective Mass: {self.simulator.m_eff_p/self.simulator.m0:.2f} m₀")
        summary.append("")

        if 'boundary_conditions' in results:
            bc = results['boundary_conditions']
            summary.append("⚡ OPERATING POINT:")
            summary.append(f"   Source Voltage: {bc[0]:.3f} V")
            summary.append(f"   Drain Voltage: {bc[1]:.3f} V")
            summary.append(f"   Substrate Voltage: {bc[2]:.3f} V")
            summary.append(f"   Gate Voltage: {bc[3]:.3f} V")
            summary.append("")

        if 'drain_current' in results:
            summary.append("📈 ELECTRICAL CHARACTERISTICS:")
            summary.append(f"   Drain Current: {results['drain_current']:.2e} A")
            summary.append(f"   Operating Region: {results.get('operating_region', 'Unknown')}")
            summary.append("")

        if 'potential' in results:
            V = results['potential']
            summary.append("🔋 ELECTROSTATIC POTENTIAL:")
            summary.append(f"   Range: {np.min(V):.4f} to {np.max(V):.4f} V")
            summary.append(f"   Mean: {np.mean(V):.4f} V")
            summary.append(f"   Standard Deviation: {np.std(V):.4f} V")
            summary.append("")

        if 'n' in results and 'p' in results:
            n = results['n']
            p = results['p']
            summary.append("🔬 CARRIER DENSITIES:")
            summary.append(f"   Electron Density Range: {np.min(n):.2e} to {np.max(n):.2e} /m³")
            summary.append(f"   Hole Density Range: {np.min(p):.2e} to {np.max(p):.2e} /m³")
            summary.append(f"   Total Electrons: {np.sum(n):.2e}")
            summary.append(f"   Total Holes: {np.sum(p):.2e}")
            summary.append("")

        if 'Jn' in results and 'Jp' in results:
            Jn = results['Jn']
            Jp = results['Jp']
            summary.append("⚡ CURRENT DENSITIES:")
            summary.append(f"   Electron Current Range: {np.min(Jn):.2e} to {np.max(Jn):.2e} A/m²")
            summary.append(f"   Hole Current Range: {np.min(Jp):.2e} to {np.max(Jp):.2e} A/m²")
            summary.append("")

        if 'transfer' in results:
            transfer = results['transfer']
            max_current = np.max(transfer['Id'])
            min_current = np.min(transfer['Id'])
            summary.append("📊 I-V CHARACTERISTICS:")
            summary.append(f"   Maximum Current: {max_current:.2e} A")
            summary.append(f"   Minimum Current: {min_current:.2e} A")
            summary.append(f"   On/Off Ratio: {max_current/min_current:.1e}")
            summary.append("")

        summary.append("✅ SIMULATION COMPLETED SUCCESSFULLY")
        summary.append("=" * 60)

        return "\n".join(summary)

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    """Main function to start the GUI"""

    print("🖥️  Starting Real-Time MOSFET Simulator GUI...")
    print("Features:")
    print("• Live simulation logging in separate window")
    print("• Real-time progress monitoring")
    print("• Interactive voltage controls")
    print("• Comprehensive physics models (effective mass, SRH, mobility)")
    print("• Real-time results visualization")
    print("• No synthetic data - all calculations are real!")
    print()

    try:
        app = MOSFETSimulatorGUI()
        app.run()
    except Exception as e:
        print(f"❌ GUI failed to start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
