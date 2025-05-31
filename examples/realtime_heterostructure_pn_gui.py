#!/usr/bin/env python3
"""
Real-time Heterostructure PN Diode Simulation with GUI
=====================================================

Advanced semiconductor device simulation featuring:
• Real-time GUI with live logging and progress tracking
• Heterostructure PN diode with GaAs/AlGaAs materials
• Live I-V and C-V characteristics plotting
• Professional visualization with white theme
• Interactive parameter controls

Author: Dr. Mazharuddin Mohammed
Institution: Advanced Semiconductor Research Lab
"""

import sys
import os
import numpy as np
import time
from datetime import datetime

# Add gui directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gui'))

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QProgressBar, QTextEdit,
        QGroupBox, QSplitter, QFrame, QSlider, QSpinBox
    )
    from PySide6.QtCore import Qt, QTimer, QThread, Signal, QPropertyAnimation
    from PySide6.QtGui import QFont, QPalette
    
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"❌ GUI dependencies not available: {e}")
    print("Please install: pip install PySide6 matplotlib")
    sys.exit(1)

class HeterostructureConfig:
    """Heterostructure PN diode configuration"""
    
    def __init__(self):
        # Device geometry
        self.length = 2e-6          # Device length (2 μm)
        self.width = 1e-6           # Device width (1 μm)
        
        # Material properties (GaAs/AlGaAs heterostructure)
        self.material_p = "GaAs"    # P-side material
        self.material_n = "AlGaAs"  # N-side material
        
        # Doping concentrations
        self.Na_p = 1e18           # P-side doping (cm⁻³)
        self.Nd_n = 1e18           # N-side doping (cm⁻³)
        
        # Material parameters
        self.epsilon_p = 12.9      # GaAs relative permittivity
        self.epsilon_n = 12.0      # AlGaAs relative permittivity
        self.bandgap_p = 1.42      # GaAs bandgap (eV)
        self.bandgap_n = 1.8       # AlGaAs bandgap (eV)
        
        # Simulation parameters
        self.temperature = 300     # Temperature (K)
        self.nx = 80              # Grid points in x
        self.ny = 60              # Grid points in y

class HeterostructureSimulationWorker(QThread):
    """Worker thread for heterostructure simulation"""
    
    progress_updated = Signal(int)
    log_message = Signal(str)
    simulation_completed = Signal(dict)
    plot_updated = Signal(dict)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = False
        
        # Physical constants
        self.q = 1.602e-19        # Elementary charge
        self.k = 1.381e-23        # Boltzmann constant
        self.eps0 = 8.854e-12     # Vacuum permittivity
        self.ni_p = 2.1e6         # GaAs intrinsic carrier density
        self.ni_n = 1.8e6         # AlGaAs intrinsic carrier density
    
    def setup_device_structure(self):
        """Setup heterostructure device geometry"""
        
        self.log_message.emit("🔧 Setting up heterostructure device geometry...")
        
        # Create coordinate grids
        x = np.linspace(0, self.config.length, self.config.nx)
        y = np.linspace(0, self.config.width, self.config.ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Define junction position (center)
        self.junction_pos = self.config.length / 2
        
        # Material and doping profiles
        self.material_map = np.where(self.X < self.junction_pos, 1, 2)  # 1=P-side, 2=N-side
        self.doping_map = np.where(self.X < self.junction_pos, 
                                  -self.config.Na_p, self.config.Nd_n)
        
        self.log_message.emit(f"✅ Device structure configured:")
        self.log_message.emit(f"   📐 Dimensions: {self.config.length*1e6:.1f} × {self.config.width*1e6:.1f} μm")
        self.log_message.emit(f"   🧬 P-side: {self.config.material_p}, Na = {self.config.Na_p:.1e} cm⁻³")
        self.log_message.emit(f"   🧬 N-side: {self.config.material_n}, Nd = {self.config.Nd_n:.1e} cm⁻³")
        self.log_message.emit(f"   📍 Junction position: {self.junction_pos*1e6:.1f} μm")
    
    def solve_poisson_equation(self, applied_voltage=0.0):
        """Solve Poisson equation for given applied voltage"""
        
        # Initialize potential
        V = np.zeros_like(self.X)
        
        # Built-in potential calculation
        Vt = self.k * self.config.temperature / self.q
        Vbi = Vt * np.log((self.config.Na_p * self.config.Nd_n) / (self.ni_p * self.ni_n))
        
        # Apply boundary conditions and solve iteratively
        for iteration in range(50):
            V_old = V.copy()
            
            # Update potential based on doping and applied voltage
            for i in range(1, self.config.ny-1):
                for j in range(1, self.config.nx-1):
                    if self.X[i, j] < self.junction_pos:  # P-side
                        V[i, j] = -Vbi/2 + applied_voltage/2
                    else:  # N-side
                        V[i, j] = Vbi/2 - applied_voltage/2
            
            # Apply boundary conditions
            V[0, :] = V[1, :]      # Top boundary
            V[-1, :] = V[-2, :]    # Bottom boundary
            V[:, 0] = applied_voltage  # Left contact
            V[:, -1] = 0.0         # Right contact (ground)
            
            # Check convergence
            if np.max(np.abs(V - V_old)) < 1e-6:
                break
        
        return V
    
    def calculate_carrier_densities(self, V):
        """Calculate electron and hole densities"""
        
        Vt = self.k * self.config.temperature / self.q
        n = np.zeros_like(V)
        p = np.zeros_like(V)
        
        for i in range(self.config.ny):
            for j in range(self.config.nx):
                if self.material_map[i, j] == 1:  # P-side (GaAs)
                    ni = self.ni_p
                    if self.doping_map[i, j] < 0:  # P-doped
                        p[i, j] = abs(self.doping_map[i, j]) * np.exp(-V[i, j] / Vt)
                        n[i, j] = ni**2 / p[i, j]
                else:  # N-side (AlGaAs)
                    ni = self.ni_n
                    if self.doping_map[i, j] > 0:  # N-doped
                        n[i, j] = self.doping_map[i, j] * np.exp(V[i, j] / Vt)
                        p[i, j] = ni**2 / n[i, j]
        
        return n, p
    
    def calculate_current_density(self, V, n, p):
        """Calculate current density"""
        
        # Electric field
        Ex = -np.gradient(V, axis=1) / (self.config.length / self.config.nx)
        Ey = -np.gradient(V, axis=0) / (self.config.width / self.config.ny)
        
        # Mobility values (material dependent)
        mu_n_p = 8500e-4  # GaAs electron mobility (m²/V·s)
        mu_p_p = 400e-4   # GaAs hole mobility (m²/V·s)
        mu_n_n = 8000e-4  # AlGaAs electron mobility (m²/V·s)
        mu_p_n = 350e-4   # AlGaAs hole mobility (m²/V·s)
        
        Jn = np.zeros_like(n)
        Jp = np.zeros_like(p)
        
        for i in range(self.config.ny):
            for j in range(self.config.nx):
                if self.material_map[i, j] == 1:  # P-side
                    Jn[i, j] = self.q * mu_n_p * n[i, j] * np.sqrt(Ex[i, j]**2 + Ey[i, j]**2)
                    Jp[i, j] = self.q * mu_p_p * p[i, j] * np.sqrt(Ex[i, j]**2 + Ey[i, j]**2)
                else:  # N-side
                    Jn[i, j] = self.q * mu_n_n * n[i, j] * np.sqrt(Ex[i, j]**2 + Ey[i, j]**2)
                    Jp[i, j] = self.q * mu_p_n * p[i, j] * np.sqrt(Ex[i, j]**2 + Ey[i, j]**2)
        
        return Jn, Jp
    
    def run_steady_state_simulation(self, voltage):
        """Run steady-state simulation for given voltage"""
        
        V = self.solve_poisson_equation(voltage)
        n, p = self.calculate_carrier_densities(V)
        Jn, Jp = self.calculate_current_density(V, n, p)
        
        # Calculate total current
        total_current = np.sum(Jn + Jp) * (self.config.length / self.config.nx) * (self.config.width / self.config.ny)
        
        return {
            'voltage': voltage,
            'current': total_current,
            'potential': V,
            'electron_density': n,
            'hole_density': p,
            'current_density_n': Jn,
            'current_density_p': Jp
        }
    
    def run(self):
        """Main simulation thread"""
        
        self.is_running = True
        self.log_message.emit("🚀 Starting real-time heterostructure PN diode simulation...")
        self.log_message.emit(f"⏰ Simulation started at: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Setup device structure
            self.setup_device_structure()
            self.progress_updated.emit(10)
            
            # Run I-V characteristics
            self.log_message.emit("📊 Running I-V characteristics simulation...")
            voltage_range = np.linspace(-2.0, 2.0, 50)
            currents = []
            
            for i, voltage in enumerate(voltage_range):
                if not self.is_running:
                    break
                
                result = self.run_steady_state_simulation(voltage)
                currents.append(result['current'])
                
                # Update progress
                progress = 10 + int((i + 1) / len(voltage_range) * 60)
                self.progress_updated.emit(progress)
                
                # Log progress every 10 points
                if (i + 1) % 10 == 0:
                    self.log_message.emit(f"   ⚡ Progress: {i+1}/{len(voltage_range)} ({(i+1)/len(voltage_range)*100:.1f}%)")
                    
                    # Send intermediate plot update
                    plot_data = {
                        'voltages': voltage_range[:i+1],
                        'currents': np.array(currents),
                        'type': 'iv_update'
                    }
                    self.plot_updated.emit(plot_data)
                
                # Small delay for real-time effect
                self.msleep(50)
            
            # Run C-V characteristics
            self.log_message.emit("📈 Running C-V characteristics simulation...")
            cv_voltages = np.linspace(-2.0, 1.0, 30)
            capacitances = []
            
            for i, voltage in enumerate(cv_voltages):
                if not self.is_running:
                    break
                
                # Calculate capacitance from charge variation
                dV = 0.01
                result1 = self.run_steady_state_simulation(voltage)
                result2 = self.run_steady_state_simulation(voltage + dV)
                
                charge1 = np.sum(result1['electron_density'] - result1['hole_density']) * self.q
                charge2 = np.sum(result2['electron_density'] - result2['hole_density']) * self.q
                
                capacitance = abs(charge2 - charge1) / dV
                capacitances.append(capacitance)
                
                # Update progress
                progress = 70 + int((i + 1) / len(cv_voltages) * 25)
                self.progress_updated.emit(progress)
                
                if (i + 1) % 5 == 0:
                    self.log_message.emit(f"   📊 C-V Progress: {i+1}/{len(cv_voltages)} ({(i+1)/len(cv_voltages)*100:.1f}%)")
                
                self.msleep(30)
            
            # Final results
            final_results = {
                'iv_voltages': voltage_range,
                'iv_currents': np.array(currents),
                'cv_voltages': cv_voltages,
                'cv_capacitances': np.array(capacitances),
                'device_structure': self.material_map,
                'final_potential': result['potential'],
                'final_n': result['electron_density'],
                'final_p': result['hole_density']
            }
            
            self.progress_updated.emit(100)
            self.log_message.emit("✅ Simulation completed successfully!")
            self.log_message.emit(f"⏰ Completed at: {datetime.now().strftime('%H:%M:%S')}")
            
            # Calculate and log key metrics
            forward_current = currents[np.argmin(np.abs(voltage_range - 1.0))]
            reverse_current = currents[np.argmin(np.abs(voltage_range + 1.0))]
            rectification_ratio = abs(forward_current / reverse_current) if reverse_current != 0 else float('inf')
            
            self.log_message.emit("📊 SIMULATION RESULTS:")
            self.log_message.emit(f"   🔋 Forward current (+1V): {forward_current:.2e} A")
            self.log_message.emit(f"   🔋 Reverse current (-1V): {reverse_current:.2e} A")
            self.log_message.emit(f"   📈 Rectification ratio: {rectification_ratio:.1e}")
            self.log_message.emit(f"   ⚡ Zero-bias capacitance: {capacitances[np.argmin(np.abs(cv_voltages))]:.2e} F")
            
            self.simulation_completed.emit(final_results)
            
        except Exception as e:
            self.log_message.emit(f"❌ Simulation failed: {str(e)}")
            import traceback
            self.log_message.emit(f"🔍 Error details: {traceback.format_exc()}")
        
        finally:
            self.is_running = False
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.log_message.emit("⏹️ Simulation stopped by user")

class ModernStyle:
    """Modern white theme styling"""
    
    # Color palette
    BACKGROUND_DARK = "#ffffff"      # White background
    BACKGROUND_MEDIUM = "#f5f5f5"    # Light gray
    BACKGROUND_LIGHT = "#e8e8e8"     # Slightly darker gray
    ACCENT_BLUE = "#007acc"          # Blue accent
    ACCENT_GREEN = "#4caf50"         # Green accent
    ACCENT_ORANGE = "#ff9800"        # Orange accent
    ACCENT_RED = "#f44336"           # Red accent
    TEXT_PRIMARY = "#000000"         # Black text
    TEXT_SECONDARY = "#666666"       # Dark gray text
    BORDER_COLOR = "#cccccc"         # Light gray borders
    
    @staticmethod
    def get_stylesheet():
        return f"""
        QMainWindow {{
            background-color: {ModernStyle.BACKGROUND_DARK};
            color: {ModernStyle.TEXT_PRIMARY};
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 12px;
        }}
        
        QGroupBox {{
            font-weight: bold;
            border: 2px solid {ModernStyle.BORDER_COLOR};
            border-radius: 8px;
            margin-top: 1ex;
            padding-top: 15px;
            background-color: {ModernStyle.BACKGROUND_MEDIUM};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px 0 8px;
            color: {ModernStyle.ACCENT_BLUE};
        }}
        
        QPushButton {{
            background-color: {ModernStyle.ACCENT_BLUE};
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
        }}
        
        QPushButton:hover {{
            background-color: #0086d9;
        }}
        
        QPushButton:pressed {{
            background-color: #005a9e;
        }}
        
        QPushButton.success {{
            background-color: {ModernStyle.ACCENT_GREEN};
        }}
        
        QPushButton.danger {{
            background-color: {ModernStyle.ACCENT_RED};
        }}
        
        QProgressBar {{
            border: 1px solid {ModernStyle.BORDER_COLOR};
            border-radius: 6px;
            text-align: center;
            background-color: {ModernStyle.BACKGROUND_LIGHT};
            color: {ModernStyle.TEXT_PRIMARY};
            font-weight: bold;
        }}
        
        QProgressBar::chunk {{
            background-color: {ModernStyle.ACCENT_GREEN};
            border-radius: 5px;
        }}
        
        QTextEdit {{
            background-color: {ModernStyle.BACKGROUND_DARK};
            color: {ModernStyle.TEXT_PRIMARY};
            border: 1px solid {ModernStyle.BORDER_COLOR};
            border-radius: 6px;
            padding: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
        }}
        
        QTabWidget::pane {{
            border: 1px solid {ModernStyle.BORDER_COLOR};
            background-color: {ModernStyle.BACKGROUND_MEDIUM};
            border-radius: 8px;
        }}
        
        QTabBar::tab {{
            background-color: {ModernStyle.BACKGROUND_LIGHT};
            color: {ModernStyle.TEXT_SECONDARY};
            padding: 12px 20px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            min-width: 120px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {ModernStyle.ACCENT_BLUE};
            color: white;
            font-weight: bold;
        }}
        """

class RealtimeHeterostructureGUI(QMainWindow):
    """Real-time heterostructure PN diode simulation GUI"""

    def __init__(self):
        super().__init__()

        # Initialize configuration
        self.config = HeterostructureConfig()
        self.simulation_worker = None
        self.results = {}

        self.setWindowTitle("Real-time Heterostructure PN Diode Simulation - Dr. Mazharuddin Mohammed")
        self.setGeometry(100, 100, 1400, 900)

        # Apply modern styling
        self.setStyleSheet(ModernStyle.get_stylesheet())

        self.setup_ui()
        self.setup_connections()

        # Welcome message
        self.log_message("🚀 Real-time Heterostructure PN Diode Simulator")
        self.log_message("Author: Dr. Mazharuddin Mohammed")
        self.log_message("Institution: Advanced Semiconductor Research Lab")
        self.log_message("Ready to simulate GaAs/AlGaAs heterostructure devices...")

    def setup_ui(self):
        """Setup the user interface"""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        self.create_control_panel(splitter)

        # Right panel - Results and logs
        self.create_results_panel(splitter)

        # Set splitter proportions
        splitter.setSizes([400, 1000])

        # Status bar
        self.statusBar().showMessage("Ready to simulate")

    def create_control_panel(self, parent):
        """Create the left control panel"""

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Title
        title_label = QLabel("🔬 Heterostructure PN Diode Simulator")
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: 18px;
                font-weight: bold;
                color: {ModernStyle.ACCENT_BLUE};
                padding: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {ModernStyle.BACKGROUND_MEDIUM},
                    stop:1 {ModernStyle.BACKGROUND_LIGHT});
                border-radius: 8px;
                margin-bottom: 10px;
            }}
        """)
        control_layout.addWidget(title_label)

        # Device parameters
        device_group = QGroupBox("🧬 Device Parameters")
        device_layout = QVBoxLayout(device_group)

        # Material selection (display only)
        material_label = QLabel("Materials: GaAs (P-side) / AlGaAs (N-side)")
        material_label.setStyleSheet(f"color: {ModernStyle.TEXT_PRIMARY}; font-weight: bold;")
        device_layout.addWidget(material_label)

        # Doping concentrations (display only)
        doping_label = QLabel(f"Doping: Na = {self.config.Na_p:.1e} cm⁻³, Nd = {self.config.Nd_n:.1e} cm⁻³")
        doping_label.setStyleSheet(f"color: {ModernStyle.TEXT_PRIMARY};")
        device_layout.addWidget(doping_label)

        # Device dimensions (display only)
        dims_label = QLabel(f"Dimensions: {self.config.length*1e6:.1f} × {self.config.width*1e6:.1f} μm")
        dims_label.setStyleSheet(f"color: {ModernStyle.TEXT_PRIMARY};")
        device_layout.addWidget(dims_label)

        control_layout.addWidget(device_group)

        # Simulation controls
        sim_group = QGroupBox("⚡ Simulation Controls")
        sim_layout = QVBoxLayout(sim_group)

        # Start simulation button
        self.start_button = QPushButton("🚀 Start Real-time Simulation")
        self.start_button.setProperty("class", "success")
        sim_layout.addWidget(self.start_button)

        # Stop simulation button
        self.stop_button = QPushButton("⏹️ Stop Simulation")
        self.stop_button.setProperty("class", "danger")
        self.stop_button.setEnabled(False)
        sim_layout.addWidget(self.stop_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        sim_layout.addWidget(self.progress_bar)

        control_layout.addWidget(sim_group)

        # Add stretch
        control_layout.addStretch()

        parent.addWidget(control_widget)

    def create_results_panel(self, parent):
        """Create the right results panel"""

        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        results_layout.addWidget(self.tab_widget)

        # Live Log tab
        self.create_log_tab()

        # Real-time Plots tab
        self.create_plots_tab()

        # Device Structure tab
        self.create_structure_tab()

        parent.addWidget(results_widget)

    def create_log_tab(self):
        """Create the live log tab"""

        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)

        # Log header
        log_header = QLabel("📋 Live Simulation Log")
        log_header.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                font-weight: bold;
                color: {ModernStyle.ACCENT_GREEN};
                padding: 10px;
                background-color: {ModernStyle.BACKGROUND_MEDIUM};
                border-radius: 6px;
                margin-bottom: 5px;
            }}
        """)
        log_layout.addWidget(log_header)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        self.tab_widget.addTab(log_widget, "📋 Live Log")

    def create_plots_tab(self):
        """Create the real-time plots tab"""

        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)

        # Plots header
        plots_header = QLabel("📊 Real-time Results")
        plots_header.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                font-weight: bold;
                color: {ModernStyle.ACCENT_BLUE};
                padding: 10px;
                background-color: {ModernStyle.BACKGROUND_MEDIUM};
                border-radius: 6px;
                margin-bottom: 5px;
            }}
        """)
        plots_layout.addWidget(plots_header)

        # Matplotlib canvas
        self.figure = Figure(figsize=(12, 8), facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        plots_layout.addWidget(self.canvas)

        self.tab_widget.addTab(plots_widget, "📊 Real-time Plots")

    def create_structure_tab(self):
        """Create the device structure tab"""

        structure_widget = QWidget()
        structure_layout = QVBoxLayout(structure_widget)

        # Structure header
        structure_header = QLabel("🏗️ Device Structure")
        structure_header.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                font-weight: bold;
                color: {ModernStyle.ACCENT_ORANGE};
                padding: 10px;
                background-color: {ModernStyle.BACKGROUND_MEDIUM};
                border-radius: 6px;
                margin-bottom: 5px;
            }}
        """)
        structure_layout.addWidget(structure_header)

        # Structure canvas
        self.structure_figure = Figure(figsize=(10, 6), facecolor='white')
        self.structure_canvas = FigureCanvas(self.structure_figure)
        structure_layout.addWidget(self.structure_canvas)

        self.tab_widget.addTab(structure_widget, "🏗️ Structure")

    def setup_connections(self):
        """Setup signal connections"""

        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)

    def log_message(self, message):
        """Add message to log"""

        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)

        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_simulation(self):
        """Start the simulation"""

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)

        # Clear previous results
        self.figure.clear()
        self.structure_figure.clear()
        self.canvas.draw()
        self.structure_canvas.draw()

        # Create and start simulation worker
        self.simulation_worker = HeterostructureSimulationWorker(self.config)
        self.simulation_worker.progress_updated.connect(self.update_progress)
        self.simulation_worker.log_message.connect(self.log_message)
        self.simulation_worker.simulation_completed.connect(self.simulation_completed)
        self.simulation_worker.plot_updated.connect(self.update_plots)

        self.simulation_worker.start()

        self.statusBar().showMessage("Simulation running...")

    def stop_simulation(self):
        """Stop the simulation"""

        if self.simulation_worker:
            self.simulation_worker.stop_simulation()
            self.simulation_worker.wait()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.statusBar().showMessage("Simulation stopped")

    def update_progress(self, value):
        """Update progress bar"""

        self.progress_bar.setValue(value)

    def update_plots(self, plot_data):
        """Update real-time plots"""

        if plot_data['type'] == 'iv_update':
            self.figure.clear()

            # I-V characteristics (real-time update)
            ax = self.figure.add_subplot(111)
            ax.semilogy(plot_data['voltages'], np.abs(plot_data['currents']), 'b-', linewidth=2, label='|Current|')
            ax.set_xlabel('Voltage (V)')
            ax.set_ylabel('Current (A)')
            ax.set_title('Real-time I-V Characteristics')
            ax.grid(True, alpha=0.3)
            ax.legend()

            self.canvas.draw()

    def simulation_completed(self, results):
        """Handle simulation completion"""

        self.results = results
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # Create comprehensive plots
        self.create_final_plots()
        self.create_structure_plot()

        self.statusBar().showMessage("Simulation completed successfully")

    def create_final_plots(self):
        """Create final comprehensive plots"""

        self.figure.clear()

        # Create subplots
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)

        # I-V characteristics (log scale)
        ax1.semilogy(self.results['iv_voltages'], np.abs(self.results['iv_currents']), 'b-', linewidth=2)
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('|Current| (A)')
        ax1.set_title('I-V Characteristics (Log)')
        ax1.grid(True, alpha=0.3)

        # I-V characteristics (linear scale)
        ax2.plot(self.results['iv_voltages'], self.results['iv_currents'], 'r-', linewidth=2)
        ax2.set_xlabel('Voltage (V)')
        ax2.set_ylabel('Current (A)')
        ax2.set_title('I-V Characteristics (Linear)')
        ax2.grid(True, alpha=0.3)

        # C-V characteristics
        ax3.plot(self.results['cv_voltages'], self.results['cv_capacitances'], 'g-', linewidth=2)
        ax3.set_xlabel('Voltage (V)')
        ax3.set_ylabel('Capacitance (F)')
        ax3.set_title('C-V Characteristics')
        ax3.grid(True, alpha=0.3)

        # Potential distribution
        im = ax4.imshow(self.results['final_potential'],
                       extent=[0, self.config.length*1e6, 0, self.config.width*1e6],
                       aspect='auto', cmap='viridis')
        ax4.set_xlabel('Length (μm)')
        ax4.set_ylabel('Width (μm)')
        ax4.set_title('Potential Distribution')
        self.figure.colorbar(im, ax=ax4, label='Potential (V)')

        self.figure.tight_layout()
        self.canvas.draw()

    def create_structure_plot(self):
        """Create device structure plot"""

        self.structure_figure.clear()

        ax = self.structure_figure.add_subplot(111)

        # Device structure
        im = ax.imshow(self.results['device_structure'],
                      extent=[0, self.config.length*1e6, 0, self.config.width*1e6],
                      aspect='auto', cmap='RdYlBu', alpha=0.8)

        ax.set_xlabel('Length (μm)')
        ax.set_ylabel('Width (μm)')
        ax.set_title('GaAs/AlGaAs Heterostructure Device')

        # Add junction line
        junction_pos_um = self.config.length * 1e6 / 2
        ax.axvline(x=junction_pos_um, color='black', linestyle='--', linewidth=2, label='Junction')

        # Add material labels
        ax.text(junction_pos_um/2, self.config.width*1e6*0.9, 'GaAs\n(P-side)',
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(junction_pos_um + junction_pos_um/2, self.config.width*1e6*0.9, 'AlGaAs\n(N-side)',
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.legend()
        self.structure_figure.colorbar(im, ax=ax, label='Material (1=GaAs, 2=AlGaAs)')

        self.structure_figure.tight_layout()
        self.structure_canvas.draw()

def main():
    """Main function"""

    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Heterostructure PN Diode Simulator")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Advanced Semiconductor Research Lab")

    # Create and show GUI
    gui = RealtimeHeterostructureGUI()
    gui.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
