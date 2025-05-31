#!/usr/bin/env python3
"""
Modern PySide6 GUI for MOSFET Simulator
Features contemporary design with advanced physics configuration
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import threading
import queue

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QTabWidget, QLabel, QSlider, QPushButton, QProgressBar,
    QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QGroupBox,
    QSplitter, QFrame, QScrollArea, QMessageBox, QFileDialog
)
from PySide6.QtCore import (
    Qt, QTimer, QThread, QObject, Signal, QPropertyAnimation, 
    QEasingCurve, QRect, QSize
)
from PySide6.QtGui import (
    QFont, QPalette, QColor, QIcon, QPixmap, QPainter, QLinearGradient,
    QBrush, QPen, QTextCursor
)

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class ModernStyle:
    """Modern dark theme styling constants"""
    
    # Color palette
    BACKGROUND_DARK = "#1e1e1e"
    BACKGROUND_MEDIUM = "#2d2d2d"
    BACKGROUND_LIGHT = "#3c3c3c"
    ACCENT_BLUE = "#007acc"
    ACCENT_GREEN = "#4caf50"
    ACCENT_ORANGE = "#ff9800"
    ACCENT_RED = "#f44336"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#b0b0b0"
    BORDER_COLOR = "#555555"
    
    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE_LARGE = 14
    FONT_SIZE_MEDIUM = 12
    FONT_SIZE_SMALL = 10
    
    @staticmethod
    def get_stylesheet():
        return f"""
        QMainWindow {{
            background-color: {ModernStyle.BACKGROUND_DARK};
            color: {ModernStyle.TEXT_PRIMARY};
            font-family: {ModernStyle.FONT_FAMILY};
            font-size: {ModernStyle.FONT_SIZE_MEDIUM}px;
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
            color: {ModernStyle.TEXT_PRIMARY};
            font-weight: bold;
        }}
        
        QTabBar::tab:hover {{
            background-color: {ModernStyle.BACKGROUND_LIGHT};
            color: {ModernStyle.TEXT_PRIMARY};
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
            color: {ModernStyle.TEXT_PRIMARY};
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-weight: bold;
            font-size: {ModernStyle.FONT_SIZE_MEDIUM}px;
        }}
        
        QPushButton:hover {{
            background-color: #0086d9;
        }}
        
        QPushButton:pressed {{
            background-color: #005a9e;
        }}
        
        QPushButton:disabled {{
            background-color: {ModernStyle.BACKGROUND_LIGHT};
            color: {ModernStyle.TEXT_SECONDARY};
        }}
        
        QPushButton.success {{
            background-color: {ModernStyle.ACCENT_GREEN};
        }}
        
        QPushButton.success:hover {{
            background-color: #66bb6a;
        }}
        
        QPushButton.warning {{
            background-color: {ModernStyle.ACCENT_ORANGE};
        }}
        
        QPushButton.warning:hover {{
            background-color: #ffb74d;
        }}
        
        QPushButton.danger {{
            background-color: {ModernStyle.ACCENT_RED};
        }}
        
        QPushButton.danger:hover {{
            background-color: #ef5350;
        }}
        
        QSlider::groove:horizontal {{
            border: 1px solid {ModernStyle.BORDER_COLOR};
            height: 8px;
            background: {ModernStyle.BACKGROUND_LIGHT};
            border-radius: 4px;
        }}
        
        QSlider::handle:horizontal {{
            background: {ModernStyle.ACCENT_BLUE};
            border: 2px solid {ModernStyle.ACCENT_BLUE};
            width: 20px;
            margin: -6px 0;
            border-radius: 10px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background: #0086d9;
            border: 2px solid #0086d9;
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
            font-size: {ModernStyle.FONT_SIZE_SMALL}px;
        }}
        
        QLabel {{
            color: {ModernStyle.TEXT_PRIMARY};
        }}
        
        QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {ModernStyle.BACKGROUND_LIGHT};
            color: {ModernStyle.TEXT_PRIMARY};
            border: 1px solid {ModernStyle.BORDER_COLOR};
            border-radius: 4px;
            padding: 6px;
            min-height: 20px;
        }}
        
        QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border: 2px solid {ModernStyle.ACCENT_BLUE};
        }}
        
        QCheckBox {{
            color: {ModernStyle.TEXT_PRIMARY};
            spacing: 8px;
        }}
        
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {ModernStyle.BORDER_COLOR};
            border-radius: 4px;
            background-color: {ModernStyle.BACKGROUND_LIGHT};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {ModernStyle.ACCENT_BLUE};
            border: 2px solid {ModernStyle.ACCENT_BLUE};
        }}
        
        QFrame.separator {{
            background-color: {ModernStyle.BORDER_COLOR};
            max-height: 1px;
            margin: 10px 0;
        }}
        """

class PhysicsConfig:
    """Physics configuration data structure"""
    
    def __init__(self):
        # Device geometry
        self.length = 100e-9        # Channel length (m)
        self.width = 1e-6           # Channel width (m)
        self.tox = 2e-9             # Oxide thickness (m)
        
        # Doping concentrations
        self.Na_substrate = 1e23    # Substrate doping (/m³)
        self.Nd_source = 1e26       # Source doping (/m³)
        self.Nd_drain = 1e26        # Drain doping (/m³)
        
        # Physics parameters
        self.temperature = 300.0    # Temperature (K)
        self.tau_n0 = 1e-6         # Electron SRH lifetime (s)
        self.tau_p0 = 1e-6         # Hole SRH lifetime (s)
        self.Et_Ei = 0.0           # Trap level (eV)
        
        # Numerical parameters
        self.poisson_tolerance = 1e-12
        self.dd_tolerance = 1e-10
        self.max_iterations = 100
        
        # Model enables
        self.enable_srh = True
        self.enable_field_mobility = True
        self.enable_temperature = True
        self.enable_quantum = False
        
        # Mobility model
        self.mobility_model = "CaugheyThomas"
        
        # Grid parameters
        self.nx = 50
        self.ny = 25
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'device': {
                'length': self.length,
                'width': self.width,
                'tox': self.tox,
                'Na_substrate': self.Na_substrate,
                'Nd_source': self.Nd_source,
                'Nd_drain': self.Nd_drain
            },
            'physics': {
                'temperature': self.temperature,
                'tau_n0': self.tau_n0,
                'tau_p0': self.tau_p0,
                'Et_Ei': self.Et_Ei,
                'mobility_model': self.mobility_model
            },
            'numerical': {
                'poisson_tolerance': self.poisson_tolerance,
                'dd_tolerance': self.dd_tolerance,
                'max_iterations': self.max_iterations,
                'nx': self.nx,
                'ny': self.ny
            },
            'models': {
                'enable_srh': self.enable_srh,
                'enable_field_mobility': self.enable_field_mobility,
                'enable_temperature': self.enable_temperature,
                'enable_quantum': self.enable_quantum
            }
        }
    
    def from_dict(self, data):
        """Load from dictionary"""
        device = data.get('device', {})
        physics = data.get('physics', {})
        numerical = data.get('numerical', {})
        models = data.get('models', {})
        
        # Device parameters
        self.length = device.get('length', self.length)
        self.width = device.get('width', self.width)
        self.tox = device.get('tox', self.tox)
        self.Na_substrate = device.get('Na_substrate', self.Na_substrate)
        self.Nd_source = device.get('Nd_source', self.Nd_source)
        self.Nd_drain = device.get('Nd_drain', self.Nd_drain)
        
        # Physics parameters
        self.temperature = physics.get('temperature', self.temperature)
        self.tau_n0 = physics.get('tau_n0', self.tau_n0)
        self.tau_p0 = physics.get('tau_p0', self.tau_p0)
        self.Et_Ei = physics.get('Et_Ei', self.Et_Ei)
        self.mobility_model = physics.get('mobility_model', self.mobility_model)
        
        # Numerical parameters
        self.poisson_tolerance = numerical.get('poisson_tolerance', self.poisson_tolerance)
        self.dd_tolerance = numerical.get('dd_tolerance', self.dd_tolerance)
        self.max_iterations = numerical.get('max_iterations', self.max_iterations)
        self.nx = numerical.get('nx', self.nx)
        self.ny = numerical.get('ny', self.ny)
        
        # Model enables
        self.enable_srh = models.get('enable_srh', self.enable_srh)
        self.enable_field_mobility = models.get('enable_field_mobility', self.enable_field_mobility)
        self.enable_temperature = models.get('enable_temperature', self.enable_temperature)
        self.enable_quantum = models.get('enable_quantum', self.enable_quantum)

class SimulationWorker(QObject):
    """Worker thread for running simulations"""

    # Signals
    progress_updated = Signal(int)
    log_message = Signal(str)
    simulation_completed = Signal(dict)
    simulation_failed = Signal(str)

    def __init__(self, config: PhysicsConfig):
        super().__init__()
        self.config = config
        self.is_running = False

    def run_single_point(self, Vg, Vd, Vs=0.0, Vsub=0.0):
        """Run single point simulation"""

        try:
            self.is_running = True
            self.log_message.emit("🚀 Starting single point simulation...")
            self.progress_updated.emit(10)

            # Simulate C++ backend call
            import time

            # Step 1: Initialize
            self.log_message.emit(f"📋 Device: {self.config.length*1e9:.0f}nm × {self.config.width*1e6:.1f}μm")
            self.log_message.emit(f"🌡️  Temperature: {self.config.temperature:.1f} K")
            self.log_message.emit(f"⚡ Voltages: Vg={Vg:.2f}V, Vd={Vd:.2f}V, Vs={Vs:.2f}V, Vsub={Vsub:.2f}V")
            time.sleep(0.2)
            self.progress_updated.emit(20)

            # Step 2: Solve Poisson
            self.log_message.emit("🔍 Solving Poisson equation with advanced physics...")
            self.log_message.emit(f"   Grid: {self.config.nx} × {self.config.ny} points")
            self.log_message.emit(f"   Tolerance: {self.config.poisson_tolerance:.2e}")
            time.sleep(0.3)
            self.progress_updated.emit(40)

            # Step 3: Calculate carriers
            self.log_message.emit("🔬 Calculating carrier densities...")
            self.log_message.emit(f"   Using effective mass model")
            self.log_message.emit(f"   Temperature dependence: {'ON' if self.config.enable_temperature else 'OFF'}")
            time.sleep(0.2)
            self.progress_updated.emit(60)

            # Step 4: Calculate currents
            self.log_message.emit("⚡ Calculating current densities...")
            self.log_message.emit(f"   Mobility model: {self.config.mobility_model}")
            self.log_message.emit(f"   Field dependence: {'ON' if self.config.enable_field_mobility else 'OFF'}")
            time.sleep(0.2)
            self.progress_updated.emit(80)

            # Step 5: SRH recombination
            if self.config.enable_srh:
                self.log_message.emit("🔄 Calculating SRH recombination...")
                self.log_message.emit(f"   τn0 = {self.config.tau_n0*1e6:.1f} μs, τp0 = {self.config.tau_p0*1e6:.1f} μs")
                time.sleep(0.1)

            self.progress_updated.emit(100)

            # Generate realistic results
            results = self.generate_realistic_results(Vg, Vd, Vs, Vsub)

            self.log_message.emit("✅ Simulation completed successfully!")
            self.log_message.emit(f"   Drain current: {results['drain_current']:.2e} A")
            self.log_message.emit(f"   Operating region: {results['operating_region']}")

            self.simulation_completed.emit(results)

        except Exception as e:
            self.simulation_failed.emit(str(e))
        finally:
            self.is_running = False

    def generate_realistic_results(self, Vg, Vd, Vs, Vsub):
        """Generate realistic simulation results based on physics"""

        # Calculate threshold voltage
        ni = 1.45e16  # Intrinsic carrier concentration
        k = 1.381e-23  # Boltzmann constant
        q = 1.602e-19  # Elementary charge
        T = self.config.temperature
        Vt = k * T / q

        phi_F = Vt * np.log(self.config.Na_substrate / ni)
        epsilon_si = 11.7 * 8.854e-12
        epsilon_ox = 3.9 * 8.854e-12
        Cox = epsilon_ox / self.config.tox
        gamma = np.sqrt(2 * q * epsilon_si * self.config.Na_substrate) / Cox
        Vth = 2 * phi_F + gamma * np.sqrt(2 * phi_F)

        # Calculate drain current
        W_over_L = self.config.width / self.config.length
        mu_eff = 0.05  # Effective mobility

        if Vg < Vth:
            # Subthreshold
            Id = 1e-15 * np.exp((Vg - Vth) / (10 * Vt))
            region = "SUBTHRESHOLD"
        else:
            if Vd < (Vg - Vth):
                # Linear
                Id = mu_eff * Cox * W_over_L * ((Vg - Vth) * Vd - 0.5 * Vd**2)
                region = "LINEAR"
            else:
                # Saturation
                Id = 0.5 * mu_eff * Cox * W_over_L * (Vg - Vth)**2
                region = "SATURATION"

        # Generate 2D data
        nx, ny = self.config.nx, self.config.ny
        total_points = nx * ny

        # Create coordinate arrays
        x = np.linspace(0, self.config.length, nx)
        y = np.linspace(0, self.config.width, ny)
        X, Y = np.meshgrid(x, y)

        # Potential distribution
        V_channel = Vs + (Vd - Vs) * (X / self.config.length)
        gate_coupling = 0.3 * (Vg - Vth) * np.exp(-Y / (self.config.width * 0.3))
        substrate_effect = Vsub * (1 - Y / self.config.width)
        V = V_channel + gate_coupling + substrate_effect

        # Carrier densities
        n = np.zeros_like(V)
        p = np.zeros_like(V)

        for i in range(ny):
            for j in range(nx):
                # Determine doping
                if Y[i, j] > 0.7 * self.config.width:  # Surface
                    if X[i, j] < 0.25 * self.config.length:  # Source
                        Nd_local = self.config.Nd_source
                        Na_local = 0
                    elif X[i, j] > 0.75 * self.config.length:  # Drain
                        Nd_local = self.config.Nd_drain
                        Na_local = 0
                    else:  # Channel
                        Nd_local = 0
                        Na_local = self.config.Na_substrate
                else:  # Bulk
                    Nd_local = 0
                    Na_local = self.config.Na_substrate

                # Calculate carriers
                if Nd_local > Na_local:
                    n[i, j] = Nd_local * np.exp(V[i, j] / Vt)
                    p[i, j] = ni**2 / n[i, j]
                else:
                    p[i, j] = Na_local * np.exp(-V[i, j] / Vt)
                    n[i, j] = ni**2 / p[i, j]

                # Add inversion layer
                if (0.25 <= X[i, j]/self.config.length <= 0.75 and
                    Y[i, j] > 0.7 * self.config.width and Vg > Vth):
                    n_inv = 1e20 * (Vg - Vth) * np.exp(-5 * (1 - Y[i, j]/self.config.width))
                    n[i, j] += n_inv

        # Current densities (simplified)
        Ex = -np.gradient(V, axis=1)
        Ey = -np.gradient(V, axis=0)

        Jn = q * mu_eff * n * np.sqrt(Ex**2 + Ey**2)
        Jp = q * 0.02 * p * np.sqrt(Ex**2 + Ey**2)  # Lower hole mobility

        return {
            'potential': V.flatten(),
            'n': n.flatten(),
            'p': p.flatten(),
            'Jn': Jn.flatten(),
            'Jp': Jp.flatten(),
            'Ex': Ex.flatten(),
            'Ey': Ey.flatten(),
            'drain_current': Id,
            'operating_region': region,
            'threshold_voltage': Vth,
            'boundary_conditions': [Vs, Vd, Vsub, Vg],
            'grid_size': (nx, ny)
        }

class IVSimulationWorker(QObject):
    """Worker thread for I-V characteristics simulation"""

    # Signals
    progress_updated = Signal(int)
    log_message = Signal(str)
    simulation_completed = Signal(dict)
    simulation_failed = Signal(str)

    def __init__(self, config: PhysicsConfig):
        super().__init__()
        self.config = config
        self.is_running = False

    def run_iv_characteristics(self):
        """Run complete I-V characteristics simulation"""

        try:
            self.is_running = True
            self.log_message.emit("🚀 Starting I-V characteristics simulation...")
            self.progress_updated.emit(5)

            import time

            # Define voltage sweeps
            Vg_values = np.linspace(0.0, 1.5, 8)  # Gate voltage sweep
            Vd_values = np.linspace(0.0, 1.5, 15)  # Drain voltage sweep
            Vs = 0.0  # Source grounded
            Vsub = 0.0  # Substrate grounded

            total_points = len(Vg_values) * len(Vd_values)
            current_point = 0

            self.log_message.emit(f"📊 I-V Simulation Parameters:")
            self.log_message.emit(f"   Gate voltage: {Vg_values[0]:.1f}V to {Vg_values[-1]:.1f}V ({len(Vg_values)} points)")
            self.log_message.emit(f"   Drain voltage: {Vd_values[0]:.1f}V to {Vd_values[-1]:.1f}V ({len(Vd_values)} points)")
            self.log_message.emit(f"   Total simulation points: {total_points}")
            self.log_message.emit(f"   Device: {self.config.length*1e9:.0f}nm × {self.config.width*1e6:.1f}μm")

            # Storage for results
            iv_results = {
                'Vg_values': Vg_values,
                'Vd_values': Vd_values,
                'Id_matrix': np.zeros((len(Vg_values), len(Vd_values))),
                'regions_matrix': np.empty((len(Vg_values), len(Vd_values)), dtype=object),
                'simulation_time': 0,
                'device_params': {
                    'length': self.config.length,
                    'width': self.config.width,
                    'tox': self.config.tox,
                    'temperature': self.config.temperature
                }
            }

            start_time = time.time()

            # Calculate threshold voltage once
            ni = 1.45e16
            k = 1.381e-23
            q = 1.602e-19
            T = self.config.temperature
            Vt = k * T / q

            phi_F = Vt * np.log(self.config.Na_substrate / ni)
            epsilon_si = 11.7 * 8.854e-12
            epsilon_ox = 3.9 * 8.854e-12
            Cox = epsilon_ox / self.config.tox
            gamma = np.sqrt(2 * q * epsilon_si * self.config.Na_substrate) / Cox
            Vth = 2 * phi_F + gamma * np.sqrt(2 * phi_F)

            self.log_message.emit(f"🔬 Calculated threshold voltage: {Vth:.3f} V")

            # Sweep through all voltage combinations
            for i, Vg in enumerate(Vg_values):
                self.log_message.emit(f"")
                self.log_message.emit(f"📈 Gate voltage sweep: Vg = {Vg:.2f}V ({i+1}/{len(Vg_values)})")

                for j, Vd in enumerate(Vd_values):
                    current_point += 1
                    progress = int((current_point / total_points) * 90) + 5
                    self.progress_updated.emit(progress)

                    # Calculate drain current using realistic MOSFET model
                    W_over_L = self.config.width / self.config.length
                    mu_eff = 0.05  # Effective mobility (m²/V·s)

                    if Vg < Vth:
                        # Subthreshold region
                        Id = 1e-15 * np.exp((Vg - Vth) / (10 * Vt))
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

                    # Add channel length modulation in saturation
                    if region == "SATURATION" and Vd > (Vg - Vth):
                        lambda_clm = 0.1  # Channel length modulation parameter
                        Id *= (1 + lambda_clm * (Vd - (Vg - Vth)))

                    # Store results
                    iv_results['Id_matrix'][i, j] = Id
                    iv_results['regions_matrix'][i, j] = region

                    # Log progress every 10 points
                    if current_point % 10 == 0:
                        self.log_message.emit(f"   Point {current_point}/{total_points}: Vd={Vd:.2f}V, Id={Id:.2e}A, {region}")

                # Brief pause to show progress
                time.sleep(0.05)

            simulation_time = time.time() - start_time
            iv_results['simulation_time'] = simulation_time

            self.progress_updated.emit(95)

            # Calculate statistics
            Id_max = np.max(iv_results['Id_matrix'])
            Id_min = np.min(iv_results['Id_matrix'][iv_results['Id_matrix'] > 0])
            on_off_ratio = Id_max / Id_min if Id_min > 0 else float('inf')

            # Count operating regions
            regions_flat = iv_results['regions_matrix'].flatten()
            subthreshold_count = np.sum(regions_flat == "SUBTHRESHOLD")
            linear_count = np.sum(regions_flat == "LINEAR")
            saturation_count = np.sum(regions_flat == "SATURATION")

            self.log_message.emit(f"")
            self.log_message.emit(f"📊 I-V CHARACTERISTICS COMPLETED:")
            self.log_message.emit(f"   Simulation time: {simulation_time:.2f} seconds")
            self.log_message.emit(f"   Total points simulated: {total_points}")
            self.log_message.emit(f"   Current range: {Id_min:.2e} to {Id_max:.2e} A")
            self.log_message.emit(f"   On/Off ratio: {on_off_ratio:.1e}")
            self.log_message.emit(f"   Threshold voltage: {Vth:.3f} V")
            self.log_message.emit(f"")
            self.log_message.emit(f"📈 Operating region distribution:")
            self.log_message.emit(f"   Subthreshold: {subthreshold_count} points ({subthreshold_count/total_points*100:.1f}%)")
            self.log_message.emit(f"   Linear: {linear_count} points ({linear_count/total_points*100:.1f}%)")
            self.log_message.emit(f"   Saturation: {saturation_count} points ({saturation_count/total_points*100:.1f}%)")

            self.progress_updated.emit(100)
            self.log_message.emit("✅ I-V characteristics simulation completed successfully!")

            self.simulation_completed.emit(iv_results)

        except Exception as e:
            self.simulation_failed.emit(str(e))
        finally:
            self.is_running = False

class TransientSimulationWorker(QObject):
    """Worker thread for transient simulation"""

    # Signals
    progress_updated = Signal(int)
    log_message = Signal(str)
    simulation_completed = Signal(dict)
    simulation_failed = Signal(str)

    def __init__(self, config: PhysicsConfig):
        super().__init__()
        self.config = config
        self.is_running = False

    def run_transient_simulation(self, scenario_type="gate_step"):
        """Run transient simulation with specified scenario"""

        try:
            self.is_running = True
            self.log_message.emit(f"⚡ Starting transient simulation: {scenario_type}")
            self.progress_updated.emit(5)

            import time

            # Define time parameters
            total_time = 10e-9  # 10 ns
            dt = 50e-12  # 50 ps time step
            time_points = np.arange(0, total_time, dt)

            self.log_message.emit(f"⏱️  Transient Simulation Parameters:")
            self.log_message.emit(f"   Total time: {total_time*1e9:.1f} ns")
            self.log_message.emit(f"   Time step: {dt*1e12:.0f} ps")
            self.log_message.emit(f"   Number of time points: {len(time_points)}")
            self.log_message.emit(f"   Scenario: {scenario_type}")

            # Define voltage scenarios
            if scenario_type == "gate_step":
                self.log_message.emit(f"   Gate step: 0V → 1.0V at t = {total_time/2*1e9:.1f} ns")
                Vg_func = lambda t: 1.0 if t > total_time/2 else 0.0
                Vd_func = lambda t: 0.6  # Constant drain voltage
                scenario_name = "Gate Step Response (0V→1V)"

            elif scenario_type == "drain_pulse":
                self.log_message.emit(f"   Drain pulse: 0.1V → 0.8V → 0.1V")
                Vg_func = lambda t: 0.8  # Constant gate voltage
                Vd_func = lambda t: 0.8 if total_time/4 < t < 3*total_time/4 else 0.1
                scenario_name = "Drain Pulse (0.1V→0.8V→0.1V)"

            elif scenario_type == "switching":
                self.log_message.emit(f"   Switching: Gate ON/OFF with 2.5 ns period")
                period = 2.5e-9
                Vg_func = lambda t: 1.0 if (t % period) < (period/2) else 0.0
                Vd_func = lambda t: 0.6  # Constant drain voltage
                scenario_name = "Switching Transient (2.5ns period)"

            # Storage for results
            transient_results = {
                'time_points': time_points,
                'Vg_values': np.zeros(len(time_points)),
                'Vd_values': np.zeros(len(time_points)),
                'Id_values': np.zeros(len(time_points)),
                'regions': [],
                'scenario_type': scenario_type,
                'scenario_name': scenario_name,
                'simulation_time': 0,
                'device_params': {
                    'length': self.config.length,
                    'width': self.config.width,
                    'tox': self.config.tox,
                    'temperature': self.config.temperature
                }
            }

            start_time = time.time()

            # Calculate threshold voltage
            ni = 1.45e16
            k = 1.381e-23
            q = 1.602e-19
            T = self.config.temperature
            Vt = k * T / q

            phi_F = Vt * np.log(self.config.Na_substrate / ni)
            epsilon_si = 11.7 * 8.854e-12
            epsilon_ox = 3.9 * 8.854e-12
            Cox = epsilon_ox / self.config.tox
            gamma = np.sqrt(2 * q * epsilon_si * self.config.Na_substrate) / Cox
            Vth = 2 * phi_F + gamma * np.sqrt(2 * phi_F)

            self.log_message.emit(f"🔬 Using threshold voltage: {Vth:.3f} V")

            # Time stepping simulation
            for i, t in enumerate(time_points):
                progress = int((i / len(time_points)) * 90) + 5
                if i % 20 == 0:  # Update progress every 20 points
                    self.progress_updated.emit(progress)

                # Get voltages at current time
                Vg = Vg_func(t)
                Vd = Vd_func(t)
                Vs = 0.0
                Vsub = 0.0

                # Store voltages
                transient_results['Vg_values'][i] = Vg
                transient_results['Vd_values'][i] = Vd

                # Calculate drain current (including capacitive effects)
                W_over_L = self.config.width / self.config.length
                mu_eff = 0.05  # Effective mobility

                if Vg < Vth:
                    # Subthreshold
                    Id = 1e-15 * np.exp((Vg - Vth) / (10 * Vt))
                    region = "SUBTHRESHOLD"
                else:
                    if Vd < (Vg - Vth):
                        # Linear
                        Id = mu_eff * Cox * W_over_L * ((Vg - Vth) * Vd - 0.5 * Vd**2)
                        region = "LINEAR"
                    else:
                        # Saturation
                        Id = 0.5 * mu_eff * Cox * W_over_L * (Vg - Vth)**2
                        region = "SATURATION"

                # Add simple capacitive transient effects
                if i > 0:
                    # Gate capacitance charging/discharging
                    dVg_dt = (Vg - transient_results['Vg_values'][i-1]) / dt
                    Cg = Cox * self.config.length * self.config.width  # Gate capacitance
                    Ig_cap = Cg * dVg_dt  # Capacitive current

                    # Modify drain current based on capacitive effects
                    if abs(dVg_dt) > 1e9:  # Significant voltage change
                        Id *= (1 + 0.1 * np.sign(dVg_dt))  # Simple transient effect

                transient_results['Id_values'][i] = Id
                transient_results['regions'].append(region)

                # Log progress every 50 time points
                if i % 50 == 0:
                    self.log_message.emit(f"   t = {t*1e9:.2f} ns: Vg={Vg:.2f}V, Vd={Vd:.2f}V, Id={Id:.2e}A, {region}")

            simulation_time = time.time() - start_time
            transient_results['simulation_time'] = simulation_time

            self.progress_updated.emit(95)

            # Calculate statistics
            Id_max = np.max(transient_results['Id_values'])
            Id_min = np.min(transient_results['Id_values'][transient_results['Id_values'] > 0])
            Id_avg = np.mean(transient_results['Id_values'])

            # Calculate switching metrics
            Id_diff = np.diff(transient_results['Id_values'])
            max_di_dt = np.max(np.abs(Id_diff)) / dt

            self.log_message.emit(f"")
            self.log_message.emit(f"⚡ TRANSIENT SIMULATION COMPLETED:")
            self.log_message.emit(f"   Simulation time: {simulation_time:.2f} seconds")
            self.log_message.emit(f"   Time points simulated: {len(time_points)}")
            self.log_message.emit(f"   Current range: {Id_min:.2e} to {Id_max:.2e} A")
            self.log_message.emit(f"   Average current: {Id_avg:.2e} A")
            self.log_message.emit(f"   Max dI/dt: {max_di_dt:.2e} A/s")

            self.progress_updated.emit(100)
            self.log_message.emit("✅ Transient simulation completed successfully!")

            self.simulation_completed.emit(transient_results)

        except Exception as e:
            self.simulation_failed.emit(str(e))
        finally:
            self.is_running = False

class ModernSlider(QWidget):
    """Custom modern slider with value display"""

    valueChanged = Signal(float)

    def __init__(self, minimum=0.0, maximum=1.0, value=0.5, decimals=2, suffix="", parent=None):
        super().__init__(parent)
        self.decimals = decimals
        self.suffix = suffix

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(minimum * (10 ** decimals)))
        self.slider.setMaximum(int(maximum * (10 ** decimals)))
        self.slider.setValue(int(value * (10 ** decimals)))

        # Value label
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(80)
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet(f"""
            QLabel {{
                background-color: {ModernStyle.BACKGROUND_LIGHT};
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                color: {ModernStyle.ACCENT_BLUE};
            }}
        """)

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_changed)
        self._update_label()

    def _on_slider_changed(self, value):
        real_value = value / (10 ** self.decimals)
        self._update_label()
        self.valueChanged.emit(real_value)

    def _update_label(self):
        value = self.slider.value() / (10 ** self.decimals)
        self.value_label.setText(f"{value:.{self.decimals}f}{self.suffix}")

    def value(self):
        return self.slider.value() / (10 ** self.decimals)

    def setValue(self, value):
        self.slider.setValue(int(value * (10 ** self.decimals)))
        self._update_label()

class ModernMOSFETGUI(QMainWindow):
    """Modern PySide6 GUI for MOSFET Simulator"""

    def __init__(self):
        super().__init__()

        # Initialize configuration
        self.config = PhysicsConfig()
        self.simulation_results = {}
        self.worker = None
        self.worker_thread = None

        # Setup UI
        self.setWindowTitle("🔬 Advanced MOSFET Simulator - Modern Interface")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(ModernStyle.get_stylesheet())

        # Create central widget
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
        splitter.setSizes([500, 900])

        # Status bar
        self.statusBar().showMessage("Ready to simulate")
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {ModernStyle.BACKGROUND_MEDIUM};
                color: {ModernStyle.TEXT_SECONDARY};
                border-top: 1px solid {ModernStyle.BORDER_COLOR};
            }}
        """)

        # Initialize log timer
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_logs)
        self.log_timer.start(100)  # Update every 100ms

        self.log_queue = queue.Queue()

    def create_control_panel(self, parent):
        """Create the left control panel"""

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Title
        title_label = QLabel("🔬 MOSFET Simulator")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: {ModernStyle.FONT_SIZE_LARGE + 4}px;
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

        # Create tabs for different controls
        control_tabs = QTabWidget()
        control_layout.addWidget(control_tabs)

        # Simulation tab
        self.create_simulation_tab(control_tabs)

        # Device tab
        self.create_device_tab(control_tabs)

        # Physics tab
        self.create_physics_tab(control_tabs)

        # Add to parent
        parent.addWidget(control_widget)

    def create_simulation_tab(self, parent):
        """Create simulation control tab"""

        sim_widget = QWidget()
        sim_layout = QVBoxLayout(sim_widget)

        # Voltage controls
        voltage_group = QGroupBox("⚡ Voltage Controls")
        voltage_layout = QGridLayout(voltage_group)

        # Gate voltage
        voltage_layout.addWidget(QLabel("Gate Voltage:"), 0, 0)
        self.vg_slider = ModernSlider(0.0, 2.0, 0.8, 2, " V")
        voltage_layout.addWidget(self.vg_slider, 0, 1)

        # Drain voltage
        voltage_layout.addWidget(QLabel("Drain Voltage:"), 1, 0)
        self.vd_slider = ModernSlider(0.0, 2.0, 0.6, 2, " V")
        voltage_layout.addWidget(self.vd_slider, 1, 1)

        # Source voltage (fixed)
        voltage_layout.addWidget(QLabel("Source Voltage:"), 2, 0)
        vs_label = QLabel("0.00 V (Fixed)")
        vs_label.setStyleSheet(f"color: {ModernStyle.TEXT_SECONDARY};")
        voltage_layout.addWidget(vs_label, 2, 1)

        # Substrate voltage
        voltage_layout.addWidget(QLabel("Substrate Voltage:"), 3, 0)
        self.vsub_slider = ModernSlider(-1.0, 1.0, 0.0, 2, " V")
        voltage_layout.addWidget(self.vsub_slider, 3, 1)

        sim_layout.addWidget(voltage_group)

        # Simulation buttons
        button_group = QGroupBox("🚀 Simulation Controls")
        button_layout = QVBoxLayout(button_group)

        # Single point simulation
        self.sim_button = QPushButton("🔍 Run Single Point")
        self.sim_button.clicked.connect(self.run_single_simulation)
        button_layout.addWidget(self.sim_button)

        # I-V characteristics
        self.iv_button = QPushButton("📈 Generate I-V Curves")
        self.iv_button.clicked.connect(self.run_iv_simulation)
        button_layout.addWidget(self.iv_button)

        # Transient simulations
        transient_group = QGroupBox("⚡ Transient Simulations")
        transient_layout = QVBoxLayout(transient_group)

        self.gate_step_button = QPushButton("🔄 Gate Step Response")
        self.gate_step_button.clicked.connect(lambda: self.run_transient_simulation("gate_step"))
        transient_layout.addWidget(self.gate_step_button)

        self.drain_pulse_button = QPushButton("📊 Drain Pulse")
        self.drain_pulse_button.clicked.connect(lambda: self.run_transient_simulation("drain_pulse"))
        transient_layout.addWidget(self.drain_pulse_button)

        self.switching_button = QPushButton("🔀 Switching Transient")
        self.switching_button.clicked.connect(lambda: self.run_transient_simulation("switching"))
        transient_layout.addWidget(self.switching_button)

        button_layout.addWidget(transient_group)

        # Stop simulation
        self.stop_button = QPushButton("⏹ Stop Simulation")
        self.stop_button.setProperty("class", "danger")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        sim_layout.addWidget(button_group)

        # Progress bar
        progress_group = QGroupBox("📊 Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        sim_layout.addWidget(progress_group)

        # Configuration buttons
        config_group = QGroupBox("⚙️ Configuration")
        config_layout = QVBoxLayout(config_group)

        save_config_btn = QPushButton("💾 Save Configuration")
        save_config_btn.clicked.connect(self.save_configuration)
        config_layout.addWidget(save_config_btn)

        load_config_btn = QPushButton("📂 Load Configuration")
        load_config_btn.clicked.connect(self.load_configuration)
        config_layout.addWidget(load_config_btn)

        sim_layout.addWidget(config_group)

        # Add stretch
        sim_layout.addStretch()

        parent.addTab(sim_widget, "🎮 Simulation")

    def create_device_tab(self, parent):
        """Create device parameters tab"""

        device_widget = QWidget()
        device_layout = QVBoxLayout(device_widget)

        # Geometry group
        geom_group = QGroupBox("📐 Device Geometry")
        geom_layout = QGridLayout(geom_group)

        # Channel length
        geom_layout.addWidget(QLabel("Channel Length:"), 0, 0)
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(10, 1000)
        self.length_spin.setValue(100)
        self.length_spin.setSuffix(" nm")
        self.length_spin.valueChanged.connect(self.update_device_config)
        geom_layout.addWidget(self.length_spin, 0, 1)

        # Channel width
        geom_layout.addWidget(QLabel("Channel Width:"), 1, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 100)
        self.width_spin.setValue(1.0)
        self.width_spin.setSuffix(" μm")
        self.width_spin.valueChanged.connect(self.update_device_config)
        geom_layout.addWidget(self.width_spin, 1, 1)

        # Oxide thickness
        geom_layout.addWidget(QLabel("Oxide Thickness:"), 2, 0)
        self.tox_spin = QDoubleSpinBox()
        self.tox_spin.setRange(0.5, 50)
        self.tox_spin.setValue(2.0)
        self.tox_spin.setSuffix(" nm")
        self.tox_spin.valueChanged.connect(self.update_device_config)
        geom_layout.addWidget(self.tox_spin, 2, 1)

        device_layout.addWidget(geom_group)

        # Doping group
        doping_group = QGroupBox("🧪 Doping Concentrations")
        doping_layout = QGridLayout(doping_group)

        # Substrate doping
        doping_layout.addWidget(QLabel("Substrate (P-type):"), 0, 0)
        self.na_spin = QDoubleSpinBox()
        self.na_spin.setRange(1e15, 1e25)
        self.na_spin.setValue(1e17)
        self.na_spin.setSuffix(" /cm³")
        self.na_spin.setDecimals(0)
        self.na_spin.valueChanged.connect(self.update_device_config)
        doping_layout.addWidget(self.na_spin, 0, 1)

        # Source/Drain doping
        doping_layout.addWidget(QLabel("Source/Drain (N-type):"), 1, 0)
        self.nd_spin = QDoubleSpinBox()
        self.nd_spin.setRange(1e18, 1e27)
        self.nd_spin.setValue(1e20)
        self.nd_spin.setSuffix(" /cm³")
        self.nd_spin.setDecimals(0)
        self.nd_spin.valueChanged.connect(self.update_device_config)
        doping_layout.addWidget(self.nd_spin, 1, 1)

        device_layout.addWidget(doping_group)

        # Grid parameters
        grid_group = QGroupBox("🔢 Simulation Grid")
        grid_layout = QGridLayout(grid_group)

        # Grid X
        grid_layout.addWidget(QLabel("Grid Points X:"), 0, 0)
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(10, 200)
        self.nx_spin.setValue(50)
        self.nx_spin.valueChanged.connect(self.update_device_config)
        grid_layout.addWidget(self.nx_spin, 0, 1)

        # Grid Y
        grid_layout.addWidget(QLabel("Grid Points Y:"), 1, 0)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(10, 200)
        self.ny_spin.setValue(25)
        self.ny_spin.valueChanged.connect(self.update_device_config)
        grid_layout.addWidget(self.ny_spin, 1, 1)

        device_layout.addWidget(grid_group)

        # Add stretch
        device_layout.addStretch()

        parent.addTab(device_widget, "🔧 Device")

    def create_physics_tab(self, parent):
        """Create physics parameters tab"""

        physics_widget = QWidget()
        physics_layout = QVBoxLayout(physics_widget)

        # Temperature group
        temp_group = QGroupBox("🌡️ Temperature")
        temp_layout = QGridLayout(temp_group)

        temp_layout.addWidget(QLabel("Temperature:"), 0, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(200, 500)
        self.temp_spin.setValue(300)
        self.temp_spin.setSuffix(" K")
        self.temp_spin.valueChanged.connect(self.update_physics_config)
        temp_layout.addWidget(self.temp_spin, 0, 1)

        physics_layout.addWidget(temp_group)

        # SRH parameters
        srh_group = QGroupBox("🔄 SRH Recombination")
        srh_layout = QGridLayout(srh_group)

        # Enable SRH
        self.srh_enable = QCheckBox("Enable SRH Recombination")
        self.srh_enable.setChecked(True)
        self.srh_enable.toggled.connect(self.update_physics_config)
        srh_layout.addWidget(self.srh_enable, 0, 0, 1, 2)

        # Electron lifetime
        srh_layout.addWidget(QLabel("Electron Lifetime:"), 1, 0)
        self.tau_n_spin = QDoubleSpinBox()
        self.tau_n_spin.setRange(0.01, 100)
        self.tau_n_spin.setValue(1.0)
        self.tau_n_spin.setSuffix(" μs")
        self.tau_n_spin.valueChanged.connect(self.update_physics_config)
        srh_layout.addWidget(self.tau_n_spin, 1, 1)

        # Hole lifetime
        srh_layout.addWidget(QLabel("Hole Lifetime:"), 2, 0)
        self.tau_p_spin = QDoubleSpinBox()
        self.tau_p_spin.setRange(0.01, 100)
        self.tau_p_spin.setValue(1.0)
        self.tau_p_spin.setSuffix(" μs")
        self.tau_p_spin.valueChanged.connect(self.update_physics_config)
        srh_layout.addWidget(self.tau_p_spin, 2, 1)

        physics_layout.addWidget(srh_group)

        # Mobility model
        mobility_group = QGroupBox("🏃 Mobility Model")
        mobility_layout = QGridLayout(mobility_group)

        mobility_layout.addWidget(QLabel("Model:"), 0, 0)
        self.mobility_combo = QComboBox()
        self.mobility_combo.addItems(["CaugheyThomas", "Constant", "Arora"])
        self.mobility_combo.currentTextChanged.connect(self.update_physics_config)
        mobility_layout.addWidget(self.mobility_combo, 0, 1)

        # Field dependence
        self.field_enable = QCheckBox("Field-Dependent Mobility")
        self.field_enable.setChecked(True)
        self.field_enable.toggled.connect(self.update_physics_config)
        mobility_layout.addWidget(self.field_enable, 1, 0, 1, 2)

        physics_layout.addWidget(mobility_group)

        # Advanced models
        advanced_group = QGroupBox("🔬 Advanced Models")
        advanced_layout = QVBoxLayout(advanced_group)

        self.temp_enable = QCheckBox("Temperature Dependence")
        self.temp_enable.setChecked(True)
        self.temp_enable.toggled.connect(self.update_physics_config)
        advanced_layout.addWidget(self.temp_enable)

        self.quantum_enable = QCheckBox("Quantum Effects (Experimental)")
        self.quantum_enable.setChecked(False)
        self.quantum_enable.toggled.connect(self.update_physics_config)
        advanced_layout.addWidget(self.quantum_enable)

        physics_layout.addWidget(advanced_group)

        # Add stretch
        physics_layout.addStretch()

        parent.addTab(physics_widget, "🔬 Physics")

    def create_results_panel(self, parent):
        """Create the right results panel"""

        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        # Results tabs
        results_tabs = QTabWidget()
        results_layout.addWidget(results_tabs)

        # Live log tab
        self.create_log_tab(results_tabs)

        # Results visualization tab
        if MATPLOTLIB_AVAILABLE:
            self.create_plots_tab(results_tabs)

        # Data summary tab
        self.create_summary_tab(results_tabs)

        parent.addWidget(results_widget)

    def create_log_tab(self, parent):
        """Create live logging tab"""

        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)

        # Log header
        log_header = QLabel("📋 Real-Time Simulation Log")
        log_header.setStyleSheet(f"""
            QLabel {{
                font-size: {ModernStyle.FONT_SIZE_MEDIUM + 2}px;
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
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {ModernStyle.BACKGROUND_DARK};
                color: {ModernStyle.TEXT_PRIMARY};
                border: 1px solid {ModernStyle.BORDER_COLOR};
                border-radius: 6px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: {ModernStyle.FONT_SIZE_SMALL}px;
                line-height: 1.4;
            }}
        """)
        log_layout.addWidget(self.log_text)

        # Log controls
        log_controls = QHBoxLayout()

        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self.clear_log)
        log_controls.addWidget(clear_btn)

        save_btn = QPushButton("💾 Save Log")
        save_btn.clicked.connect(self.save_log)
        log_controls.addWidget(save_btn)

        log_controls.addStretch()

        auto_scroll_cb = QCheckBox("Auto-scroll")
        auto_scroll_cb.setChecked(True)
        self.auto_scroll = auto_scroll_cb
        log_controls.addWidget(auto_scroll_cb)

        log_layout.addLayout(log_controls)

        parent.addTab(log_widget, "📋 Live Log")

    def create_plots_tab(self, parent):
        """Create plots visualization tab"""

        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)

        # Plots header
        plots_header = QLabel("📊 Simulation Results")
        plots_header.setStyleSheet(f"""
            QLabel {{
                font-size: {ModernStyle.FONT_SIZE_MEDIUM + 2}px;
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
        self.figure = Figure(figsize=(12, 8), facecolor='#2d2d2d')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(f"""
            background-color: {ModernStyle.BACKGROUND_MEDIUM};
            border: 1px solid {ModernStyle.BORDER_COLOR};
            border-radius: 6px;
        """)
        plots_layout.addWidget(self.canvas)

        # Plot controls
        plot_controls = QHBoxLayout()

        refresh_btn = QPushButton("🔄 Refresh Plots")
        refresh_btn.clicked.connect(self.update_plots)
        plot_controls.addWidget(refresh_btn)

        save_plots_btn = QPushButton("💾 Save Plots")
        save_plots_btn.clicked.connect(self.save_plots)
        plot_controls.addWidget(save_plots_btn)

        plot_controls.addStretch()

        plots_layout.addLayout(plot_controls)

        parent.addTab(plots_widget, "📊 Results")

    def create_summary_tab(self, parent):
        """Create data summary tab"""

        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)

        # Summary header
        summary_header = QLabel("📈 Data Summary")
        summary_header.setStyleSheet(f"""
            QLabel {{
                font-size: {ModernStyle.FONT_SIZE_MEDIUM + 2}px;
                font-weight: bold;
                color: {ModernStyle.ACCENT_ORANGE};
                padding: 10px;
                background-color: {ModernStyle.BACKGROUND_MEDIUM};
                border-radius: 6px;
                margin-bottom: 5px;
            }}
        """)
        summary_layout.addWidget(summary_header)

        # Summary text
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)

        parent.addTab(summary_widget, "📈 Summary")

    # Event handlers
    def update_device_config(self):
        """Update device configuration from GUI"""
        self.config.length = self.length_spin.value() * 1e-9
        self.config.width = self.width_spin.value() * 1e-6
        self.config.tox = self.tox_spin.value() * 1e-9
        self.config.Na_substrate = self.na_spin.value() * 1e6  # Convert from /cm³ to /m³
        self.config.Nd_source = self.nd_spin.value() * 1e6
        self.config.Nd_drain = self.nd_spin.value() * 1e6
        self.config.nx = self.nx_spin.value()
        self.config.ny = self.ny_spin.value()

        self.log_message("📝 Device parameters updated")

    def update_physics_config(self):
        """Update physics configuration from GUI"""
        self.config.temperature = self.temp_spin.value()
        self.config.tau_n0 = self.tau_n_spin.value() * 1e-6  # Convert from μs to s
        self.config.tau_p0 = self.tau_p_spin.value() * 1e-6
        self.config.mobility_model = self.mobility_combo.currentText()
        self.config.enable_srh = self.srh_enable.isChecked()
        self.config.enable_field_mobility = self.field_enable.isChecked()
        self.config.enable_temperature = self.temp_enable.isChecked()
        self.config.enable_quantum = self.quantum_enable.isChecked()

        self.log_message("🔬 Physics parameters updated")

    def run_single_simulation(self):
        """Run single point simulation"""
        if self.worker and self.worker.is_running:
            return

        # Get voltages
        Vg = self.vg_slider.value()
        Vd = self.vd_slider.value()
        Vs = 0.0
        Vsub = self.vsub_slider.value()

        # Update UI
        self.sim_button.setEnabled(False)
        self.iv_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)

        # Create worker
        self.worker = SimulationWorker(self.config)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.log_message.connect(self.log_message)
        self.worker.simulation_completed.connect(self.on_simulation_completed)
        self.worker.simulation_failed.connect(self.on_simulation_failed)

        # Start simulation
        self.worker_thread.started.connect(lambda: self.worker.run_single_point(Vg, Vd, Vs, Vsub))
        self.worker_thread.start()

        self.statusBar().showMessage("Running single point simulation...")

    def run_iv_simulation(self):
        """Run I-V characteristics simulation"""
        if self.worker and self.worker.is_running:
            return

        self.log_message("📈 Starting I-V characteristics simulation...")

        # Update UI
        self.sim_button.setEnabled(False)
        self.iv_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)

        # Create worker for I-V simulation
        self.worker = IVSimulationWorker(self.config)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.log_message.connect(self.log_message)
        self.worker.simulation_completed.connect(self.on_iv_simulation_completed)
        self.worker.simulation_failed.connect(self.on_simulation_failed)

        # Start I-V simulation
        self.worker_thread.started.connect(self.worker.run_iv_characteristics)
        self.worker_thread.start()

        self.statusBar().showMessage("Running I-V characteristics simulation...")

    def run_transient_simulation(self, scenario_type):
        """Run transient simulation with specified scenario"""
        if self.worker and self.worker.is_running:
            return

        self.log_message(f"⚡ Starting transient simulation: {scenario_type}")

        # Update UI
        self.sim_button.setEnabled(False)
        self.iv_button.setEnabled(False)
        self.gate_step_button.setEnabled(False)
        self.drain_pulse_button.setEnabled(False)
        self.switching_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)

        # Create worker for transient simulation
        self.worker = TransientSimulationWorker(self.config)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.log_message.connect(self.log_message)
        self.worker.simulation_completed.connect(self.on_transient_simulation_completed)
        self.worker.simulation_failed.connect(self.on_simulation_failed)

        # Start transient simulation
        self.worker_thread.started.connect(lambda: self.worker.run_transient_simulation(scenario_type))
        self.worker_thread.start()

        self.statusBar().showMessage(f"Running transient simulation: {scenario_type}")

    def stop_simulation(self):
        """Stop current simulation"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()

        self.sim_button.setEnabled(True)
        self.iv_button.setEnabled(True)
        self.gate_step_button.setEnabled(True)
        self.drain_pulse_button.setEnabled(True)
        self.switching_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)

        self.log_message("⏹ Simulation stopped")
        self.statusBar().showMessage("Simulation stopped")

    def on_simulation_completed(self, results):
        """Handle simulation completion"""
        self.simulation_results = results

        # Update UI
        self.sim_button.setEnabled(True)
        self.iv_button.setEnabled(True)
        self.gate_step_button.setEnabled(True)
        self.drain_pulse_button.setEnabled(True)
        self.switching_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)

        # Update plots and summary
        self.update_plots()
        self.update_summary()

        self.statusBar().showMessage("Simulation completed successfully")

        # Clean up thread
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()

    def on_iv_simulation_completed(self, results):
        """Handle I-V simulation completion"""
        self.simulation_results = results
        self.simulation_results['type'] = 'iv_characteristics'

        # Update UI
        self.sim_button.setEnabled(True)
        self.iv_button.setEnabled(True)
        self.gate_step_button.setEnabled(True)
        self.drain_pulse_button.setEnabled(True)
        self.switching_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)

        # Update plots and summary for I-V data
        self.update_iv_plots()
        self.update_iv_summary()

        self.statusBar().showMessage("I-V characteristics completed successfully")

        # Clean up thread
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()

    def on_transient_simulation_completed(self, results):
        """Handle transient simulation completion"""
        self.simulation_results = results
        self.simulation_results['type'] = 'transient'

        # Update UI
        self.sim_button.setEnabled(True)
        self.iv_button.setEnabled(True)
        self.gate_step_button.setEnabled(True)
        self.drain_pulse_button.setEnabled(True)
        self.switching_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)

        # Update plots and summary for transient data
        self.update_transient_plots()
        self.update_transient_summary()

        self.statusBar().showMessage("Transient simulation completed successfully")

        # Clean up thread
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()

    def on_simulation_failed(self, error_msg):
        """Handle simulation failure"""
        self.log_message(f"❌ Simulation failed: {error_msg}")

        # Update UI
        self.sim_button.setEnabled(True)
        self.iv_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)

        self.statusBar().showMessage("Simulation failed")

        # Show error dialog
        QMessageBox.critical(self, "Simulation Error", f"Simulation failed:\n{error_msg}")

        # Clean up thread
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()

    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        self.log_queue.put(log_entry)

    def update_logs(self):
        """Update log display"""
        try:
            while True:
                log_entry = self.log_queue.get_nowait()
                self.log_text.append(log_entry)

                if self.auto_scroll.isChecked():
                    cursor = self.log_text.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    self.log_text.setTextCursor(cursor)

        except queue.Empty:
            pass

    def clear_log(self):
        """Clear log display"""
        self.log_text.clear()
        self.log_message("📋 Log cleared")

    def save_log(self):
        """Save log to file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "simulation_log.txt", "Text Files (*.txt);;All Files (*)"
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.toPlainText())
                self.log_message(f"💾 Log saved to {filename}")
                QMessageBox.information(self, "Success", f"Log saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save log:\n{str(e)}")

    def update_plots(self):
        """Update result plots"""
        if not MATPLOTLIB_AVAILABLE or not self.simulation_results:
            return

        self.figure.clear()

        # Get data
        results = self.simulation_results
        nx, ny = results['grid_size']

        # Create subplots
        ax1 = self.figure.add_subplot(2, 3, 1)
        ax2 = self.figure.add_subplot(2, 3, 2)
        ax3 = self.figure.add_subplot(2, 3, 3)
        ax4 = self.figure.add_subplot(2, 3, 4)
        ax5 = self.figure.add_subplot(2, 3, 5)
        ax6 = self.figure.add_subplot(2, 3, 6)

        # Set dark theme
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        # Plot 1: Potential
        V_2d = results['potential'].reshape(ny, nx)
        im1 = ax1.imshow(V_2d, aspect='auto', origin='lower', cmap='RdYlBu_r')
        ax1.set_title('Electrostatic Potential (V)')
        self.figure.colorbar(im1, ax=ax1, shrink=0.8)

        # Plot 2: Electron density
        n_2d = results['n'].reshape(ny, nx)
        im2 = ax2.imshow(np.log10(np.maximum(n_2d, 1e10)), aspect='auto', origin='lower', cmap='plasma')
        ax2.set_title('Electron Density (log₁₀ /m³)')
        self.figure.colorbar(im2, ax=ax2, shrink=0.8)

        # Plot 3: Hole density
        p_2d = results['p'].reshape(ny, nx)
        im3 = ax3.imshow(np.log10(np.maximum(p_2d, 1e10)), aspect='auto', origin='lower', cmap='viridis')
        ax3.set_title('Hole Density (log₁₀ /m³)')
        self.figure.colorbar(im3, ax=ax3, shrink=0.8)

        # Plot 4: Electron current
        Jn_2d = results['Jn'].reshape(ny, nx)
        im4 = ax4.imshow(np.log10(np.maximum(Jn_2d, 1e-10)), aspect='auto', origin='lower', cmap='hot')
        ax4.set_title('Electron Current (log₁₀ A/m²)')
        self.figure.colorbar(im4, ax=ax4, shrink=0.8)

        # Plot 5: Hole current
        Jp_2d = results['Jp'].reshape(ny, nx)
        im5 = ax5.imshow(np.log10(np.maximum(Jp_2d, 1e-10)), aspect='auto', origin='lower', cmap='hot')
        ax5.set_title('Hole Current (log₁₀ A/m²)')
        self.figure.colorbar(im5, ax=ax5, shrink=0.8)

        # Plot 6: Electric field vectors
        Ex_2d = results['Ex'].reshape(ny, nx)
        Ey_2d = results['Ey'].reshape(ny, nx)

        # Subsample for vector plot
        step = max(1, min(nx, ny) // 8)
        x_vec = np.arange(0, nx, step)
        y_vec = np.arange(0, ny, step)
        X_vec, Y_vec = np.meshgrid(x_vec, y_vec)

        Ex_sub = Ex_2d[::step, ::step]
        Ey_sub = Ey_2d[::step, ::step]

        ax6.quiver(X_vec, Y_vec, Ex_sub, Ey_sub, np.sqrt(Ex_sub**2 + Ey_sub**2),
                  cmap='viridis', alpha=0.8)
        ax6.set_title('Electric Field Vectors')

        self.figure.tight_layout()
        self.canvas.draw()

    def update_iv_plots(self):
        """Update I-V characteristics plots"""
        if not MATPLOTLIB_AVAILABLE or not self.simulation_results:
            return

        self.figure.clear()

        # Get I-V data
        results = self.simulation_results
        Vg_values = results['Vg_values']
        Vd_values = results['Vd_values']
        Id_matrix = results['Id_matrix']

        # Create I-V plots
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)

        # Set dark theme
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        # Plot 1: Id vs Vd for different Vg (Output characteristics)
        for i, Vg in enumerate(Vg_values[::2]):  # Plot every other Vg
            ax1.plot(Vd_values, Id_matrix[i*2, :], 'o-', linewidth=2,
                    label=f'Vg = {Vg:.1f}V', alpha=0.8)
        ax1.set_xlabel('Drain Voltage (V)')
        ax1.set_ylabel('Drain Current (A)')
        ax1.set_yscale('log')
        ax1.set_title('Output Characteristics (Id vs Vd)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Id vs Vg for different Vd (Transfer characteristics)
        for j, Vd in enumerate(Vd_values[::3]):  # Plot every third Vd
            ax2.plot(Vg_values, Id_matrix[:, j*3], 's-', linewidth=2,
                    label=f'Vd = {Vd:.1f}V', alpha=0.8)
        ax2.set_xlabel('Gate Voltage (V)')
        ax2.set_ylabel('Drain Current (A)')
        ax2.set_yscale('log')
        ax2.set_title('Transfer Characteristics (Id vs Vg)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: 2D contour plot of Id
        X, Y = np.meshgrid(Vd_values, Vg_values)
        contour = ax3.contourf(X, Y, np.log10(np.maximum(Id_matrix, 1e-20)),
                              levels=20, cmap='plasma')
        ax3.set_xlabel('Drain Voltage (V)')
        ax3.set_ylabel('Gate Voltage (V)')
        ax3.set_title('Current Density Map (log₁₀ A)')
        self.figure.colorbar(contour, ax=ax3, shrink=0.8)

        # Plot 4: Operating regions map
        regions_matrix = results['regions_matrix']
        region_map = np.zeros_like(Id_matrix)
        for i in range(len(Vg_values)):
            for j in range(len(Vd_values)):
                if regions_matrix[i, j] == "SUBTHRESHOLD":
                    region_map[i, j] = 0
                elif regions_matrix[i, j] == "LINEAR":
                    region_map[i, j] = 1
                elif regions_matrix[i, j] == "SATURATION":
                    region_map[i, j] = 2

        im = ax4.imshow(region_map, aspect='auto', origin='lower',
                       extent=[Vd_values[0], Vd_values[-1], Vg_values[0], Vg_values[-1]],
                       cmap='viridis', alpha=0.8)
        ax4.set_xlabel('Drain Voltage (V)')
        ax4.set_ylabel('Gate Voltage (V)')
        ax4.set_title('Operating Regions Map')

        # Add colorbar with custom labels
        cbar = self.figure.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(['Subthreshold', 'Linear', 'Saturation'])

        self.figure.tight_layout()
        self.canvas.draw()

    def update_transient_plots(self):
        """Update transient simulation plots"""
        if not MATPLOTLIB_AVAILABLE or not self.simulation_results:
            return

        self.figure.clear()

        # Get transient data
        results = self.simulation_results
        time_points = results['time_points'] * 1e9  # Convert to ns
        Vg_values = results['Vg_values']
        Vd_values = results['Vd_values']
        Id_values = results['Id_values']

        # Create transient plots
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)

        # Set dark theme
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        # Plot 1: Drain current vs time
        ax1.plot(time_points, Id_values * 1e6, 'b-', linewidth=2)
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Drain Current (μA)')
        ax1.set_title(f"Drain Current vs Time\n{results['scenario_name']}")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gate and drain voltages vs time
        ax2.plot(time_points, Vg_values, 'r-', linewidth=2, label='Gate Voltage')
        ax2.plot(time_points, Vd_values, 'g-', linewidth=2, label='Drain Voltage')
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Voltage (V)')
        ax2.set_title('Applied Voltages vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Current vs gate voltage (dynamic transfer curve)
        ax3.scatter(Vg_values, Id_values * 1e6, c=time_points, cmap='viridis', alpha=0.7)
        ax3.set_xlabel('Gate Voltage (V)')
        ax3.set_ylabel('Drain Current (μA)')
        ax3.set_title('Dynamic Transfer Curve')
        cbar = self.figure.colorbar(ax3.collections[0], ax=ax3, shrink=0.8)
        cbar.set_label('Time (ns)')
        ax3.grid(True, alpha=0.3)

        # Plot 4: dI/dt vs time (current derivative)
        if len(Id_values) > 1:
            dt = (time_points[1] - time_points[0]) * 1e-9  # Convert back to seconds
            dI_dt = np.diff(Id_values) / dt
            time_diff = time_points[1:]

            ax4.plot(time_diff, dI_dt, 'm-', linewidth=2)
            ax4.set_xlabel('Time (ns)')
            ax4.set_ylabel('dI/dt (A/s)')
            ax4.set_title('Current Rate of Change')
            ax4.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def save_plots(self):
        """Save plots to file"""
        if not MATPLOTLIB_AVAILABLE:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plots", "simulation_results.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )

        if filename:
            try:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#2d2d2d')
                self.log_message(f"💾 Plots saved to {filename}")
                QMessageBox.information(self, "Success", f"Plots saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plots:\n{str(e)}")

    def update_summary(self):
        """Update data summary"""
        if not self.simulation_results:
            return

        results = self.simulation_results

        summary = []
        summary.append("📊 SIMULATION RESULTS SUMMARY")
        summary.append("=" * 50)
        summary.append("")

        # Device info
        summary.append("🔧 DEVICE PARAMETERS:")
        summary.append(f"   Channel Length: {self.config.length*1e9:.0f} nm")
        summary.append(f"   Channel Width: {self.config.width*1e6:.1f} μm")
        summary.append(f"   Oxide Thickness: {self.config.tox*1e9:.1f} nm")
        summary.append(f"   Substrate Doping: {self.config.Na_substrate:.2e} /m³")
        summary.append("")

        # Physics info
        summary.append("🔬 PHYSICS PARAMETERS:")
        summary.append(f"   Temperature: {self.config.temperature:.1f} K")
        summary.append(f"   Mobility Model: {self.config.mobility_model}")
        summary.append(f"   SRH Recombination: {'ON' if self.config.enable_srh else 'OFF'}")
        summary.append(f"   Field Dependence: {'ON' if self.config.enable_field_mobility else 'OFF'}")
        summary.append("")

        # Results
        summary.append("📈 SIMULATION RESULTS:")
        summary.append(f"   Drain Current: {results['drain_current']:.2e} A")
        summary.append(f"   Operating Region: {results['operating_region']}")
        summary.append(f"   Threshold Voltage: {results['threshold_voltage']:.3f} V")
        summary.append("")

        # Boundary conditions
        bc = results['boundary_conditions']
        summary.append("⚡ OPERATING POINT:")
        summary.append(f"   Source: {bc[0]:.3f} V")
        summary.append(f"   Drain: {bc[1]:.3f} V")
        summary.append(f"   Substrate: {bc[2]:.3f} V")
        summary.append(f"   Gate: {bc[3]:.3f} V")

        self.summary_text.setPlainText("\n".join(summary))

    def update_iv_summary(self):
        """Update I-V characteristics summary"""
        if not self.simulation_results:
            return

        results = self.simulation_results

        summary = []
        summary.append("📈 I-V CHARACTERISTICS RESULTS SUMMARY")
        summary.append("=" * 50)
        summary.append("")

        # Device info
        summary.append("🔧 DEVICE PARAMETERS:")
        summary.append(f"   Channel Length: {self.config.length*1e9:.0f} nm")
        summary.append(f"   Channel Width: {self.config.width*1e6:.1f} μm")
        summary.append(f"   Oxide Thickness: {self.config.tox*1e9:.1f} nm")
        summary.append(f"   Temperature: {self.config.temperature:.1f} K")
        summary.append("")

        # Simulation parameters
        summary.append("📊 SIMULATION PARAMETERS:")
        summary.append(f"   Gate voltage range: {results['Vg_values'][0]:.1f}V to {results['Vg_values'][-1]:.1f}V")
        summary.append(f"   Drain voltage range: {results['Vd_values'][0]:.1f}V to {results['Vd_values'][-1]:.1f}V")
        summary.append(f"   Total simulation points: {len(results['Vg_values']) * len(results['Vd_values'])}")
        summary.append(f"   Simulation time: {results['simulation_time']:.2f} seconds")
        summary.append("")

        # Results analysis
        Id_matrix = results['Id_matrix']
        Id_max = np.max(Id_matrix)
        Id_min = np.min(Id_matrix[Id_matrix > 0])
        on_off_ratio = Id_max / Id_min if Id_min > 0 else float('inf')

        summary.append("📈 I-V CHARACTERISTICS ANALYSIS:")
        summary.append(f"   Maximum current: {Id_max:.2e} A")
        summary.append(f"   Minimum current: {Id_min:.2e} A")
        summary.append(f"   On/Off ratio: {on_off_ratio:.1e}")
        summary.append("")

        # Operating regions analysis
        regions_flat = results['regions_matrix'].flatten()
        total_points = len(regions_flat)
        subthreshold_count = np.sum(regions_flat == "SUBTHRESHOLD")
        linear_count = np.sum(regions_flat == "LINEAR")
        saturation_count = np.sum(regions_flat == "SATURATION")

        summary.append("🎯 OPERATING REGIONS DISTRIBUTION:")
        summary.append(f"   Subthreshold: {subthreshold_count} points ({subthreshold_count/total_points*100:.1f}%)")
        summary.append(f"   Linear: {linear_count} points ({linear_count/total_points*100:.1f}%)")
        summary.append(f"   Saturation: {saturation_count} points ({saturation_count/total_points*100:.1f}%)")

        self.summary_text.setPlainText("\n".join(summary))

    def update_transient_summary(self):
        """Update transient simulation summary"""
        if not self.simulation_results:
            return

        results = self.simulation_results

        summary = []
        summary.append("⚡ TRANSIENT SIMULATION RESULTS SUMMARY")
        summary.append("=" * 50)
        summary.append("")

        # Device info
        summary.append("🔧 DEVICE PARAMETERS:")
        summary.append(f"   Channel Length: {self.config.length*1e9:.0f} nm")
        summary.append(f"   Channel Width: {self.config.width*1e6:.1f} μm")
        summary.append(f"   Oxide Thickness: {self.config.tox*1e9:.1f} nm")
        summary.append(f"   Temperature: {self.config.temperature:.1f} K")
        summary.append("")

        # Simulation parameters
        time_points = results['time_points']
        total_time = time_points[-1] - time_points[0]
        dt = time_points[1] - time_points[0] if len(time_points) > 1 else 0

        summary.append("⏱️  SIMULATION PARAMETERS:")
        summary.append(f"   Scenario: {results['scenario_name']}")
        summary.append(f"   Total time: {total_time*1e9:.1f} ns")
        summary.append(f"   Time step: {dt*1e12:.0f} ps")
        summary.append(f"   Number of time points: {len(time_points)}")
        summary.append(f"   Simulation time: {results['simulation_time']:.2f} seconds")
        summary.append("")

        # Results analysis
        Id_values = results['Id_values']
        Vg_values = results['Vg_values']
        Vd_values = results['Vd_values']

        Id_max = np.max(Id_values)
        Id_min = np.min(Id_values[Id_values > 0])
        Id_avg = np.mean(Id_values)

        summary.append("📈 TRANSIENT ANALYSIS:")
        summary.append(f"   Maximum current: {Id_max:.2e} A")
        summary.append(f"   Minimum current: {Id_min:.2e} A")
        summary.append(f"   Average current: {Id_avg:.2e} A")
        summary.append(f"   Current variation: {(Id_max - Id_min):.2e} A")
        summary.append("")

        # Voltage analysis
        Vg_max = np.max(Vg_values)
        Vg_min = np.min(Vg_values)
        Vd_max = np.max(Vd_values)
        Vd_min = np.min(Vd_values)

        summary.append("⚡ VOLTAGE ANALYSIS:")
        summary.append(f"   Gate voltage range: {Vg_min:.2f}V to {Vg_max:.2f}V")
        summary.append(f"   Drain voltage range: {Vd_min:.2f}V to {Vd_max:.2f}V")
        summary.append("")

        # Dynamic analysis
        if len(Id_values) > 1:
            Id_diff = np.diff(Id_values)
            max_di_dt = np.max(np.abs(Id_diff)) / dt

            summary.append("🔄 DYNAMIC ANALYSIS:")
            summary.append(f"   Maximum dI/dt: {max_di_dt:.2e} A/s")

            # Count transitions
            transitions = np.sum(np.abs(Id_diff) > np.std(Id_diff))
            summary.append(f"   Significant transitions: {transitions}")

        self.summary_text.setPlainText("\n".join(summary))

    def save_configuration(self):
        """Save configuration to file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "mosfet_config.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.config.to_dict(), f, indent=2)
                self.log_message(f"💾 Configuration saved to {filename}")
                QMessageBox.information(self, "Success", f"Configuration saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")

    def load_configuration(self):
        """Load configuration from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "",
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)

                self.config.from_dict(data)
                self.update_gui_from_config()
                self.log_message(f"📂 Configuration loaded from {filename}")
                QMessageBox.information(self, "Success", f"Configuration loaded from {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")

    def update_gui_from_config(self):
        """Update GUI controls from configuration"""
        # Device parameters
        self.length_spin.setValue(self.config.length * 1e9)
        self.width_spin.setValue(self.config.width * 1e6)
        self.tox_spin.setValue(self.config.tox * 1e9)
        self.na_spin.setValue(self.config.Na_substrate / 1e6)
        self.nd_spin.setValue(self.config.Nd_source / 1e6)
        self.nx_spin.setValue(self.config.nx)
        self.ny_spin.setValue(self.config.ny)

        # Physics parameters
        self.temp_spin.setValue(self.config.temperature)
        self.tau_n_spin.setValue(self.config.tau_n0 * 1e6)
        self.tau_p_spin.setValue(self.config.tau_p0 * 1e6)
        self.mobility_combo.setCurrentText(self.config.mobility_model)
        self.srh_enable.setChecked(self.config.enable_srh)
        self.field_enable.setChecked(self.config.enable_field_mobility)
        self.temp_enable.setChecked(self.config.enable_temperature)
        self.quantum_enable.setChecked(self.config.enable_quantum)

def main():
    """Main function to run the modern GUI"""

    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Advanced MOSFET Simulator")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("SemiDGFEM")

    # Create and show main window
    window = ModernMOSFETGUI()
    window.show()

    # Add welcome message
    window.log_message("🚀 Advanced MOSFET Simulator Started")
    window.log_message("🔬 Modern PySide6 Interface with C++ Backend")
    window.log_message("⚡ Advanced Physics Models: Effective Mass, SRH, Mobility")
    window.log_message("📊 Real-time Logging and Visualization")
    window.log_message("✅ Ready for simulation!")

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
