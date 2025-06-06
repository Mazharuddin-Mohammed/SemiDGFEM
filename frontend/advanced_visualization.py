#!/usr/bin/env python3
"""
Advanced Visualization Components for SemiDGFEM Frontend
Provides specialized visualization widgets for transport simulation results

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QComboBox, QPushButton, QSlider, QCheckBox, QGroupBox,
    QGridLayout, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class InteractiveVisualizationWidget(QWidget):
    """Interactive visualization widget with advanced controls"""
    
    def __init__(self):
        super().__init__()
        self.current_results = None
        self.animation = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup interactive visualization UI"""
        layout = QVBoxLayout(self)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Visualization area
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(14, 10))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
            
            # Animation timer
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self.update_animation)
            
        else:
            placeholder = QLabel("Advanced visualization requires matplotlib")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder)
    
    def create_control_panel(self):
        """Create comprehensive control panel"""
        control_group = QGroupBox("Visualization Controls")
        layout = QGridLayout(control_group)
        
        # Visualization type
        layout.addWidget(QLabel("Visualization Type:"), 0, 0)
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "2D Contour Plot", "3D Surface Plot", "Vector Field Plot",
            "Streamline Plot", "Animation", "Comparative Analysis",
            "Cross-Section Analysis", "Time Evolution", "Phase Space"
        ])
        self.viz_type_combo.currentTextChanged.connect(self.update_visualization)
        layout.addWidget(self.viz_type_combo, 0, 1)
        
        # Data field selection
        layout.addWidget(QLabel("Data Field:"), 0, 2)
        self.field_combo = QComboBox()
        self.field_combo.addItems([
            "Electron Density", "Hole Density", "Electric Potential",
            "Energy Density", "Momentum", "Current Density",
            "Temperature", "Quasi-Fermi Levels"
        ])
        self.field_combo.currentTextChanged.connect(self.update_visualization)
        layout.addWidget(self.field_combo, 0, 3)
        
        # Colormap selection
        layout.addWidget(QLabel("Colormap:"), 1, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis", "plasma", "inferno", "magma", "jet", "hot", "cool",
            "seismic", "RdBu", "coolwarm", "rainbow"
        ])
        self.colormap_combo.currentTextChanged.connect(self.update_visualization)
        layout.addWidget(self.colormap_combo, 1, 1)
        
        # Scale selection
        layout.addWidget(QLabel("Scale:"), 1, 2)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Linear", "Logarithmic", "Symmetric Log"])
        self.scale_combo.currentTextChanged.connect(self.update_visualization)
        layout.addWidget(self.scale_combo, 1, 3)
        
        # Animation controls
        self.animation_check = QCheckBox("Enable Animation")
        self.animation_check.toggled.connect(self.toggle_animation)
        layout.addWidget(self.animation_check, 2, 0)
        
        self.animation_speed_slider = QSlider(Qt.Horizontal)
        self.animation_speed_slider.setRange(1, 10)
        self.animation_speed_slider.setValue(5)
        self.animation_speed_slider.valueChanged.connect(self.update_animation_speed)
        layout.addWidget(QLabel("Speed:"), 2, 1)
        layout.addWidget(self.animation_speed_slider, 2, 2, 1, 2)
        
        # View controls
        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self.update_visualization)
        layout.addWidget(self.grid_check, 3, 0)
        
        self.contour_check = QCheckBox("Show Contours")
        self.contour_check.setChecked(True)
        self.contour_check.toggled.connect(self.update_visualization)
        layout.addWidget(self.contour_check, 3, 1)
        
        self.colorbar_check = QCheckBox("Show Colorbar")
        self.colorbar_check.setChecked(True)
        self.colorbar_check.toggled.connect(self.update_visualization)
        layout.addWidget(self.colorbar_check, 3, 2)
        
        # Action buttons
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.update_visualization)
        layout.addWidget(self.refresh_btn, 3, 3)
        
        return control_group
    
    def update_results(self, results):
        """Update with new simulation results"""
        self.current_results = results
        self.update_field_options()
        self.update_visualization()
    
    def update_field_options(self):
        """Update available field options based on results"""
        if not self.current_results:
            return
        
        self.field_combo.clear()
        
        # Add available fields from results
        for model_name, model_data in self.current_results.items():
            if isinstance(model_data, dict):
                for field_name in model_data.keys():
                    display_name = f"{model_name}: {field_name}"
                    self.field_combo.addItem(display_name)
    
    def update_visualization(self):
        """Update the visualization based on current settings"""
        if not MATPLOTLIB_AVAILABLE or not self.current_results:
            return
        
        self.figure.clear()
        
        viz_type = self.viz_type_combo.currentText()
        field = self.field_combo.currentText()
        colormap = self.colormap_combo.currentText()
        scale = self.scale_combo.currentText()
        
        try:
            if viz_type == "2D Contour Plot":
                self.create_2d_contour_plot(field, colormap, scale)
            elif viz_type == "3D Surface Plot":
                self.create_3d_surface_plot(field, colormap, scale)
            elif viz_type == "Vector Field Plot":
                self.create_vector_field_plot(field, colormap)
            elif viz_type == "Streamline Plot":
                self.create_streamline_plot(field, colormap)
            elif viz_type == "Comparative Analysis":
                self.create_comparative_analysis(colormap)
            elif viz_type == "Cross-Section Analysis":
                self.create_cross_section_analysis(field, colormap)
            elif viz_type == "Time Evolution":
                self.create_time_evolution_plot(field, colormap)
            elif viz_type == "Phase Space":
                self.create_phase_space_plot(colormap)
            else:
                self.create_overview_plot()
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Visualization update failed: {e}")
    
    def create_2d_contour_plot(self, field, colormap, scale):
        """Create 2D contour plot"""
        ax = self.figure.add_subplot(1, 1, 1)
        
        # Generate sample 2D data
        x = np.linspace(0, 2e-6, 100)
        y = np.linspace(0, 1e-6, 50)
        X, Y = np.meshgrid(x, y)
        Z = self.generate_sample_2d_data(X, Y, field)
        
        # Apply scaling
        if scale == "Logarithmic":
            Z = np.log10(np.abs(Z) + 1e-20)
        elif scale == "Symmetric Log":
            Z = np.sign(Z) * np.log10(np.abs(Z) + 1)
        
        # Create contour plot
        if self.contour_check.isChecked():
            contour = ax.contourf(X * 1e6, Y * 1e6, Z, levels=20, cmap=colormap)
            if self.colorbar_check.isChecked():
                self.figure.colorbar(contour, ax=ax)
        
        # Add contour lines
        contour_lines = ax.contour(X * 1e6, Y * 1e6, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Y Position (μm)')
        ax.set_title(f'2D Contour Plot: {field}')
        
        if self.grid_check.isChecked():
            ax.grid(True, alpha=0.3)
    
    def create_3d_surface_plot(self, field, colormap, scale):
        """Create 3D surface plot"""
        ax = self.figure.add_subplot(1, 1, 1, projection='3d')
        
        # Generate sample 3D data
        x = np.linspace(0, 2e-6, 50)
        y = np.linspace(0, 1e-6, 25)
        X, Y = np.meshgrid(x, y)
        Z = self.generate_sample_2d_data(X, Y, field)
        
        # Apply scaling
        if scale == "Logarithmic":
            Z = np.log10(np.abs(Z) + 1e-20)
        
        # Create surface plot
        surf = ax.plot_surface(X * 1e6, Y * 1e6, Z, cmap=colormap, alpha=0.8)
        
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Y Position (μm)')
        ax.set_zlabel('Value')
        ax.set_title(f'3D Surface Plot: {field}')
        
        if self.colorbar_check.isChecked():
            self.figure.colorbar(surf, shrink=0.5, aspect=5)
    
    def create_vector_field_plot(self, field, colormap):
        """Create vector field plot"""
        ax = self.figure.add_subplot(1, 1, 1)
        
        # Generate sample vector field data
        x = np.linspace(0, 2e-6, 20)
        y = np.linspace(0, 1e-6, 10)
        X, Y = np.meshgrid(x, y)
        
        # Sample vector components
        U = np.sin(X * 1e6) * np.cos(Y * 1e6)
        V = np.cos(X * 1e6) * np.sin(Y * 1e6)
        
        # Create vector field plot
        quiver = ax.quiver(X * 1e6, Y * 1e6, U, V, np.sqrt(U**2 + V**2), 
                          cmap=colormap, scale=20, alpha=0.8)
        
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Y Position (μm)')
        ax.set_title(f'Vector Field Plot: {field}')
        
        if self.colorbar_check.isChecked():
            self.figure.colorbar(quiver, ax=ax)
        
        if self.grid_check.isChecked():
            ax.grid(True, alpha=0.3)
    
    def create_streamline_plot(self, field, colormap):
        """Create streamline plot"""
        ax = self.figure.add_subplot(1, 1, 1)
        
        # Generate sample flow field
        x = np.linspace(0, 2e-6, 50)
        y = np.linspace(0, 1e-6, 25)
        X, Y = np.meshgrid(x, y)
        
        U = -np.gradient(np.sin(X * 1e6) * np.cos(Y * 1e6))[1]
        V = -np.gradient(np.sin(X * 1e6) * np.cos(Y * 1e6))[0]
        
        # Create streamline plot
        stream = ax.streamplot(X * 1e6, Y * 1e6, U, V, color=np.sqrt(U**2 + V**2), 
                              cmap=colormap, density=2, linewidth=1.5)
        
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Y Position (μm)')
        ax.set_title(f'Streamline Plot: {field}')
        
        if self.colorbar_check.isChecked():
            self.figure.colorbar(stream.lines, ax=ax)
        
        if self.grid_check.isChecked():
            ax.grid(True, alpha=0.3)
    
    def create_comparative_analysis(self, colormap):
        """Create comparative analysis of multiple transport models"""
        if not self.current_results:
            return
        
        # Create subplots for comparison
        n_models = len([k for k in self.current_results.keys() if k != 'mesh_info'])
        if n_models == 0:
            return
        
        fig_rows = int(np.ceil(n_models / 2))
        fig_cols = min(2, n_models)
        
        plot_idx = 1
        for model_name, model_data in self.current_results.items():
            if model_name == 'mesh_info':
                continue
            
            ax = self.figure.add_subplot(fig_rows, fig_cols, plot_idx)
            
            # Plot first available field from each model
            if isinstance(model_data, dict) and model_data:
                first_field = list(model_data.keys())[0]
                data = model_data[first_field]
                
                if isinstance(data, np.ndarray):
                    if data.ndim == 1:
                        ax.plot(data, linewidth=2, label=first_field)
                        ax.set_ylabel('Value')
                        ax.set_xlabel('Position')
                    else:
                        im = ax.imshow(data, cmap=colormap, aspect='auto')
                        self.figure.colorbar(im, ax=ax)
                
                ax.set_title(f'{model_name.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        self.figure.suptitle('Comparative Analysis of Transport Models', fontsize=14, fontweight='bold')
        self.figure.tight_layout()
    
    def create_cross_section_analysis(self, field, colormap):
        """Create cross-section analysis"""
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)
        
        # Generate sample data
        x = np.linspace(0, 2e-6, 100)
        y = np.linspace(0, 1e-6, 50)
        X, Y = np.meshgrid(x, y)
        Z = self.generate_sample_2d_data(X, Y, field)
        
        # 2D plot
        im = ax1.imshow(Z, extent=[0, 2, 0, 1], cmap=colormap, aspect='auto')
        ax1.set_title('2D Distribution')
        ax1.set_xlabel('X (μm)')
        ax1.set_ylabel('Y (μm)')
        
        # X cross-section at middle Y
        mid_y = Z.shape[0] // 2
        ax2.plot(x * 1e6, Z[mid_y, :], 'b-', linewidth=2)
        ax2.set_title('X Cross-Section (Y = 0.5 μm)')
        ax2.set_xlabel('X (μm)')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        
        # Y cross-section at middle X
        mid_x = Z.shape[1] // 2
        ax3.plot(y * 1e6, Z[:, mid_x], 'r-', linewidth=2)
        ax3.set_title('Y Cross-Section (X = 1.0 μm)')
        ax3.set_xlabel('Y (μm)')
        ax3.set_ylabel('Value')
        ax3.grid(True, alpha=0.3)
        
        # Diagonal cross-section
        diag_indices = np.linspace(0, min(Z.shape) - 1, min(Z.shape)).astype(int)
        diag_values = Z[diag_indices, diag_indices]
        diag_pos = np.linspace(0, min(2, 1), len(diag_values))
        ax4.plot(diag_pos, diag_values, 'g-', linewidth=2)
        ax4.set_title('Diagonal Cross-Section')
        ax4.set_xlabel('Position (μm)')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
    
    def create_time_evolution_plot(self, field, colormap):
        """Create time evolution visualization"""
        ax = self.figure.add_subplot(1, 1, 1)
        
        # Generate sample time evolution data
        time_steps = 50
        x = np.linspace(0, 2e-6, 100)
        
        for t in range(0, time_steps, 5):
            phase = 2 * np.pi * t / time_steps
            y = np.sin(x * 1e6 + phase) * np.exp(-t / 20)
            alpha = 0.3 + 0.7 * (time_steps - t) / time_steps
            ax.plot(x * 1e6, y, alpha=alpha, linewidth=1.5, 
                   color=plt.cm.get_cmap(colormap)(t / time_steps))
        
        ax.set_xlabel('X Position (μm)')
        ax.set_ylabel('Value')
        ax.set_title(f'Time Evolution: {field}')
        ax.grid(True, alpha=0.3)
    
    def create_phase_space_plot(self, colormap):
        """Create phase space visualization"""
        ax = self.figure.add_subplot(1, 1, 1)
        
        # Generate sample phase space data
        n_points = 1000
        theta = np.linspace(0, 4 * np.pi, n_points)
        r = np.linspace(0.1, 2, n_points)
        
        x = r * np.cos(theta) * np.exp(-theta / 10)
        y = r * np.sin(theta) * np.exp(-theta / 10)
        
        scatter = ax.scatter(x, y, c=theta, cmap=colormap, alpha=0.7, s=20)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Momentum')
        ax.set_title('Phase Space Visualization')
        
        if self.colorbar_check.isChecked():
            self.figure.colorbar(scatter, ax=ax, label='Time')
        
        if self.grid_check.isChecked():
            ax.grid(True, alpha=0.3)
    
    def create_overview_plot(self):
        """Create overview plot when no specific visualization is selected"""
        ax = self.figure.add_subplot(1, 1, 1)
        
        ax.text(0.5, 0.5, 'Advanced Visualization\n\nSelect visualization type and data field\nto display simulation results', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=16, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def generate_sample_2d_data(self, X, Y, field):
        """Generate sample 2D data based on field type"""
        if "density" in field.lower():
            return 1e22 * np.exp(-(X - 1e-6)**2 / (0.5e-6)**2 - (Y - 0.5e-6)**2 / (0.2e-6)**2)
        elif "potential" in field.lower():
            return np.sin(X * 1e6) * np.cos(Y * 1e6)
        elif "energy" in field.lower():
            return 1e-19 * (1 + 0.5 * np.sin(X * 2e6) * np.cos(Y * 2e6))
        elif "momentum" in field.lower():
            return 1e-25 * np.gradient(np.sin(X * 1e6))[0]
        else:
            return np.sin(X * 1e6) * np.cos(Y * 1e6)
    
    def toggle_animation(self, enabled):
        """Toggle animation on/off"""
        if enabled and MATPLOTLIB_AVAILABLE:
            self.animation_timer.start(100)  # 100ms interval
        else:
            self.animation_timer.stop()
    
    def update_animation_speed(self, speed):
        """Update animation speed"""
        if self.animation_timer.isActive():
            interval = max(50, 500 - speed * 45)  # 50ms to 500ms
            self.animation_timer.setInterval(interval)
    
    def update_animation(self):
        """Update animation frame"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Simple animation - rotate phase
        self.update_visualization()
        # Add animation-specific updates here
