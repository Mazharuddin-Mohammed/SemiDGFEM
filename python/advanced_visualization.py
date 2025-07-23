#!/usr/bin/env python3
"""
Advanced Visualization and Analysis Tools
Comprehensive plotting and analysis capabilities for semiconductor device simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Optional seaborn import
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    # Set up plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            pass
    sns.set_palette("husl")
except ImportError:
    SEABORN_AVAILABLE = False
    # Use matplotlib default style
    plt.style.use('default')

class DeviceVisualizer:
    """
    Advanced visualization tools for semiconductor device simulation results
    """
    
    def __init__(self, device_width, device_height, nx=50, ny=25):
        """
        Initialize device visualizer
        
        Parameters:
        -----------
        device_width : float
            Device width in meters
        device_height : float
            Device height in meters
        nx : int
            Number of grid points in x direction
        ny : int
            Number of grid points in y direction
        """
        self.device_width = device_width
        self.device_height = device_height
        self.nx = nx
        self.ny = ny
        
        # Create coordinate grids
        self.x = np.linspace(0, device_width, nx)
        self.y = np.linspace(0, device_height, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        logger.info(f"Device visualizer initialized: {device_width*1e6:.1f}x{device_height*1e6:.1f} μm")
    
    def plot_potential_2d(self, potential, title="Electrostatic Potential", save_path=None):
        """
        Plot 2D potential distribution
        
        Parameters:
        -----------
        potential : array_like
            Potential values
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Reshape potential to 2D grid
        if len(potential) == self.nx * self.ny:
            V_2d = potential.reshape(self.ny, self.nx)
        else:
            # Interpolate to grid
            V_2d = self._interpolate_to_grid(potential)
        
        # Create contour plot
        levels = 20
        contour = ax.contourf(self.X * 1e6, self.Y * 1e6, V_2d, levels=levels, cmap='RdYlBu_r')
        contour_lines = ax.contour(self.X * 1e6, self.Y * 1e6, V_2d, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Potential (V)', fontsize=12)
        
        # Add contour labels
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        
        # Formatting
        ax.set_xlabel('x (μm)', fontsize=12)
        ax.set_ylabel('y (μm)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Potential plot saved to {save_path}")
        
        return fig, ax
    
    def plot_carrier_densities(self, n, p, title="Carrier Densities", save_path=None):
        """
        Plot electron and hole density distributions
        
        Parameters:
        -----------
        n : array_like
            Electron density (m^-3)
        p : array_like
            Hole density (m^-3)
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Reshape densities to 2D grid
        if len(n) == self.nx * self.ny:
            n_2d = n.reshape(self.ny, self.nx)
            p_2d = p.reshape(self.ny, self.nx)
        else:
            n_2d = self._interpolate_to_grid(n)
            p_2d = self._interpolate_to_grid(p)
        
        # Electron density plot
        n_min, n_max = np.min(n_2d[n_2d > 0]), np.max(n_2d)
        im1 = ax1.imshow(n_2d, extent=[0, self.device_width*1e6, 0, self.device_height*1e6],
                        origin='lower', cmap='Blues', norm=LogNorm(vmin=n_min, vmax=n_max))
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Electron Density (m⁻³)', fontsize=12)
        
        ax1.set_xlabel('x (μm)', fontsize=12)
        ax1.set_ylabel('y (μm)', fontsize=12)
        ax1.set_title('Electron Density', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Hole density plot
        p_min, p_max = np.min(p_2d[p_2d > 0]), np.max(p_2d)
        im2 = ax2.imshow(p_2d, extent=[0, self.device_width*1e6, 0, self.device_height*1e6],
                        origin='lower', cmap='Reds', norm=LogNorm(vmin=p_min, vmax=p_max))
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Hole Density (m⁻³)', fontsize=12)
        
        ax2.set_xlabel('x (μm)', fontsize=12)
        ax2.set_ylabel('y (μm)', fontsize=12)
        ax2.set_title('Hole Density', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Carrier density plot saved to {save_path}")
        
        return fig, (ax1, ax2)

    def plot_1d_profiles(self, data_dict, x_positions=None, title="1D Profiles", save_path=None):
        """
        Plot 1D profiles of various quantities

        Parameters:
        -----------
        data_dict : dict
            Dictionary with data arrays and labels
        x_positions : array_like, optional
            X positions for the data
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        if x_positions is None:
            x_positions = np.linspace(0, self.device_width * 1e6, len(list(data_dict.values())[0]))
        else:
            x_positions = np.array(x_positions) * 1e6  # Convert to μm

        plot_idx = 0
        for key, data in data_dict.items():
            if plot_idx >= 4:
                break

            ax = axes[plot_idx]

            if 'density' in key.lower() or 'concentration' in key.lower() or key in ['n', 'p']:
                ax.semilogy(x_positions, np.abs(data), 'o-', linewidth=2, markersize=4)
                ax.set_ylabel(f'{key} (m⁻³)', fontsize=11)
            elif 'temperature' in key.lower():
                ax.plot(x_positions, data, 's-', linewidth=2, markersize=4)
                ax.set_ylabel(f'{key} (K)', fontsize=11)
            elif 'potential' in key.lower():
                ax.plot(x_positions, data, '^-', linewidth=2, markersize=4)
                ax.set_ylabel(f'{key} (V)', fontsize=11)
            else:
                ax.plot(x_positions, data, 'o-', linewidth=2, markersize=4)
                ax.set_ylabel(f'{key}', fontsize=11)

            ax.set_xlabel('Position (μm)', fontsize=11)
            ax.set_title(key, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"1D profiles plot saved to {save_path}")

        return fig, axes

    def plot_current_density(self, Jn_x, Jn_y, Jp_x, Jp_y, title="Current Density", save_path=None):
        """
        Plot current density vectors and magnitude
        
        Parameters:
        -----------
        Jn_x, Jn_y : array_like
            Electron current density components (A/m²)
        Jp_x, Jp_y : array_like
            Hole current density components (A/m²)
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Reshape current densities to 2D grid
        if len(Jn_x) == self.nx * self.ny:
            Jn_x_2d = Jn_x.reshape(self.ny, self.nx)
            Jn_y_2d = Jn_y.reshape(self.ny, self.nx)
            Jp_x_2d = Jp_x.reshape(self.ny, self.nx)
            Jp_y_2d = Jp_y.reshape(self.ny, self.nx)
        else:
            Jn_x_2d = self._interpolate_to_grid(Jn_x)
            Jn_y_2d = self._interpolate_to_grid(Jn_y)
            Jp_x_2d = self._interpolate_to_grid(Jp_x)
            Jp_y_2d = self._interpolate_to_grid(Jp_y)
        
        # Calculate magnitudes
        Jn_mag = np.sqrt(Jn_x_2d**2 + Jn_y_2d**2)
        Jp_mag = np.sqrt(Jp_x_2d**2 + Jp_y_2d**2)
        J_total_mag = np.sqrt((Jn_x_2d + Jp_x_2d)**2 + (Jn_y_2d + Jp_y_2d)**2)
        
        # Electron current magnitude
        im1 = ax1.imshow(Jn_mag, extent=[0, self.device_width*1e6, 0, self.device_height*1e6],
                        origin='lower', cmap='Blues')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('|Jn| (A/m²)', fontsize=10)
        ax1.set_title('Electron Current Magnitude', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x (μm)')
        ax1.set_ylabel('y (μm)')
        
        # Hole current magnitude
        im2 = ax2.imshow(Jp_mag, extent=[0, self.device_width*1e6, 0, self.device_height*1e6],
                        origin='lower', cmap='Reds')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('|Jp| (A/m²)', fontsize=10)
        ax2.set_title('Hole Current Magnitude', fontsize=12, fontweight='bold')
        ax2.set_xlabel('x (μm)')
        ax2.set_ylabel('y (μm)')
        
        # Total current magnitude
        im3 = ax3.imshow(J_total_mag, extent=[0, self.device_width*1e6, 0, self.device_height*1e6],
                        origin='lower', cmap='viridis')
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('|J_total| (A/m²)', fontsize=10)
        ax3.set_title('Total Current Magnitude', fontsize=12, fontweight='bold')
        ax3.set_xlabel('x (μm)')
        ax3.set_ylabel('y (μm)')
        
        # Current vector field (subsampled)
        skip = max(1, min(self.nx, self.ny) // 15)  # Subsample for clarity
        X_sub = self.X[::skip, ::skip] * 1e6
        Y_sub = self.Y[::skip, ::skip] * 1e6
        Jx_sub = (Jn_x_2d + Jp_x_2d)[::skip, ::skip]
        Jy_sub = (Jn_y_2d + Jp_y_2d)[::skip, ::skip]
        
        ax4.quiver(X_sub, Y_sub, Jx_sub, Jy_sub, J_total_mag[::skip, ::skip], 
                  cmap='plasma', scale_units='xy', angles='xy', scale=1)
        ax4.set_title('Current Vector Field', fontsize=12, fontweight='bold')
        ax4.set_xlabel('x (μm)')
        ax4.set_ylabel('y (μm)')
        ax4.set_aspect('equal')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Current density plot saved to {save_path}")
        
        return fig, ((ax1, ax2), (ax3, ax4))
    
    def plot_energy_transport(self, T_n, T_p, title="Carrier Temperatures", save_path=None):
        """
        Plot carrier temperature distributions for energy transport
        
        Parameters:
        -----------
        T_n : array_like
            Electron temperature (K)
        T_p : array_like
            Hole temperature (K)
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Reshape temperatures to 2D grid
        if len(T_n) == self.nx * self.ny:
            T_n_2d = T_n.reshape(self.ny, self.nx)
            T_p_2d = T_p.reshape(self.ny, self.nx)
        else:
            T_n_2d = self._interpolate_to_grid(T_n)
            T_p_2d = self._interpolate_to_grid(T_p)
        
        # Electron temperature plot
        im1 = ax1.imshow(T_n_2d, extent=[0, self.device_width*1e6, 0, self.device_height*1e6],
                        origin='lower', cmap='hot')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Electron Temperature (K)', fontsize=12)
        
        ax1.set_xlabel('x (μm)', fontsize=12)
        ax1.set_ylabel('y (μm)', fontsize=12)
        ax1.set_title('Electron Temperature', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Hole temperature plot
        im2 = ax2.imshow(T_p_2d, extent=[0, self.device_width*1e6, 0, self.device_height*1e6],
                        origin='lower', cmap='hot')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Hole Temperature (K)', fontsize=12)
        
        ax2.set_xlabel('x (μm)', fontsize=12)
        ax2.set_ylabel('y (μm)', fontsize=12)
        ax2.set_title('Hole Temperature', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Energy transport plot saved to {save_path}")
        
        return fig, (ax1, ax2)
    
    def plot_recombination_rates(self, R_srh, R_rad, R_auger, title="Recombination Rates", save_path=None):
        """
        Plot recombination rate distributions
        
        Parameters:
        -----------
        R_srh : array_like
            SRH recombination rate (m^-3/s)
        R_rad : array_like
            Radiative recombination rate (m^-3/s)
        R_auger : array_like
            Auger recombination rate (m^-3/s)
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Reshape recombination rates to 2D grid
        if len(R_srh) == self.nx * self.ny:
            R_srh_2d = R_srh.reshape(self.ny, self.nx)
            R_rad_2d = R_rad.reshape(self.ny, self.nx)
            R_auger_2d = R_auger.reshape(self.ny, self.nx)
        else:
            R_srh_2d = self._interpolate_to_grid(R_srh)
            R_rad_2d = self._interpolate_to_grid(R_rad)
            R_auger_2d = self._interpolate_to_grid(R_auger)
        
        # SRH recombination
        im1 = ax1.imshow(np.abs(R_srh_2d), extent=[0, self.device_width*1e6, 0, self.device_height*1e6],
                        origin='lower', cmap='Oranges', norm=LogNorm())
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('SRH Rate (m⁻³/s)', fontsize=10)
        ax1.set_title('SRH Recombination', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x (μm)')
        ax1.set_ylabel('y (μm)')
        
        # Radiative recombination
        im2 = ax2.imshow(np.abs(R_rad_2d), extent=[0, self.device_width*1e6, 0, self.device_height*1e6],
                        origin='lower', cmap='Greens', norm=LogNorm())
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Radiative Rate (m⁻³/s)', fontsize=10)
        ax2.set_title('Radiative Recombination', fontsize=12, fontweight='bold')
        ax2.set_xlabel('x (μm)')
        ax2.set_ylabel('y (μm)')
        
        # Auger recombination
        im3 = ax3.imshow(np.abs(R_auger_2d), extent=[0, self.device_width*1e6, 0, self.device_height*1e6],
                        origin='lower', cmap='Purples', norm=LogNorm())
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Auger Rate (m⁻³/s)', fontsize=10)
        ax3.set_title('Auger Recombination', fontsize=12, fontweight='bold')
        ax3.set_xlabel('x (μm)')
        ax3.set_ylabel('y (μm)')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Recombination rates plot saved to {save_path}")
        
        return fig, (ax1, ax2, ax3)
    
    def plot_3d_surface(self, data, title="3D Surface Plot", save_path=None):
        """
        Create 3D surface plot of data
        
        Parameters:
        -----------
        data : array_like
            Data to plot
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Reshape data to 2D grid
        if len(data) == self.nx * self.ny:
            data_2d = data.reshape(self.ny, self.nx)
        else:
            data_2d = self._interpolate_to_grid(data)
        
        # Create 3D surface
        surf = ax.plot_surface(self.X * 1e6, self.Y * 1e6, data_2d, 
                              cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
        
        # Add colorbar
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Value', fontsize=12)
        
        # Formatting
        ax.set_xlabel('x (μm)', fontsize=12)
        ax.set_ylabel('y (μm)', fontsize=12)
        ax.set_zlabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"3D surface plot saved to {save_path}")
        
        return fig, ax
    
    def _interpolate_to_grid(self, data):
        """Interpolate 1D data to 2D grid"""
        # Simple interpolation - in practice, would use proper interpolation
        if len(data) < self.nx * self.ny:
            # Pad with zeros or repeat values
            data_padded = np.pad(data, (0, self.nx * self.ny - len(data)), mode='edge')
        else:
            # Truncate
            data_padded = data[:self.nx * self.ny]
        
        return data_padded.reshape(self.ny, self.nx)

class AnalysisTools:
    """
    Advanced analysis tools for semiconductor device simulation
    """
    
    @staticmethod
    def calculate_iv_characteristics(results_list, voltages):
        """
        Calculate I-V characteristics from simulation results
        
        Parameters:
        -----------
        results_list : list
            List of simulation results at different voltages
        voltages : array_like
            Applied voltages
            
        Returns:
        --------
        dict
            I-V characteristics data
        """
        currents = []
        
        for results in results_list:
            if 'Jn' in results and 'Jp' in results:
                # Calculate total current (simplified)
                J_total = np.abs(results['Jn']) + np.abs(results['Jp'])
                I_total = np.mean(J_total)  # Simplified current calculation
                currents.append(I_total)
            else:
                currents.append(0.0)
        
        return {
            'voltages': np.array(voltages),
            'currents': np.array(currents),
            'resistance': np.gradient(voltages) / np.gradient(currents) if len(currents) > 1 else [0],
            'conductance': np.gradient(currents) / np.gradient(voltages) if len(voltages) > 1 else [0]
        }
    
    @staticmethod
    def analyze_convergence(convergence_history):
        """
        Analyze convergence behavior
        
        Parameters:
        -----------
        convergence_history : array_like
            Convergence residuals vs iteration
            
        Returns:
        --------
        dict
            Convergence analysis
        """
        if len(convergence_history) < 2:
            return {'converged': False, 'rate': 0.0}
        
        # Calculate convergence rate
        log_residuals = np.log10(np.array(convergence_history))
        if len(log_residuals) > 1:
            rate = np.mean(np.diff(log_residuals))
        else:
            rate = 0.0
        
        # Check if converged
        final_residual = convergence_history[-1]
        converged = final_residual < 1e-6
        
        return {
            'converged': converged,
            'final_residual': final_residual,
            'iterations': len(convergence_history),
            'convergence_rate': rate,
            'linear_convergence': rate > -0.1,
            'superlinear_convergence': rate < -0.5
        }
    
    @staticmethod
    def calculate_device_metrics(results, device_width, device_height):
        """
        Calculate important device metrics
        
        Parameters:
        -----------
        results : dict
            Simulation results
        device_width : float
            Device width
        device_height : float
            Device height
            
        Returns:
        --------
        dict
            Device metrics
        """
        metrics = {}
        
        if 'potential' in results:
            V = results['potential']
            metrics['max_potential'] = np.max(V)
            metrics['min_potential'] = np.min(V)
            metrics['potential_range'] = np.max(V) - np.min(V)
        
        if 'n' in results and 'p' in results:
            n = results['n']
            p = results['p']
            
            metrics['max_electron_density'] = np.max(n)
            metrics['max_hole_density'] = np.max(p)
            metrics['min_electron_density'] = np.min(n)
            metrics['min_hole_density'] = np.min(p)
            
            # Calculate depletion width (simplified)
            ni = 1.45e16  # Intrinsic concentration for Si
            depletion_mask = (n < 10 * ni) & (p < 10 * ni)
            if np.any(depletion_mask):
                metrics['depletion_fraction'] = np.sum(depletion_mask) / len(n)
            else:
                metrics['depletion_fraction'] = 0.0
        
        if 'Jn' in results and 'Jp' in results:
            Jn = results['Jn']
            Jp = results['Jp']
            
            metrics['max_electron_current'] = np.max(np.abs(Jn))
            metrics['max_hole_current'] = np.max(np.abs(Jp))
            metrics['total_current'] = np.mean(np.abs(Jn) + np.abs(Jp))
        
        # Device area
        metrics['device_area'] = device_width * device_height
        
        return metrics

# Convenience functions
def create_device_visualizer(device_width, device_height, nx=50, ny=25):
    """Create a device visualizer instance"""
    return DeviceVisualizer(device_width, device_height, nx, ny)

def plot_comprehensive_results(results, device_width, device_height, save_dir=None):
    """
    Create comprehensive plots of simulation results
    
    Parameters:
    -----------
    results : dict
        Simulation results
    device_width : float
        Device width
    device_height : float
        Device height
    save_dir : str, optional
        Directory to save plots
        
    Returns:
    --------
    dict
        Dictionary of created figures
    """
    visualizer = create_device_visualizer(device_width, device_height)
    figures = {}
    
    # Plot potential
    if 'potential' in results:
        fig, ax = visualizer.plot_potential_2d(
            results['potential'], 
            save_path=f"{save_dir}/potential.png" if save_dir else None
        )
        figures['potential'] = fig
    
    # Plot carrier densities
    if 'n' in results and 'p' in results:
        fig, axes = visualizer.plot_carrier_densities(
            results['n'], results['p'],
            save_path=f"{save_dir}/carrier_densities.png" if save_dir else None
        )
        figures['carrier_densities'] = fig
    
    # Plot current densities
    if all(key in results for key in ['Jn', 'Jp']):
        # Assume 1D current for simplicity, split into x and y components
        Jn = results['Jn']
        Jp = results['Jp']
        Jn_x = Jn * 0.7  # Simplified split
        Jn_y = Jn * 0.3
        Jp_x = Jp * 0.7
        Jp_y = Jp * 0.3
        
        fig, axes = visualizer.plot_current_density(
            Jn_x, Jn_y, Jp_x, Jp_y,
            save_path=f"{save_dir}/current_density.png" if save_dir else None
        )
        figures['current_density'] = fig
    
    # Plot recombination rates if available
    if all(key in results for key in ['recombination_srh', 'recombination_radiative', 'recombination_auger']):
        fig, axes = visualizer.plot_recombination_rates(
            results['recombination_srh'], 
            results['recombination_radiative'], 
            results['recombination_auger'],
            save_path=f"{save_dir}/recombination.png" if save_dir else None
        )
        figures['recombination'] = fig
    
    logger.info(f"Created {len(figures)} comprehensive plots")
    return figures

def validate_visualization():
    """Validate visualization capabilities"""
    try:
        # Test visualizer creation
        visualizer = create_device_visualizer(2e-6, 1e-6)

        # Test with dummy data
        dummy_potential = np.random.randn(1250) * 0.5
        dummy_n = np.random.exponential(1e16, 1250)
        dummy_p = np.random.exponential(1e16, 1250)

        # Test plotting functions
        fig1, ax1 = visualizer.plot_potential_2d(dummy_potential)
        fig2, axes2 = visualizer.plot_carrier_densities(dummy_n, dummy_p)

        plt.close('all')  # Clean up

        return {
            "visualizer_creation": True,
            "potential_plotting": True,
            "carrier_density_plotting": True,
            "validation_passed": True
        }
    except Exception as e:
        return {
            "visualizer_creation": False,
            "error": str(e),
            "validation_passed": False
        }
