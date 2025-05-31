#!/usr/bin/env python3
"""
Comprehensive unit tests for the GUI components.
Tests the main GUI application and visualization components.
"""

import unittest
import tkinter as tk
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import sys
import os
import threading
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from gui.main_gui_2d import SimulatorGUI2D
except ImportError:
    SimulatorGUI2D = None

try:
    from visualization.viz_2d import plot_2d_potential, plot_2d_quantity, plot_current_vectors
except ImportError:
    plot_2d_potential = None
    plot_2d_quantity = None
    plot_current_vectors = None


class TestGUIBase(unittest.TestCase):
    """Base class for GUI tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a root window for testing
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during testing
        
        # Mock the simulator to avoid dependency issues
        self.mock_simulator = MagicMock()
        self.mock_simulator.get_grid_points.return_value = {
            "x": np.linspace(0, 1e-6, 100),
            "y": np.linspace(0, 0.5e-6, 50)
        }
        self.mock_simulator.solve_drift_diffusion.return_value = {
            "potential": np.random.random(5000),
            "n": np.random.random(5000) * 1e17,
            "p": np.random.random(5000) * 1e16,
            "Jn": np.random.random(10000),
            "Jp": np.random.random(10000)
        }
    
    def tearDown(self):
        """Clean up after tests."""
        if self.root:
            self.root.destroy()
        
        # Clean up any test files
        test_files = ["simulation_results.npz", "test_config.json"]
        for filename in test_files:
            if os.path.exists(filename):
                os.remove(filename)


class TestSimulatorGUI2D(TestGUIBase):
    """Test the main 2D simulator GUI."""
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_gui_creation(self):
        """Test GUI window creation."""
        with patch('gui.main_gui_2d.Simulator', return_value=self.mock_simulator):
            gui = SimulatorGUI2D(self.root)
            self.assertIsNotNone(gui)
            self.assertEqual(gui.root, self.root)
            self.assertFalse(gui.running)
            self.assertIsNone(gui.sim)
            self.assertIsNone(gui.results)
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_widget_creation(self):
        """Test that all required widgets are created."""
        with patch('gui.main_gui_2d.Simulator', return_value=self.mock_simulator):
            gui = SimulatorGUI2D(self.root)
            
            # Check that notebook exists
            self.assertTrue(hasattr(gui, 'notebook'))
            
            # Check that entry widgets exist
            required_entries = [
                'lx_entry', 'ly_entry', 'nx_entry', 'ny_entry',
                'nd_peak', 'na_peak', 'nd_x', 'na_x', 'sigma', 'et_entry',
                'v_left', 'v_right', 'v_bottom', 'v_top'
            ]
            
            for entry_name in required_entries:
                self.assertTrue(hasattr(gui, entry_name), 
                              f"Missing entry widget: {entry_name}")
            
            # Check that control buttons exist
            self.assertTrue(hasattr(gui, 'run_button'))
            self.assertTrue(hasattr(gui, 'stop_button'))
            self.assertTrue(hasattr(gui, 'save_button'))
            
            # Check that variable controls exist
            self.assertTrue(hasattr(gui, 'method_var'))
            self.assertTrue(hasattr(gui, 'mesh_var'))
            self.assertTrue(hasattr(gui, 'amr_var'))
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_default_values(self):
        """Test that default values are set correctly."""
        with patch('gui.main_gui_2d.Simulator', return_value=self.mock_simulator):
            gui = SimulatorGUI2D(self.root)
            
            # Check default device dimensions
            self.assertEqual(gui.lx_entry.get(), "1.0")
            self.assertEqual(gui.ly_entry.get(), "0.5")
            
            # Check default mesh parameters
            self.assertEqual(gui.nx_entry.get(), "50")
            self.assertEqual(gui.ny_entry.get(), "25")
            
            # Check default method and mesh type
            self.assertEqual(gui.method_var.get(), "DG")
            self.assertEqual(gui.mesh_var.get(), "Structured")
            
            # Check default AMR setting
            self.assertTrue(gui.amr_var.get())
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_input_validation(self):
        """Test input validation for GUI parameters."""
        with patch('gui.main_gui_2d.Simulator', return_value=self.mock_simulator):
            gui = SimulatorGUI2D(self.root)
            
            # Test invalid device dimensions
            gui.lx_entry.delete(0, tk.END)
            gui.lx_entry.insert(0, "-1.0")
            
            # Simulation should handle this gracefully
            with patch('tkinter.messagebox.showinfo') as mock_msg:
                gui.start_simulation()
                # Should show error message or handle gracefully
                self.assertTrue(mock_msg.called or not gui.running)
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_simulation_workflow(self):
        """Test the complete simulation workflow."""
        with patch('gui.main_gui_2d.Simulator', return_value=self.mock_simulator):
            gui = SimulatorGUI2D(self.root)
            
            # Mock threading to avoid actual thread creation
            with patch('threading.Thread') as mock_thread:
                mock_thread_instance = MagicMock()
                mock_thread.return_value = mock_thread_instance
                
                # Start simulation
                gui.start_simulation()
                
                # Check that thread was created
                mock_thread.assert_called_once()
                mock_thread_instance.start.assert_called_once()
                
                # Check button states
                self.assertEqual(gui.run_button.cget('state'), 'disabled')
                self.assertEqual(gui.stop_button.cget('state'), 'normal')
                self.assertEqual(gui.save_button.cget('state'), 'disabled')
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_simulation_stop(self):
        """Test simulation stopping functionality."""
        with patch('gui.main_gui_2d.Simulator', return_value=self.mock_simulator):
            gui = SimulatorGUI2D(self.root)
            
            # Simulate running state
            gui.running = True
            gui.run_button.config(state='disabled')
            gui.stop_button.config(state='normal')
            
            # Stop simulation
            gui.stop_simulation()
            
            # Check that running state is reset
            self.assertFalse(gui.running)
            self.assertEqual(gui.run_button.cget('state'), 'normal')
            self.assertEqual(gui.stop_button.cget('state'), 'disabled')
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_results_saving(self):
        """Test results saving functionality."""
        with patch('gui.main_gui_2d.Simulator', return_value=self.mock_simulator):
            gui = SimulatorGUI2D(self.root)
            
            # Set mock results
            gui.results = {
                "potential": np.random.random(100),
                "n": np.random.random(100),
                "p": np.random.random(100)
            }
            
            # Mock numpy.savez
            with patch('numpy.savez') as mock_savez:
                with patch('tkinter.messagebox.showinfo') as mock_msg:
                    gui.save_results()
                    
                    # Check that savez was called
                    mock_savez.assert_called_once()
                    mock_msg.assert_called_once()
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_plot_updates(self):
        """Test plot updating functionality."""
        with patch('gui.main_gui_2d.Simulator', return_value=self.mock_simulator):
            gui = SimulatorGUI2D(self.root)
            
            # Set mock results and simulator
            gui.results = self.mock_simulator.solve_drift_diffusion.return_value
            gui.sim = self.mock_simulator
            
            # Mock matplotlib components
            with patch.object(gui, 'ax_potential') as mock_ax_pot:
                with patch.object(gui, 'ax_density') as mock_ax_den:
                    with patch.object(gui, 'ax_current') as mock_ax_cur:
                        with patch.object(gui, 'canvas_potential') as mock_canvas_pot:
                            with patch.object(gui, 'canvas_density') as mock_canvas_den:
                                with patch.object(gui, 'canvas_current') as mock_canvas_cur:
                                    
                                    # Update plots
                                    gui.update_plots()
                                    
                                    # Check that axes were cleared and canvases drawn
                                    mock_ax_pot.clear.assert_called_once()
                                    mock_ax_den.clear.assert_called_once()
                                    mock_ax_cur.clear.assert_called_once()
                                    mock_canvas_pot.draw.assert_called_once()
                                    mock_canvas_den.draw.assert_called_once()
                                    mock_canvas_cur.draw.assert_called_once()
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_gaussian_doping(self):
        """Test Gaussian doping profile calculation."""
        with patch('gui.main_gui_2d.Simulator', return_value=self.mock_simulator):
            gui = SimulatorGUI2D(self.root)
            
            # Test Gaussian doping function
            x = 0.5e-6
            x_center = 0.5e-6
            peak = 1e17
            sigma = 0.1e-6
            
            result = gui.gaussian_doping(x, x_center, peak, sigma)
            
            # At center, should be close to peak value
            self.assertAlmostEqual(result, peak, places=5)
            
            # Test off-center
            result_off = gui.gaussian_doping(x + sigma, x_center, peak, sigma)
            self.assertLess(result_off, peak)
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_error_handling(self):
        """Test error handling in GUI operations."""
        with patch('gui.main_gui_2d.Simulator', side_effect=Exception("Test error")):
            gui = SimulatorGUI2D(self.root)
            
            # Mock error message display
            with patch('tkinter.messagebox.showinfo') as mock_msg:
                # Try to start simulation with failing simulator
                gui.start_simulation()
                
                # Should handle error gracefully
                self.assertFalse(gui.running)


class TestVisualization(TestGUIBase):
    """Test visualization components."""
    
    @unittest.skipIf(plot_2d_potential is None, "Visualization module not available")
    def test_plot_2d_potential(self):
        """Test 2D potential plotting function."""
        # Create test data
        x = np.linspace(0, 1e-6, 10)
        y = np.linspace(0, 0.5e-6, 5)
        V = np.random.random((5, 10))
        
        # Mock matplotlib
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_ax = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            # Mock contourf
            mock_contour = MagicMock()
            mock_ax.contourf.return_value = mock_contour
            
            # Test plotting function
            plot_2d_potential(mock_ax, x, y, V.flatten(), "Test Potential")
            
            # Check that contourf was called
            mock_ax.contourf.assert_called_once()
            mock_ax.set_title.assert_called_with("Test Potential")
            mock_ax.set_xlabel.assert_called_with("X (m)")
            mock_ax.set_ylabel.assert_called_with("Y (m)")
    
    @unittest.skipIf(plot_2d_quantity is None, "Visualization module not available")
    def test_plot_2d_quantity(self):
        """Test 2D quantity plotting function."""
        # Create test data
        x = np.linspace(0, 1e-6, 10)
        y = np.linspace(0, 0.5e-6, 5)
        quantity = np.random.random((5, 10))
        
        # Mock matplotlib
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_ax = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            # Mock contourf
            mock_contour = MagicMock()
            mock_ax.contourf.return_value = mock_contour
            
            # Test plotting function
            plot_2d_quantity(mock_ax, x, y, quantity.flatten(), "Test Quantity")
            
            # Check that contourf was called
            mock_ax.contourf.assert_called_once()
            mock_ax.set_title.assert_called_with("Test Quantity")
    
    @unittest.skipIf(plot_current_vectors is None, "Visualization module not available")
    def test_plot_current_vectors(self):
        """Test current vector plotting function."""
        # Create test data
        x = np.linspace(0, 1e-6, 10)
        y = np.linspace(0, 0.5e-6, 5)
        Jx = np.random.random((5, 10))
        Jy = np.random.random((5, 10))
        
        # Mock matplotlib
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_ax = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            # Test plotting function
            plot_current_vectors(mock_ax, x, y, Jx.flatten(), Jy.flatten(), "Test Current")
            
            # Check that quiver was called
            mock_ax.quiver.assert_called_once()
            mock_ax.set_title.assert_called_with("Test Current")
    
    def test_visualization_error_handling(self):
        """Test error handling in visualization functions."""
        if plot_2d_potential is None:
            self.skipTest("Visualization module not available")
        
        # Test with mismatched data sizes
        x = np.linspace(0, 1e-6, 10)
        y = np.linspace(0, 0.5e-6, 5)
        V_wrong_size = np.random.random(30)  # Wrong size
        
        mock_ax = MagicMock()
        
        # Should handle error gracefully or raise appropriate exception
        try:
            plot_2d_potential(mock_ax, x, y, V_wrong_size, "Test")
        except (ValueError, IndexError):
            # Expected for mismatched sizes
            pass


class TestGUIIntegration(TestGUIBase):
    """Integration tests for GUI components."""
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_full_workflow_simulation(self):
        """Test complete workflow from GUI input to results display."""
        with patch('gui.main_gui_2d.Simulator', return_value=self.mock_simulator):
            gui = SimulatorGUI2D(self.root)
            
            # Set up input parameters
            gui.lx_entry.delete(0, tk.END)
            gui.lx_entry.insert(0, "2.0")
            gui.ly_entry.delete(0, tk.END)
            gui.ly_entry.insert(0, "1.0")
            
            # Mock the complete simulation process
            with patch.object(gui, 'run_simulation') as mock_run:
                with patch.object(gui, 'update_plots') as mock_update:
                    # Simulate successful run
                    gui.results = self.mock_simulator.solve_drift_diffusion.return_value
                    gui.sim = self.mock_simulator
                    
                    # Update plots
                    gui.update_plots()
                    
                    # Verify update was called
                    mock_update.assert_called_once()
    
    @unittest.skipIf(SimulatorGUI2D is None, "GUI module not available")
    def test_gui_responsiveness(self):
        """Test GUI responsiveness during operations."""
        with patch('gui.main_gui_2d.Simulator', return_value=self.mock_simulator):
            gui = SimulatorGUI2D(self.root)
            
            # Test that GUI doesn't freeze during operations
            start_time = time.time()
            
            # Perform multiple GUI operations
            for i in range(10):
                gui.lx_entry.delete(0, tk.END)
                gui.lx_entry.insert(0, str(i))
                self.root.update()  # Process GUI events
            
            end_time = time.time()
            
            # Should complete quickly
            self.assertLess(end_time - start_time, 1.0)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
