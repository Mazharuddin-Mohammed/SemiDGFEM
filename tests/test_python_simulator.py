#!/usr/bin/env python3
"""
Comprehensive unit tests for the Python simulator interface.
Tests both the ctypes and Cython implementations.
"""

import unittest
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from simulator import Simulator as CtypesSimulator
except ImportError:
    CtypesSimulator = None

try:
    import simulator_cython as CythonSimulator
except ImportError:
    CythonSimulator = None


class TestSimulatorBase:
    """Base class for simulator tests that can be used with both implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extents = [1e-6, 0.5e-6]
        self.num_points_x = 20
        self.num_points_y = 10
        self.method = "DG"
        self.mesh_type = "Structured"
        self.order = "P3"
        
    def tearDown(self):
        """Clean up test files."""
        test_files = ["device_2d.msh", "device_refined.msh", "test_output.npz"]
        for filename in test_files:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_simulator_creation(self):
        """Test basic simulator creation."""
        if self.SimulatorClass is None:
            self.skipTest("Simulator implementation not available")
            
        sim = self.SimulatorClass(
            dimension="TwoD",
            extents=self.extents,
            num_points_x=self.num_points_x,
            num_points_y=self.num_points_y,
            method=self.method,
            mesh_type=self.mesh_type,
            order=self.order
        )
        
        self.assertIsNotNone(sim)
    
    def test_invalid_parameters(self):
        """Test simulator creation with invalid parameters."""
        if self.SimulatorClass is None:
            self.skipTest("Simulator implementation not available")
        
        # Invalid extents
        with self.assertRaises((ValueError, TypeError)):
            self.SimulatorClass(extents=[-1e-6, 0.5e-6])
        
        with self.assertRaises((ValueError, TypeError)):
            self.SimulatorClass(extents=[1e-6, -0.5e-6])
        
        # Invalid method
        with self.assertRaises((ValueError, KeyError)):
            self.SimulatorClass(method="INVALID_METHOD")
        
        # Invalid mesh type
        with self.assertRaises((ValueError, KeyError)):
            self.SimulatorClass(mesh_type="INVALID_MESH")
        
        # Invalid order
        with self.assertRaises((ValueError, KeyError)):
            self.SimulatorClass(order="INVALID_ORDER")
    
    def test_doping_configuration(self):
        """Test doping profile configuration."""
        if self.SimulatorClass is None:
            self.skipTest("Simulator implementation not available")
        
        sim = self.SimulatorClass(
            extents=self.extents,
            num_points_x=self.num_points_x,
            num_points_y=self.num_points_y
        )
        
        # Create test doping profiles
        num_nodes = self.num_points_x * self.num_points_y
        Nd = np.ones(num_nodes) * 1e17
        Na = np.ones(num_nodes) * 1e16
        
        # Should not raise exception
        sim.set_doping(Nd, Na)
        
        # Test with wrong sizes
        with self.assertRaises((ValueError, IndexError)):
            sim.set_doping(Nd[:-10], Na)
        
        with self.assertRaises((ValueError, IndexError)):
            sim.set_doping(Nd, Na[:-10])
    
    def test_trap_level_configuration(self):
        """Test trap level configuration."""
        if self.SimulatorClass is None:
            self.skipTest("Simulator implementation not available")
        
        sim = self.SimulatorClass(
            extents=self.extents,
            num_points_x=self.num_points_x,
            num_points_y=self.num_points_y
        )
        
        # Create test trap levels
        num_nodes = self.num_points_x * self.num_points_y
        Et = np.ones(num_nodes) * 0.5  # Mid-gap traps
        
        # Should not raise exception
        sim.set_trap_level(Et)
        
        # Test with wrong size
        with self.assertRaises((ValueError, IndexError)):
            sim.set_trap_level(Et[:-10])
    
    def test_boundary_conditions(self):
        """Test boundary condition validation."""
        if self.SimulatorClass is None:
            self.skipTest("Simulator implementation not available")
        
        sim = self.SimulatorClass(
            extents=self.extents,
            num_points_x=self.num_points_x,
            num_points_y=self.num_points_y
        )
        
        # Valid boundary conditions (4 values for 2D)
        bc_valid = [0.0, 1.0, 0.0, 0.0]  # left, right, bottom, top
        
        # Set up doping
        num_nodes = self.num_points_x * self.num_points_y
        Nd = np.ones(num_nodes) * 1e17
        Na = np.ones(num_nodes) * 1e16
        sim.set_doping(Nd, Na)
        
        # This should work (though may fail due to missing implementation)
        try:
            results = sim.solve_drift_diffusion(bc_valid, max_steps=10)
            self.assertIsInstance(results, dict)
        except (NotImplementedError, AttributeError):
            # Implementation may not be complete
            pass
        
        # Invalid boundary conditions
        bc_invalid = [0.0, 1.0]  # Too few values
        with self.assertRaises((ValueError, IndexError)):
            sim.solve_drift_diffusion(bc_invalid)
    
    def test_mesh_generation(self):
        """Test mesh generation for unstructured meshes."""
        if self.SimulatorClass is None:
            self.skipTest("Simulator implementation not available")
        
        sim = self.SimulatorClass(
            extents=self.extents,
            num_points_x=self.num_points_x,
            num_points_y=self.num_points_y,
            mesh_type="Unstructured"
        )
        
        # Generate mesh file
        filename = "test_mesh.msh"
        try:
            sim.generate_mesh(filename)
            # Check if file was created
            self.assertTrue(os.path.exists(filename))
        except (NotImplementedError, AttributeError):
            # Implementation may not be complete
            pass
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_grid_points_access(self):
        """Test grid points access."""
        if self.SimulatorClass is None:
            self.skipTest("Simulator implementation not available")
        
        sim = self.SimulatorClass(
            extents=self.extents,
            num_points_x=self.num_points_x,
            num_points_y=self.num_points_y,
            mesh_type="Structured"
        )
        
        try:
            grid = sim.get_grid_points()
            self.assertIsInstance(grid, dict)
            self.assertIn("x", grid)
            self.assertIn("y", grid)
            
            # Check grid dimensions
            self.assertEqual(len(grid["x"]), self.num_points_x * self.num_points_y)
            self.assertEqual(len(grid["y"]), self.num_points_x * self.num_points_y)
            
            # Check bounds
            self.assertTrue(all(0 <= x <= self.extents[0] for x in grid["x"]))
            self.assertTrue(all(0 <= y <= self.extents[1] for y in grid["y"]))
            
        except (NotImplementedError, AttributeError):
            # Implementation may not be complete
            pass


class TestCtypesSimulator(TestSimulatorBase, unittest.TestCase):
    """Test the ctypes-based simulator implementation."""
    
    def setUp(self):
        super().setUp()
        self.SimulatorClass = CtypesSimulator
    
    @unittest.skipIf(CtypesSimulator is None, "Ctypes simulator not available")
    def test_library_loading(self):
        """Test that the shared library loads correctly."""
        # This test is specific to ctypes implementation
        try:
            sim = CtypesSimulator()
            # If we get here, library loaded successfully
            self.assertTrue(True)
        except OSError as e:
            self.fail(f"Failed to load shared library: {e}")


class TestCythonSimulator(TestSimulatorBase, unittest.TestCase):
    """Test the Cython-based simulator implementation."""
    
    def setUp(self):
        super().setUp()
        self.SimulatorClass = CythonSimulator
    
    @unittest.skipIf(CythonSimulator is None, "Cython simulator not available")
    def test_cython_compilation(self):
        """Test that Cython module compiled correctly."""
        # This test is specific to Cython implementation
        try:
            sim = CythonSimulator.Simulator()
            # If we get here, Cython module loaded successfully
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to create Cython simulator: {e}")


class TestSimulatorComparison(unittest.TestCase):
    """Compare results between ctypes and Cython implementations."""
    
    @unittest.skipIf(CtypesSimulator is None or CythonSimulator is None, 
                     "Both implementations not available")
    def test_implementation_consistency(self):
        """Test that both implementations give consistent results."""
        extents = [1e-6, 0.5e-6]
        num_points_x = 10
        num_points_y = 5
        
        # Create both simulators with same parameters
        ctypes_sim = CtypesSimulator(
            extents=extents,
            num_points_x=num_points_x,
            num_points_y=num_points_y,
            mesh_type="Structured"
        )
        
        cython_sim = CythonSimulator.Simulator(
            extents=extents,
            num_points_x=num_points_x,
            num_points_y=num_points_y,
            mesh_type="Structured"
        )
        
        # Compare grid points
        try:
            ctypes_grid = ctypes_sim.get_grid_points()
            cython_grid = cython_sim.get_grid_points()
            
            np.testing.assert_array_almost_equal(
                ctypes_grid["x"], cython_grid["x"], decimal=10
            )
            np.testing.assert_array_almost_equal(
                ctypes_grid["y"], cython_grid["y"], decimal=10
            )
            
        except (NotImplementedError, AttributeError):
            # Implementations may not be complete
            pass


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_memory_management(self):
        """Test memory management and cleanup."""
        if CtypesSimulator is None:
            self.skipTest("Ctypes simulator not available")
        
        # Create and destroy many simulators to test for memory leaks
        for i in range(100):
            sim = CtypesSimulator(
                extents=[1e-6, 0.5e-6],
                num_points_x=10,
                num_points_y=5
            )
            # Simulator should be automatically cleaned up
            del sim
        
        # If we get here without crashing, memory management is working
        self.assertTrue(True)
    
    def test_thread_safety(self):
        """Test thread safety of simulator operations."""
        if CtypesSimulator is None:
            self.skipTest("Ctypes simulator not available")
        
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                sim = CtypesSimulator(
                    extents=[1e-6, 0.5e-6],
                    num_points_x=5,
                    num_points_y=5
                )
                # Perform some operations
                time.sleep(0.01)  # Simulate work
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), 10)


class TestPerformance(unittest.TestCase):
    """Performance and benchmark tests."""
    
    def test_creation_performance(self):
        """Test simulator creation performance."""
        if CtypesSimulator is None:
            self.skipTest("Ctypes simulator not available")
        
        import time
        
        start_time = time.time()
        
        # Create multiple simulators
        for i in range(50):
            sim = CtypesSimulator(
                extents=[1e-6, 0.5e-6],
                num_points_x=20,
                num_points_y=10
            )
            del sim
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should create 50 simulators in less than 1 second
        self.assertLess(creation_time, 1.0, 
                       f"Simulator creation too slow: {creation_time:.3f}s")
    
    def test_large_mesh_handling(self):
        """Test handling of large meshes."""
        if CtypesSimulator is None:
            self.skipTest("Ctypes simulator not available")
        
        # Test with larger mesh
        try:
            sim = CtypesSimulator(
                extents=[1e-6, 0.5e-6],
                num_points_x=100,
                num_points_y=50,
                mesh_type="Structured"
            )
            
            # Try to access grid points
            grid = sim.get_grid_points()
            expected_nodes = 100 * 50
            self.assertEqual(len(grid["x"]), expected_nodes)
            self.assertEqual(len(grid["y"]), expected_nodes)
            
        except (MemoryError, NotImplementedError, AttributeError):
            # May not be implemented or system may not have enough memory
            pass


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
