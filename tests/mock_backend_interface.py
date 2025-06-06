#!/usr/bin/env python3
"""
Mock Backend Interface for Testing
Provides mock implementations when real backend is not available

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
from typing import Dict, List, Any, Optional

class MockBackendInterface:
    """Mock backend interface for testing frontend integration"""
    
    def __init__(self):
        self.backend_available = False
        self.modules_available = {
            'simulator': False,
            'complete_dg': False,
            'unstructured_transport': False,
            'performance_bindings': False,
            'advanced_transport': False
        }
        
    def check_backend_availability(self):
        """Check if backend is available (always False for mock)"""
        return False
    
    def test_module_connection(self, module_name: str):
        """Test connection to specific module (always False for mock)"""
        return False
    
    def get_default_configuration(self):
        """Get default configuration"""
        return {
            'device': {
                'length': 2e-6,
                'width': 1e-6,
                'nx': 20,
                'ny': 10
            },
            'physics': {
                'enable_energy_transport': True,
                'enable_hydrodynamic': True,
                'enable_non_equilibrium_dd': True
            },
            'numerical': {
                'max_iterations': 100,
                'tolerance': 1e-10,
                'time_step': 1e-12
            }
        }
    
    def setup_simulation(self, config):
        """Setup simulation (mock implementation)"""
        # Always succeeds for mock
        return True
    
    def run_simulation(self, config, max_steps=10):
        """Run simulation (mock implementation)"""
        
        # Generate mock results
        size = config.get('device', {}).get('nx', 20) * config.get('device', {}).get('ny', 10)
        
        results = {
            'potential': np.random.random(size) * 2.0 - 1.0,  # -1 to 1 V
            'n': np.random.random(size) * 1e17 + 1e15,        # 1e15 to 1e17 m^-3
            'p': np.random.random(size) * 1e16 + 1e14,        # 1e14 to 1e16 m^-3
        }
        
        # Add transport model results if enabled
        physics = config.get('physics', {})
        
        if physics.get('enable_energy_transport', False):
            results['energy_n'] = np.random.random(size) * 1e-19 + 1e-20
            results['energy_p'] = np.random.random(size) * 1e-20 + 1e-21
            results['T_n'] = np.random.random(size) * 100 + 300  # 300-400 K
            results['T_p'] = np.random.random(size) * 50 + 300   # 300-350 K
        
        if physics.get('enable_hydrodynamic', False):
            results['velocity_n'] = np.random.random(size) * 1e5  # 0-1e5 m/s
            results['velocity_p'] = np.random.random(size) * 1e4  # 0-1e4 m/s
            results['momentum_n'] = results['n'] * 9.11e-31 * results['velocity_n']
            results['momentum_p'] = results['p'] * 9.11e-31 * results['velocity_p']
        
        if physics.get('enable_non_equilibrium_dd', False):
            kT = 1.381e-23 * 300 / 1.602e-19  # Thermal voltage at 300K
            results['quasi_fermi_n'] = kT * np.log(results['n'] / 1e16)
            results['quasi_fermi_p'] = -kT * np.log(results['p'] / 1e16)
        
        return results

class MockTransportModelConfig:
    """Mock transport model configuration"""
    
    def __init__(self):
        self.device_length = 2e-6
        self.device_width = 1e-6
        self.nx = 20
        self.ny = 10
        self.enable_energy_transport = True
        self.enable_hydrodynamic = True
        self.enable_non_equilibrium_dd = True
        self.temperature = 300.0
        self.max_iterations = 100
        self.tolerance = 1e-10

# Mock classes for missing modules
class MockDevice:
    """Mock device class"""
    
    def __init__(self, width, height):
        self._width = width
        self._height = height
    
    def get_width(self):
        return self._width
    
    def get_height(self):
        return self._height

class MockSimulator:
    """Mock simulator class"""
    
    def __init__(self, extents, num_points_x, num_points_y, method="DG", mesh_type="Structured"):
        self.extents = extents
        self.nx = num_points_x
        self.ny = num_points_y
        self.method = method
        self.mesh_type = mesh_type
        self._doping_set = False
    
    def set_doping(self, Nd, Na):
        """Set doping concentrations"""
        self._doping_set = True
        self._Nd = Nd
        self._Na = Na
    
    def solve_drift_diffusion(self, bc, Vg=0.0, max_steps=10, use_amr=False, **kwargs):
        """Solve drift-diffusion equations (mock)"""
        
        if not self._doping_set:
            raise RuntimeError("Doping must be set before solving")
        
        size = self.nx * self.ny
        
        # Generate realistic-looking results
        x = np.linspace(0, 1, size)
        
        # Potential with applied bias
        potential = Vg * (1 - x) + bc[1] * x + 0.1 * np.sin(2 * np.pi * x)
        
        # Carrier densities
        ni = 1.45e16  # Intrinsic density
        n = ni * np.exp(potential / 0.026) + np.mean(self._Nd)
        p = ni**2 / n + np.mean(self._Na)
        
        return {
            'potential': potential,
            'n': n,
            'p': p,
            'Jn': np.gradient(n) * 1350e-4 * 1.602e-19,  # Electron current
            'Jp': -np.gradient(p) * 480e-4 * 1.602e-19   # Hole current
        }

class MockMethod:
    """Mock method enumeration"""
    DG = 0
    FEM = 1

class MockMeshType:
    """Mock mesh type enumeration"""
    Structured = 0
    Unstructured = 1

# Create mock modules
class MockSimulatorModule:
    """Mock simulator module"""
    Device = MockDevice
    Simulator = MockSimulator
    Method = MockMethod
    MeshType = MockMeshType

class MockCompleteDGModule:
    """Mock complete DG module"""
    
    class CompleteDGSolver:
        def __init__(self):
            pass
    
    @staticmethod
    def test_basis_functions(order):
        """Mock basis function test"""
        return f"P{order} basis functions working"

class MockUnstructuredTransportModule:
    """Mock unstructured transport module"""
    
    class UnstructuredTransportSuite:
        def __init__(self, device, order):
            self.device = device
            self.order = order
    
    class EnergyTransportSolver:
        def __init__(self):
            pass
    
    class HydrodynamicSolver:
        def __init__(self):
            pass
    
    class NonEquilibriumDDSolver:
        def __init__(self):
            pass

class MockPerformanceBindingsModule:
    """Mock performance bindings module"""
    
    class SIMDKernels:
        @staticmethod
        def vector_add(a, b):
            return a + b
        
        @staticmethod
        def vector_multiply(a, b):
            return a * b
        
        @staticmethod
        def dot_product(a, b):
            return np.dot(a, b)
    
    class GPUAcceleration:
        def __init__(self):
            pass
        
        def is_available(self):
            return False  # Mock GPU not available
        
        def vector_add(self, a, b):
            return a + b

class MockAdvancedTransportModule:
    """Mock advanced transport module"""
    
    class TransportModel:
        DRIFT_DIFFUSION = 0
        ENERGY_TRANSPORT = 1
        HYDRODYNAMIC = 2
        NON_EQUILIBRIUM_STATISTICS = 3
    
    class AdvancedTransport:
        def __init__(self, device, model_id, order=2):
            self.device = device
            self.model_id = model_id
            self.order = order

def install_mock_modules():
    """Install mock modules in sys.modules for testing"""
    
    import sys
    
    # Install mock modules
    sys.modules['simulator'] = MockSimulatorModule()
    sys.modules['complete_dg'] = MockCompleteDGModule()
    sys.modules['unstructured_transport'] = MockUnstructuredTransportModule()
    sys.modules['performance_bindings'] = MockPerformanceBindingsModule()
    sys.modules['advanced_transport'] = MockAdvancedTransportModule()
    
    print("🔧 Mock backend modules installed for testing")

def remove_mock_modules():
    """Remove mock modules from sys.modules"""
    
    import sys
    
    mock_modules = [
        'simulator',
        'complete_dg', 
        'unstructured_transport',
        'performance_bindings',
        'advanced_transport'
    ]
    
    for module in mock_modules:
        if module in sys.modules:
            del sys.modules[module]
    
    print("🔧 Mock backend modules removed")
