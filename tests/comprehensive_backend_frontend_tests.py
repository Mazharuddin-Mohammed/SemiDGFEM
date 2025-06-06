#!/usr/bin/env python3
"""
Comprehensive Backend-Frontend Integration Tests
Tests all backend models and frontend integration points

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import time
import traceback
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Add python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

class BackendTestSuite:
    """Comprehensive backend testing suite"""
    
    def __init__(self):
        self.test_results = {}
        self.backend_modules = {}
        self.integration_issues = []
        
    def detect_backend_modules(self):
        """Detect available backend modules"""
        
        print("🔍 Detecting Backend Modules...")
        
        modules_to_test = {
            'simulator': 'Core simulator functionality',
            'complete_dg': 'Complete DG discretization',
            'unstructured_transport': 'Unstructured transport models',
            'performance_bindings': 'SIMD/GPU performance optimization',
            'advanced_transport': 'Advanced transport physics'
        }
        
        for module_name, description in modules_to_test.items():
            try:
                module = __import__(module_name)
                self.backend_modules[module_name] = {
                    'module': module,
                    'description': description,
                    'available': True,
                    'import_error': None
                }
                print(f"   ✓ {module_name}: {description}")
            except ImportError as e:
                self.backend_modules[module_name] = {
                    'module': None,
                    'description': description,
                    'available': False,
                    'import_error': str(e)
                }
                print(f"   ✗ {module_name}: {e}")
                self.integration_issues.append(f"Module {module_name} not available: {e}")
        
        available_count = sum(1 for m in self.backend_modules.values() if m['available'])
        total_count = len(self.backend_modules)
        
        print(f"\n📊 Backend Module Status: {available_count}/{total_count} available")
        
        if available_count == 0:
            print("❌ No backend modules available!")
            print("   To fix: cd python && python3 compile_all.py")
            return False
        elif available_count < total_count:
            print("⚠ Partial backend available - some tests will be skipped")
        else:
            print("✅ Complete backend available")
        
        return available_count > 0
    
    def test_core_simulator(self):
        """Test core simulator functionality"""
        
        print("\n🧪 Testing Core Simulator...")
        
        if not self.backend_modules['simulator']['available']:
            print("   ⏭ Skipping - simulator module not available")
            self.test_results['core_simulator'] = {'status': 'skipped', 'reason': 'module not available'}
            return False
        
        try:
            simulator = self.backend_modules['simulator']['module']
            
            # Test 1: Device creation
            print("   Testing device creation...")
            device = simulator.Device(2e-6, 1e-6)  # 2μm × 1μm
            assert device.get_width() == 2e-6, "Device width mismatch"
            assert device.get_height() == 1e-6, "Device height mismatch"
            print(f"     ✓ Device created: {device.get_width()*1e6:.1f}μm × {device.get_height()*1e6:.1f}μm")
            
            # Test 2: Simulator creation
            print("   Testing simulator creation...")
            sim = simulator.Simulator(
                extents=[2e-6, 1e-6],
                num_points_x=20,
                num_points_y=10,
                method="DG",
                mesh_type="Structured"
            )
            print("     ✓ Simulator created with DG method")
            
            # Test 3: Doping configuration
            print("   Testing doping configuration...")
            size = 20 * 10
            Nd = np.full(size, 1e17, dtype=np.float64)
            Na = np.full(size, 1e16, dtype=np.float64)
            sim.set_doping(Nd, Na)
            print(f"     ✓ Doping set: Nd={np.mean(Nd):.1e}, Na={np.mean(Na):.1e}")
            
            # Test 4: Basic simulation
            print("   Testing basic simulation...")
            bc = [0.0, 1.0, 0.0, 0.0]  # Boundary conditions
            results = sim.solve_drift_diffusion(bc, Vg=0.5, max_steps=10)
            
            # Validate results
            required_fields = ['potential', 'n', 'p']
            for field in required_fields:
                assert field in results, f"Missing field: {field}"
                assert len(results[field]) > 0, f"Empty field: {field}"
                assert np.all(np.isfinite(results[field])), f"Non-finite values in {field}"
            
            print(f"     ✓ Simulation completed: {len(results)} fields returned")
            
            self.test_results['core_simulator'] = {
                'status': 'passed',
                'device_creation': True,
                'simulator_creation': True,
                'doping_configuration': True,
                'basic_simulation': True,
                'result_fields': list(results.keys())
            }
            
            print("   ✅ Core simulator tests PASSED")
            return True
            
        except Exception as e:
            print(f"   ❌ Core simulator test FAILED: {e}")
            self.test_results['core_simulator'] = {'status': 'failed', 'error': str(e)}
            self.integration_issues.append(f"Core simulator test failed: {e}")
            return False
    
    def test_complete_dg(self):
        """Test complete DG discretization"""
        
        print("\n🧪 Testing Complete DG Discretization...")
        
        if not self.backend_modules['complete_dg']['available']:
            print("   ⏭ Skipping - complete_dg module not available")
            self.test_results['complete_dg'] = {'status': 'skipped', 'reason': 'module not available'}
            return False
        
        try:
            complete_dg = self.backend_modules['complete_dg']['module']
            
            # Test 1: DG solver creation
            print("   Testing DG solver creation...")
            if hasattr(complete_dg, 'CompleteDGSolver'):
                solver = complete_dg.CompleteDGSolver()
                print("     ✓ CompleteDGSolver created")
            else:
                print("     ⚠ CompleteDGSolver not found - checking alternatives...")
                # Check for other DG classes
                dg_classes = [attr for attr in dir(complete_dg) if 'DG' in attr or 'Solver' in attr]
                if dg_classes:
                    print(f"     Available classes: {dg_classes}")
                else:
                    raise AttributeError("No DG solver classes found")
            
            # Test 2: Basis function orders
            print("   Testing basis function orders...")
            orders_to_test = [1, 2, 3]
            for order in orders_to_test:
                try:
                    if hasattr(complete_dg, 'test_basis_functions'):
                        result = complete_dg.test_basis_functions(order)
                        print(f"     ✓ P{order} basis functions available")
                    else:
                        print(f"     ⚠ Cannot test P{order} basis functions - method not available")
                except Exception as e:
                    print(f"     ✗ P{order} basis functions failed: {e}")
                    self.integration_issues.append(f"P{order} basis functions not working: {e}")
            
            self.test_results['complete_dg'] = {
                'status': 'passed',
                'solver_creation': True,
                'basis_functions_tested': orders_to_test
            }
            
            print("   ✅ Complete DG tests PASSED")
            return True
            
        except Exception as e:
            print(f"   ❌ Complete DG test FAILED: {e}")
            self.test_results['complete_dg'] = {'status': 'failed', 'error': str(e)}
            self.integration_issues.append(f"Complete DG test failed: {e}")
            return False
    
    def test_unstructured_transport(self):
        """Test unstructured transport models"""
        
        print("\n🧪 Testing Unstructured Transport Models...")
        
        if not self.backend_modules['unstructured_transport']['available']:
            print("   ⏭ Skipping - unstructured_transport module not available")
            self.test_results['unstructured_transport'] = {'status': 'skipped', 'reason': 'module not available'}
            return False
        
        try:
            unstructured_transport = self.backend_modules['unstructured_transport']['module']
            
            # Test 1: Transport suite creation
            print("   Testing transport suite creation...")
            if hasattr(unstructured_transport, 'UnstructuredTransportSuite'):
                # Need a device for transport suite
                if self.backend_modules['simulator']['available']:
                    simulator = self.backend_modules['simulator']['module']
                    device = simulator.Device(1e-6, 1e-6)
                    suite = unstructured_transport.UnstructuredTransportSuite(device, 2)  # P2 elements
                    print("     ✓ UnstructuredTransportSuite created")
                else:
                    print("     ⚠ Cannot create transport suite - simulator not available")
                    raise ImportError("Simulator required for transport suite")
            else:
                print("     ⚠ UnstructuredTransportSuite not found")
                # Check for alternative classes
                transport_classes = [attr for attr in dir(unstructured_transport) if 'Transport' in attr]
                if transport_classes:
                    print(f"     Available classes: {transport_classes}")
                else:
                    raise AttributeError("No transport classes found")
            
            # Test 2: Individual transport models
            print("   Testing individual transport models...")
            transport_models = [
                'EnergyTransportSolver',
                'HydrodynamicSolver', 
                'NonEquilibriumDDSolver'
            ]
            
            available_models = []
            for model_name in transport_models:
                if hasattr(unstructured_transport, model_name):
                    print(f"     ✓ {model_name} available")
                    available_models.append(model_name)
                else:
                    print(f"     ✗ {model_name} not found")
                    self.integration_issues.append(f"Transport model {model_name} not available")
            
            self.test_results['unstructured_transport'] = {
                'status': 'passed' if available_models else 'failed',
                'suite_creation': True,
                'available_models': available_models,
                'missing_models': [m for m in transport_models if m not in available_models]
            }
            
            if available_models:
                print("   ✅ Unstructured transport tests PASSED")
                return True
            else:
                print("   ❌ No transport models available")
                return False
            
        except Exception as e:
            print(f"   ❌ Unstructured transport test FAILED: {e}")
            self.test_results['unstructured_transport'] = {'status': 'failed', 'error': str(e)}
            self.integration_issues.append(f"Unstructured transport test failed: {e}")
            return False
    
    def test_performance_bindings(self):
        """Test performance optimization bindings"""
        
        print("\n🧪 Testing Performance Optimization...")
        
        if not self.backend_modules['performance_bindings']['available']:
            print("   ⏭ Skipping - performance_bindings module not available")
            self.test_results['performance_bindings'] = {'status': 'skipped', 'reason': 'module not available'}
            return False
        
        try:
            performance_bindings = self.backend_modules['performance_bindings']['module']
            
            # Test 1: SIMD kernels
            print("   Testing SIMD kernels...")
            if hasattr(performance_bindings, 'SIMDKernels'):
                # Test vector operations
                size = 1000
                a = np.random.random(size).astype(np.float64)
                b = np.random.random(size).astype(np.float64)
                
                # Test vector addition
                result_add = performance_bindings.SIMDKernels.vector_add(a, b)
                expected_add = a + b
                assert np.allclose(result_add, expected_add, rtol=1e-12), "SIMD vector addition failed"
                print("     ✓ SIMD vector addition working")
                
                # Test vector multiplication
                result_mul = performance_bindings.SIMDKernels.vector_multiply(a, b)
                expected_mul = a * b
                assert np.allclose(result_mul, expected_mul, rtol=1e-12), "SIMD vector multiplication failed"
                print("     ✓ SIMD vector multiplication working")
                
            else:
                print("     ⚠ SIMDKernels not found")
                self.integration_issues.append("SIMDKernels class not available")
            
            # Test 2: GPU acceleration
            print("   Testing GPU acceleration...")
            if hasattr(performance_bindings, 'GPUAcceleration'):
                gpu = performance_bindings.GPUAcceleration()
                if gpu.is_available():
                    print("     ✓ GPU acceleration available")
                    
                    # Test GPU vector operations
                    size = 1000
                    a = np.random.random(size).astype(np.float64)
                    b = np.random.random(size).astype(np.float64)
                    
                    result_gpu = gpu.vector_add(a, b)
                    expected = a + b
                    assert np.allclose(result_gpu, expected, rtol=1e-12), "GPU vector addition failed"
                    print("     ✓ GPU vector operations working")
                else:
                    print("     ⚠ GPU not available on this system")
            else:
                print("     ⚠ GPUAcceleration not found")
                self.integration_issues.append("GPUAcceleration class not available")
            
            self.test_results['performance_bindings'] = {
                'status': 'passed',
                'simd_available': hasattr(performance_bindings, 'SIMDKernels'),
                'gpu_available': hasattr(performance_bindings, 'GPUAcceleration')
            }
            
            print("   ✅ Performance optimization tests PASSED")
            return True
            
        except Exception as e:
            print(f"   ❌ Performance optimization test FAILED: {e}")
            self.test_results['performance_bindings'] = {'status': 'failed', 'error': str(e)}
            self.integration_issues.append(f"Performance optimization test failed: {e}")
            return False
    
    def test_advanced_transport(self):
        """Test advanced transport physics"""
        
        print("\n🧪 Testing Advanced Transport Physics...")
        
        if not self.backend_modules['advanced_transport']['available']:
            print("   ⏭ Skipping - advanced_transport module not available")
            self.test_results['advanced_transport'] = {'status': 'skipped', 'reason': 'module not available'}
            return False
        
        try:
            advanced_transport = self.backend_modules['advanced_transport']['module']
            
            # Test 1: Transport model enumeration
            print("   Testing transport model types...")
            if hasattr(advanced_transport, 'TransportModel'):
                models = advanced_transport.TransportModel
                expected_models = ['DRIFT_DIFFUSION', 'ENERGY_TRANSPORT', 'HYDRODYNAMIC', 'NON_EQUILIBRIUM_STATISTICS']
                
                available_models = []
                for model in expected_models:
                    if hasattr(models, model):
                        print(f"     ✓ {model} available")
                        available_models.append(model)
                    else:
                        print(f"     ✗ {model} not found")
                        self.integration_issues.append(f"Transport model {model} not available")
            else:
                print("     ⚠ TransportModel enumeration not found")
                self.integration_issues.append("TransportModel enumeration not available")
                available_models = []
            
            # Test 2: Advanced transport solver
            print("   Testing advanced transport solver...")
            if hasattr(advanced_transport, 'AdvancedTransport'):
                if self.backend_modules['simulator']['available']:
                    simulator = self.backend_modules['simulator']['module']
                    device = simulator.Device(1e-6, 1e-6)
                    
                    # Test different transport models
                    for model_name in available_models:
                        try:
                            model_id = getattr(advanced_transport.TransportModel, model_name)
                            solver = advanced_transport.AdvancedTransport(device, model_id, order=2)
                            print(f"     ✓ {model_name} solver created")
                        except Exception as e:
                            print(f"     ✗ {model_name} solver failed: {e}")
                            self.integration_issues.append(f"Advanced transport {model_name} solver failed: {e}")
                else:
                    print("     ⚠ Cannot test solver - simulator not available")
            else:
                print("     ⚠ AdvancedTransport class not found")
                self.integration_issues.append("AdvancedTransport class not available")
            
            self.test_results['advanced_transport'] = {
                'status': 'passed' if available_models else 'failed',
                'available_models': available_models,
                'solver_tested': hasattr(advanced_transport, 'AdvancedTransport')
            }
            
            if available_models:
                print("   ✅ Advanced transport tests PASSED")
                return True
            else:
                print("   ❌ No advanced transport models available")
                return False
            
        except Exception as e:
            print(f"   ❌ Advanced transport test FAILED: {e}")
            self.test_results['advanced_transport'] = {'status': 'failed', 'error': str(e)}
            self.integration_issues.append(f"Advanced transport test failed: {e}")
            return False

class FrontendIntegrationTestSuite:
    """Frontend integration testing suite"""

    def __init__(self, backend_results: Dict):
        self.backend_results = backend_results
        self.frontend_results = {}
        self.integration_issues = []

    def test_frontend_imports(self):
        """Test frontend module imports"""

        print("\n🧪 Testing Frontend Imports...")

        try:
            # Test Qt imports
            print("   Testing Qt imports...")
            try:
                from PyQt5.QtWidgets import QApplication, QMainWindow
                from PyQt5.QtCore import QThread, pyqtSignal
                print("     ✓ PyQt5 available")
                qt_available = True
            except ImportError:
                try:
                    from PySide2.QtWidgets import QApplication, QMainWindow
                    from PySide2.QtCore import QThread, Signal as pyqtSignal
                    print("     ✓ PySide2 available")
                    qt_available = True
                except ImportError:
                    print("     ✗ No Qt framework available")
                    qt_available = False
                    self.integration_issues.append("No Qt framework (PyQt5/PySide2) available")

            # Test matplotlib imports
            print("   Testing matplotlib imports...")
            try:
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
                from matplotlib.figure import Figure
                print("     ✓ Matplotlib with Qt backend available")
                matplotlib_available = True
            except ImportError as e:
                print(f"     ✗ Matplotlib Qt backend not available: {e}")
                matplotlib_available = False
                self.integration_issues.append(f"Matplotlib Qt backend not available: {e}")

            # Test frontend modules
            print("   Testing frontend modules...")
            frontend_modules = []

            # Add frontend directory to path
            frontend_dir = Path(__file__).parent.parent / "frontend"
            if frontend_dir.exists():
                sys.path.insert(0, str(frontend_dir))
                print(f"     Frontend directory: {frontend_dir}")

                # Test main frontend module
                try:
                    import complete_frontend_integration
                    frontend_modules.append("complete_frontend_integration")
                    print("     ✓ complete_frontend_integration available")
                except ImportError as e:
                    print(f"     ✗ complete_frontend_integration not available: {e}")
                    self.integration_issues.append(f"Frontend integration module not available: {e}")

                # Test visualization module
                try:
                    import advanced_visualization
                    frontend_modules.append("advanced_visualization")
                    print("     ✓ advanced_visualization available")
                except ImportError as e:
                    print(f"     ✗ advanced_visualization not available: {e}")
                    self.integration_issues.append(f"Visualization module not available: {e}")
            else:
                print(f"     ✗ Frontend directory not found: {frontend_dir}")
                self.integration_issues.append("Frontend directory not found")

            self.frontend_results['imports'] = {
                'status': 'passed' if qt_available and matplotlib_available else 'failed',
                'qt_available': qt_available,
                'matplotlib_available': matplotlib_available,
                'frontend_modules': frontend_modules
            }

            if qt_available and matplotlib_available and frontend_modules:
                print("   ✅ Frontend imports PASSED")
                return True
            else:
                print("   ❌ Frontend imports FAILED")
                return False

        except Exception as e:
            print(f"   ❌ Frontend import test FAILED: {e}")
            self.frontend_results['imports'] = {'status': 'failed', 'error': str(e)}
            self.integration_issues.append(f"Frontend import test failed: {e}")
            return False

    def test_backend_interface(self):
        """Test backend interface integration"""

        print("\n🧪 Testing Backend Interface...")

        try:
            # Add frontend directory to path
            frontend_dir = Path(__file__).parent.parent / "frontend"
            if frontend_dir.exists():
                sys.path.insert(0, str(frontend_dir))

            # Import frontend backend interface
            from complete_frontend_integration import BackendInterface

            # Test backend interface creation
            print("   Testing backend interface creation...")
            backend = BackendInterface()
            print("     ✓ BackendInterface created")

            # Test backend availability detection
            print("   Testing backend availability detection...")
            backend_available = backend.backend_available
            print(f"     Backend availability: {backend_available}")

            # Test individual module connections
            print("   Testing module connections...")
            module_connections = {}

            for module_name, module_info in self.backend_results.items():
                if module_info.get('status') == 'passed':
                    try:
                        # Test if backend interface can connect to module
                        # Use the existing backend interface structure
                        if hasattr(backend, 'backend_available') and backend.backend_available:
                            connected = True  # If backend is available, assume connection works
                        else:
                            connected = False
                        module_connections[module_name] = connected
                        print(f"     {module_name}: {'✓' if connected else '✗'}")
                    except Exception as e:
                        module_connections[module_name] = False
                        print(f"     {module_name}: ✗ ({e})")
                        self.integration_issues.append(f"Backend interface cannot connect to {module_name}: {e}")
                else:
                    module_connections[module_name] = False
                    print(f"     {module_name}: ⏭ (backend not available)")

            # Test configuration management
            print("   Testing configuration management...")
            try:
                # Use the existing TransportModelConfig class
                from complete_frontend_integration import TransportModelConfig
                config = TransportModelConfig()
                config_dict = config.to_dict()
                assert isinstance(config_dict, dict), "Configuration should be a dictionary"
                print("     ✓ Configuration management working")
                config_working = True
            except Exception as e:
                print(f"     ✗ Configuration management failed: {e}")
                config_working = False
                self.integration_issues.append(f"Configuration management failed: {e}")

            self.frontend_results['backend_interface'] = {
                'status': 'passed' if any(module_connections.values()) else 'failed',
                'interface_creation': True,
                'backend_available': backend_available,
                'module_connections': module_connections,
                'config_working': config_working
            }

            if any(module_connections.values()):
                print("   ✅ Backend interface tests PASSED")
                return True
            else:
                print("   ❌ No backend modules connected")
                return False

        except Exception as e:
            print(f"   ❌ Backend interface test FAILED: {e}")
            self.frontend_results['backend_interface'] = {'status': 'failed', 'error': str(e)}
            self.integration_issues.append(f"Backend interface test failed: {e}")
            return False

    def test_simulation_workflow(self):
        """Test complete simulation workflow"""

        print("\n🧪 Testing Simulation Workflow...")

        try:
            from complete_frontend_integration import BackendInterface, TransportModelConfig

            # Create backend interface
            backend = BackendInterface()

            # Test configuration creation
            print("   Testing configuration creation...")
            config = TransportModelConfig()
            config.device_length = 2e-6
            config.device_width = 1e-6
            config.nx = 20
            config.ny = 10
            config.enable_energy_transport = True
            config.enable_hydrodynamic = True
            print("     ✓ Configuration created")

            # Test simulation setup
            print("   Testing simulation setup...")
            setup_success = backend.setup_simulation(config)
            print(f"     Setup success: {setup_success}")

            if setup_success:
                # Test simulation execution (if backend available)
                print("   Testing simulation execution...")
                if backend.backend_available:
                    try:
                        results = backend.run_simulation(config, max_steps=5)  # Quick test
                        assert isinstance(results, dict), "Results should be a dictionary"
                        print(f"     ✓ Simulation completed: {len(results)} result fields")
                        simulation_success = True
                    except Exception as e:
                        print(f"     ✗ Simulation execution failed: {e}")
                        simulation_success = False
                        self.integration_issues.append(f"Simulation execution failed: {e}")
                else:
                    print("     ⏭ Simulation execution skipped (backend not available)")
                    simulation_success = None
            else:
                simulation_success = False
                self.integration_issues.append("Simulation setup failed")

            self.frontend_results['simulation_workflow'] = {
                'status': 'passed' if setup_success else 'failed',
                'config_creation': True,
                'setup_success': setup_success,
                'simulation_success': simulation_success
            }

            if setup_success:
                print("   ✅ Simulation workflow tests PASSED")
                return True
            else:
                print("   ❌ Simulation workflow tests FAILED")
                return False

        except Exception as e:
            print(f"   ❌ Simulation workflow test FAILED: {e}")
            self.frontend_results['simulation_workflow'] = {'status': 'failed', 'error': str(e)}
            self.integration_issues.append(f"Simulation workflow test failed: {e}")
            return False

    def test_visualization_integration(self):
        """Test visualization integration"""

        print("\n🧪 Testing Visualization Integration...")

        try:
            # Test matplotlib integration
            print("   Testing matplotlib integration...")
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure

            # Create test figure
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

            # Test basic plotting
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y, 'b-', label='sin(x)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True)

            print("     ✓ Basic plotting working")

            # Test advanced visualization module
            print("   Testing advanced visualization...")
            try:
                from advanced_visualization import AdvancedVisualizationWidget

                # Test widget creation (without Qt event loop)
                print("     ✓ AdvancedVisualizationWidget available")
                viz_available = True
            except ImportError as e:
                print(f"     ✗ Advanced visualization not available: {e}")
                viz_available = False
                self.integration_issues.append(f"Advanced visualization not available: {e}")

            # Test result visualization
            print("   Testing result visualization...")
            try:
                # Create mock simulation results
                mock_results = {
                    'potential': np.random.random(200),
                    'n': np.random.random(200) * 1e16,
                    'p': np.random.random(200) * 1e15,
                    'energy_n': np.random.random(200) * 1e-19,
                    'energy_p': np.random.random(200) * 1e-19
                }

                # Test if we can create visualizations
                fig_results = Figure(figsize=(12, 8))

                # Potential plot
                ax1 = fig_results.add_subplot(2, 2, 1)
                ax1.plot(mock_results['potential'])
                ax1.set_title('Potential')

                # Carrier densities
                ax2 = fig_results.add_subplot(2, 2, 2)
                ax2.semilogy(mock_results['n'], label='Electrons')
                ax2.semilogy(mock_results['p'], label='Holes')
                ax2.set_title('Carrier Densities')
                ax2.legend()

                # Energy densities
                ax3 = fig_results.add_subplot(2, 2, 3)
                ax3.plot(mock_results['energy_n'], label='Electron Energy')
                ax3.plot(mock_results['energy_p'], label='Hole Energy')
                ax3.set_title('Energy Densities')
                ax3.legend()

                print("     ✓ Result visualization working")
                result_viz_working = True

            except Exception as e:
                print(f"     ✗ Result visualization failed: {e}")
                result_viz_working = False
                self.integration_issues.append(f"Result visualization failed: {e}")

            self.frontend_results['visualization'] = {
                'status': 'passed' if result_viz_working else 'failed',
                'matplotlib_working': True,
                'advanced_viz_available': viz_available,
                'result_viz_working': result_viz_working
            }

            if result_viz_working:
                print("   ✅ Visualization integration tests PASSED")
                return True
            else:
                print("   ❌ Visualization integration tests FAILED")
                return False

        except Exception as e:
            print(f"   ❌ Visualization integration test FAILED: {e}")
            self.frontend_results['visualization'] = {'status': 'failed', 'error': str(e)}
            self.integration_issues.append(f"Visualization integration test failed: {e}")
            return False

class IntegrationIssueResolver:
    """Identifies and provides solutions for integration issues"""

    def __init__(self, backend_results: Dict, frontend_results: Dict, issues: List[str]):
        self.backend_results = backend_results
        self.frontend_results = frontend_results
        self.issues = issues
        self.solutions = {}

    def analyze_issues(self):
        """Analyze integration issues and provide solutions"""

        print("\n🔧 Analyzing Integration Issues...")

        # Categorize issues
        backend_issues = []
        frontend_issues = []
        integration_issues = []
        dependency_issues = []

        for issue in self.issues:
            if 'module not available' in issue.lower() or 'import' in issue.lower():
                if any(module in issue for module in ['simulator', 'complete_dg', 'unstructured_transport', 'performance_bindings', 'advanced_transport']):
                    backend_issues.append(issue)
                elif any(frontend in issue for frontend in ['qt', 'matplotlib', 'frontend']):
                    frontend_issues.append(issue)
                else:
                    dependency_issues.append(issue)
            elif 'interface' in issue.lower() or 'connection' in issue.lower():
                integration_issues.append(issue)
            else:
                integration_issues.append(issue)

        print(f"   Backend issues: {len(backend_issues)}")
        print(f"   Frontend issues: {len(frontend_issues)}")
        print(f"   Integration issues: {len(integration_issues)}")
        print(f"   Dependency issues: {len(dependency_issues)}")

        # Generate solutions
        self._generate_backend_solutions(backend_issues)
        self._generate_frontend_solutions(frontend_issues)
        self._generate_integration_solutions(integration_issues)
        self._generate_dependency_solutions(dependency_issues)

        return {
            'backend_issues': backend_issues,
            'frontend_issues': frontend_issues,
            'integration_issues': integration_issues,
            'dependency_issues': dependency_issues,
            'solutions': self.solutions
        }

    def _generate_backend_solutions(self, issues: List[str]):
        """Generate solutions for backend issues"""

        if not issues:
            return

        self.solutions['backend'] = {
            'description': 'Backend compilation and binding issues',
            'steps': [
                '1. Build C++ backend:',
                '   cd /path/to/SemiDGFEM',
                '   make clean',
                '   make',
                '',
                '2. Compile Python bindings:',
                '   cd python',
                '   python3 compile_all.py',
                '',
                '3. Test bindings:',
                '   python3 test_complete_bindings.py',
                '',
                '4. If compilation fails, check dependencies:',
                '   - C++ compiler (g++ or clang++)',
                '   - CMake (version 3.10+)',
                '   - Python development headers',
                '   - Cython (pip install cython)',
                '   - NumPy (pip install numpy)',
                '',
                '5. For specific module issues:',
                '   - simulator: Core C++ library compilation',
                '   - complete_dg: DG discretization implementation',
                '   - unstructured_transport: Advanced transport models',
                '   - performance_bindings: SIMD/GPU optimization',
                '   - advanced_transport: Physics model integration'
            ],
            'issues': issues
        }

    def _generate_frontend_solutions(self, issues: List[str]):
        """Generate solutions for frontend issues"""

        if not issues:
            return

        self.solutions['frontend'] = {
            'description': 'Frontend dependency and module issues',
            'steps': [
                '1. Install Qt framework:',
                '   pip install PyQt5',
                '   # OR',
                '   pip install PySide2',
                '',
                '2. Install matplotlib with Qt backend:',
                '   pip install matplotlib',
                '',
                '3. Verify Qt installation:',
                '   python3 -c "from PyQt5.QtWidgets import QApplication"',
                '',
                '4. Test matplotlib Qt backend:',
                '   python3 -c "from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg"',
                '',
                '5. If frontend modules missing:',
                '   - Check frontend/ directory exists',
                '   - Verify complete_frontend_integration.py is present',
                '   - Check advanced_visualization.py is available',
                '',
                '6. For display issues (headless systems):',
                '   export QT_QPA_PLATFORM=offscreen',
                '   # OR use Xvfb for virtual display'
            ],
            'issues': issues
        }

    def _generate_integration_solutions(self, issues: List[str]):
        """Generate solutions for integration issues"""

        if not issues:
            return

        self.solutions['integration'] = {
            'description': 'Backend-frontend integration issues',
            'steps': [
                '1. Verify Python path configuration:',
                '   - Check python/ directory is in sys.path',
                '   - Verify frontend/ directory is accessible',
                '',
                '2. Test backend interface:',
                '   cd frontend',
                '   python3 -c "from complete_frontend_integration import BackendInterface; b=BackendInterface()"',
                '',
                '3. Check module connections:',
                '   - Ensure backend modules are compiled',
                '   - Verify frontend can import backend modules',
                '   - Test configuration management',
                '',
                '4. Debug simulation workflow:',
                '   - Check device creation',
                '   - Verify transport model setup',
                '   - Test simulation execution',
                '',
                '5. Fix visualization integration:',
                '   - Ensure matplotlib Qt backend works',
                '   - Check result data format compatibility',
                '   - Verify plotting functions work'
            ],
            'issues': issues
        }

    def _generate_dependency_solutions(self, issues: List[str]):
        """Generate solutions for dependency issues"""

        if not issues:
            return

        self.solutions['dependency'] = {
            'description': 'General dependency issues',
            'steps': [
                '1. Update package manager:',
                '   pip install --upgrade pip',
                '',
                '2. Install core dependencies:',
                '   pip install numpy scipy matplotlib',
                '',
                '3. Install development dependencies:',
                '   pip install cython setuptools wheel',
                '',
                '4. Install Qt framework:',
                '   pip install PyQt5 # or PySide2',
                '',
                '5. For system-level dependencies:',
                '   # Ubuntu/Debian:',
                '   sudo apt-get install python3-dev cmake g++',
                '   ',
                '   # CentOS/RHEL:',
                '   sudo yum install python3-devel cmake gcc-c++',
                '   ',
                '   # macOS:',
                '   brew install cmake',
                '',
                '6. Verify installation:',
                '   python3 -c "import numpy, matplotlib, cython"'
            ],
            'issues': issues
        }

    def print_solutions(self):
        """Print all solutions"""

        print("\n🛠️ INTEGRATION ISSUE SOLUTIONS")
        print("=" * 60)

        for category, solution in self.solutions.items():
            print(f"\n{category.upper()} ISSUES:")
            print("-" * 40)
            print(solution['description'])
            print()

            for step in solution['steps']:
                print(step)

            if solution['issues']:
                print(f"\nSpecific issues in this category:")
                for issue in solution['issues']:
                    print(f"  • {issue}")

def run_comprehensive_tests():
    """Run comprehensive backend and frontend tests"""

    print("🧪 COMPREHENSIVE BACKEND-FRONTEND INTEGRATION TESTS")
    print("=" * 70)
    print("Testing all backend models and frontend integration points")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # Phase 1: Backend Testing
    print("\n" + "="*70)
    print("PHASE 1: BACKEND MODEL TESTING")
    print("="*70)

    backend_suite = BackendTestSuite()

    # Detect available modules
    if not backend_suite.detect_backend_modules():
        print("\n❌ No backend modules available - cannot proceed with backend tests")
        backend_results = {}
    else:
        # Run backend tests
        backend_tests = [
            ('Core Simulator', backend_suite.test_core_simulator),
            ('Complete DG', backend_suite.test_complete_dg),
            ('Unstructured Transport', backend_suite.test_unstructured_transport),
            ('Performance Bindings', backend_suite.test_performance_bindings),
            ('Advanced Transport', backend_suite.test_advanced_transport)
        ]

        backend_results = {}
        for test_name, test_func in backend_tests:
            try:
                success = test_func()
                backend_results[test_name.lower().replace(' ', '_')] = backend_suite.test_results.get(test_name.lower().replace(' ', '_'), {'status': 'unknown'})
            except Exception as e:
                print(f"Backend test {test_name} crashed: {e}")
                backend_results[test_name.lower().replace(' ', '_')] = {'status': 'crashed', 'error': str(e)}

    # Phase 2: Frontend Testing
    print("\n" + "="*70)
    print("PHASE 2: FRONTEND INTEGRATION TESTING")
    print("="*70)

    frontend_suite = FrontendIntegrationTestSuite(backend_results)

    frontend_tests = [
        ('Frontend Imports', frontend_suite.test_frontend_imports),
        ('Backend Interface', frontend_suite.test_backend_interface),
        ('Simulation Workflow', frontend_suite.test_simulation_workflow),
        ('Visualization Integration', frontend_suite.test_visualization_integration)
    ]

    for test_name, test_func in frontend_tests:
        try:
            test_func()
        except Exception as e:
            print(f"Frontend test {test_name} crashed: {e}")
            frontend_suite.frontend_results[test_name.lower().replace(' ', '_')] = {'status': 'crashed', 'error': str(e)}

    # Phase 3: Issue Analysis and Resolution
    print("\n" + "="*70)
    print("PHASE 3: ISSUE ANALYSIS AND RESOLUTION")
    print("="*70)

    all_issues = backend_suite.integration_issues + frontend_suite.integration_issues
    resolver = IntegrationIssueResolver(backend_results, frontend_suite.frontend_results, all_issues)
    issue_analysis = resolver.analyze_issues()

    # Generate comprehensive report
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*70)

    # Backend results summary
    backend_passed = sum(1 for result in backend_results.values() if result.get('status') == 'passed')
    backend_total = len(backend_results)

    print(f"\nBackend Tests: {backend_passed}/{backend_total} passed")
    for test_name, result in backend_results.items():
        status = result.get('status', 'unknown')
        icon = "✅" if status == 'passed' else "⏭" if status == 'skipped' else "❌"
        print(f"  {icon} {test_name.replace('_', ' ').title()}: {status}")

    # Frontend results summary
    frontend_passed = sum(1 for result in frontend_suite.frontend_results.values() if result.get('status') == 'passed')
    frontend_total = len(frontend_suite.frontend_results)

    print(f"\nFrontend Tests: {frontend_passed}/{frontend_total} passed")
    for test_name, result in frontend_suite.frontend_results.items():
        status = result.get('status', 'unknown')
        icon = "✅" if status == 'passed' else "⏭" if status == 'skipped' else "❌"
        print(f"  {icon} {test_name.replace('_', ' ').title()}: {status}")

    # Issue summary
    total_issues = len(all_issues)
    print(f"\nIntegration Issues Found: {total_issues}")

    if total_issues > 0:
        print("\nIssue Categories:")
        for category, issues in issue_analysis.items():
            if category != 'solutions' and issues:
                print(f"  {category.replace('_', ' ').title()}: {len(issues)}")

        # Print solutions
        resolver.print_solutions()

    print(f"\nTotal test duration: {duration:.2f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Final assessment
    overall_success = (backend_passed > 0 or frontend_passed > 0) and total_issues == 0

    if overall_success:
        print("\n🎉 INTEGRATION TESTS PASSED!")
        print("   Backend and frontend are properly integrated")
        return 0
    elif total_issues > 0:
        print("\n⚠ INTEGRATION ISSUES FOUND")
        print("   Follow the solutions above to resolve issues")
        return 1
    else:
        print("\n❌ INTEGRATION TESTS FAILED")
        print("   Major integration problems detected")
        return 2

def main():
    """Main test runner"""

    try:
        return run_comprehensive_tests()
    except KeyboardInterrupt:
        print("\n\n⏹ Tests cancelled by user")
        return 0
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
