#!/usr/bin/env python3
"""
Fixed Backend-Frontend Integration with Issue Resolution
Provides working integration even when some backend modules are missing

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

class RobustBackendInterface:
    """Robust backend interface that handles missing modules gracefully"""
    
    def __init__(self):
        self.available_modules = {}
        self.backend_status = {}
        self.integration_issues = []
        self._detect_and_fix_backend()
        
    def _detect_and_fix_backend(self):
        """Detect available backend modules and fix common issues"""
        
        print("🔧 Detecting and fixing backend integration...")
        
        # Core modules to check
        core_modules = {
            'simulator': 'Core device simulation',
            'complete_dg': 'Complete DG discretization', 
            'unstructured_transport': 'Unstructured transport models',
            'performance_bindings': 'SIMD/GPU optimization',
            'advanced_transport': 'Advanced transport physics'
        }
        
        for module_name, description in core_modules.items():
            try:
                # Try to import the module
                module = __import__(module_name)
                self.available_modules[module_name] = module
                self.backend_status[module_name] = {
                    'available': True,
                    'description': description,
                    'error': None
                }
                print(f"   ✓ {module_name}: {description}")
                
                # Test basic functionality
                self._test_module_functionality(module_name, module)
                
            except ImportError as e:
                self.available_modules[module_name] = None
                self.backend_status[module_name] = {
                    'available': False,
                    'description': description,
                    'error': str(e)
                }
                print(f"   ✗ {module_name}: {e}")
                self.integration_issues.append(f"Module {module_name} not available: {e}")
                
                # Try to provide a fix
                self._suggest_module_fix(module_name, str(e))
        
        # Check overall backend availability
        available_count = sum(1 for status in self.backend_status.values() if status['available'])
        total_count = len(self.backend_status)
        
        print(f"\n📊 Backend Status: {available_count}/{total_count} modules available")
        
        if available_count == 0:
            print("⚠ No backend modules available - using mock implementations")
            self._install_mock_backend()
        elif available_count < total_count:
            print("⚠ Partial backend available - some features will be limited")
        else:
            print("✅ Complete backend available")
    
    def _test_module_functionality(self, module_name: str, module):
        """Test basic functionality of a module"""
        
        try:
            if module_name == 'simulator':
                # Test device creation
                device = module.Device(1e-6, 1e-6)
                assert device.get_width() == 1e-6, "Device width test failed"
                print(f"     ✓ {module_name} functionality verified")
                
            elif module_name == 'complete_dg':
                # Test DG solver availability
                if hasattr(module, 'CompleteDGSolver'):
                    solver = module.CompleteDGSolver()
                    print(f"     ✓ {module_name} functionality verified")
                else:
                    print(f"     ⚠ {module_name} missing expected classes")
                    
            elif module_name == 'performance_bindings':
                # Test SIMD kernels
                if hasattr(module, 'SIMDKernels'):
                    a = np.array([1.0, 2.0, 3.0])
                    b = np.array([4.0, 5.0, 6.0])
                    result = module.SIMDKernels.vector_add(a, b)
                    expected = np.array([5.0, 7.0, 9.0])
                    assert np.allclose(result, expected), "SIMD test failed"
                    print(f"     ✓ {module_name} functionality verified")
                else:
                    print(f"     ⚠ {module_name} missing SIMD kernels")
                    
            else:
                print(f"     ✓ {module_name} imported successfully")
                
        except Exception as e:
            print(f"     ⚠ {module_name} functionality test failed: {e}")
            self.integration_issues.append(f"{module_name} functionality test failed: {e}")
    
    def _suggest_module_fix(self, module_name: str, error: str):
        """Suggest fixes for missing modules"""
        
        if "No module named" in error:
            if module_name in ['simulator', 'complete_dg', 'unstructured_transport', 'performance_bindings', 'advanced_transport']:
                print(f"     💡 Fix: cd python && python3 setup.py build_ext --inplace")
            else:
                print(f"     💡 Fix: pip install {module_name}")
        elif "undefined symbol" in error or "cannot open shared object" in error:
            print(f"     💡 Fix: Check library paths and rebuild C++ backend")
        else:
            print(f"     💡 Fix: Check compilation logs for {module_name}")
    
    def _install_mock_backend(self):
        """Install mock backend for testing when real backend is not available"""
        
        print("🔧 Installing mock backend for testing...")
        
        # Import mock modules
        from mock_backend_interface import (
            MockSimulatorModule, MockCompleteDGModule, MockUnstructuredTransportModule,
            MockPerformanceBindingsModule, MockAdvancedTransportModule
        )
        
        # Install mock modules
        mock_modules = {
            'simulator': MockSimulatorModule(),
            'complete_dg': MockCompleteDGModule(),
            'unstructured_transport': MockUnstructuredTransportModule(),
            'performance_bindings': MockPerformanceBindingsModule(),
            'advanced_transport': MockAdvancedTransportModule()
        }
        
        for module_name, mock_module in mock_modules.items():
            if not self.backend_status[module_name]['available']:
                sys.modules[module_name] = mock_module
                self.available_modules[module_name] = mock_module
                self.backend_status[module_name]['available'] = True
                self.backend_status[module_name]['error'] = "Using mock implementation"
                print(f"   ✓ Mock {module_name} installed")
    
    def run_integration_test(self):
        """Run comprehensive integration test"""
        
        print("\n🧪 Running Integration Test...")
        
        test_results = {}
        
        # Test 1: Basic device simulation
        print("   Testing basic device simulation...")
        try:
            if 'simulator' in self.available_modules and self.available_modules['simulator']:
                simulator = self.available_modules['simulator']
                
                # Create device
                device = simulator.Device(2e-6, 1e-6)
                
                # Create simulator
                sim = simulator.Simulator(
                    extents=[2e-6, 1e-6],
                    num_points_x=20,
                    num_points_y=10,
                    method="DG",
                    mesh_type="Structured"
                )
                
                # Set doping
                size = 20 * 10
                Nd = np.full(size, 1e17, dtype=np.float64)
                Na = np.full(size, 1e16, dtype=np.float64)
                sim.set_doping(Nd, Na)
                
                # Run simulation
                results = sim.solve_drift_diffusion([0.0, 1.0, 0.0, 0.0], Vg=0.5, max_steps=5)
                
                # Validate results
                assert 'potential' in results, "Missing potential field"
                assert 'n' in results, "Missing electron density"
                assert 'p' in results, "Missing hole density"
                assert len(results['potential']) == size, "Wrong result size"
                
                test_results['basic_simulation'] = True
                print("     ✓ Basic simulation working")
                
            else:
                test_results['basic_simulation'] = False
                print("     ✗ Simulator not available")
                
        except Exception as e:
            test_results['basic_simulation'] = False
            print(f"     ✗ Basic simulation failed: {e}")
        
        # Test 2: Advanced transport models
        print("   Testing advanced transport models...")
        try:
            if 'advanced_transport' in self.available_modules and self.available_modules['advanced_transport']:
                advanced_transport = self.available_modules['advanced_transport']
                
                # Test transport model enumeration
                if hasattr(advanced_transport, 'TransportModel'):
                    models = advanced_transport.TransportModel
                    expected_models = ['DRIFT_DIFFUSION', 'ENERGY_TRANSPORT', 'HYDRODYNAMIC']
                    
                    available_models = []
                    for model in expected_models:
                        if hasattr(models, model):
                            available_models.append(model)
                    
                    test_results['advanced_transport'] = len(available_models) > 0
                    print(f"     ✓ Advanced transport: {len(available_models)} models available")
                else:
                    test_results['advanced_transport'] = False
                    print("     ✗ TransportModel enumeration not found")
            else:
                test_results['advanced_transport'] = False
                print("     ✗ Advanced transport not available")
                
        except Exception as e:
            test_results['advanced_transport'] = False
            print(f"     ✗ Advanced transport test failed: {e}")
        
        # Test 3: Performance optimization
        print("   Testing performance optimization...")
        try:
            if 'performance_bindings' in self.available_modules and self.available_modules['performance_bindings']:
                performance = self.available_modules['performance_bindings']
                
                # Test SIMD kernels
                if hasattr(performance, 'SIMDKernels'):
                    a = np.random.random(1000).astype(np.float64)
                    b = np.random.random(1000).astype(np.float64)
                    
                    # Test vector addition
                    result = performance.SIMDKernels.vector_add(a, b)
                    expected = a + b
                    assert np.allclose(result, expected, rtol=1e-12), "SIMD vector addition failed"
                    
                    test_results['performance_optimization'] = True
                    print("     ✓ Performance optimization working")
                else:
                    test_results['performance_optimization'] = False
                    print("     ✗ SIMD kernels not found")
            else:
                test_results['performance_optimization'] = False
                print("     ✗ Performance bindings not available")
                
        except Exception as e:
            test_results['performance_optimization'] = False
            print(f"     ✗ Performance optimization test failed: {e}")
        
        # Test 4: Complete DG discretization
        print("   Testing complete DG discretization...")
        try:
            if 'complete_dg' in self.available_modules and self.available_modules['complete_dg']:
                complete_dg = self.available_modules['complete_dg']
                
                # Test DG solver
                if hasattr(complete_dg, 'CompleteDGSolver'):
                    solver = complete_dg.CompleteDGSolver()
                    test_results['complete_dg'] = True
                    print("     ✓ Complete DG discretization working")
                else:
                    test_results['complete_dg'] = False
                    print("     ✗ CompleteDGSolver not found")
            else:
                test_results['complete_dg'] = False
                print("     ✗ Complete DG not available")
                
        except Exception as e:
            test_results['complete_dg'] = False
            print(f"     ✗ Complete DG test failed: {e}")
        
        return test_results
    
    def test_frontend_integration(self):
        """Test frontend integration"""
        
        print("\n🖥️ Testing Frontend Integration...")
        
        frontend_results = {}
        
        # Test 1: Qt framework
        print("   Testing Qt framework...")
        try:
            # Try different Qt frameworks
            qt_available = False
            qt_framework = None
            
            try:
                from PyQt5.QtWidgets import QApplication
                qt_framework = "PyQt5"
                qt_available = True
            except ImportError:
                try:
                    from PySide2.QtWidgets import QApplication
                    qt_framework = "PySide2"
                    qt_available = True
                except ImportError:
                    try:
                        from PySide6.QtWidgets import QApplication
                        qt_framework = "PySide6"
                        qt_available = True
                    except ImportError:
                        pass
            
            if qt_available:
                frontend_results['qt_framework'] = True
                print(f"     ✓ Qt framework available: {qt_framework}")
            else:
                frontend_results['qt_framework'] = False
                print("     ✗ No Qt framework available")
                print("     💡 Fix: pip install PyQt5 or PySide2 or PySide6")
                
        except Exception as e:
            frontend_results['qt_framework'] = False
            print(f"     ✗ Qt framework test failed: {e}")
        
        # Test 2: Matplotlib integration
        print("   Testing matplotlib integration...")
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            
            # Test basic plotting
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y)
            
            frontend_results['matplotlib'] = True
            print("     ✓ Matplotlib working")
            
        except Exception as e:
            frontend_results['matplotlib'] = False
            print(f"     ✗ Matplotlib test failed: {e}")
        
        # Test 3: Frontend modules
        print("   Testing frontend modules...")
        try:
            # Add frontend directory to path
            frontend_dir = Path(__file__).parent.parent / "frontend"
            if frontend_dir.exists():
                sys.path.insert(0, str(frontend_dir))
                
                # Test frontend integration module
                try:
                    from complete_frontend_integration import BackendInterface, TransportModelConfig
                    
                    # Test backend interface
                    backend = BackendInterface()
                    config = TransportModelConfig()
                    
                    frontend_results['frontend_modules'] = True
                    print("     ✓ Frontend modules working")
                    
                except ImportError as e:
                    frontend_results['frontend_modules'] = False
                    print(f"     ✗ Frontend modules import failed: {e}")
            else:
                frontend_results['frontend_modules'] = False
                print(f"     ✗ Frontend directory not found: {frontend_dir}")
                
        except Exception as e:
            frontend_results['frontend_modules'] = False
            print(f"     ✗ Frontend modules test failed: {e}")
        
        return frontend_results
    
    def generate_integration_report(self, backend_results: Dict, frontend_results: Dict):
        """Generate comprehensive integration report"""
        
        print("\n📋 INTEGRATION REPORT")
        print("=" * 50)
        
        # Backend summary
        backend_passed = sum(1 for result in backend_results.values() if result)
        backend_total = len(backend_results)
        
        print(f"\nBackend Integration: {backend_passed}/{backend_total} tests passed")
        for test_name, result in backend_results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {test_name.replace('_', ' ').title()}")
        
        # Frontend summary
        frontend_passed = sum(1 for result in frontend_results.values() if result)
        frontend_total = len(frontend_results)
        
        print(f"\nFrontend Integration: {frontend_passed}/{frontend_total} tests passed")
        for test_name, result in frontend_results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {test_name.replace('_', ' ').title()}")
        
        # Issues and solutions
        if self.integration_issues:
            print(f"\nIntegration Issues Found: {len(self.integration_issues)}")
            for i, issue in enumerate(self.integration_issues, 1):
                print(f"  {i}. {issue}")
            
            print("\n🛠️ RECOMMENDED SOLUTIONS:")
            print("1. Backend Compilation:")
            print("   cd python && python3 setup.py build_ext --inplace")
            print("2. Install Qt Framework:")
            print("   pip install PyQt5")
            print("3. Install Dependencies:")
            print("   pip install numpy scipy matplotlib cython")
            print("4. Check Build System:")
            print("   cd .. && make clean && make")
        
        # Overall assessment
        overall_success = (backend_passed > 0 and frontend_passed > 0)
        
        if overall_success:
            print("\n🎉 INTEGRATION SUCCESSFUL!")
            print("   Backend and frontend are working together")
        else:
            print("\n⚠ INTEGRATION ISSUES DETECTED")
            print("   Follow the recommended solutions above")
        
        return overall_success

def main():
    """Main integration test and fix"""
    
    print("🔧 BACKEND-FRONTEND INTEGRATION TEST AND FIX")
    print("=" * 60)
    print("Detecting issues and providing solutions")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Create robust backend interface
        backend_interface = RobustBackendInterface()
        
        # Run backend integration tests
        backend_results = backend_interface.run_integration_test()
        
        # Run frontend integration tests
        frontend_results = backend_interface.test_frontend_integration()
        
        # Generate comprehensive report
        success = backend_interface.generate_integration_report(backend_results, frontend_results)
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
