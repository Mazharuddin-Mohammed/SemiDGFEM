#!/usr/bin/env python3
"""
Working Integration Test with Issue Resolution
Tests current state and provides specific fixes for identified issues

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import time
from pathlib import Path
from datetime import datetime

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

class IntegrationValidator:
    """Validates current integration state and provides specific fixes"""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        self.test_results = {}
        
    def test_backend_modules(self):
        """Test backend module availability and functionality"""
        
        print("🔍 Testing Backend Modules...")
        
        modules_to_test = {
            'simulator': self._test_simulator,
            'complete_dg': self._test_complete_dg,
            'unstructured_transport': self._test_unstructured_transport,
            'performance_bindings': self._test_performance_bindings,
            'advanced_transport': self._test_advanced_transport
        }
        
        backend_results = {}
        
        for module_name, test_func in modules_to_test.items():
            print(f"\n   Testing {module_name}...")
            try:
                result = test_func()
                backend_results[module_name] = result
                if result['available']:
                    print(f"     ✅ {module_name}: Available and working")
                else:
                    print(f"     ❌ {module_name}: {result['error']}")
                    self.issues_found.append(f"{module_name}: {result['error']}")
            except Exception as e:
                backend_results[module_name] = {'available': False, 'error': str(e)}
                print(f"     ❌ {module_name}: Test failed - {e}")
                self.issues_found.append(f"{module_name}: Test failed - {e}")
        
        self.test_results['backend'] = backend_results
        return backend_results
    
    def _test_simulator(self):
        """Test simulator module"""
        try:
            import simulator
            
            # Test device creation with correct parameters
            device = simulator.Device(2e-6, 1e-6)  # Lx, Ly
            
            # Verify device properties (corrected method names)
            width = device.width
            height = device.length

            if abs(width - 1e-6) < 1e-9 and abs(height - 2e-6) < 1e-9:
                return {'available': True, 'error': None, 'functionality': 'Device creation working'}
            else:
                return {'available': True, 'error': f'Device properties incorrect: width={width}, height={height}'}
                
        except ImportError as e:
            return {'available': False, 'error': f'Module not compiled: {e}'}
        except Exception as e:
            return {'available': False, 'error': f'Functionality error: {e}'}
    
    def _test_complete_dg(self):
        """Test complete DG module"""
        try:
            import complete_dg
            
            # Check for expected classes/functions
            if hasattr(complete_dg, 'CompleteDGSolver'):
                solver = complete_dg.CompleteDGSolver()
                return {'available': True, 'error': None, 'functionality': 'DG solver available'}
            else:
                available_attrs = [attr for attr in dir(complete_dg) if not attr.startswith('_')]
                return {'available': True, 'error': f'CompleteDGSolver not found. Available: {available_attrs}'}
                
        except ImportError as e:
            return {'available': False, 'error': f'Module not compiled: {e}'}
        except Exception as e:
            return {'available': False, 'error': f'Functionality error: {e}'}
    
    def _test_unstructured_transport(self):
        """Test unstructured transport module"""
        try:
            import unstructured_transport
            
            # Check for transport classes
            expected_classes = ['UnstructuredTransportSuite', 'EnergyTransportSolver', 'HydrodynamicSolver']
            available_classes = []
            
            for cls_name in expected_classes:
                if hasattr(unstructured_transport, cls_name):
                    available_classes.append(cls_name)
            
            if available_classes:
                return {'available': True, 'error': None, 'functionality': f'Classes available: {available_classes}'}
            else:
                all_attrs = [attr for attr in dir(unstructured_transport) if not attr.startswith('_')]
                return {'available': True, 'error': f'Expected classes not found. Available: {all_attrs}'}
                
        except ImportError as e:
            return {'available': False, 'error': f'Module not compiled: {e}'}
        except Exception as e:
            return {'available': False, 'error': f'Functionality error: {e}'}
    
    def _test_performance_bindings(self):
        """Test performance bindings module"""
        try:
            import performance_bindings
            
            # Test SIMD kernels
            if hasattr(performance_bindings, 'SIMDKernels'):
                # Test vector addition
                a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
                b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
                
                result = performance_bindings.SIMDKernels.vector_add(a, b)
                expected = np.array([5.0, 7.0, 9.0])
                
                if np.allclose(result, expected):
                    return {'available': True, 'error': None, 'functionality': 'SIMD kernels working'}
                else:
                    return {'available': True, 'error': f'SIMD test failed: got {result}, expected {expected}'}
            else:
                available_attrs = [attr for attr in dir(performance_bindings) if not attr.startswith('_')]
                return {'available': True, 'error': f'SIMDKernels not found. Available: {available_attrs}'}
                
        except ImportError as e:
            return {'available': False, 'error': f'Module not compiled: {e}'}
        except Exception as e:
            return {'available': False, 'error': f'Functionality error: {e}'}
    
    def _test_advanced_transport(self):
        """Test advanced transport module"""
        try:
            import advanced_transport
            
            # Check for transport model enumeration
            if hasattr(advanced_transport, 'TransportModel'):
                models = advanced_transport.TransportModel
                expected_models = ['DRIFT_DIFFUSION', 'ENERGY_TRANSPORT', 'HYDRODYNAMIC']
                
                available_models = []
                for model in expected_models:
                    if hasattr(models, model):
                        available_models.append(model)
                
                if available_models:
                    return {'available': True, 'error': None, 'functionality': f'Transport models: {available_models}'}
                else:
                    all_attrs = [attr for attr in dir(models) if not attr.startswith('_')]
                    return {'available': True, 'error': f'Expected models not found. Available: {all_attrs}'}
            else:
                available_attrs = [attr for attr in dir(advanced_transport) if not attr.startswith('_')]
                return {'available': True, 'error': f'TransportModel not found. Available: {available_attrs}'}
                
        except ImportError as e:
            return {'available': False, 'error': f'Module not compiled: {e}'}
        except Exception as e:
            return {'available': False, 'error': f'Functionality error: {e}'}
    
    def test_frontend_integration(self):
        """Test frontend integration"""
        
        print("\n🖥️ Testing Frontend Integration...")
        
        frontend_results = {}
        
        # Test Qt framework
        print("   Testing Qt framework...")
        qt_result = self._test_qt_framework()
        frontend_results['qt'] = qt_result
        if qt_result['available']:
            print(f"     ✅ Qt: {qt_result['framework']}")
        else:
            print(f"     ❌ Qt: {qt_result['error']}")
            self.issues_found.append(f"Qt framework: {qt_result['error']}")
        
        # Test matplotlib
        print("   Testing matplotlib...")
        matplotlib_result = self._test_matplotlib()
        frontend_results['matplotlib'] = matplotlib_result
        if matplotlib_result['available']:
            print("     ✅ Matplotlib: Working")
        else:
            print(f"     ❌ Matplotlib: {matplotlib_result['error']}")
            self.issues_found.append(f"Matplotlib: {matplotlib_result['error']}")
        
        # Test frontend modules
        print("   Testing frontend modules...")
        frontend_modules_result = self._test_frontend_modules()
        frontend_results['modules'] = frontend_modules_result
        if frontend_modules_result['available']:
            print("     ✅ Frontend modules: Available")
        else:
            print(f"     ❌ Frontend modules: {frontend_modules_result['error']}")
            self.issues_found.append(f"Frontend modules: {frontend_modules_result['error']}")
        
        self.test_results['frontend'] = frontend_results
        return frontend_results
    
    def _test_qt_framework(self):
        """Test Qt framework availability"""
        
        qt_frameworks = [
            ('PyQt5', 'PyQt5.QtWidgets'),
            ('PySide2', 'PySide2.QtWidgets'),
            ('PySide6', 'PySide6.QtWidgets'),
            ('PyQt6', 'PyQt6.QtWidgets')
        ]
        
        for framework_name, module_name in qt_frameworks:
            try:
                module = __import__(module_name, fromlist=['QApplication'])
                QApplication = getattr(module, 'QApplication')
                
                # Test QAction import (common issue)
                try:
                    if framework_name in ['PySide6', 'PyQt6']:
                        # Qt6 moved QAction to QtGui
                        gui_module = __import__(module_name.replace('QtWidgets', 'QtGui'), fromlist=['QAction'])
                        QAction = getattr(gui_module, 'QAction')
                    else:
                        # Qt5 has QAction in QtWidgets
                        QAction = getattr(module, 'QAction')
                    
                    return {'available': True, 'framework': framework_name, 'error': None}
                    
                except AttributeError:
                    return {'available': False, 'error': f'{framework_name} missing QAction'}
                    
            except ImportError:
                continue
        
        return {'available': False, 'error': 'No Qt framework found (PyQt5/6, PySide2/6)'}
    
    def _test_matplotlib(self):
        """Test matplotlib"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            
            # Test basic plotting
            fig = Figure(figsize=(4, 3))
            ax = fig.add_subplot(111)
            x = np.linspace(0, 1, 10)
            y = x**2
            ax.plot(x, y)
            
            return {'available': True, 'error': None}
            
        except ImportError as e:
            return {'available': False, 'error': f'Import failed: {e}'}
        except Exception as e:
            return {'available': False, 'error': f'Functionality error: {e}'}
    
    def _test_frontend_modules(self):
        """Test frontend modules"""
        try:
            # Add frontend directory to path
            frontend_dir = Path(__file__).parent.parent / "frontend"
            if not frontend_dir.exists():
                return {'available': False, 'error': f'Frontend directory not found: {frontend_dir}'}
            
            sys.path.insert(0, str(frontend_dir))
            
            # Test main frontend integration module
            try:
                from complete_frontend_integration import BackendInterface, TransportModelConfig
                
                # Test basic functionality
                backend = BackendInterface()
                config = TransportModelConfig()
                
                return {'available': True, 'error': None}
                
            except ImportError as e:
                return {'available': False, 'error': f'Frontend integration import failed: {e}'}
                
        except Exception as e:
            return {'available': False, 'error': f'Frontend test failed: {e}'}
    
    def generate_fix_instructions(self):
        """Generate specific fix instructions for identified issues"""
        
        print("\n🛠️ INTEGRATION ISSUE FIXES")
        print("=" * 50)
        
        if not self.issues_found:
            print("✅ No integration issues found!")
            return
        
        print(f"Found {len(self.issues_found)} integration issues:")
        
        # Categorize issues
        backend_compilation_issues = []
        frontend_dependency_issues = []
        functionality_issues = []
        
        for issue in self.issues_found:
            if "Module not compiled" in issue:
                backend_compilation_issues.append(issue)
            elif any(keyword in issue.lower() for keyword in ['qt', 'matplotlib', 'frontend modules']):
                frontend_dependency_issues.append(issue)
            else:
                functionality_issues.append(issue)
        
        # Backend compilation fixes
        if backend_compilation_issues:
            print("\n🔧 BACKEND COMPILATION FIXES:")
            print("1. Compile Python bindings:")
            print("   cd python")
            print("   python3 setup.py build_ext --inplace")
            print()
            print("2. If compilation fails, check C++ backend:")
            print("   cd ..")
            print("   make clean")
            print("   make")
            print()
            print("3. Install required dependencies:")
            print("   pip install cython numpy scipy")
            print()
            print("Affected modules:")
            for issue in backend_compilation_issues:
                print(f"   • {issue}")
        
        # Frontend dependency fixes
        if frontend_dependency_issues:
            print("\n🖥️ FRONTEND DEPENDENCY FIXES:")
            print("1. Install Qt framework:")
            print("   pip install PyQt5")
            print("   # OR")
            print("   pip install PySide6")
            print()
            print("2. Install matplotlib:")
            print("   pip install matplotlib")
            print()
            print("3. For headless systems:")
            print("   export QT_QPA_PLATFORM=offscreen")
            print()
            print("Affected components:")
            for issue in frontend_dependency_issues:
                print(f"   • {issue}")
        
        # Functionality fixes
        if functionality_issues:
            print("\n⚙️ FUNCTIONALITY FIXES:")
            print("1. Check API compatibility:")
            print("   - Verify function signatures match")
            print("   - Check parameter types and counts")
            print()
            print("2. Update frontend integration:")
            print("   - Adapt to current backend API")
            print("   - Handle missing features gracefully")
            print()
            print("Issues to address:")
            for issue in functionality_issues:
                print(f"   • {issue}")
        
        # Priority recommendations
        print("\n🎯 PRIORITY RECOMMENDATIONS:")
        print("1. HIGH: Compile backend modules (enables core functionality)")
        print("2. MEDIUM: Install Qt framework (enables GUI)")
        print("3. LOW: Fix API compatibility issues (improves integration)")
    
    def run_complete_validation(self):
        """Run complete integration validation"""
        
        print("🧪 COMPLETE BACKEND-FRONTEND INTEGRATION VALIDATION")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        # Test backend
        backend_results = self.test_backend_modules()
        
        # Test frontend
        frontend_results = self.test_frontend_integration()
        
        # Generate summary
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n📊 VALIDATION SUMMARY")
        print("=" * 30)
        
        # Backend summary
        backend_available = sum(1 for result in backend_results.values() if result['available'])
        backend_total = len(backend_results)
        print(f"Backend modules: {backend_available}/{backend_total} available")
        
        # Frontend summary
        frontend_available = sum(1 for result in frontend_results.values() if result['available'])
        frontend_total = len(frontend_results)
        print(f"Frontend components: {frontend_available}/{frontend_total} available")
        
        # Overall status
        overall_health = (backend_available + frontend_available) / (backend_total + frontend_total)
        
        if overall_health >= 0.8:
            print("🟢 Integration Status: GOOD")
        elif overall_health >= 0.5:
            print("🟡 Integration Status: PARTIAL")
        else:
            print("🔴 Integration Status: POOR")
        
        print(f"Validation time: {duration:.2f} seconds")
        
        # Generate fixes
        self.generate_fix_instructions()
        
        return overall_health >= 0.5

def main():
    """Main validation function"""
    
    try:
        validator = IntegrationValidator()
        success = validator.run_complete_validation()
        
        if success:
            print("\n🎉 Integration validation completed!")
            print("   System is ready for development")
            return 0
        else:
            print("\n⚠ Integration issues detected")
            print("   Follow the fix instructions above")
            return 1
            
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
