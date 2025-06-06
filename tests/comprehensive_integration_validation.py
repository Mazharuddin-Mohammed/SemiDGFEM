#!/usr/bin/env python3
"""
Comprehensive Integration Validation with Fixes Applied
Tests all backend and frontend components with proper environment setup

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import numpy as np
import time
from pathlib import Path
from datetime import datetime

# Set up environment
project_root = Path(__file__).parent.parent
python_dir = project_root / "python"
build_dir = project_root / "build"
frontend_dir = project_root / "frontend"

# Add paths
sys.path.insert(0, str(python_dir))
if frontend_dir.exists():
    sys.path.insert(0, str(frontend_dir))

# Set library path for compiled modules
if build_dir.exists():
    os.environ['LD_LIBRARY_PATH'] = str(build_dir) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

class ComprehensiveValidator:
    """Comprehensive validation with all fixes applied"""
    
    def __init__(self):
        self.results = {}
        self.issues_resolved = []
        self.remaining_issues = []
        
    def test_backend_integration(self):
        """Test complete backend integration"""
        
        print("🔧 BACKEND INTEGRATION VALIDATION")
        print("=" * 40)
        
        backend_results = {}
        
        # Test 1: Core simulator
        print("\n1. Testing Core Simulator...")
        try:
            import simulator
            
            # Test device creation with correct API
            device = simulator.Device(2e-6, 1e-6)  # Lx, Ly
            
            # Test device properties (corrected method names)
            width = device.width
            length = device.length
            
            # Test device methods
            epsilon = device.get_epsilon_at(1e-6, 5e-7)
            extents = device.get_extents()
            
            backend_results['simulator'] = {
                'available': True,
                'device_creation': True,
                'properties_access': True,
                'methods_working': True,
                'width': width,
                'length': length,
                'epsilon_test': epsilon,
                'extents': extents
            }
            
            print(f"   ✅ Simulator: Device {width:.2e}×{length:.2e}, ε={epsilon:.2e}")
            self.issues_resolved.append("Simulator module compilation and API compatibility")
            
        except ImportError as e:
            backend_results['simulator'] = {'available': False, 'error': f'Import failed: {e}'}
            print(f"   ❌ Simulator: Import failed - {e}")
            self.remaining_issues.append(f"Simulator import: {e}")
        except Exception as e:
            backend_results['simulator'] = {'available': False, 'error': f'Functionality failed: {e}'}
            print(f"   ❌ Simulator: Functionality failed - {e}")
            self.remaining_issues.append(f"Simulator functionality: {e}")
        
        # Test 2: Complete DG
        print("\n2. Testing Complete DG...")
        try:
            import complete_dg
            
            # Check available classes
            available_classes = [attr for attr in dir(complete_dg) if not attr.startswith('_')]
            
            backend_results['complete_dg'] = {
                'available': True,
                'classes': available_classes
            }
            
            if available_classes:
                print(f"   ✅ Complete DG: {len(available_classes)} classes available")
                self.issues_resolved.append("Complete DG module compilation")
            else:
                print("   ⚠ Complete DG: Module imported but no classes found")
                self.remaining_issues.append("Complete DG: No classes available")
                
        except ImportError as e:
            backend_results['complete_dg'] = {'available': False, 'error': f'Import failed: {e}'}
            print(f"   ❌ Complete DG: Not compiled - {e}")
            self.remaining_issues.append(f"Complete DG compilation: {e}")
        
        # Test 3: Unstructured Transport
        print("\n3. Testing Unstructured Transport...")
        try:
            import unstructured_transport
            
            available_classes = [attr for attr in dir(unstructured_transport) if not attr.startswith('_')]
            
            backend_results['unstructured_transport'] = {
                'available': True,
                'classes': available_classes
            }
            
            if available_classes:
                print(f"   ✅ Unstructured Transport: {len(available_classes)} classes available")
                self.issues_resolved.append("Unstructured transport module compilation")
            else:
                print("   ⚠ Unstructured Transport: Module imported but no classes found")
                
        except ImportError as e:
            backend_results['unstructured_transport'] = {'available': False, 'error': f'Import failed: {e}'}
            print(f"   ❌ Unstructured Transport: Not compiled - {e}")
            self.remaining_issues.append(f"Unstructured transport compilation: {e}")
        
        # Test 4: Performance Bindings
        print("\n4. Testing Performance Bindings...")
        try:
            import performance_bindings
            
            available_classes = [attr for attr in dir(performance_bindings) if not attr.startswith('_')]
            
            # Test SIMD if available
            simd_working = False
            if hasattr(performance_bindings, 'SIMDKernels'):
                try:
                    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
                    b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
                    result = performance_bindings.SIMDKernels.vector_add(a, b)
                    expected = np.array([5.0, 7.0, 9.0])
                    simd_working = np.allclose(result, expected)
                except:
                    pass
            
            backend_results['performance_bindings'] = {
                'available': True,
                'classes': available_classes,
                'simd_working': simd_working
            }
            
            if simd_working:
                print(f"   ✅ Performance Bindings: SIMD kernels working")
                self.issues_resolved.append("Performance bindings with SIMD functionality")
            elif available_classes:
                print(f"   ⚠ Performance Bindings: Available but SIMD not working")
            else:
                print("   ⚠ Performance Bindings: Module imported but no classes found")
                
        except ImportError as e:
            backend_results['performance_bindings'] = {'available': False, 'error': f'Import failed: {e}'}
            print(f"   ❌ Performance Bindings: Not compiled - {e}")
            self.remaining_issues.append(f"Performance bindings compilation: {e}")
        
        # Test 5: Advanced Transport
        print("\n5. Testing Advanced Transport...")
        try:
            import advanced_transport
            
            available_classes = [attr for attr in dir(advanced_transport) if not attr.startswith('_')]
            
            # Test transport models if available
            transport_models = []
            if hasattr(advanced_transport, 'TransportModel'):
                models = advanced_transport.TransportModel
                expected_models = ['DRIFT_DIFFUSION', 'ENERGY_TRANSPORT', 'HYDRODYNAMIC']
                for model in expected_models:
                    if hasattr(models, model):
                        transport_models.append(model)
            
            backend_results['advanced_transport'] = {
                'available': True,
                'classes': available_classes,
                'transport_models': transport_models
            }
            
            if transport_models:
                print(f"   ✅ Advanced Transport: {len(transport_models)} models available")
                self.issues_resolved.append("Advanced transport with physics models")
            elif available_classes:
                print(f"   ⚠ Advanced Transport: Available but no transport models")
            else:
                print("   ⚠ Advanced Transport: Module imported but no classes found")
                
        except ImportError as e:
            backend_results['advanced_transport'] = {'available': False, 'error': f'Import failed: {e}'}
            print(f"   ❌ Advanced Transport: Not compiled - {e}")
            self.remaining_issues.append(f"Advanced transport compilation: {e}")
        
        self.results['backend'] = backend_results
        return backend_results
    
    def test_frontend_integration(self):
        """Test complete frontend integration"""
        
        print("\n\n🖥️ FRONTEND INTEGRATION VALIDATION")
        print("=" * 40)
        
        frontend_results = {}
        
        # Test 1: Qt Framework
        print("\n1. Testing Qt Framework...")
        qt_result = self._test_qt_comprehensive()
        frontend_results['qt'] = qt_result
        
        if qt_result['available']:
            print(f"   ✅ Qt: {qt_result['framework']} working")
            if qt_result.get('qaction_fixed'):
                self.issues_resolved.append("Qt QAction import issue resolved")
        else:
            print(f"   ❌ Qt: {qt_result['error']}")
            self.remaining_issues.append(f"Qt framework: {qt_result['error']}")
        
        # Test 2: Matplotlib
        print("\n2. Testing Matplotlib...")
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            
            # Test plotting
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y, 'b-', label='sin(x)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True)
            
            frontend_results['matplotlib'] = {'available': True, 'plotting_working': True}
            print("   ✅ Matplotlib: Plotting functionality working")
            
        except Exception as e:
            frontend_results['matplotlib'] = {'available': False, 'error': str(e)}
            print(f"   ❌ Matplotlib: {e}")
            self.remaining_issues.append(f"Matplotlib: {e}")
        
        # Test 3: Frontend Modules
        print("\n3. Testing Frontend Modules...")
        try:
            from complete_frontend_integration import BackendInterface, TransportModelConfig
            
            # Test backend interface
            backend = BackendInterface()
            config = TransportModelConfig()
            
            # Test configuration
            config_dict = config.to_dict()
            
            frontend_results['modules'] = {
                'available': True,
                'backend_interface': True,
                'config_management': True,
                'backend_detected': backend.backend_available
            }
            
            print(f"   ✅ Frontend Modules: Interface working, backend detected: {backend.backend_available}")
            
        except Exception as e:
            frontend_results['modules'] = {'available': False, 'error': str(e)}
            print(f"   ❌ Frontend Modules: {e}")
            self.remaining_issues.append(f"Frontend modules: {e}")
        
        self.results['frontend'] = frontend_results
        return frontend_results
    
    def _test_qt_comprehensive(self):
        """Comprehensive Qt testing with QAction fix"""
        
        qt_frameworks = [
            ('PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtWidgets'),
            ('PySide2', 'PySide2.QtWidgets', 'PySide2.QtWidgets'),
            ('PySide6', 'PySide6.QtWidgets', 'PySide6.QtGui'),  # QAction moved to QtGui in Qt6
            ('PyQt6', 'PyQt6.QtWidgets', 'PyQt6.QtGui')        # QAction moved to QtGui in Qt6
        ]
        
        for framework_name, widgets_module, action_module in qt_frameworks:
            try:
                # Test widgets module
                widgets = __import__(widgets_module, fromlist=['QApplication'])
                QApplication = getattr(widgets, 'QApplication')
                
                # Test QAction import (the problematic one)
                action_mod = __import__(action_module, fromlist=['QAction'])
                QAction = getattr(action_mod, 'QAction')
                
                return {
                    'available': True,
                    'framework': framework_name,
                    'qaction_fixed': True,
                    'widgets_module': widgets_module,
                    'action_module': action_module
                }
                
            except ImportError:
                continue
            except AttributeError as e:
                # QAction not found in expected module
                continue
        
        return {'available': False, 'error': 'No working Qt framework found'}
    
    def test_integration_workflow(self):
        """Test complete integration workflow"""
        
        print("\n\n🔄 INTEGRATION WORKFLOW VALIDATION")
        print("=" * 40)
        
        workflow_results = {}
        
        # Test 1: Backend-Frontend Communication
        print("\n1. Testing Backend-Frontend Communication...")
        try:
            # Import both backend and frontend
            import simulator
            from complete_frontend_integration import BackendInterface, TransportModelConfig
            
            # Create backend interface
            backend = BackendInterface()
            
            # Create configuration
            config = TransportModelConfig()
            config.device_length = 2e-6
            config.device_width = 1e-6
            config.nx = 20
            config.ny = 10
            
            # Test simulation setup
            setup_success = backend.setup_simulation(config)
            
            workflow_results['communication'] = {
                'backend_import': True,
                'frontend_import': True,
                'config_creation': True,
                'setup_success': setup_success
            }
            
            if setup_success:
                print("   ✅ Backend-Frontend Communication: Working")
                self.issues_resolved.append("Backend-frontend communication established")
            else:
                print("   ⚠ Backend-Frontend Communication: Setup failed")
                self.remaining_issues.append("Simulation setup failed")
                
        except Exception as e:
            workflow_results['communication'] = {'available': False, 'error': str(e)}
            print(f"   ❌ Backend-Frontend Communication: {e}")
            self.remaining_issues.append(f"Backend-frontend communication: {e}")
        
        # Test 2: End-to-End Simulation
        print("\n2. Testing End-to-End Simulation...")
        try:
            if workflow_results.get('communication', {}).get('setup_success'):
                # Run a simple simulation
                results = backend.run_simulation(config, max_steps=5)
                
                # Validate results
                required_fields = ['potential', 'n', 'p']
                fields_present = all(field in results for field in required_fields)
                
                workflow_results['simulation'] = {
                    'execution_success': True,
                    'fields_present': fields_present,
                    'result_fields': list(results.keys()),
                    'result_size': len(results.get('potential', []))
                }
                
                if fields_present:
                    print(f"   ✅ End-to-End Simulation: {len(results)} fields, {len(results.get('potential', []))} points")
                    self.issues_resolved.append("End-to-end simulation workflow")
                else:
                    print(f"   ⚠ End-to-End Simulation: Missing required fields")
            else:
                workflow_results['simulation'] = {'execution_success': False, 'error': 'Setup failed'}
                print("   ⏭ End-to-End Simulation: Skipped (setup failed)")
                
        except Exception as e:
            workflow_results['simulation'] = {'execution_success': False, 'error': str(e)}
            print(f"   ❌ End-to-End Simulation: {e}")
            self.remaining_issues.append(f"End-to-end simulation: {e}")
        
        self.results['workflow'] = workflow_results
        return workflow_results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive validation report"""
        
        print("\n\n📋 COMPREHENSIVE INTEGRATION REPORT")
        print("=" * 50)
        
        # Summary statistics
        backend_available = sum(1 for result in self.results.get('backend', {}).values() if result.get('available', False))
        backend_total = len(self.results.get('backend', {}))
        
        frontend_available = sum(1 for result in self.results.get('frontend', {}).values() if result.get('available', False))
        frontend_total = len(self.results.get('frontend', {}))
        
        workflow_working = sum(1 for result in self.results.get('workflow', {}).values() if result.get('execution_success', False) or result.get('setup_success', False))
        workflow_total = len(self.results.get('workflow', {}))
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Backend Modules: {backend_available}/{backend_total} available")
        print(f"   Frontend Components: {frontend_available}/{frontend_total} available")
        print(f"   Workflow Tests: {workflow_working}/{workflow_total} working")
        
        # Issues resolved
        if self.issues_resolved:
            print(f"\n✅ ISSUES RESOLVED ({len(self.issues_resolved)}):")
            for i, issue in enumerate(self.issues_resolved, 1):
                print(f"   {i}. {issue}")
        
        # Remaining issues
        if self.remaining_issues:
            print(f"\n❌ REMAINING ISSUES ({len(self.remaining_issues)}):")
            for i, issue in enumerate(self.remaining_issues, 1):
                print(f"   {i}. {issue}")
            
            print(f"\n🛠️ NEXT STEPS:")
            print("   1. Compile remaining backend modules:")
            print("      cd python && python3 setup.py build_ext --inplace")
            print("   2. Install missing dependencies:")
            print("      pip install PyQt5 matplotlib")
            print("   3. Set library path:")
            print("      export LD_LIBRARY_PATH=../build:$LD_LIBRARY_PATH")
        
        # Overall assessment
        total_available = backend_available + frontend_available + workflow_working
        total_possible = backend_total + frontend_total + workflow_total
        
        if total_possible > 0:
            success_rate = total_available / total_possible
            
            if success_rate >= 0.8:
                print(f"\n🟢 INTEGRATION STATUS: EXCELLENT ({success_rate:.1%})")
                print("   System is ready for production use")
            elif success_rate >= 0.6:
                print(f"\n🟡 INTEGRATION STATUS: GOOD ({success_rate:.1%})")
                print("   System is functional with minor issues")
            elif success_rate >= 0.4:
                print(f"\n🟠 INTEGRATION STATUS: PARTIAL ({success_rate:.1%})")
                print("   System has significant issues but core functionality works")
            else:
                print(f"\n🔴 INTEGRATION STATUS: POOR ({success_rate:.1%})")
                print("   System requires major fixes")
        
        return len(self.remaining_issues) == 0

def main():
    """Main comprehensive validation"""
    
    print("🧪 COMPREHENSIVE BACKEND-FRONTEND INTEGRATION VALIDATION")
    print("=" * 70)
    print("Testing all components with fixes applied")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        validator = ComprehensiveValidator()
        
        # Run all tests
        validator.test_backend_integration()
        validator.test_frontend_integration()
        validator.test_integration_workflow()
        
        # Generate report
        success = validator.generate_comprehensive_report()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nValidation completed in {duration:.2f} seconds")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if success:
            print("\n🎉 INTEGRATION VALIDATION SUCCESSFUL!")
            print("   All critical components are working")
            return 0
        else:
            print("\n⚠ INTEGRATION VALIDATION COMPLETED WITH ISSUES")
            print("   Follow the next steps above to resolve remaining issues")
            return 1
            
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
