#!/usr/bin/env python3
"""
Comprehensive test runner for the SemiDGFEM project.
Runs both C++ and Python tests with detailed reporting.
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path


class TestRunner:
    """Main test runner class."""
    
    def __init__(self, verbose=False, coverage=False, memcheck=False):
        self.verbose = verbose
        self.coverage = coverage
        self.memcheck = memcheck
        self.results = {
            'cpp_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'python_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'total_time': 0
        }
        
        # Find project root
        self.project_root = Path(__file__).parent.parent
        self.test_dir = Path(__file__).parent
        
    def run_cpp_tests(self):
        """Run C++ unit tests using Google Test."""
        print("=" * 60)
        print("Running C++ Unit Tests")
        print("=" * 60)
        
        # Check if test executable exists
        test_executable = self.test_dir / "build" / "run_tests"
        if not test_executable.exists():
            print("Building C++ tests...")
            if not self._build_cpp_tests():
                self.results['cpp_tests']['errors'].append("Failed to build C++ tests")
                return False
        
        try:
            # Run tests
            cmd = [str(test_executable)]
            if self.verbose:
                cmd.append("--gtest_output=xml:cpp_test_results.xml")
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir)
            end_time = time.time()
            
            if self.verbose:
                print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
            
            # Parse results
            if result.returncode == 0:
                self.results['cpp_tests']['passed'] = self._count_cpp_tests_passed(result.stdout)
                print(f"✓ C++ tests passed ({end_time - start_time:.2f}s)")
                return True
            else:
                self.results['cpp_tests']['failed'] = self._count_cpp_tests_failed(result.stdout)
                self.results['cpp_tests']['errors'].append(result.stderr)
                print(f"✗ C++ tests failed ({end_time - start_time:.2f}s)")
                return False
                
        except Exception as e:
            self.results['cpp_tests']['errors'].append(str(e))
            print(f"✗ Error running C++ tests: {e}")
            return False
    
    def run_python_tests(self):
        """Run Python unit tests using unittest."""
        print("\n" + "=" * 60)
        print("Running Python Unit Tests")
        print("=" * 60)
        
        # List of Python test modules
        test_modules = [
            'test_python_simulator.py',
            'test_gui.py'
        ]
        
        total_passed = 0
        total_failed = 0
        
        for test_module in test_modules:
            test_file = self.test_dir / test_module
            if not test_file.exists():
                print(f"Warning: Test file {test_module} not found")
                continue
            
            print(f"\nRunning {test_module}...")
            
            try:
                cmd = [sys.executable, str(test_file)]
                if self.verbose:
                    cmd.append('-v')
                
                start_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir)
                end_time = time.time()
                
                if self.verbose:
                    print("STDOUT:", result.stdout)
                    if result.stderr:
                        print("STDERR:", result.stderr)
                
                # Parse results
                passed, failed = self._parse_python_test_results(result.stdout)
                total_passed += passed
                total_failed += failed
                
                if result.returncode == 0:
                    print(f"  ✓ {test_module}: {passed} passed ({end_time - start_time:.2f}s)")
                else:
                    print(f"  ✗ {test_module}: {passed} passed, {failed} failed ({end_time - start_time:.2f}s)")
                    if result.stderr:
                        self.results['python_tests']['errors'].append(f"{test_module}: {result.stderr}")
                
            except Exception as e:
                print(f"  ✗ Error running {test_module}: {e}")
                self.results['python_tests']['errors'].append(f"{test_module}: {str(e)}")
                total_failed += 1
        
        self.results['python_tests']['passed'] = total_passed
        self.results['python_tests']['failed'] = total_failed
        
        return total_failed == 0
    
    def run_integration_tests(self):
        """Run integration tests."""
        print("\n" + "=" * 60)
        print("Running Integration Tests")
        print("=" * 60)
        
        # Integration tests would test the complete pipeline
        # For now, we'll just verify that the main components can be imported
        
        try:
            # Test C++ library loading
            print("Testing C++ library loading...")
            sys.path.insert(0, str(self.project_root / "python"))
            
            try:
                from simulator import Simulator
                print("  ✓ Ctypes simulator import successful")
            except ImportError as e:
                print(f"  ✗ Ctypes simulator import failed: {e}")
            
            try:
                import simulator_cython
                print("  ✓ Cython simulator import successful")
            except ImportError as e:
                print(f"  ✗ Cython simulator import failed: {e}")
            
            # Test GUI components
            print("Testing GUI components...")
            try:
                import tkinter as tk
                from gui.main_gui_2d import SimulatorGUI2D
                print("  ✓ GUI components import successful")
            except ImportError as e:
                print(f"  ✗ GUI components import failed: {e}")
            
            # Test visualization
            print("Testing visualization components...")
            try:
                from visualization.viz_2d import plot_2d_potential
                print("  ✓ Visualization components import successful")
            except ImportError as e:
                print(f"  ✗ Visualization components import failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"✗ Integration test error: {e}")
            return False
    
    def run_memory_checks(self):
        """Run memory leak detection using valgrind."""
        if not self.memcheck:
            return True
        
        print("\n" + "=" * 60)
        print("Running Memory Leak Detection")
        print("=" * 60)
        
        # Check if valgrind is available
        try:
            subprocess.run(['valgrind', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Valgrind not available, skipping memory checks")
            return True
        
        test_executable = self.test_dir / "build" / "run_tests"
        if not test_executable.exists():
            print("C++ test executable not found, skipping memory checks")
            return True
        
        try:
            cmd = [
                'valgrind',
                '--tool=memcheck',
                '--leak-check=full',
                '--show-leak-kinds=all',
                '--track-origins=yes',
                '--error-exitcode=1',
                str(test_executable)
            ]
            
            print("Running valgrind (this may take a while)...")
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir)
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"✓ No memory leaks detected ({end_time - start_time:.2f}s)")
                return True
            else:
                print(f"✗ Memory leaks detected ({end_time - start_time:.2f}s)")
                if self.verbose:
                    print("Valgrind output:", result.stderr)
                return False
                
        except Exception as e:
            print(f"✗ Error running valgrind: {e}")
            return False
    
    def generate_coverage_report(self):
        """Generate code coverage report."""
        if not self.coverage:
            return True
        
        print("\n" + "=" * 60)
        print("Generating Coverage Report")
        print("=" * 60)
        
        try:
            # Run C++ coverage
            build_dir = self.test_dir / "build"
            if build_dir.exists():
                subprocess.run(['make', 'coverage'], cwd=build_dir, check=True)
                print("✓ C++ coverage report generated")
            
            # Run Python coverage
            cmd = [
                'python', '-m', 'coverage', 'run', '--source=../python',
                'test_python_simulator.py'
            ]
            subprocess.run(cmd, cwd=self.test_dir, check=True)
            
            cmd = ['python', '-m', 'coverage', 'html']
            subprocess.run(cmd, cwd=self.test_dir, check=True)
            print("✓ Python coverage report generated")
            
            return True
            
        except Exception as e:
            print(f"✗ Error generating coverage report: {e}")
            return False
    
    def _build_cpp_tests(self):
        """Build C++ tests."""
        build_dir = self.test_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        try:
            # Configure
            subprocess.run([
                'cmake', '..',
                '-DCMAKE_BUILD_TYPE=Debug',
                '-DENABLE_COVERAGE=ON' if self.coverage else '-DENABLE_COVERAGE=OFF'
            ], cwd=build_dir, check=True)
            
            # Build
            subprocess.run(['make', '-j4'], cwd=build_dir, check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Build failed: {e}")
            return False
    
    def _count_cpp_tests_passed(self, output):
        """Count passed C++ tests from output."""
        # Parse Google Test output
        lines = output.split('\n')
        for line in lines:
            if 'tests from' in line and 'test cases' in line:
                # Extract number of passed tests
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i > 0:
                        return int(part)
        return 0
    
    def _count_cpp_tests_failed(self, output):
        """Count failed C++ tests from output."""
        # Parse Google Test output for failures
        lines = output.split('\n')
        failed_count = 0
        for line in lines:
            if 'FAILED' in line:
                failed_count += 1
        return failed_count
    
    def _parse_python_test_results(self, output):
        """Parse Python unittest output."""
        lines = output.split('\n')
        passed = 0
        failed = 0
        
        for line in lines:
            if line.startswith('Ran '):
                # Extract test count
                parts = line.split()
                if len(parts) >= 2:
                    total_tests = int(parts[1])
            elif 'FAILED' in line:
                # Count failures
                if 'failures=' in line:
                    failures = int(line.split('failures=')[1].split(',')[0].split(')')[0])
                    failed += failures
                if 'errors=' in line:
                    errors = int(line.split('errors=')[1].split(',')[0].split(')')[0])
                    failed += errors
            elif line == 'OK':
                # All tests passed
                passed = total_tests if 'total_tests' in locals() else 1
        
        if passed == 0 and failed == 0:
            # Fallback parsing
            passed = output.count('ok')
            failed = output.count('FAIL') + output.count('ERROR')
        
        return passed, failed
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        cpp_total = self.results['cpp_tests']['passed'] + self.results['cpp_tests']['failed']
        python_total = self.results['python_tests']['passed'] + self.results['python_tests']['failed']
        
        print(f"C++ Tests:    {self.results['cpp_tests']['passed']}/{cpp_total} passed")
        print(f"Python Tests: {self.results['python_tests']['passed']}/{python_total} passed")
        
        total_passed = self.results['cpp_tests']['passed'] + self.results['python_tests']['passed']
        total_tests = cpp_total + python_total
        
        print(f"Overall:      {total_passed}/{total_tests} passed")
        
        if self.results['cpp_tests']['errors'] or self.results['python_tests']['errors']:
            print("\nErrors:")
            for error in self.results['cpp_tests']['errors']:
                print(f"  C++: {error}")
            for error in self.results['python_tests']['errors']:
                print(f"  Python: {error}")
        
        # Save results to file
        with open(self.test_dir / 'test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return total_passed == total_tests
    
    def run_all(self):
        """Run all tests."""
        start_time = time.time()
        
        print("SemiDGFEM Test Suite")
        print("=" * 60)
        
        success = True
        
        # Run C++ tests
        if not self.run_cpp_tests():
            success = False
        
        # Run Python tests
        if not self.run_python_tests():
            success = False
        
        # Run integration tests
        if not self.run_integration_tests():
            success = False
        
        # Run memory checks
        if not self.run_memory_checks():
            success = False
        
        # Generate coverage report
        if not self.generate_coverage_report():
            success = False
        
        end_time = time.time()
        self.results['total_time'] = end_time - start_time
        
        # Print summary
        overall_success = self.print_summary()
        
        return overall_success and success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run SemiDGFEM test suite')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-c', '--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('-m', '--memcheck', action='store_true', help='Run memory leak detection')
    parser.add_argument('--cpp-only', action='store_true', help='Run only C++ tests')
    parser.add_argument('--python-only', action='store_true', help='Run only Python tests')
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose, coverage=args.coverage, memcheck=args.memcheck)
    
    if args.cpp_only:
        success = runner.run_cpp_tests()
    elif args.python_only:
        success = runner.run_python_tests()
    else:
        success = runner.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
