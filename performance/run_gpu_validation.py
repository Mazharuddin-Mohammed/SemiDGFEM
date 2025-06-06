#!/usr/bin/env python3
"""
Complete GPU Validation Launcher
Runs comprehensive GPU validation and performance benchmarking

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_section(title):
    """Print formatted section"""
    print(f"\n{'='*20} {title} {'='*20}")

def check_dependencies():
    """Check if all required dependencies are available"""
    print_section("Dependency Check")
    
    dependencies = [
        ("numpy", "NumPy"),
        ("psutil", "System information"),
        ("json", "JSON support"),
        ("time", "Timing utilities")
    ]
    
    missing_deps = []
    
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✓ {description}: Available")
        except ImportError:
            print(f"✗ {description}: Missing")
            missing_deps.append(module)
    
    # Check optional dependencies
    optional_deps = [
        ("pynvml", "NVIDIA GPU monitoring"),
        ("matplotlib", "Plotting (optional)")
    ]
    
    for module, description in optional_deps:
        try:
            __import__(module)
            print(f"✓ {description}: Available")
        except ImportError:
            print(f"⚠ {description}: Not available (optional)")
    
    if missing_deps:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    print("\n✅ All required dependencies available")
    return True

def check_backend_availability():
    """Check availability of SemiDGFEM backends"""
    print_section("Backend Availability Check")
    
    # Add python directory to path
    python_dir = Path(__file__).parent.parent / "python"
    if python_dir.exists():
        sys.path.insert(0, str(python_dir))
        print(f"✓ Python bindings path: {python_dir}")
    else:
        print(f"⚠ Python bindings path not found: {python_dir}")
    
    backends = {
        "simulator": "Core simulator functionality",
        "complete_dg": "Complete DG discretization",
        "unstructured_transport": "Unstructured transport models",
        "performance_bindings": "Performance optimization"
    }
    
    available_backends = {}
    
    for module, description in backends.items():
        try:
            __import__(module)
            print(f"✓ {description}: Available")
            available_backends[module] = True
        except ImportError as e:
            print(f"⚠ {description}: Not available ({e})")
            available_backends[module] = False
    
    # Check GPU availability specifically
    gpu_available = False
    if available_backends.get("performance_bindings", False):
        try:
            import performance_bindings
            gpu_available = performance_bindings.GPUAcceleration.is_available()
            if gpu_available:
                print("✅ GPU acceleration: Available and functional")
            else:
                print("⚠ GPU acceleration: Backend available but GPU not detected")
        except Exception as e:
            print(f"⚠ GPU acceleration: Error checking availability ({e})")
    
    return available_backends, gpu_available

def run_gpu_validation_suite():
    """Run the complete GPU validation suite"""
    print_section("GPU Validation Suite")
    
    try:
        from gpu_validation_suite import ComprehensiveGPUValidation
        
        validator = ComprehensiveGPUValidation()
        summary = validator.run_complete_validation()
        
        return summary["validation_completed"]
        
    except ImportError as e:
        print(f"❌ Cannot import GPU validation suite: {e}")
        return False
    except Exception as e:
        print(f"❌ GPU validation suite failed: {e}")
        return False

def run_gpu_kernel_validation():
    """Run GPU kernel correctness validation"""
    print_section("GPU Kernel Validation")
    
    try:
        from gpu_kernel_validation import GPUKernelValidator
        
        validator = GPUKernelValidator()
        success = validator.run_complete_validation()
        
        return success
        
    except ImportError as e:
        print(f"❌ Cannot import GPU kernel validator: {e}")
        return False
    except Exception as e:
        print(f"❌ GPU kernel validation failed: {e}")
        return False

def run_performance_analysis():
    """Run performance analysis and comparison"""
    print_section("Performance Analysis")
    
    try:
        from performance_analysis import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        analyzer.run_complete_analysis()
        
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import performance analyzer: {e}")
        return False
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        return False

def generate_summary_report(results):
    """Generate summary report of all validation results"""
    print_header("COMPLETE GPU VALIDATION SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Total validation categories: {total_tests}")
    print(f"Passed validations: {passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL GPU VALIDATIONS PASSED!")
        print("   SemiDGFEM GPU acceleration is fully validated and ready for production use.")
        return True
    else:
        print(f"\n⚠ {total_tests - passed_tests} validation(s) failed or had issues.")
        print("   Check the detailed output above for specific problems.")
        return False

def main():
    """Main validation launcher"""
    parser = argparse.ArgumentParser(
        description="Complete GPU Validation and Performance Benchmarking for SemiDGFEM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_gpu_validation.py                    # Run all validations
  python3 run_gpu_validation.py --quick           # Run quick validation only
  python3 run_gpu_validation.py --performance     # Run performance analysis only
  python3 run_gpu_validation.py --check-only      # Check dependencies and backends only
        """
    )
    
    parser.add_argument("--quick", action="store_true",
                       help="Run quick validation (skip comprehensive benchmarks)")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance analysis only")
    parser.add_argument("--check-only", action="store_true",
                       help="Check dependencies and backends only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    print_header("SemiDGFEM Complete GPU Validation and Performance Benchmarking")
    print("Comprehensive testing of SIMD/GPU performance optimization capabilities")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check backend availability
    backends, gpu_available = check_backend_availability()
    
    if args.check_only:
        print("\n✅ Dependency and backend check completed")
        return 0
    
    # Run validations based on arguments
    results = {}
    
    if args.performance:
        # Performance analysis only
        results["Performance Analysis"] = run_performance_analysis()
    
    elif args.quick:
        # Quick validation
        if gpu_available:
            results["GPU Kernel Validation"] = run_gpu_kernel_validation()
        else:
            print("⚠ Skipping GPU validation - GPU not available")
        
        results["Performance Analysis"] = run_performance_analysis()
    
    else:
        # Complete validation
        results["GPU Validation Suite"] = run_gpu_validation_suite()
        
        if gpu_available:
            results["GPU Kernel Validation"] = run_gpu_kernel_validation()
        else:
            print("⚠ Skipping GPU kernel validation - GPU not available")
            results["GPU Kernel Validation"] = False
        
        results["Performance Analysis"] = run_performance_analysis()
    
    # Generate summary
    end_time = time.time()
    duration = end_time - start_time
    
    success = generate_summary_report(results)
    
    print(f"\nTotal validation time: {duration:.2f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\n🚀 SemiDGFEM GPU validation completed successfully!")
        print("   All performance optimization capabilities are validated and ready.")
        return 0
    else:
        print("\n⚠ Some validations failed. Check the output for details.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹ Validation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Validation launcher failed: {e}")
        sys.exit(1)
