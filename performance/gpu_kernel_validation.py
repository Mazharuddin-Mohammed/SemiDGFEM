#!/usr/bin/env python3
"""
GPU Kernel Validation and Correctness Testing
Validates GPU kernel implementations against CPU reference implementations

Author: Dr. Mazharuddin Mohammed
"""

import sys
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Add python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

class GPUKernelValidator:
    """Validates GPU kernel correctness and performance"""
    
    def __init__(self):
        self.backend_available = self._check_backend()
        self.gpu_available = self._check_gpu()
        self.validation_results = {}
        
    def _check_backend(self):
        """Check if performance bindings are available"""
        try:
            import performance_bindings
            return True
        except ImportError:
            return False
    
    def _check_gpu(self):
        """Check if GPU is available"""
        if not self.backend_available:
            return False
        
        try:
            import performance_bindings
            return performance_bindings.GPUAcceleration.is_available()
        except:
            return False
    
    def validate_vector_operations(self, sizes: List[int] = None, tolerance: float = 1e-12):
        """Validate GPU vector operations against CPU reference"""
        if not self.gpu_available:
            print("⚠ GPU not available - skipping GPU kernel validation")
            return False
        
        if sizes is None:
            sizes = [1000, 10000, 100000, 1000000]
        
        print("=== GPU Vector Operations Validation ===")
        
        try:
            import performance_bindings
            gpu = performance_bindings.GPUAcceleration()
            
            all_passed = True
            
            for size in sizes:
                print(f"\nValidating size: {size:,} elements")
                
                # Generate test data with various patterns
                test_cases = [
                    ("Random", np.random.random(size), np.random.random(size)),
                    ("Zeros", np.zeros(size), np.zeros(size)),
                    ("Ones", np.ones(size), np.ones(size)),
                    ("Sequential", np.arange(size, dtype=float), np.arange(size, dtype=float) * 0.5),
                    ("Large Values", np.full(size, 1e10), np.full(size, 2e10)),
                    ("Small Values", np.full(size, 1e-10), np.full(size, 2e-10)),
                    ("Mixed Signs", np.random.random(size) - 0.5, np.random.random(size) - 0.5)
                ]
                
                for test_name, a, b in test_cases:
                    # Ensure correct data type
                    a = a.astype(np.float64)
                    b = b.astype(np.float64)
                    
                    # Test vector addition
                    cpu_result = a + b
                    gpu_result = gpu.vector_add(a, b)
                    
                    max_diff = np.max(np.abs(cpu_result - gpu_result))
                    rel_error = max_diff / (np.max(np.abs(cpu_result)) + 1e-16)
                    
                    if max_diff < tolerance and rel_error < tolerance:
                        status = "✓ PASS"
                    else:
                        status = "✗ FAIL"
                        all_passed = False
                    
                    print(f"  {test_name} Vector Add: {status} (max_diff: {max_diff:.2e}, rel_error: {rel_error:.2e})")
                    
                    # Store validation result
                    key = f"vector_add_{size}_{test_name.lower()}"
                    self.validation_results[key] = {
                        "passed": max_diff < tolerance and rel_error < tolerance,
                        "max_absolute_error": max_diff,
                        "max_relative_error": rel_error,
                        "tolerance": tolerance
                    }
            
            return all_passed
            
        except Exception as e:
            print(f"✗ GPU vector validation failed: {e}")
            return False
    
    def validate_matrix_operations(self, sizes: List[Tuple[int, int]] = None, tolerance: float = 1e-12):
        """Validate GPU matrix operations against CPU reference"""
        if not self.gpu_available:
            return False
        
        if sizes is None:
            sizes = [(100, 100), (500, 500), (1000, 1000)]
        
        print("\n=== GPU Matrix Operations Validation ===")
        
        try:
            import performance_bindings
            gpu = performance_bindings.GPUAcceleration()
            
            all_passed = True
            
            for rows, cols in sizes:
                print(f"\nValidating matrix size: {rows}×{cols}")
                
                # Generate test matrices with various patterns
                test_cases = [
                    ("Random", np.random.random((rows, cols)), np.random.random(cols)),
                    ("Identity", np.eye(rows, cols), np.ones(cols)),
                    ("Zeros", np.zeros((rows, cols)), np.zeros(cols)),
                    ("Diagonal", np.diag(np.arange(min(rows, cols), dtype=float)), np.ones(cols)),
                    ("Sparse", self._create_sparse_matrix(rows, cols, 0.1), np.random.random(cols))
                ]
                
                for test_name, matrix, vector in test_cases:
                    # Ensure correct data types
                    matrix = matrix.astype(np.float64)
                    vector = vector.astype(np.float64)
                    
                    # Test matrix-vector multiplication
                    cpu_result = matrix @ vector
                    gpu_result = gpu.matrix_vector_multiply(matrix, vector)
                    
                    max_diff = np.max(np.abs(cpu_result - gpu_result))
                    rel_error = max_diff / (np.max(np.abs(cpu_result)) + 1e-16)
                    
                    if max_diff < tolerance and rel_error < tolerance:
                        status = "✓ PASS"
                    else:
                        status = "✗ FAIL"
                        all_passed = False
                    
                    print(f"  {test_name} MatVec: {status} (max_diff: {max_diff:.2e}, rel_error: {rel_error:.2e})")
                    
                    # Store validation result
                    key = f"matrix_vector_{rows}x{cols}_{test_name.lower()}"
                    self.validation_results[key] = {
                        "passed": max_diff < tolerance and rel_error < tolerance,
                        "max_absolute_error": max_diff,
                        "max_relative_error": rel_error,
                        "tolerance": tolerance
                    }
            
            return all_passed
            
        except Exception as e:
            print(f"✗ GPU matrix validation failed: {e}")
            return False
    
    def _create_sparse_matrix(self, rows: int, cols: int, density: float):
        """Create a sparse matrix for testing"""
        matrix = np.zeros((rows, cols))
        n_nonzeros = int(rows * cols * density)
        
        for _ in range(n_nonzeros):
            i = np.random.randint(0, rows)
            j = np.random.randint(0, cols)
            matrix[i, j] = np.random.random() * 10 - 5  # Random value between -5 and 5
        
        return matrix
    
    def validate_numerical_stability(self, n_iterations: int = 100):
        """Test numerical stability of GPU operations"""
        if not self.gpu_available:
            return False
        
        print("\n=== GPU Numerical Stability Validation ===")
        
        try:
            import performance_bindings
            gpu = performance_bindings.GPUAcceleration()
            
            all_passed = True
            
            # Test with various numerical challenges
            test_cases = [
                ("Large Numbers", 1e10, 1e10),
                ("Small Numbers", 1e-10, 1e-10),
                ("Mixed Scale", 1e10, 1e-10),
                ("Near Zero", 1e-15, 1e-15),
                ("Infinity Handling", 1e308, 1e308)
            ]
            
            for test_name, scale_a, scale_b in test_cases:
                print(f"\nTesting {test_name}:")
                
                errors = []
                
                for i in range(n_iterations):
                    size = 1000
                    a = np.random.random(size) * scale_a
                    b = np.random.random(size) * scale_b
                    
                    # Ensure finite values
                    a = np.clip(a, -1e100, 1e100)
                    b = np.clip(b, -1e100, 1e100)
                    
                    try:
                        cpu_result = a + b
                        gpu_result = gpu.vector_add(a, b)
                        
                        # Check for NaN or Inf
                        if np.any(np.isnan(gpu_result)) or np.any(np.isinf(gpu_result)):
                            if not (np.any(np.isnan(cpu_result)) or np.any(np.isinf(cpu_result))):
                                errors.append(f"GPU produced NaN/Inf when CPU didn't (iteration {i})")
                        
                        # Check relative error for finite values
                        finite_mask = np.isfinite(cpu_result) & np.isfinite(gpu_result)
                        if np.any(finite_mask):
                            rel_error = np.max(np.abs(cpu_result[finite_mask] - gpu_result[finite_mask]) / 
                                             (np.abs(cpu_result[finite_mask]) + 1e-16))
                            if rel_error > 1e-10:
                                errors.append(f"High relative error: {rel_error:.2e} (iteration {i})")
                    
                    except Exception as e:
                        errors.append(f"Exception in iteration {i}: {e}")
                
                if not errors:
                    print(f"  ✓ PASS - No stability issues in {n_iterations} iterations")
                else:
                    print(f"  ✗ FAIL - {len(errors)} issues found:")
                    for error in errors[:5]:  # Show first 5 errors
                        print(f"    {error}")
                    if len(errors) > 5:
                        print(f"    ... and {len(errors) - 5} more")
                    all_passed = False
                
                # Store validation result
                self.validation_results[f"stability_{test_name.lower().replace(' ', '_')}"] = {
                    "passed": len(errors) == 0,
                    "error_count": len(errors),
                    "iterations_tested": n_iterations
                }
            
            return all_passed
            
        except Exception as e:
            print(f"✗ GPU stability validation failed: {e}")
            return False
    
    def validate_performance_consistency(self, n_runs: int = 10):
        """Validate that GPU performance is consistent across runs"""
        if not self.gpu_available:
            return False
        
        print("\n=== GPU Performance Consistency Validation ===")
        
        try:
            import performance_bindings
            gpu = performance_bindings.GPUAcceleration()
            
            size = 1000000
            a = np.random.random(size).astype(np.float64)
            b = np.random.random(size).astype(np.float64)
            
            # Warm up GPU
            for _ in range(3):
                _ = gpu.vector_add(a, b)
            
            # Measure performance consistency
            times = []
            for i in range(n_runs):
                start_time = time.perf_counter()
                result = gpu.vector_add(a, b)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            cv = std_time / mean_time  # Coefficient of variation
            
            print(f"Performance over {n_runs} runs:")
            print(f"  Mean time: {mean_time*1000:.3f} ms")
            print(f"  Std deviation: {std_time*1000:.3f} ms")
            print(f"  Coefficient of variation: {cv:.3f}")
            
            # Performance should be consistent (CV < 0.1 for well-behaved GPU code)
            consistent = cv < 0.1
            
            if consistent:
                print(f"  ✓ PASS - Performance is consistent")
            else:
                print(f"  ⚠ WARNING - High performance variation (CV > 0.1)")
            
            self.validation_results["performance_consistency"] = {
                "passed": consistent,
                "coefficient_of_variation": cv,
                "mean_time_ms": mean_time * 1000,
                "std_time_ms": std_time * 1000,
                "n_runs": n_runs
            }
            
            return consistent
            
        except Exception as e:
            print(f"✗ GPU performance consistency validation failed: {e}")
            return False
    
    def run_complete_validation(self):
        """Run complete GPU kernel validation"""
        print("🔍 GPU Kernel Validation and Correctness Testing")
        print("=" * 60)
        
        if not self.gpu_available:
            print("❌ GPU not available - cannot run GPU kernel validation")
            return False
        
        validation_passed = True
        
        # Run all validation tests
        tests = [
            ("Vector Operations", self.validate_vector_operations),
            ("Matrix Operations", self.validate_matrix_operations),
            ("Numerical Stability", self.validate_numerical_stability),
            ("Performance Consistency", self.validate_performance_consistency)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_func()
                if not result:
                    validation_passed = False
            except Exception as e:
                print(f"✗ {test_name} failed with exception: {e}")
                validation_passed = False
        
        # Summary
        print(f"\n{'='*60}")
        print("GPU KERNEL VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        if validation_passed:
            print("✅ All GPU kernel validations PASSED")
            print("   GPU implementations are correct and stable")
        else:
            print("❌ Some GPU kernel validations FAILED")
            print("   Check the detailed output above for issues")
        
        # Detailed results
        passed_tests = sum(1 for result in self.validation_results.values() if result.get("passed", False))
        total_tests = len(self.validation_results)
        
        print(f"\nDetailed Results: {passed_tests}/{total_tests} tests passed")
        
        for test_name, result in self.validation_results.items():
            status = "✓" if result.get("passed", False) else "✗"
            print(f"  {status} {test_name}")
        
        return validation_passed

def main():
    """Main validation entry point"""
    try:
        validator = GPUKernelValidator()
        success = validator.run_complete_validation()
        
        if success:
            print("\n🎉 GPU kernel validation completed successfully!")
            return 0
        else:
            print("\n⚠ GPU kernel validation completed with issues!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹ Validation cancelled by user")
        return 0
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
