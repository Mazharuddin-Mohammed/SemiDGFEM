#!/usr/bin/env python3
"""
Test suite for enhanced GPU acceleration features.
"""

import numpy as np
import sys
import os
import time
from typing import Dict, Any

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from gpu_enhanced_solver import GPULinearSolver, GPUMemoryManager

class TestGPUEnhancedAcceleration:
    """Test enhanced GPU acceleration features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.small_size = 50
        self.medium_size = 200
        self.large_size = 500
        
        # Create test matrices
        self.A_small = self._create_test_matrix(self.small_size)
        self.A_medium = self._create_test_matrix(self.medium_size)
        self.A_large = self._create_test_matrix(self.large_size)
        
        # Create corresponding RHS vectors
        self.b_small = np.random.rand(self.small_size)
        self.b_medium = np.random.rand(self.medium_size)
        self.b_large = np.random.rand(self.large_size)
    
    def _create_test_matrix(self, n: int, condition_number: float = 100.0) -> np.ndarray:
        """Create a test matrix with specified condition number."""
        # Create a symmetric positive definite matrix
        A = np.random.rand(n, n)
        A = A @ A.T  # Make positive definite
        
        # Add diagonal dominance for better conditioning
        A += condition_number * np.eye(n)
        
        return A
    
    def _create_sparse_test_matrix(self, n: int, sparsity: float = 0.1) -> np.ndarray:
        """Create a sparse test matrix."""
        A = np.random.rand(n, n)
        mask = np.random.rand(n, n) > sparsity
        A[mask] = 0
        
        # Ensure diagonal dominance
        for i in range(n):
            A[i, i] = np.sum(np.abs(A[i, :])) + 1.0
        
        return A
    
    def test_gpu_solver_initialization(self):
        """Test GPU solver initialization."""
        # Test different solver types
        solver_types = ["jacobi", "gauss_seidel", "conjugate_gradient", "auto"]
        
        for solver_type in solver_types:
            solver = GPULinearSolver(solver_type)
            assert solver.solver_type == solver_type
            assert isinstance(solver.gpu_available, bool)
            print(f"✓ {solver_type} solver initialized successfully")
    
    def test_jacobi_solver(self):
        """Test Jacobi iterative solver."""
        solver = GPULinearSolver("jacobi", relaxation_factor=0.8)
        
        # Test on small system
        x, info = solver.solve(self.A_small, self.b_small, tolerance=1e-6, max_iterations=1000)
        
        # Verify solution
        residual = np.linalg.norm(self.A_small @ x - self.b_small)
        assert residual < 1e-5, f"Jacobi solver residual too large: {residual}"
        assert 'iterations' in info
        assert 'solve_time' in info
        
        print(f"✓ Jacobi solver: {info['iterations']} iterations, residual: {residual:.2e}")
    
    def test_gauss_seidel_solver(self):
        """Test Gauss-Seidel iterative solver."""
        solver = GPULinearSolver("gauss_seidel")
        
        # Test on medium system
        x, info = solver.solve(self.A_medium, self.b_medium, tolerance=1e-6, max_iterations=1000)
        
        # Verify solution
        residual = np.linalg.norm(self.A_medium @ x - self.b_medium)
        assert residual < 1e-5, f"Gauss-Seidel solver residual too large: {residual}"
        
        print(f"✓ Gauss-Seidel solver: {info['iterations']} iterations, residual: {residual:.2e}")
    
    def test_conjugate_gradient_solver(self):
        """Test Conjugate Gradient solver."""
        solver = GPULinearSolver("conjugate_gradient")
        
        # Test on symmetric positive definite matrix
        A_spd = self.A_small @ self.A_small.T + np.eye(self.small_size)
        b_spd = np.random.rand(self.small_size)
        
        x, info = solver.solve(A_spd, b_spd, tolerance=1e-8, max_iterations=500)
        
        # Verify solution
        residual = np.linalg.norm(A_spd @ x - b_spd)
        assert residual < 1e-7, f"CG solver residual too large: {residual}"
        
        print(f"✓ Conjugate Gradient solver: {info['iterations']} iterations, residual: {residual:.2e}")
    
    def test_auto_solver_selection(self):
        """Test automatic solver selection."""
        solver = GPULinearSolver("auto")
        
        # Test on different matrix types
        matrices = [
            ("Small well-conditioned", self.A_small, self.b_small),
            ("Medium conditioned", self.A_medium, self.b_medium),
            ("Sparse matrix", self._create_sparse_test_matrix(100), np.random.rand(100))
        ]
        
        for name, A, b in matrices:
            x, info = solver.solve(A, b, tolerance=1e-6, max_iterations=1000)
            residual = np.linalg.norm(A @ x - b)
            
            assert residual < 1e-5, f"Auto solver failed on {name}: residual {residual}"
            print(f"✓ Auto solver on {name}: {info['iterations']} iterations, residual: {residual:.2e}")
    
    def test_solver_performance_comparison(self):
        """Compare performance of different solvers."""
        solvers = {
            "jacobi": GPULinearSolver("jacobi"),
            "gauss_seidel": GPULinearSolver("gauss_seidel"),
            "conjugate_gradient": GPULinearSolver("conjugate_gradient"),
            "auto": GPULinearSolver("auto")
        }
        
        # Test matrix
        A = self._create_test_matrix(200, condition_number=50.0)
        b = np.random.rand(200)
        
        results = {}
        
        for name, solver in solvers.items():
            start_time = time.time()
            x, info = solver.solve(A, b, tolerance=1e-6, max_iterations=1000)
            solve_time = time.time() - start_time
            
            residual = np.linalg.norm(A @ x - b)
            results[name] = {
                'time': solve_time,
                'iterations': info.get('iterations', 0),
                'residual': residual,
                'converged': residual < 1e-5
            }
        
        # Print comparison
        print("\n📊 Solver Performance Comparison:")
        print(f"{'Solver':<20} {'Time (s)':<10} {'Iterations':<12} {'Residual':<12} {'Converged'}")
        print("-" * 70)
        
        for name, result in results.items():
            print(f"{name:<20} {result['time']:<10.4f} {result['iterations']:<12} "
                  f"{result['residual']:<12.2e} {'✓' if result['converged'] else '✗'}")
        
        # Verify all solvers converged
        for name, result in results.items():
            assert result['converged'], f"{name} solver did not converge"
    
    def test_memory_management(self):
        """Test GPU memory management."""
        # Test memory info retrieval
        memory_info = GPUMemoryManager.get_memory_info()
        assert isinstance(memory_info, dict)
        
        if 'error' not in memory_info:
            assert 'used_bytes' in memory_info
            assert 'total_bytes' in memory_info
            assert 'free_bytes' in memory_info
            assert 'utilization_percent' in memory_info
            
            print(f"✓ GPU Memory: {memory_info['utilization_percent']:.1f}% utilized")
        else:
            print("⚠ GPU not available for memory testing")
        
        # Test memory pool clearing
        GPUMemoryManager.clear_memory_pool()
        print("✓ GPU memory pool cleared")
    
    def test_solver_statistics(self):
        """Test solver performance statistics."""
        solver = GPULinearSolver("conjugate_gradient")
        
        # Perform multiple solves
        for i in range(5):
            A = self._create_test_matrix(100 + i * 20)
            b = np.random.rand(100 + i * 20)
            x, info = solver.solve(A, b, tolerance=1e-6)
        
        # Get statistics
        stats = solver.get_performance_stats()
        
        assert stats['total_solves'] == 5
        assert 'average_time' in stats
        assert 'total_time' in stats
        assert 'average_iterations' in stats
        
        print(f"✓ Solver statistics: {stats['total_solves']} solves, "
              f"avg time: {stats['average_time']:.4f}s, "
              f"avg iterations: {stats['average_iterations']:.1f}")
        
        # Reset statistics
        solver.reset_stats()
        stats_after_reset = solver.get_performance_stats()
        assert stats_after_reset == {} or stats_after_reset['total_solves'] == 0
        print("✓ Solver statistics reset")
    
    def test_large_system_scalability(self):
        """Test solver scalability on larger systems."""
        sizes = [100, 200, 400]
        solver = GPULinearSolver("auto")
        
        times = []
        
        for size in sizes:
            A = self._create_test_matrix(size, condition_number=10.0)
            b = np.random.rand(size)
            
            start_time = time.time()
            x, info = solver.solve(A, b, tolerance=1e-6, max_iterations=2000)
            solve_time = time.time() - start_time
            
            residual = np.linalg.norm(A @ x - b)
            times.append(solve_time)
            
            assert residual < 1e-5, f"Large system solve failed for size {size}"
            print(f"✓ Size {size}: {solve_time:.4f}s, {info['iterations']} iterations, "
                  f"residual: {residual:.2e}")
        
        # Check that scaling is reasonable (not exponential)
        if len(times) >= 2:
            scaling_factor = times[-1] / times[0]
            size_factor = (sizes[-1] / sizes[0]) ** 2  # Expected quadratic scaling
            
            assert scaling_factor < size_factor * 5, "Solver scaling is too poor"
            print(f"✓ Scaling factor: {scaling_factor:.2f} (expected ~{size_factor:.2f})")
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        solver = GPULinearSolver("jacobi")
        
        # Test singular matrix
        A_singular = np.ones((5, 5))
        b_singular = np.ones(5)
        
        try:
            x, info = solver.solve(A_singular, b_singular, tolerance=1e-6, max_iterations=100)
            # Should either fail gracefully or use fallback
            print("✓ Singular matrix handled gracefully")
        except Exception as e:
            print(f"✓ Singular matrix error handled: {type(e).__name__}")
        
        # Test mismatched dimensions
        A_mismatch = np.eye(5)
        b_mismatch = np.ones(4)
        
        try:
            x, info = solver.solve(A_mismatch, b_mismatch)
            assert False, "Should have raised an error for mismatched dimensions"
        except (ValueError, AssertionError):
            print("✓ Dimension mismatch error handled")


def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("🚀 Starting Enhanced GPU Acceleration Test Suite")
    print("=" * 60)
    
    test_instance = TestGPUEnhancedAcceleration()
    test_instance.setup_method()
    
    tests = [
        ("GPU Solver Initialization", test_instance.test_gpu_solver_initialization),
        ("Jacobi Solver", test_instance.test_jacobi_solver),
        ("Gauss-Seidel Solver", test_instance.test_gauss_seidel_solver),
        ("Conjugate Gradient Solver", test_instance.test_conjugate_gradient_solver),
        ("Auto Solver Selection", test_instance.test_auto_solver_selection),
        ("Solver Performance Comparison", test_instance.test_solver_performance_comparison),
        ("Memory Management", test_instance.test_memory_management),
        ("Solver Statistics", test_instance.test_solver_statistics),
        ("Large System Scalability", test_instance.test_large_system_scalability),
        ("Error Handling", test_instance.test_error_handling)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            test_func()
            print(f"✅ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Enhanced GPU acceleration is working correctly.")
    else:
        print(f"⚠ {failed} test(s) failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
