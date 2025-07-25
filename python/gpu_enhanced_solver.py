"""
Enhanced GPU-accelerated solver with advanced linear solvers and memory optimization.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List
import time

logger = logging.getLogger(__name__)

class GPULinearSolver:
    """Python interface for GPU-accelerated linear solvers."""
    
    def __init__(self, solver_type: str = "auto", **kwargs):
        """
        Initialize GPU linear solver.
        
        Args:
            solver_type: Type of solver ("jacobi", "gauss_seidel", "conjugate_gradient", "auto")
            **kwargs: Solver-specific parameters
        """
        self.solver_type = solver_type.lower()
        self.solver_params = kwargs
        self.gpu_available = self._check_gpu_availability()
        
        # Performance tracking
        self.solve_times = []
        self.iteration_counts = []
        
        logger.info(f"Initialized GPU linear solver: {solver_type}")
        logger.info(f"GPU available: {self.gpu_available}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            # Try to import CUDA-related modules or check for GPU
            import cupy
            return cupy.cuda.is_available()
        except ImportError:
            try:
                import pycuda.driver as cuda
                cuda.init()
                return cuda.Device.count() > 0
            except ImportError:
                return False
    
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
              tolerance: float = 1e-8, max_iterations: int = 1000) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve linear system Ax = b using GPU acceleration.
        
        Args:
            A: Coefficient matrix (sparse or dense)
            b: Right-hand side vector
            x0: Initial guess (optional)
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (solution vector, solver info)
        """
        start_time = time.time()
        
        # Convert to appropriate format
        if hasattr(A, 'toarray'):
            A_dense = A.toarray()
        else:
            A_dense = np.array(A)
        
        n = A_dense.shape[0]
        if x0 is None:
            x0 = np.zeros(n)
        
        # Choose solver based on matrix properties and availability
        if self.gpu_available:
            result, info = self._solve_gpu(A_dense, b, x0, tolerance, max_iterations)
        else:
            logger.warning("GPU not available, falling back to CPU solver")
            result, info = self._solve_cpu(A_dense, b, x0, tolerance, max_iterations)
        
        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)
        self.iteration_counts.append(info.get('iterations', 0))
        
        info['solve_time'] = solve_time
        info['solver_type'] = self.solver_type
        info['gpu_used'] = self.gpu_available
        
        return result, info
    
    def _solve_gpu(self, A: np.ndarray, b: np.ndarray, x0: np.ndarray,
                   tolerance: float, max_iterations: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """GPU-accelerated solve implementation."""
        try:
            import cupy as cp
            
            # Transfer data to GPU
            A_gpu = cp.asarray(A)
            b_gpu = cp.asarray(b)
            x_gpu = cp.asarray(x0)
            
            if self.solver_type == "jacobi":
                result, info = self._jacobi_gpu(A_gpu, b_gpu, x_gpu, tolerance, max_iterations)
            elif self.solver_type == "gauss_seidel":
                result, info = self._gauss_seidel_gpu(A_gpu, b_gpu, x_gpu, tolerance, max_iterations)
            elif self.solver_type == "conjugate_gradient":
                result, info = self._conjugate_gradient_gpu(A_gpu, b_gpu, x_gpu, tolerance, max_iterations)
            else:  # auto
                result, info = self._auto_solve_gpu(A_gpu, b_gpu, x_gpu, tolerance, max_iterations)
            
            # Transfer result back to CPU
            return cp.asnumpy(result), info
            
        except Exception as e:
            logger.error(f"GPU solve failed: {e}")
            return self._solve_cpu(A, b, x0, tolerance, max_iterations)
    
    def _jacobi_gpu(self, A_gpu, b_gpu, x_gpu, tolerance: float, max_iterations: int):
        """GPU Jacobi iteration."""
        import cupy as cp
        
        # Extract diagonal
        D = cp.diag(cp.diag(A_gpu))
        D_inv = cp.linalg.inv(D)
        R = A_gpu - D
        
        omega = self.solver_params.get('relaxation_factor', 1.0)
        x = x_gpu.copy()
        
        for iteration in range(max_iterations):
            x_new = (1 - omega) * x + omega * D_inv @ (b_gpu - R @ x)
            
            # Check convergence
            if iteration % 10 == 0:
                residual = cp.linalg.norm(x_new - x)
                if residual < tolerance:
                    return x_new, {'iterations': iteration + 1, 'residual': float(residual)}
            
            x = x_new
        
        residual = cp.linalg.norm(A_gpu @ x - b_gpu)
        return x, {'iterations': max_iterations, 'residual': float(residual), 'converged': False}
    
    def _gauss_seidel_gpu(self, A_gpu, b_gpu, x_gpu, tolerance: float, max_iterations: int):
        """GPU Gauss-Seidel iteration with graph coloring."""
        import cupy as cp
        
        # For simplicity, use a basic implementation
        # In practice, this would use graph coloring for parallelization
        x = x_gpu.copy()
        n = len(x)
        
        for iteration in range(max_iterations):
            x_old = x.copy()
            
            # Sequential update (simplified for demonstration)
            for i in range(n):
                sum_ax = cp.sum(A_gpu[i, :i] * x[:i]) + cp.sum(A_gpu[i, i+1:] * x[i+1:])
                if abs(A_gpu[i, i]) > 1e-14:
                    x[i] = (b_gpu[i] - sum_ax) / A_gpu[i, i]
            
            # Check convergence
            if iteration % 10 == 0:
                residual = cp.linalg.norm(x - x_old)
                if residual < tolerance:
                    return x, {'iterations': iteration + 1, 'residual': float(residual)}
        
        residual = cp.linalg.norm(A_gpu @ x - b_gpu)
        return x, {'iterations': max_iterations, 'residual': float(residual), 'converged': False}
    
    def _conjugate_gradient_gpu(self, A_gpu, b_gpu, x_gpu, tolerance: float, max_iterations: int):
        """GPU Conjugate Gradient method."""
        import cupy as cp
        
        x = x_gpu.copy()
        r = b_gpu - A_gpu @ x
        p = r.copy()
        rsold = cp.dot(r, r)
        
        for iteration in range(max_iterations):
            Ap = A_gpu @ p
            alpha = rsold / cp.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = cp.dot(r, r)
            
            residual = cp.sqrt(rsnew)
            if residual < tolerance:
                return x, {'iterations': iteration + 1, 'residual': float(residual)}
            
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        residual = cp.linalg.norm(A_gpu @ x - b_gpu)
        return x, {'iterations': max_iterations, 'residual': float(residual), 'converged': False}
    
    def _auto_solve_gpu(self, A_gpu, b_gpu, x_gpu, tolerance: float, max_iterations: int):
        """Automatically choose best GPU solver based on matrix properties."""
        import cupy as cp
        
        # Analyze matrix properties
        n = A_gpu.shape[0]
        is_symmetric = cp.allclose(A_gpu, A_gpu.T, rtol=1e-10)
        
        # Estimate condition number (simplified)
        eigenvals = cp.linalg.eigvals(A_gpu)
        condition_number = cp.max(cp.real(eigenvals)) / cp.min(cp.real(eigenvals))
        
        # Choose solver
        if is_symmetric and condition_number < 1e6:
            logger.info("Auto-selected Conjugate Gradient (symmetric, well-conditioned)")
            return self._conjugate_gradient_gpu(A_gpu, b_gpu, x_gpu, tolerance, max_iterations)
        elif condition_number < 1e4:
            logger.info("Auto-selected Gauss-Seidel (moderately conditioned)")
            return self._gauss_seidel_gpu(A_gpu, b_gpu, x_gpu, tolerance, max_iterations)
        else:
            logger.info("Auto-selected Jacobi (ill-conditioned)")
            return self._jacobi_gpu(A_gpu, b_gpu, x_gpu, tolerance, max_iterations)
    
    def _solve_cpu(self, A: np.ndarray, b: np.ndarray, x0: np.ndarray,
                   tolerance: float, max_iterations: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """CPU fallback solver."""
        try:
            # Use scipy sparse solvers if available
            from scipy.sparse.linalg import spsolve, cg, gmres
            from scipy.sparse import csr_matrix
            
            A_sparse = csr_matrix(A)
            
            if self.solver_type == "conjugate_gradient":
                result, info_code = cg(A_sparse, b, x0=x0, rtol=tolerance, maxiter=max_iterations)
                return result, {'iterations': max_iterations, 'converged': info_code == 0}
            else:
                # Use direct solver as fallback
                result = spsolve(A_sparse, b)
                return result, {'iterations': 1, 'method': 'direct'}
                
        except ImportError:
            # Basic NumPy fallback
            result = np.linalg.solve(A, b)
            return result, {'iterations': 1, 'method': 'numpy_direct'}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get solver performance statistics."""
        if not self.solve_times:
            return {}
        
        return {
            'total_solves': len(self.solve_times),
            'average_time': np.mean(self.solve_times),
            'total_time': np.sum(self.solve_times),
            'average_iterations': np.mean(self.iteration_counts) if self.iteration_counts else 0,
            'solver_type': self.solver_type,
            'gpu_available': self.gpu_available
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.solve_times.clear()
        self.iteration_counts.clear()


class GPUMemoryManager:
    """GPU memory management utilities."""
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get GPU memory information."""
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            return {
                'used_bytes': mempool.used_bytes(),
                'total_bytes': mempool.total_bytes(),
                'free_bytes': mempool.total_bytes() - mempool.used_bytes(),
                'utilization_percent': (mempool.used_bytes() / mempool.total_bytes()) * 100
            }
        except ImportError:
            return {'error': 'CuPy not available'}
    
    @staticmethod
    def clear_memory_pool():
        """Clear GPU memory pool."""
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            logger.info("GPU memory pool cleared")
        except ImportError:
            logger.warning("CuPy not available, cannot clear GPU memory pool")
    
    @staticmethod
    def print_memory_info():
        """Print GPU memory information."""
        info = GPUMemoryManager.get_memory_info()
        if 'error' in info:
            print(f"GPU Memory Info: {info['error']}")
        else:
            print(f"GPU Memory Info:")
            print(f"  Used: {info['used_bytes'] / (1024**3):.2f} GB")
            print(f"  Total: {info['total_bytes'] / (1024**3):.2f} GB")
            print(f"  Free: {info['free_bytes'] / (1024**3):.2f} GB")
            print(f"  Utilization: {info['utilization_percent']:.1f}%")
