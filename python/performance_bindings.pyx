# distutils: language = c++
# cython: language_level=3

"""
Performance Optimization Python Interface
Provides access to SIMD kernels, parallel computing, and GPU acceleration
"""

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from cython.parallel import prange
cimport openmp

# C interface declarations for performance modules
cdef extern from "src/performance/simd_kernels.cpp":
    # SIMD optimized kernels
    void simd_vector_add(double* a, double* b, double* result, int size)
    void simd_vector_multiply(double* a, double* b, double* result, int size)
    void simd_matrix_vector_multiply(double* matrix, double* vector, double* result, int rows, int cols)
    double simd_dot_product(double* a, double* b, int size)
    void simd_basis_function_evaluation(double* xi, double* eta, int* indices, double* results, int size, int order)

cdef extern from "src/performance/parallel_computing.cpp":
    # Parallel computing utilities
    void set_num_threads(int num_threads)
    int get_num_threads()
    void parallel_element_assembly(double* elements, double* matrices, int num_elements, int dofs_per_element)
    void parallel_mesh_refinement(double* mesh_data, int* refinement_flags, int num_elements)

cdef extern from "src/gpu/gpu_context.cpp":
    # GPU acceleration
    bool gpu_is_available()
    void gpu_initialize()
    void gpu_finalize()
    void gpu_vector_add(double* a, double* b, double* result, int size)
    void gpu_matrix_vector_multiply(double* matrix, double* vector, double* result, int rows, int cols)
    void gpu_dg_assembly(double* elements, double* matrices, int num_elements, int dofs_per_element)

# Python wrapper for SIMD operations
cdef class SIMDKernels:
    """
    SIMD-optimized computational kernels for high-performance computing.
    """
    
    @staticmethod
    def vector_add(np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] b):
        """
        SIMD-optimized vector addition.
        
        Parameters:
        -----------
        a : array_like
            First vector
        b : array_like
            Second vector
            
        Returns:
        --------
        numpy.ndarray
            Result vector a + b
        """
        if a.size != b.size:
            raise ValueError("Vectors must have the same size")
        
        cdef int size = a.size
        cdef np.ndarray[double, ndim=1] result = np.zeros(size)
        
        simd_vector_add(&a[0], &b[0], &result[0], size)
        return result
    
    @staticmethod
    def vector_multiply(np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] b):
        """
        SIMD-optimized element-wise vector multiplication.
        
        Parameters:
        -----------
        a : array_like
            First vector
        b : array_like
            Second vector
            
        Returns:
        --------
        numpy.ndarray
            Result vector a * b (element-wise)
        """
        if a.size != b.size:
            raise ValueError("Vectors must have the same size")
        
        cdef int size = a.size
        cdef np.ndarray[double, ndim=1] result = np.zeros(size)
        
        simd_vector_multiply(&a[0], &b[0], &result[0], size)
        return result
    
    @staticmethod
    def matrix_vector_multiply(np.ndarray[double, ndim=2] matrix, np.ndarray[double, ndim=1] vector):
        """
        SIMD-optimized matrix-vector multiplication.
        
        Parameters:
        -----------
        matrix : array_like
            Matrix (rows x cols)
        vector : array_like
            Vector (cols,)
            
        Returns:
        --------
        numpy.ndarray
            Result vector (rows,)
        """
        if matrix.shape[1] != vector.size:
            raise ValueError("Matrix columns must match vector size")
        
        cdef int rows = matrix.shape[0]
        cdef int cols = matrix.shape[1]
        cdef np.ndarray[double, ndim=1] result = np.zeros(rows)
        
        simd_matrix_vector_multiply(&matrix[0, 0], &vector[0], &result[0], rows, cols)
        return result
    
    @staticmethod
    def dot_product(np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] b):
        """
        SIMD-optimized dot product.
        
        Parameters:
        -----------
        a : array_like
            First vector
        b : array_like
            Second vector
            
        Returns:
        --------
        float
            Dot product a · b
        """
        if a.size != b.size:
            raise ValueError("Vectors must have the same size")
        
        return simd_dot_product(&a[0], &b[0], a.size)
    
    @staticmethod
    def basis_function_evaluation(np.ndarray[double, ndim=1] xi,
                                 np.ndarray[double, ndim=1] eta,
                                 np.ndarray[int, ndim=1] indices,
                                 int order):
        """
        SIMD-optimized basis function evaluation.
        
        Parameters:
        -----------
        xi : array_like
            First barycentric coordinates
        eta : array_like
            Second barycentric coordinates
        indices : array_like
            Basis function indices
        order : int
            Polynomial order
            
        Returns:
        --------
        numpy.ndarray
            Basis function values
        """
        if not (xi.size == eta.size == indices.size):
            raise ValueError("All input arrays must have the same size")
        
        cdef int size = xi.size
        cdef np.ndarray[double, ndim=1] results = np.zeros(size)
        
        simd_basis_function_evaluation(&xi[0], &eta[0], &indices[0], &results[0], size, order)
        return results

# Python wrapper for parallel computing
cdef class ParallelComputing:
    """
    Parallel computing utilities for multi-threaded operations.
    """
    
    @staticmethod
    def set_num_threads(int num_threads):
        """
        Set the number of OpenMP threads.
        
        Parameters:
        -----------
        num_threads : int
            Number of threads to use
        """
        set_num_threads(num_threads)
    
    @staticmethod
    def get_num_threads():
        """
        Get the current number of OpenMP threads.
        
        Returns:
        --------
        int
            Number of threads
        """
        return get_num_threads()
    
    @staticmethod
    def parallel_element_assembly(np.ndarray[double, ndim=3] elements,
                                 int dofs_per_element):
        """
        Parallel assembly of element matrices.
        
        Parameters:
        -----------
        elements : array_like
            Element data (num_elements x dofs_per_element x dofs_per_element)
        dofs_per_element : int
            Degrees of freedom per element
            
        Returns:
        --------
        numpy.ndarray
            Assembled matrices
        """
        cdef int num_elements = elements.shape[0]
        cdef np.ndarray[double, ndim=3] matrices = np.zeros_like(elements)
        
        parallel_element_assembly(&elements[0, 0, 0], &matrices[0, 0, 0], 
                                num_elements, dofs_per_element)
        return matrices
    
    @staticmethod
    def parallel_mesh_refinement(np.ndarray[double, ndim=2] mesh_data,
                                np.ndarray[int, ndim=1] refinement_flags):
        """
        Parallel mesh refinement.
        
        Parameters:
        -----------
        mesh_data : array_like
            Mesh element data
        refinement_flags : array_like
            Refinement flags for each element
            
        Returns:
        --------
        numpy.ndarray
            Refined mesh data
        """
        cdef int num_elements = mesh_data.shape[0]
        
        parallel_mesh_refinement(&mesh_data[0, 0], &refinement_flags[0], num_elements)
        return mesh_data

# Python wrapper for GPU acceleration
cdef class GPUAcceleration:
    """
    GPU acceleration utilities for CUDA-based computations.
    """
    
    def __init__(self):
        """Initialize GPU context."""
        if self.is_available():
            gpu_initialize()
        else:
            print("Warning: GPU not available, falling back to CPU")
    
    def __dealloc__(self):
        """Cleanup GPU context."""
        if self.is_available():
            gpu_finalize()
    
    @staticmethod
    def is_available():
        """
        Check if GPU acceleration is available.
        
        Returns:
        --------
        bool
            True if GPU is available
        """
        return gpu_is_available()
    
    def vector_add(self, np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] b):
        """
        GPU-accelerated vector addition.
        
        Parameters:
        -----------
        a : array_like
            First vector
        b : array_like
            Second vector
            
        Returns:
        --------
        numpy.ndarray
            Result vector a + b
        """
        if not self.is_available():
            return SIMDKernels.vector_add(a, b)
        
        if a.size != b.size:
            raise ValueError("Vectors must have the same size")
        
        cdef int size = a.size
        cdef np.ndarray[double, ndim=1] result = np.zeros(size)
        
        gpu_vector_add(&a[0], &b[0], &result[0], size)
        return result
    
    def matrix_vector_multiply(self, np.ndarray[double, ndim=2] matrix, 
                              np.ndarray[double, ndim=1] vector):
        """
        GPU-accelerated matrix-vector multiplication.
        
        Parameters:
        -----------
        matrix : array_like
            Matrix (rows x cols)
        vector : array_like
            Vector (cols,)
            
        Returns:
        --------
        numpy.ndarray
            Result vector (rows,)
        """
        if not self.is_available():
            return SIMDKernels.matrix_vector_multiply(matrix, vector)
        
        if matrix.shape[1] != vector.size:
            raise ValueError("Matrix columns must match vector size")
        
        cdef int rows = matrix.shape[0]
        cdef int cols = matrix.shape[1]
        cdef np.ndarray[double, ndim=1] result = np.zeros(rows)
        
        gpu_matrix_vector_multiply(&matrix[0, 0], &vector[0], &result[0], rows, cols)
        return result
    
    def dg_assembly(self, np.ndarray[double, ndim=3] elements, int dofs_per_element):
        """
        GPU-accelerated DG assembly.
        
        Parameters:
        -----------
        elements : array_like
            Element data (num_elements x dofs_per_element x dofs_per_element)
        dofs_per_element : int
            Degrees of freedom per element
            
        Returns:
        --------
        numpy.ndarray
            Assembled matrices
        """
        if not self.is_available():
            return ParallelComputing.parallel_element_assembly(elements, dofs_per_element)
        
        cdef int num_elements = elements.shape[0]
        cdef np.ndarray[double, ndim=3] matrices = np.zeros_like(elements)
        
        gpu_dg_assembly(&elements[0, 0, 0], &matrices[0, 0, 0], 
                       num_elements, dofs_per_element)
        return matrices

# High-level performance interface
cdef class PerformanceOptimizer:
    """
    High-level interface for performance optimization.
    """
    
    cdef GPUAcceleration _gpu
    cdef bool _use_gpu
    cdef int _num_threads
    
    def __init__(self, bool use_gpu=True, int num_threads=0):
        """
        Initialize performance optimizer.
        
        Parameters:
        -----------
        use_gpu : bool, optional
            Whether to use GPU acceleration (default: True)
        num_threads : int, optional
            Number of threads (0 = auto-detect)
        """
        self._use_gpu = use_gpu and GPUAcceleration.is_available()
        
        if self._use_gpu:
            self._gpu = GPUAcceleration()
        
        if num_threads > 0:
            ParallelComputing.set_num_threads(num_threads)
            self._num_threads = num_threads
        else:
            self._num_threads = ParallelComputing.get_num_threads()
    
    def get_performance_info(self):
        """
        Get performance configuration information.
        
        Returns:
        --------
        dict
            Performance configuration
        """
        return {
            "gpu_available": GPUAcceleration.is_available(),
            "gpu_enabled": self._use_gpu,
            "num_threads": self._num_threads,
            "simd_enabled": True,
            "parallel_enabled": True
        }
    
    def optimize_vector_operation(self, operation, *args):
        """
        Optimize vector operation using best available method.
        
        Parameters:
        -----------
        operation : str
            Operation type ("add", "multiply", "dot", "matvec")
        *args : array_like
            Operation arguments
            
        Returns:
        --------
        numpy.ndarray or float
            Operation result
        """
        if self._use_gpu:
            if operation == "add":
                return self._gpu.vector_add(args[0], args[1])
            elif operation == "matvec":
                return self._gpu.matrix_vector_multiply(args[0], args[1])
        
        # Fall back to SIMD
        if operation == "add":
            return SIMDKernels.vector_add(args[0], args[1])
        elif operation == "multiply":
            return SIMDKernels.vector_multiply(args[0], args[1])
        elif operation == "dot":
            return SIMDKernels.dot_product(args[0], args[1])
        elif operation == "matvec":
            return SIMDKernels.matrix_vector_multiply(args[0], args[1])
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def optimize_dg_assembly(self, np.ndarray[double, ndim=3] elements, int dofs_per_element):
        """
        Optimize DG assembly using best available method.
        
        Parameters:
        -----------
        elements : array_like
            Element data
        dofs_per_element : int
            DOFs per element
            
        Returns:
        --------
        numpy.ndarray
            Assembled matrices
        """
        if self._use_gpu:
            return self._gpu.dg_assembly(elements, dofs_per_element)
        else:
            return ParallelComputing.parallel_element_assembly(elements, dofs_per_element)

# Convenience functions
def create_performance_optimizer(use_gpu=True, num_threads=0):
    """Create performance optimizer with optimal settings."""
    return PerformanceOptimizer(use_gpu, num_threads)

def benchmark_performance():
    """
    Benchmark performance of different optimization methods.
    
    Returns:
    --------
    dict
        Benchmark results
    """
    import time
    
    # Test data
    size = 10000
    a = np.random.random(size)
    b = np.random.random(size)
    matrix = np.random.random((1000, 1000))
    vector = np.random.random(1000)
    
    results = {}
    
    # Benchmark SIMD operations
    start_time = time.time()
    for _ in range(100):
        SIMDKernels.vector_add(a, b)
    results["simd_vector_add"] = time.time() - start_time
    
    start_time = time.time()
    for _ in range(10):
        SIMDKernels.matrix_vector_multiply(matrix, vector)
    results["simd_matvec"] = time.time() - start_time
    
    # Benchmark GPU operations (if available)
    if GPUAcceleration.is_available():
        gpu = GPUAcceleration()
        
        start_time = time.time()
        for _ in range(100):
            gpu.vector_add(a, b)
        results["gpu_vector_add"] = time.time() - start_time
        
        start_time = time.time()
        for _ in range(10):
            gpu.matrix_vector_multiply(matrix, vector)
        results["gpu_matvec"] = time.time() - start_time
    
    # Performance info
    results["performance_info"] = {
        "gpu_available": GPUAcceleration.is_available(),
        "num_threads": ParallelComputing.get_num_threads(),
        "test_size": size
    }
    
    return results
