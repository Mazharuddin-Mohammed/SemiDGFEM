#pragma once

#include <vector>
#include <array>
#include <memory>
#include <chrono>
#include <string>
#include <unordered_map>
#include <functional>
#include <immintrin.h>  // For SIMD intrinsics
#include <omp.h>        // For OpenMP

namespace simulator {
namespace performance {

/**
 * @brief Performance profiler for identifying bottlenecks
 */
class Profiler {
public:
    struct ProfileData {
        std::string name;
        double total_time;
        double average_time;
        size_t call_count;
        double percentage;
    };
    
    static Profiler& instance() {
        static Profiler instance_;
        return instance_;
    }
    
    void start_timer(const std::string& name);
    void end_timer(const std::string& name);
    void reset();
    std::vector<ProfileData> get_profile_data() const;
    void print_profile() const;
    
private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
    std::unordered_map<std::string, double> total_times_;
    std::unordered_map<std::string, size_t> call_counts_;
};

/**
 * @brief RAII timer for automatic profiling
 */
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name) : name_(name) {
        Profiler::instance().start_timer(name_);
    }
    
    ~ScopedTimer() {
        Profiler::instance().end_timer(name_);
    }
    
private:
    std::string name_;
};

#define PROFILE_SCOPE(name) ScopedTimer timer(name)
#define PROFILE_FUNCTION() ScopedTimer timer(__FUNCTION__)

/**
 * @brief SIMD-optimized mathematical operations
 */
namespace simd {

/**
 * @brief SIMD-optimized vector operations
 */
class VectorOps {
public:
    // AVX2-optimized dot product
    static double dot_product_avx2(const double* a, const double* b, size_t n);
    
    // AVX2-optimized vector addition
    static void vector_add_avx2(const double* a, const double* b, double* result, size_t n);
    
    // AVX2-optimized vector scaling
    static void vector_scale_avx2(const double* a, double scale, double* result, size_t n);
    
    // AVX2-optimized matrix-vector multiplication
    static void matvec_avx2(const double* matrix, const double* vector, 
                           double* result, size_t rows, size_t cols);
    
    // Check CPU capabilities
    static bool has_avx2();
    static bool has_fma();
    static bool has_avx512();
};

/**
 * @brief SIMD-optimized basis function evaluation
 */
class BasisOps {
public:
    // Vectorized P3 basis function evaluation
    static void evaluate_p3_basis_vectorized(
        const double* xi_array, const double* eta_array, size_t n_points,
        double* basis_values, double* basis_gradients);
    
    // Vectorized quadrature integration
    static void integrate_element_vectorized(
        const double* basis_values, const double* weights, 
        const double* jacobians, size_t n_quad_points,
        double* element_matrix, size_t matrix_size);
};

} // namespace simd

/**
 * @brief Memory optimization utilities
 */
namespace memory {

/**
 * @brief Cache-friendly data structures
 */
template<typename T>
class AlignedVector {
public:
    explicit AlignedVector(size_t size, size_t alignment = 32);
    ~AlignedVector();
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    
private:
    T* data_;
    size_t size_;
    size_t alignment_;
};

/**
 * @brief Memory pool for frequent allocations
 */
class MemoryPool {
public:
    explicit MemoryPool(size_t block_size, size_t num_blocks);
    ~MemoryPool();
    
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void reset();
    
    size_t get_allocated_bytes() const { return allocated_bytes_; }
    size_t get_peak_usage() const { return peak_usage_; }
    
private:
    std::vector<char*> blocks_;
    std::vector<void*> free_list_;
    size_t block_size_;
    size_t allocated_bytes_;
    size_t peak_usage_;
};

/**
 * @brief Cache-optimized matrix storage
 */
class BlockMatrix {
public:
    BlockMatrix(size_t rows, size_t cols, size_t block_size = 64);
    
    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;
    
    void multiply(const BlockMatrix& other, BlockMatrix& result) const;
    void multiply_vector(const double* vec, double* result) const;
    
private:
    std::vector<double> data_;
    size_t rows_, cols_, block_size_;
    size_t blocks_per_row_, blocks_per_col_;
    
    size_t get_block_index(size_t block_row, size_t block_col, 
                          size_t local_row, size_t local_col) const;
};

} // namespace memory

/**
 * @brief Parallel computing utilities
 */
namespace parallel {

/**
 * @brief Thread pool for parallel execution
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);  // 0 = auto-detect
    ~ThreadPool();
    
    template<typename F>
    void parallel_for(size_t start, size_t end, F&& func);
    
    template<typename F>
    void parallel_reduce(size_t start, size_t end, double& result, F&& func);
    
    size_t get_num_threads() const { return num_threads_; }
    
private:
    size_t num_threads_;
    bool initialized_;
};

/**
 * @brief OpenMP-optimized operations
 */
class OMPOps {
public:
    // Parallel matrix assembly
    static void assemble_matrix_parallel(
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices,
        const std::function<void(int, std::vector<std::vector<double>>&)>& element_func,
        std::vector<std::vector<double>>& global_matrix);
    
    // Parallel vector operations
    static void vector_add_parallel(const double* a, const double* b, 
                                   double* result, size_t n);
    
    static double dot_product_parallel(const double* a, const double* b, size_t n);
    
    // Parallel error estimation
    static void compute_error_parallel(
        const std::vector<double>& solution,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices,
        std::vector<double>& error_indicators);
};

} // namespace parallel

/**
 * @brief Algorithmic optimizations
 */
namespace algorithms {

/**
 * @brief Fast multipole method for long-range interactions
 */
class FastMultipole {
public:
    FastMultipole(size_t max_particles_per_box = 50, int max_levels = 8);
    
    void build_tree(const std::vector<std::array<double, 2>>& positions);
    void compute_interactions(const std::vector<double>& charges,
                             std::vector<double>& potentials);
    
private:
    struct Box {
        std::array<double, 2> center;
        double size;
        std::vector<int> particles;
        std::vector<std::unique_ptr<Box>> children;
        int level;
    };
    
    std::unique_ptr<Box> root_;
    size_t max_particles_per_box_;
    int max_levels_;
};

/**
 * @brief Sparse matrix optimizations
 */
class SparseMatrix {
public:
    // Compressed Sparse Row (CSR) format
    struct CSRMatrix {
        std::vector<double> values;
        std::vector<int> col_indices;
        std::vector<int> row_ptr;
        size_t rows, cols, nnz;
    };
    
    // Convert from dense to CSR
    static CSRMatrix dense_to_csr(const std::vector<std::vector<double>>& dense);
    
    // Optimized SpMV (Sparse Matrix-Vector multiplication)
    static void spmv_csr(const CSRMatrix& matrix, const double* x, double* y);
    
    // Matrix reordering for better cache performance
    static std::vector<int> compute_rcm_ordering(const CSRMatrix& matrix);
    static CSRMatrix reorder_matrix(const CSRMatrix& matrix, 
                                   const std::vector<int>& ordering);
};

/**
 * @brief Preconditioner optimizations
 */
class Preconditioners {
public:
    // Incomplete LU factorization
    static algorithms::SparseMatrix::CSRMatrix compute_ilu(
        const algorithms::SparseMatrix::CSRMatrix& matrix, int fill_level = 0);
    
    // Algebraic multigrid setup
    static void setup_amg(const algorithms::SparseMatrix::CSRMatrix& matrix);
    
    // Apply preconditioner
    static void apply_preconditioner(const algorithms::SparseMatrix::CSRMatrix& precond,
                                   const double* rhs, double* solution);
};

} // namespace algorithms

/**
 * @brief Performance monitoring and optimization suggestions
 */
class PerformanceAnalyzer {
public:
    struct OptimizationSuggestion {
        std::string category;
        std::string description;
        double potential_speedup;
        int priority;  // 1 = high, 2 = medium, 3 = low
    };
    
    void analyze_performance();
    std::vector<OptimizationSuggestion> get_suggestions() const;
    void apply_automatic_optimizations();
    
    // Hardware detection
    struct HardwareInfo {
        int num_cores;
        size_t cache_l1_size;
        size_t cache_l2_size;
        size_t cache_l3_size;
        bool has_avx2;
        bool has_avx512;
        bool has_fma;
        size_t memory_bandwidth;  // GB/s
    };
    
    static HardwareInfo detect_hardware();
    
private:
    std::vector<OptimizationSuggestion> suggestions_;
    HardwareInfo hardware_info_;
};

} // namespace performance
} // namespace simulator
