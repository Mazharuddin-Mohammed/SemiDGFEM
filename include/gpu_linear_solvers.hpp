#pragma once

#include "gpu_acceleration.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace simulator {
namespace gpu {

/**
 * @brief GPU-accelerated linear solver interface
 */
class GPULinearSolver {
public:
    virtual ~GPULinearSolver() = default;
    
    /**
     * @brief Solve linear system Ax = b
     * @param A_values CSR matrix values
     * @param row_ptr CSR row pointers
     * @param col_indices CSR column indices
     * @param b Right-hand side vector
     * @param x Solution vector (input/output)
     * @param n Matrix size
     * @param tolerance Convergence tolerance
     * @param max_iterations Maximum iterations
     * @return Number of iterations to convergence (-1 if failed)
     */
    virtual int solve(const double* A_values, const int* row_ptr, const int* col_indices,
                     const double* b, double* x, size_t n, 
                     double tolerance = 1e-8, int max_iterations = 1000) = 0;
    
    virtual std::string get_solver_name() const = 0;
    virtual bool supports_symmetric_matrices() const = 0;
    virtual bool supports_positive_definite() const = 0;
};

/**
 * @brief GPU-accelerated Jacobi iterative solver
 */
class GPUJacobiSolver : public GPULinearSolver {
public:
    GPUJacobiSolver(double relaxation_factor = 1.0);
    
    int solve(const double* A_values, const int* row_ptr, const int* col_indices,
             const double* b, double* x, size_t n, 
             double tolerance = 1e-8, int max_iterations = 1000) override;
    
    std::string get_solver_name() const override { return "GPU Jacobi"; }
    bool supports_symmetric_matrices() const override { return true; }
    bool supports_positive_definite() const override { return true; }
    
    void set_relaxation_factor(double omega) { omega_ = omega; }
    
private:
    double omega_;  // Relaxation factor
    
    // GPU memory management
    mutable double* d_A_diag_ = nullptr;
    mutable double* d_A_off_diag_ = nullptr;
    mutable int* d_row_ptr_ = nullptr;
    mutable int* d_col_indices_ = nullptr;
    mutable double* d_b_ = nullptr;
    mutable double* d_x_old_ = nullptr;
    mutable double* d_x_new_ = nullptr;
    mutable size_t allocated_size_ = 0;
    
    void allocate_gpu_memory(size_t n, size_t nnz) const;
    void deallocate_gpu_memory() const;
    void extract_diagonal(const double* A_values, const int* row_ptr, 
                         const int* col_indices, size_t n) const;
};

/**
 * @brief GPU-accelerated Gauss-Seidel solver with graph coloring
 */
class GPUGaussSeidelSolver : public GPULinearSolver {
public:
    GPUGaussSeidelSolver();
    
    int solve(const double* A_values, const int* row_ptr, const int* col_indices,
             const double* b, double* x, size_t n, 
             double tolerance = 1e-8, int max_iterations = 1000) override;
    
    std::string get_solver_name() const override { return "GPU Gauss-Seidel"; }
    bool supports_symmetric_matrices() const override { return true; }
    bool supports_positive_definite() const override { return true; }
    
private:
    // Graph coloring for parallel Gauss-Seidel
    std::vector<std::pair<int, int>> compute_graph_coloring(
        const int* row_ptr, const int* col_indices, size_t n) const;
    
    // GPU memory management
    mutable double* d_A_values_ = nullptr;
    mutable int* d_row_ptr_ = nullptr;
    mutable int* d_col_indices_ = nullptr;
    mutable double* d_b_ = nullptr;
    mutable double* d_x_ = nullptr;
    mutable size_t allocated_size_ = 0;
    
    void allocate_gpu_memory(size_t n, size_t nnz) const;
    void deallocate_gpu_memory() const;
};

/**
 * @brief GPU-accelerated Conjugate Gradient solver
 */
class GPUConjugateGradientSolver : public GPULinearSolver {
public:
    GPUConjugateGradientSolver();
    
    int solve(const double* A_values, const int* row_ptr, const int* col_indices,
             const double* b, double* x, size_t n, 
             double tolerance = 1e-8, int max_iterations = 1000) override;
    
    std::string get_solver_name() const override { return "GPU Conjugate Gradient"; }
    bool supports_symmetric_matrices() const override { return true; }
    bool supports_positive_definite() const override { return true; }
    
private:
    // GPU memory for CG algorithm
    mutable double* d_A_values_ = nullptr;
    mutable int* d_row_ptr_ = nullptr;
    mutable int* d_col_indices_ = nullptr;
    mutable double* d_b_ = nullptr;
    mutable double* d_x_ = nullptr;
    mutable double* d_r_ = nullptr;
    mutable double* d_p_ = nullptr;
    mutable double* d_Ap_ = nullptr;
    mutable size_t allocated_size_ = 0;
    
    void allocate_gpu_memory(size_t n, size_t nnz) const;
    void deallocate_gpu_memory() const;
};

/**
 * @brief GPU linear solver factory
 */
class GPULinearSolverFactory {
public:
    enum class SolverType {
        JACOBI,
        GAUSS_SEIDEL,
        CONJUGATE_GRADIENT,
        AUTO  // Automatically choose based on matrix properties
    };
    
    static std::unique_ptr<GPULinearSolver> create_solver(SolverType type);
    static std::unique_ptr<GPULinearSolver> create_auto_solver(
        const double* A_values, const int* row_ptr, const int* col_indices, size_t n);
    
    static std::vector<std::string> get_available_solvers();
    static SolverType get_solver_type_from_string(const std::string& name);
    
private:
    static bool is_symmetric(const double* A_values, const int* row_ptr, 
                           const int* col_indices, size_t n);
    static bool is_positive_definite_estimate(const double* A_values, const int* row_ptr, 
                                            const int* col_indices, size_t n);
    static double estimate_condition_number(const double* A_values, const int* row_ptr, 
                                          const int* col_indices, size_t n);
};

/**
 * @brief GPU memory pool for efficient memory management
 */
class GPUMemoryPool {
public:
    static GPUMemoryPool& instance();
    
    void* allocate(size_t size);
    void deallocate(void* ptr, size_t size);
    
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_peak_usage() const { return peak_usage_; }
    
    void clear_pool();
    
private:
    GPUMemoryPool() = default;
    ~GPUMemoryPool();
    
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<MemoryBlock> memory_blocks_;
    size_t total_allocated_ = 0;
    size_t peak_usage_ = 0;
    
    void* allocate_new_block(size_t size);
    MemoryBlock* find_free_block(size_t size);
};

} // namespace gpu
} // namespace simulator
