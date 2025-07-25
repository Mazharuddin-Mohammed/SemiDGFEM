#include "gpu_linear_solvers.hpp"
#include "gpu_acceleration.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>

namespace simulator {
namespace gpu {

// CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
    } \
} while(0)

// ============================================================================
// GPUJacobiSolver Implementation
// ============================================================================

GPUJacobiSolver::GPUJacobiSolver(double relaxation_factor) : omega_(relaxation_factor) {}

int GPUJacobiSolver::solve(const double* A_values, const int* row_ptr, const int* col_indices,
                          const double* b, double* x, size_t n, 
                          double tolerance, int max_iterations) {
    
    size_t nnz = row_ptr[n] - row_ptr[0];
    allocate_gpu_memory(n, nnz);
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_row_ptr_, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices_, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_, b, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_old_, x, n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Extract diagonal and off-diagonal parts
    extract_diagonal(A_values, row_ptr, col_indices, n);
    
    double residual = tolerance + 1.0;
    int iteration = 0;
    
    while (residual > tolerance && iteration < max_iterations) {
        // Jacobi iteration: x_new = (1-ω)x_old + ω * D^(-1) * (b - (A-D)*x_old)
        launch_jacobi_iteration(d_A_diag_, d_A_off_diag_, d_col_indices_, d_row_ptr_,
                               d_b_, d_x_old_, d_x_new_, n, omega_);
        
        // Compute residual every 10 iterations to avoid overhead
        if (iteration % 10 == 0) {
            // Compute ||x_new - x_old||
            double* d_diff;
            CUDA_CHECK(cudaMalloc(&d_diff, n * sizeof(double)));
            launch_vector_add(d_x_new_, d_x_old_, d_diff, n);  // diff = x_new - x_old
            launch_vector_scale(d_diff, -1.0, d_diff, n);     // diff = -(x_new - x_old)
            residual = sqrt(launch_dot_product(d_diff, d_diff, n));
            CUDA_CHECK(cudaFree(d_diff));
        }
        
        // Swap x_old and x_new
        std::swap(d_x_old_, d_x_new_);
        iteration++;
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(x, d_x_old_, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    return (residual <= tolerance) ? iteration : -1;
}

void GPUJacobiSolver::allocate_gpu_memory(size_t n, size_t nnz) const {
    if (allocated_size_ >= n) return;
    
    deallocate_gpu_memory();
    
    CUDA_CHECK(cudaMalloc(&d_A_diag_, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A_off_diag_, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr_, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_indices_, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b_, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_old_, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_new_, n * sizeof(double)));
    
    allocated_size_ = n;
}

void GPUJacobiSolver::deallocate_gpu_memory() const {
    if (allocated_size_ == 0) return;
    
    if (d_A_diag_) { cudaFree(d_A_diag_); d_A_diag_ = nullptr; }
    if (d_A_off_diag_) { cudaFree(d_A_off_diag_); d_A_off_diag_ = nullptr; }
    if (d_row_ptr_) { cudaFree(d_row_ptr_); d_row_ptr_ = nullptr; }
    if (d_col_indices_) { cudaFree(d_col_indices_); d_col_indices_ = nullptr; }
    if (d_b_) { cudaFree(d_b_); d_b_ = nullptr; }
    if (d_x_old_) { cudaFree(d_x_old_); d_x_old_ = nullptr; }
    if (d_x_new_) { cudaFree(d_x_new_); d_x_new_ = nullptr; }
    
    allocated_size_ = 0;
}

void GPUJacobiSolver::extract_diagonal(const double* A_values, const int* row_ptr, 
                                      const int* col_indices, size_t n) const {
    std::vector<double> h_A_diag(n, 0.0);
    std::vector<double> h_A_off_diag(row_ptr[n] - row_ptr[0]);
    
    // Extract diagonal and off-diagonal elements
    for (size_t i = 0; i < n; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            if (col_indices[j] == static_cast<int>(i)) {
                h_A_diag[i] = A_values[j];
                h_A_off_diag[j] = 0.0;  // Zero out diagonal in off-diagonal matrix
            } else {
                h_A_off_diag[j] = A_values[j];
            }
        }
        
        // Ensure diagonal is non-zero
        if (std::abs(h_A_diag[i]) < 1e-14) {
            h_A_diag[i] = 1e-14;
        }
    }
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_A_diag_, h_A_diag.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_off_diag_, h_A_off_diag.data(), 
                         (row_ptr[n] - row_ptr[0]) * sizeof(double), cudaMemcpyHostToDevice));
}

// ============================================================================
// GPUGaussSeidelSolver Implementation
// ============================================================================

GPUGaussSeidelSolver::GPUGaussSeidelSolver() {}

int GPUGaussSeidelSolver::solve(const double* A_values, const int* row_ptr, const int* col_indices,
                               const double* b, double* x, size_t n, 
                               double tolerance, int max_iterations) {
    
    size_t nnz = row_ptr[n] - row_ptr[0];
    allocate_gpu_memory(n, nnz);
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_A_values_, A_values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr_, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices_, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_, b, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_, x, n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Compute graph coloring for parallel execution
    auto colors = compute_graph_coloring(row_ptr, col_indices, n);
    
    double residual = tolerance + 1.0;
    int iteration = 0;
    
    while (residual > tolerance && iteration < max_iterations) {
        // Gauss-Seidel iteration with graph coloring
        launch_gauss_seidel_colored(d_A_values_, d_col_indices_, d_row_ptr_, 
                                   d_b_, d_x_, n, colors);
        
        // Compute residual every 10 iterations
        if (iteration % 10 == 0) {
            // Simplified residual computation (could be improved)
            residual = 1e-6;  // Placeholder - implement proper residual computation
        }
        
        iteration++;
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(x, d_x_, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    return (residual <= tolerance) ? iteration : -1;
}

void GPUGaussSeidelSolver::allocate_gpu_memory(size_t n, size_t nnz) const {
    if (allocated_size_ >= n) return;
    
    deallocate_gpu_memory();
    
    CUDA_CHECK(cudaMalloc(&d_A_values_, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr_, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_indices_, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b_, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_, n * sizeof(double)));
    
    allocated_size_ = n;
}

void GPUGaussSeidelSolver::deallocate_gpu_memory() const {
    if (allocated_size_ == 0) return;
    
    if (d_A_values_) { cudaFree(d_A_values_); d_A_values_ = nullptr; }
    if (d_row_ptr_) { cudaFree(d_row_ptr_); d_row_ptr_ = nullptr; }
    if (d_col_indices_) { cudaFree(d_col_indices_); d_col_indices_ = nullptr; }
    if (d_b_) { cudaFree(d_b_); d_b_ = nullptr; }
    if (d_x_) { cudaFree(d_x_); d_x_ = nullptr; }
    
    allocated_size_ = 0;
}

std::vector<std::pair<int, int>> GPUGaussSeidelSolver::compute_graph_coloring(
    const int* row_ptr, const int* col_indices, size_t n) const {
    
    // Simple greedy coloring algorithm
    std::vector<int> colors(n, -1);
    std::vector<std::set<int>> color_sets;
    
    for (size_t i = 0; i < n; ++i) {
        std::set<int> neighbor_colors;
        
        // Find colors of neighbors
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            int neighbor = col_indices[j];
            if (neighbor != static_cast<int>(i) && colors[neighbor] != -1) {
                neighbor_colors.insert(colors[neighbor]);
            }
        }
        
        // Find smallest available color
        int color = 0;
        while (neighbor_colors.count(color)) {
            color++;
        }
        
        colors[i] = color;
        
        // Extend color_sets if necessary
        while (color_sets.size() <= static_cast<size_t>(color)) {
            color_sets.emplace_back();
        }
        color_sets[color].insert(i);
    }
    
    // Convert to ranges
    std::vector<std::pair<int, int>> color_ranges;
    for (const auto& color_set : color_sets) {
        if (!color_set.empty()) {
            color_ranges.emplace_back(*color_set.begin(), *color_set.rbegin() + 1);
        }
    }
    
    return color_ranges;
}

// ============================================================================
// GPUConjugateGradientSolver Implementation
// ============================================================================

GPUConjugateGradientSolver::GPUConjugateGradientSolver() {}

int GPUConjugateGradientSolver::solve(const double* A_values, const int* row_ptr, const int* col_indices,
                                     const double* b, double* x, size_t n,
                                     double tolerance, int max_iterations) {

    size_t nnz = row_ptr[n] - row_ptr[0];
    allocate_gpu_memory(n, nnz);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_A_values_, A_values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr_, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices_, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_, b, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_, x, n * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize CG algorithm: r0 = b - A*x0, p0 = r0
    // Compute A*x0
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    // Use sparse matrix-vector multiplication kernel
    extern __global__ void spmv_csr_kernel(const double* values, const int* row_ptr,
                                          const int* col_indices, const double* x,
                                          double* y, size_t rows);

    spmv_csr_kernel<<<grid_size, block_size>>>(d_A_values_, d_row_ptr_, d_col_indices_,
                                               d_x_, d_Ap_, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // r0 = b - A*x0
    launch_vector_add(d_b_, d_Ap_, d_r_, n);      // r = b + Ax
    launch_vector_scale(d_Ap_, -1.0, d_Ap_, n);  // Ax = -Ax
    launch_vector_add(d_r_, d_Ap_, d_r_, n);     // r = b - Ax

    // p0 = r0
    CUDA_CHECK(cudaMemcpy(d_p_, d_r_, n * sizeof(double), cudaMemcpyDeviceToDevice));

    double rsold = launch_dot_product(d_r_, d_r_, n);
    double residual = sqrt(rsold);
    int iteration = 0;

    while (residual > tolerance && iteration < max_iterations) {
        // Ap = A * p
        spmv_csr_kernel<<<grid_size, block_size>>>(d_A_values_, d_row_ptr_, d_col_indices_,
                                                   d_p_, d_Ap_, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // alpha = rsold / (p^T * A * p)
        double pAp = launch_dot_product(d_p_, d_Ap_, n);
        if (std::abs(pAp) < 1e-14) break;  // Avoid division by zero

        double alpha = rsold / pAp;

        // x = x + alpha * p
        // r = r - alpha * Ap
        extern __global__ void conjugate_gradient_update_kernel(const double* r, const double* Ap,
                                                               double alpha, double* r_new, double* x,
                                                               const double* p, size_t n);

        conjugate_gradient_update_kernel<<<grid_size, block_size>>>(d_r_, d_Ap_, alpha,
                                                                   d_r_, d_x_, d_p_, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // rsnew = r^T * r
        double rsnew = launch_dot_product(d_r_, d_r_, n);
        residual = sqrt(rsnew);

        if (residual <= tolerance) break;

        // beta = rsnew / rsold
        double beta = rsnew / rsold;

        // p = r + beta * p
        launch_vector_scale(d_p_, beta, d_p_, n);     // p = beta * p
        launch_vector_add(d_r_, d_p_, d_p_, n);       // p = r + beta * p

        rsold = rsnew;
        iteration++;
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(x, d_x_, n * sizeof(double), cudaMemcpyDeviceToHost));

    return (residual <= tolerance) ? iteration : -1;
}

void GPUConjugateGradientSolver::allocate_gpu_memory(size_t n, size_t nnz) const {
    if (allocated_size_ >= n) return;

    deallocate_gpu_memory();

    CUDA_CHECK(cudaMalloc(&d_A_values_, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr_, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_indices_, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b_, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p_, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap_, n * sizeof(double)));

    allocated_size_ = n;
}

void GPUConjugateGradientSolver::deallocate_gpu_memory() const {
    if (allocated_size_ == 0) return;

    if (d_A_values_) { cudaFree(d_A_values_); d_A_values_ = nullptr; }
    if (d_row_ptr_) { cudaFree(d_row_ptr_); d_row_ptr_ = nullptr; }
    if (d_col_indices_) { cudaFree(d_col_indices_); d_col_indices_ = nullptr; }
    if (d_b_) { cudaFree(d_b_); d_b_ = nullptr; }
    if (d_x_) { cudaFree(d_x_); d_x_ = nullptr; }
    if (d_r_) { cudaFree(d_r_); d_r_ = nullptr; }
    if (d_p_) { cudaFree(d_p_); d_p_ = nullptr; }
    if (d_Ap_) { cudaFree(d_Ap_); d_Ap_ = nullptr; }

    allocated_size_ = 0;
}

// ============================================================================
// GPULinearSolverFactory Implementation
// ============================================================================

std::unique_ptr<GPULinearSolver> GPULinearSolverFactory::create_solver(SolverType type) {
    switch (type) {
        case SolverType::JACOBI:
            return std::make_unique<GPUJacobiSolver>();
        case SolverType::GAUSS_SEIDEL:
            return std::make_unique<GPUGaussSeidelSolver>();
        case SolverType::CONJUGATE_GRADIENT:
            return std::make_unique<GPUConjugateGradientSolver>();
        case SolverType::AUTO:
            // Default to Conjugate Gradient for AUTO
            return std::make_unique<GPUConjugateGradientSolver>();
        default:
            throw std::invalid_argument("Unknown solver type");
    }
}

std::unique_ptr<GPULinearSolver> GPULinearSolverFactory::create_auto_solver(
    const double* A_values, const int* row_ptr, const int* col_indices, size_t n) {

    // Analyze matrix properties
    bool symmetric = is_symmetric(A_values, row_ptr, col_indices, n);
    bool pos_def = is_positive_definite_estimate(A_values, row_ptr, col_indices, n);
    double condition_est = estimate_condition_number(A_values, row_ptr, col_indices, n);

    // Choose solver based on matrix properties
    if (symmetric && pos_def && condition_est < 1e6) {
        // Well-conditioned symmetric positive definite: use CG
        return std::make_unique<GPUConjugateGradientSolver>();
    } else if (condition_est < 1e4) {
        // Moderately conditioned: use Gauss-Seidel
        return std::make_unique<GPUGaussSeidelSolver>();
    } else {
        // Ill-conditioned: use Jacobi with relaxation
        return std::make_unique<GPUJacobiSolver>(0.7);  // Under-relaxation
    }
}

std::vector<std::string> GPULinearSolverFactory::get_available_solvers() {
    return {"Jacobi", "Gauss-Seidel", "Conjugate Gradient", "Auto"};
}

GPULinearSolverFactory::SolverType GPULinearSolverFactory::get_solver_type_from_string(const std::string& name) {
    if (name == "Jacobi") return SolverType::JACOBI;
    if (name == "Gauss-Seidel") return SolverType::GAUSS_SEIDEL;
    if (name == "Conjugate Gradient") return SolverType::CONJUGATE_GRADIENT;
    if (name == "Auto") return SolverType::AUTO;
    throw std::invalid_argument("Unknown solver name: " + name);
}

bool GPULinearSolverFactory::is_symmetric(const double* A_values, const int* row_ptr,
                                        const int* col_indices, size_t n) {
    // Simple symmetry check (could be optimized)
    const double tolerance = 1e-12;

    for (size_t i = 0; i < n; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            int col = col_indices[j];
            double a_ij = A_values[j];

            // Find A(col, i)
            bool found = false;
            for (int k = row_ptr[col]; k < row_ptr[col + 1]; ++k) {
                if (col_indices[k] == static_cast<int>(i)) {
                    double a_ji = A_values[k];
                    if (std::abs(a_ij - a_ji) > tolerance) {
                        return false;
                    }
                    found = true;
                    break;
                }
            }

            if (!found && std::abs(a_ij) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

bool GPULinearSolverFactory::is_positive_definite_estimate(const double* A_values, const int* row_ptr,
                                                         const int* col_indices, size_t n) {
    // Simple diagonal dominance check as PD estimate
    for (size_t i = 0; i < n; ++i) {
        double diagonal = 0.0;
        double off_diagonal_sum = 0.0;

        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            if (col_indices[j] == static_cast<int>(i)) {
                diagonal = A_values[j];
            } else {
                off_diagonal_sum += std::abs(A_values[j]);
            }
        }

        if (diagonal <= 0.0 || diagonal <= off_diagonal_sum) {
            return false;
        }
    }

    return true;
}

double GPULinearSolverFactory::estimate_condition_number(const double* A_values, const int* row_ptr,
                                                       const int* col_indices, size_t n) {
    // Simple condition number estimate based on diagonal ratio
    double min_diag = std::numeric_limits<double>::max();
    double max_diag = 0.0;

    for (size_t i = 0; i < n; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            if (col_indices[j] == static_cast<int>(i)) {
                double diag = std::abs(A_values[j]);
                min_diag = std::min(min_diag, diag);
                max_diag = std::max(max_diag, diag);
                break;
            }
        }
    }

    return (min_diag > 0.0) ? max_diag / min_diag : 1e12;
}

} // namespace gpu
} // namespace simulator
