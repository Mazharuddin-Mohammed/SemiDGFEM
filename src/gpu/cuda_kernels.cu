#include "gpu_acceleration.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace simulator {
namespace gpu {

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error: " + std::to_string(status)); \
    } \
} while(0)

// CUDA kernels for vector operations
__global__ void vector_add_kernel(const double* a, const double* b, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        result[i] = a[i] + b[i];
    }
}

__global__ void vector_scale_kernel(const double* a, double scale, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        result[i] = a[i] * scale;
    }
}

// Optimized reduction kernel for dot product
__global__ void dot_product_kernel(const double* a, const double* b, double* result, size_t n) {
    extern __shared__ double sdata[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Initialize shared memory
    sdata[tid] = 0.0;
    
    // Grid-stride loop for coalesced memory access
    while (idx < n) {
        sdata[tid] += a[idx] * b[idx];
        idx += stride;
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// Physics kernels for semiconductor simulation
__global__ void compute_carrier_densities_kernel(
    const double* potential, const double* doping_nd, const double* doping_na,
    double* n, double* p, size_t n_points, double temperature) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    const double kT = 8.617e-5 * temperature;  // Boltzmann constant * temperature (eV)
    const double ni = 1e10 * 1e6;  // Intrinsic carrier concentration (1/m³)
    const double q = 1.602e-19;    // Elementary charge
    
    for (size_t i = idx; i < n_points; i += stride) {
        // Electrostatic potential in eV
        double phi = potential[i] * q / (kT * q);  // Convert to thermal voltage units
        
        // Boltzmann statistics
        double n_intrinsic = ni * exp(phi);
        double p_intrinsic = ni * exp(-phi);
        
        // Apply doping
        double nd = doping_nd[i];
        double na = doping_na[i];
        
        if (nd > na) {
            // n-type material
            n[i] = n_intrinsic + (nd - na);
            p[i] = p_intrinsic;
        } else if (na > nd) {
            // p-type material
            n[i] = n_intrinsic;
            p[i] = p_intrinsic + (na - nd);
        } else {
            // Intrinsic material
            n[i] = n_intrinsic;
            p[i] = p_intrinsic;
        }
        
        // Ensure positive concentrations
        n[i] = fmax(n[i], ni);
        p[i] = fmax(p[i], ni);
    }
}

__global__ void compute_current_densities_kernel(
    const double* potential, const double* n, const double* p,
    const double* mobility_n, const double* mobility_p,
    double* jn, double* jp, size_t n_points, double temperature) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    const double kT = 8.617e-5 * temperature;  // Thermal voltage (eV)
    const double q = 1.602e-19;               // Elementary charge
    
    for (size_t i = idx; i < n_points; i += stride) {
        if (i > 0 && i < n_points - 1) {
            // Simple finite difference for electric field
            double ex = -(potential[i + 1] - potential[i - 1]) / 2.0;  // Simplified spacing
            
            // Einstein relation: D = μ * kT / q
            double dn = mobility_n[i] * kT / q;
            double dp = mobility_p[i] * kT / q;
            
            // Current density: J = q * μ * n * E + q * D * ∇n
            double grad_n = (n[i + 1] - n[i - 1]) / 2.0;  // Simplified gradient
            double grad_p = (p[i + 1] - p[i - 1]) / 2.0;
            
            jn[i] = q * mobility_n[i] * n[i] * ex + q * dn * grad_n;
            jp[i] = q * mobility_p[i] * p[i] * ex - q * dp * grad_p;  // Note: holes move opposite to field
        } else {
            jn[i] = 0.0;
            jp[i] = 0.0;
        }
    }
}

__global__ void compute_recombination_kernel(
    const double* n, const double* p, const double* ni,
    double* recombination, size_t n_points, double tau_n, double tau_p) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n_points; i += stride) {
        // Shockley-Read-Hall recombination
        double np = n[i] * p[i];
        double ni_sq = ni[i] * ni[i];
        double denominator = tau_p * (n[i] + ni[i]) + tau_n * (p[i] + ni[i]);
        
        if (denominator > 1e-20) {
            recombination[i] = (np - ni_sq) / denominator;
        } else {
            recombination[i] = 0.0;
        }
    }
}

// P3 basis function evaluation kernel
__global__ void evaluate_p3_basis_kernel(
    const double* xi, const double* eta, size_t n_points,
    double* basis_values, double* basis_gradients) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    const int num_basis = 10;  // P3 has 10 basis functions
    
    for (size_t p = idx; p < n_points; p += stride) {
        double xi_val = xi[p];
        double eta_val = eta[p];
        double zeta = 1.0 - xi_val - eta_val;
        
        // Vertex basis functions
        basis_values[p * num_basis + 0] = 0.5 * zeta * (3.0 * zeta - 1.0) * (3.0 * zeta - 2.0);
        basis_values[p * num_basis + 1] = 0.5 * xi_val * (3.0 * xi_val - 1.0) * (3.0 * xi_val - 2.0);
        basis_values[p * num_basis + 2] = 0.5 * eta_val * (3.0 * eta_val - 1.0) * (3.0 * eta_val - 2.0);
        
        // Edge basis functions
        basis_values[p * num_basis + 3] = 4.5 * zeta * xi_val * (3.0 * zeta - 1.0);
        basis_values[p * num_basis + 4] = 4.5 * zeta * xi_val * (3.0 * xi_val - 1.0);
        basis_values[p * num_basis + 5] = 4.5 * xi_val * eta_val * (3.0 * xi_val - 1.0);
        basis_values[p * num_basis + 6] = 4.5 * xi_val * eta_val * (3.0 * eta_val - 1.0);
        basis_values[p * num_basis + 7] = 4.5 * eta_val * zeta * (3.0 * eta_val - 1.0);
        basis_values[p * num_basis + 8] = 4.5 * eta_val * zeta * (3.0 * zeta - 1.0);
        
        // Interior basis function
        basis_values[p * num_basis + 9] = 27.0 * zeta * xi_val * eta_val;
        
        // Complete gradient computations for all P3 basis functions
        // Vertex gradients
        basis_gradients[p * num_basis * 2 + 0 * 2 + 0] = -0.5 * (27.0 * zeta * zeta - 18.0 * zeta + 2.0);
        basis_gradients[p * num_basis * 2 + 0 * 2 + 1] = -0.5 * (27.0 * zeta * zeta - 18.0 * zeta + 2.0);

        basis_gradients[p * num_basis * 2 + 1 * 2 + 0] = 0.5 * (27.0 * xi_val * xi_val - 12.0 * xi_val + 1.0);
        basis_gradients[p * num_basis * 2 + 1 * 2 + 1] = 0.0;

        basis_gradients[p * num_basis * 2 + 2 * 2 + 0] = 0.0;
        basis_gradients[p * num_basis * 2 + 2 * 2 + 1] = 0.5 * (27.0 * eta_val * eta_val - 12.0 * eta_val + 1.0);

        // Edge gradients (complete implementation)
        basis_gradients[p * num_basis * 2 + 3 * 2 + 0] = 4.5 * (zeta * (3.0 * zeta - 1.0) - xi_val * (6.0 * zeta - 1.0));
        basis_gradients[p * num_basis * 2 + 3 * 2 + 1] = -4.5 * xi_val * (6.0 * zeta - 1.0);

        basis_gradients[p * num_basis * 2 + 4 * 2 + 0] = 4.5 * (zeta * (6.0 * xi_val - 1.0) + xi_val * (3.0 * xi_val - 1.0));
        basis_gradients[p * num_basis * 2 + 4 * 2 + 1] = -4.5 * xi_val * (3.0 * xi_val - 1.0);

        basis_gradients[p * num_basis * 2 + 5 * 2 + 0] = 4.5 * eta_val * (6.0 * xi_val - 1.0);
        basis_gradients[p * num_basis * 2 + 5 * 2 + 1] = 4.5 * xi_val * (3.0 * xi_val - 1.0);

        basis_gradients[p * num_basis * 2 + 6 * 2 + 0] = 4.5 * eta_val * (3.0 * eta_val - 1.0);
        basis_gradients[p * num_basis * 2 + 6 * 2 + 1] = 4.5 * xi_val * (6.0 * eta_val - 1.0);

        basis_gradients[p * num_basis * 2 + 7 * 2 + 0] = -4.5 * eta_val * (3.0 * eta_val - 1.0);
        basis_gradients[p * num_basis * 2 + 7 * 2 + 1] = 4.5 * (zeta * (6.0 * eta_val - 1.0) + eta_val * (3.0 * eta_val - 1.0));

        basis_gradients[p * num_basis * 2 + 8 * 2 + 0] = -4.5 * eta_val * (6.0 * zeta - 1.0);
        basis_gradients[p * num_basis * 2 + 8 * 2 + 1] = 4.5 * (zeta * (3.0 * zeta - 1.0) - eta_val * (6.0 * zeta - 1.0));

        // Interior gradient
        basis_gradients[p * num_basis * 2 + 9 * 2 + 0] = 27.0 * (zeta * eta_val - xi_val * eta_val);
        basis_gradients[p * num_basis * 2 + 9 * 2 + 1] = 27.0 * (zeta * xi_val - eta_val * xi_val);
    }
}

// Element matrix assembly kernel
__global__ void assemble_element_matrices_kernel(
    const double* vertices, const int* elements, size_t n_elements,
    double* element_matrices, int dofs_per_element) {
    
    size_t elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (elem_idx >= n_elements) return;
    
    // Get element vertices
    int v1 = elements[elem_idx * 3 + 0];
    int v2 = elements[elem_idx * 3 + 1];
    int v3 = elements[elem_idx * 3 + 2];
    
    double x1 = vertices[v1 * 2 + 0], y1 = vertices[v1 * 2 + 1];
    double x2 = vertices[v2 * 2 + 0], y2 = vertices[v2 * 2 + 1];
    double x3 = vertices[v3 * 2 + 0], y3 = vertices[v3 * 2 + 1];
    
    // Compute Jacobian
    double J11 = x2 - x1, J12 = x3 - x1;
    double J21 = y2 - y1, J22 = y3 - y1;
    double det_J = J11 * J22 - J12 * J21;
    double area = 0.5 * fabs(det_J);
    
    if (area < 1e-12) return;  // Skip degenerate elements
    
    // Simplified element matrix (mass matrix for demonstration)
    double* elem_matrix = &element_matrices[elem_idx * dofs_per_element * dofs_per_element];
    
    for (int i = 0; i < dofs_per_element; ++i) {
        for (int j = 0; j < dofs_per_element; ++j) {
            if (i == j) {
                elem_matrix[i * dofs_per_element + j] = area / 3.0;  // Diagonal mass matrix
            } else {
                elem_matrix[i * dofs_per_element + j] = area / 12.0; // Off-diagonal
            }
        }
    }
}

// Sparse matrix-vector multiplication kernel
__global__ void spmv_csr_kernel(const double* values, const int* row_ptr, const int* col_indices,
                                const double* x, double* y, size_t rows) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= rows) return;
    
    double sum = 0.0;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    
    for (int j = start; j < end; ++j) {
        sum += values[j] * x[col_indices[j]];
    }
    
    y[row] = sum;
}

// Host functions for kernel launches
void launch_vector_add(const double* a, const double* b, double* result, size_t n) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    vector_add_kernel<<<grid_size, block_size>>>(a, b, result, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_vector_scale(const double* a, double scale, double* result, size_t n) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    vector_scale_kernel<<<grid_size, block_size>>>(a, scale, result, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

double launch_dot_product(const double* a, const double* b, size_t n) {
    const int block_size = 256;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), 1024);
    
    // Allocate temporary memory for partial results
    double* d_partial_results;
    CUDA_CHECK(cudaMalloc(&d_partial_results, grid_size * sizeof(double)));
    
    // Launch kernel with shared memory
    size_t shared_mem_size = block_size * sizeof(double);
    dot_product_kernel<<<grid_size, block_size, shared_mem_size>>>(a, b, d_partial_results, n);
    
    // Copy partial results to host and sum
    std::vector<double> h_partial_results(grid_size);
    CUDA_CHECK(cudaMemcpy(h_partial_results.data(), d_partial_results, 
                         grid_size * sizeof(double), cudaMemcpyDeviceToHost));
    
    double result = 0.0;
    for (double partial : h_partial_results) {
        result += partial;
    }
    
    CUDA_CHECK(cudaFree(d_partial_results));
    return result;
}

void launch_compute_carrier_densities(const double* potential, const double* doping_nd,
                                     const double* doping_na, double* n, double* p,
                                     size_t n_points, double temperature) {
    const int block_size = 256;
    const int grid_size = (n_points + block_size - 1) / block_size;
    
    compute_carrier_densities_kernel<<<grid_size, block_size>>>(
        potential, doping_nd, doping_na, n, p, n_points, temperature);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_evaluate_p3_basis(const double* xi, const double* eta, size_t n_points,
                              double* basis_values, double* basis_gradients) {
    const int block_size = 256;
    const int grid_size = (n_points + block_size - 1) / block_size;
    
    evaluate_p3_basis_kernel<<<grid_size, block_size>>>(
        xi, eta, n_points, basis_values, basis_gradients);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Advanced GPU-accelerated linear solvers
__global__ void jacobi_iteration_kernel(const double* A_diag, const double* A_off_diag,
                                       const int* col_indices, const int* row_ptr,
                                       const double* b, const double* x_old, double* x_new,
                                       size_t n, double omega) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    double sum = 0.0;
    int start = row_ptr[i];
    int end = row_ptr[i + 1];

    // Compute off-diagonal contribution
    for (int j = start; j < end; ++j) {
        int col = col_indices[j];
        if (col != i) {
            sum += A_off_diag[j] * x_old[col];
        }
    }

    // Jacobi update with relaxation
    x_new[i] = (1.0 - omega) * x_old[i] + omega * (b[i] - sum) / A_diag[i];
}

__global__ void gauss_seidel_kernel(const double* A_values, const int* col_indices,
                                   const int* row_ptr, const double* b, double* x,
                                   size_t n, int color_start, int color_end) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + color_start;

    if (i >= color_end || i >= n) return;

    double sum = 0.0;
    double diag = 0.0;
    int start = row_ptr[i];
    int end = row_ptr[i + 1];

    for (int j = start; j < end; ++j) {
        int col = col_indices[j];
        if (col == i) {
            diag = A_values[j];
        } else {
            sum += A_values[j] * x[col];
        }
    }

    if (fabs(diag) > 1e-14) {
        x[i] = (b[i] - sum) / diag;
    }
}

__global__ void conjugate_gradient_axpy_kernel(const double* x, const double* p,
                                              double alpha, double* result, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = i; idx < n; idx += stride) {
        result[idx] = x[idx] + alpha * p[idx];
    }
}

__global__ void conjugate_gradient_update_kernel(const double* r, const double* Ap,
                                                 double alpha, double* r_new, double* x,
                                                 const double* p, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = i; idx < n; idx += stride) {
        r_new[idx] = r[idx] - alpha * Ap[idx];
        x[idx] = x[idx] + alpha * p[idx];
    }
}

// Memory optimization kernels
__global__ void memory_coalescing_transpose_kernel(const double* input, double* output,
                                                  size_t rows, size_t cols) {
    __shared__ double tile[32][33];  // +1 to avoid bank conflicts

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    // Load data into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Write transposed data
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;

    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void prefetch_kernel(const double* data, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Prefetch data into L2 cache
    for (size_t idx = i; idx < n; idx += stride) {
        __ldg(&data[idx]);  // Read-only cache load
    }
}

// Advanced physics kernels with optimizations
__global__ void compute_mobility_temperature_kernel(const double* temperature,
                                                   const double* doping_total,
                                                   double* mobility_n, double* mobility_p,
                                                   size_t n_points, int material_type) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Material-specific parameters (Silicon as example)
    const double mu_n_min = 65.0;    // cm²/V·s
    const double mu_n_max = 1350.0;
    const double mu_p_min = 48.0;
    const double mu_p_max = 480.0;
    const double N_ref_n = 1.26e17;  // cm⁻³
    const double N_ref_p = 2.35e17;
    const double alpha_n = 0.88;
    const double alpha_p = 0.88;
    const double T_ref = 300.0;      // K

    for (size_t idx = i; idx < n_points; idx += stride) {
        double T = temperature[idx];
        double N_total = doping_total[idx] * 1e-6;  // Convert to cm⁻³

        // Temperature dependence
        double T_ratio = T / T_ref;
        double temp_factor = pow(T_ratio, -2.3);

        // Caughey-Thomas model
        double mu_n_lattice = (mu_n_max - mu_n_min) * temp_factor + mu_n_min;
        double mu_p_lattice = (mu_p_max - mu_p_min) * temp_factor + mu_p_min;

        // Doping dependence
        double mu_n_doping = mu_n_lattice / (1.0 + pow(N_total / N_ref_n, alpha_n));
        double mu_p_doping = mu_p_lattice / (1.0 + pow(N_total / N_ref_p, alpha_p));

        mobility_n[idx] = mu_n_doping * 1e-4;  // Convert to m²/V·s
        mobility_p[idx] = mu_p_doping * 1e-4;
    }
}

__global__ void compute_generation_recombination_kernel(const double* n, const double* p,
                                                       const double* ni, const double* temperature,
                                                       double* generation, double* recombination,
                                                       size_t n_points, double optical_power) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    const double tau_n0 = 1e-6;  // Electron lifetime (s)
    const double tau_p0 = 1e-6;  // Hole lifetime (s)
    const double B_rad = 1e-15;  // Radiative recombination coefficient
    const double C_n = 1e-43;    // Auger coefficient for electrons
    const double C_p = 1e-43;    // Auger coefficient for holes

    for (size_t idx = i; idx < n_points; idx += stride) {
        double n_val = n[idx];
        double p_val = p[idx];
        double ni_val = ni[idx];
        double T = temperature[idx];

        // SRH recombination
        double np = n_val * p_val;
        double ni_sq = ni_val * ni_val;
        double R_srh = (np - ni_sq) / (tau_p0 * (n_val + ni_val) + tau_n0 * (p_val + ni_val));

        // Radiative recombination
        double R_rad = B_rad * (np - ni_sq);

        // Auger recombination
        double R_auger = (C_n * n_val + C_p * p_val) * (np - ni_sq);

        // Optical generation (simplified)
        double G_opt = optical_power * 1e21;  // Simplified generation rate

        recombination[idx] = R_srh + R_rad + R_auger;
        generation[idx] = G_opt;
    }
}

// Host functions for advanced GPU operations
void launch_jacobi_iteration(const double* A_diag, const double* A_off_diag,
                           const int* col_indices, const int* row_ptr,
                           const double* b, const double* x_old, double* x_new,
                           size_t n, double omega) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    jacobi_iteration_kernel<<<grid_size, block_size>>>(
        A_diag, A_off_diag, col_indices, row_ptr, b, x_old, x_new, n, omega);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_gauss_seidel_colored(const double* A_values, const int* col_indices,
                               const int* row_ptr, const double* b, double* x,
                               size_t n, const std::vector<std::pair<int, int>>& colors) {
    const int block_size = 256;

    for (const auto& color : colors) {
        int color_size = color.second - color.first;
        int grid_size = (color_size + block_size - 1) / block_size;

        gauss_seidel_kernel<<<grid_size, block_size>>>(
            A_values, col_indices, row_ptr, b, x, n, color.first, color.second);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void launch_memory_optimized_transpose(const double* input, double* output,
                                     size_t rows, size_t cols) {
    dim3 block_size(32, 32);
    dim3 grid_size((cols + block_size.x - 1) / block_size.x,
                   (rows + block_size.y - 1) / block_size.y);

    memory_coalescing_transpose_kernel<<<grid_size, block_size>>>(
        input, output, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_advanced_physics_computation(const double* n, const double* p,
                                       const double* temperature, const double* doping,
                                       double* mobility_n, double* mobility_p,
                                       double* generation, double* recombination,
                                       size_t n_points, int material_type, double optical_power) {
    const int block_size = 256;
    const int grid_size = (n_points + block_size - 1) / block_size;

    // Compute temperature-dependent mobility
    compute_mobility_temperature_kernel<<<grid_size, block_size>>>(
        temperature, doping, mobility_n, mobility_p, n_points, material_type);

    // Compute generation and recombination
    double* ni;
    CUDA_CHECK(cudaMalloc(&ni, n_points * sizeof(double)));

    // Initialize intrinsic concentration (simplified)
    launch_vector_scale(temperature, 1e16, ni, n_points);  // Simplified ni calculation

    compute_generation_recombination_kernel<<<grid_size, block_size>>>(
        n, p, ni, temperature, generation, recombination, n_points, optical_power);

    CUDA_CHECK(cudaFree(ni));
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gpu
} // namespace simulator
