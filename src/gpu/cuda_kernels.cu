#include "gpu_acceleration.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <device_launch_parameters.h>
#include <cmath>

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

} // namespace gpu
} // namespace simulator
