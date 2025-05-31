#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <map>

// Forward declarations for CUDA/OpenCL
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#endif

#ifdef ENABLE_OPENCL
#include <CL/cl.h>
#endif

namespace simulator {
namespace gpu {

/**
 * @brief GPU backend types
 */
enum class GPUBackend {
    NONE,     ///< No GPU acceleration
    CUDA,     ///< NVIDIA CUDA
    OPENCL,   ///< OpenCL (cross-platform)
    AUTO      ///< Automatically detect best available
};

/**
 * @brief GPU device information
 */
struct GPUDeviceInfo {
    std::string name;
    size_t global_memory;      ///< Global memory in bytes
    size_t shared_memory;      ///< Shared memory per block in bytes
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    double memory_bandwidth;   ///< Memory bandwidth in GB/s
    bool supports_double_precision;
    GPUBackend backend;
};

/**
 * @brief GPU memory management
 */
template<typename T>
class GPUMemory {
public:
    GPUMemory(size_t size, GPUBackend backend = GPUBackend::AUTO);
    ~GPUMemory();
    
    // Non-copyable but movable
    GPUMemory(const GPUMemory&) = delete;
    GPUMemory& operator=(const GPUMemory&) = delete;
    GPUMemory(GPUMemory&& other) noexcept;
    GPUMemory& operator=(GPUMemory&& other) noexcept;
    
    // Memory operations
    void copy_to_device(const T* host_data, size_t count = 0);
    void copy_to_host(T* host_data, size_t count = 0) const;
    void memset(int value);
    
    // Accessors
    void* device_ptr() const { return device_ptr_; }
    size_t size() const { return size_; }
    GPUBackend backend() const { return backend_; }
    
private:
    void* device_ptr_;
    size_t size_;
    GPUBackend backend_;
    
    void allocate();
    void deallocate();
};

/**
 * @brief GPU context manager
 */
class GPUContext {
public:
    static GPUContext& instance();
    
    bool initialize(GPUBackend preferred_backend = GPUBackend::AUTO);
    void finalize();
    
    bool is_initialized() const { return initialized_; }
    GPUBackend get_backend() const { return backend_; }
    const GPUDeviceInfo& get_device_info() const { return device_info_; }
    
    // Performance monitoring
    void start_timer(const std::string& name);
    void end_timer(const std::string& name);
    double get_elapsed_time(const std::string& name) const;
    
private:
    GPUContext() = default;
    ~GPUContext() = default;
    
    bool initialized_ = false;
    GPUBackend backend_ = GPUBackend::NONE;
    GPUDeviceInfo device_info_;
    
    // Backend-specific contexts
#ifdef ENABLE_CUDA
    cublasHandle_t cublas_handle_ = nullptr;
    cusparseHandle_t cusparse_handle_ = nullptr;
    cusolverDnHandle_t cusolver_handle_ = nullptr;
    cudaStream_t stream_ = nullptr;
#endif
    
#ifdef ENABLE_OPENCL
    cl_context cl_context_ = nullptr;
    cl_command_queue cl_queue_ = nullptr;
    cl_device_id cl_device_ = nullptr;
#endif
    
    std::vector<GPUDeviceInfo> detect_devices();
    bool initialize_cuda();
    bool initialize_opencl();
};

/**
 * @brief GPU-accelerated linear algebra operations
 */
class GPULinearAlgebra {
public:
    explicit GPULinearAlgebra(GPUBackend backend = GPUBackend::AUTO);
    ~GPULinearAlgebra();
    
    // Vector operations
    void vector_add(const double* a, const double* b, double* result, size_t n);
    void vector_scale(const double* a, double scale, double* result, size_t n);
    double dot_product(const double* a, const double* b, size_t n);
    double vector_norm(const double* a, size_t n);
    
    // Matrix operations
    void matrix_vector_multiply(const double* matrix, const double* vector, 
                               double* result, size_t rows, size_t cols);
    void matrix_matrix_multiply(const double* a, const double* b, double* c,
                               size_t m, size_t n, size_t k);
    
    // Sparse matrix operations
    void sparse_matrix_vector_multiply(const double* values, const int* row_ptr,
                                     const int* col_indices, const double* x,
                                     double* y, size_t rows, size_t nnz);
    
    // Linear solvers
    void solve_triangular(const double* matrix, const double* rhs, double* solution,
                         size_t n, bool upper = true);
    void lu_factorization(double* matrix, int* pivot, size_t n);
    void solve_lu(const double* lu_matrix, const int* pivot, const double* rhs,
                  double* solution, size_t n);
    
private:
    GPUBackend backend_;
    bool initialized_;
    
    // GPU memory pools
    std::unique_ptr<GPUMemory<double>> temp_memory_;
    std::unique_ptr<GPUMemory<int>> int_memory_;
    
    void ensure_temp_memory(size_t size);
    void ensure_int_memory(size_t size);
};

/**
 * @brief GPU-accelerated finite element operations
 */
class GPUFiniteElement {
public:
    explicit GPUFiniteElement(GPUBackend backend = GPUBackend::AUTO);
    ~GPUFiniteElement();
    
    // Basis function evaluation
    void evaluate_basis_functions(const double* xi, const double* eta, size_t n_points,
                                 double* basis_values, double* basis_gradients,
                                 int polynomial_order);
    
    // Element matrix assembly
    void assemble_element_matrices(const double* vertices, const int* elements,
                                  size_t n_elements, double* element_matrices,
                                  int dofs_per_element);
    
    // Global assembly
    void assemble_global_matrix(const double* element_matrices, const int* element_dofs,
                               size_t n_elements, int dofs_per_element,
                               double* global_matrix, size_t matrix_size);
    
    // Numerical integration
    void integrate_elements(const double* basis_values, const double* weights,
                           const double* jacobians, size_t n_quad_points,
                           size_t n_elements, double* integrals);
    
private:
    GPUBackend backend_;
    bool initialized_;
    
    // Compiled GPU kernels
    void* basis_kernel_;
    void* assembly_kernel_;
    void* integration_kernel_;
    
    void compile_kernels();
    void cleanup_kernels();
};

/**
 * @brief GPU-accelerated physics computations
 */
class GPUPhysics {
public:
    explicit GPUPhysics(GPUBackend backend = GPUBackend::AUTO);
    ~GPUPhysics();
    
    // Semiconductor physics
    void compute_carrier_densities(const double* potential, const double* doping_nd,
                                  const double* doping_na, double* n, double* p,
                                  size_t n_points, double temperature = 300.0);
    
    void compute_current_densities(const double* potential, const double* n, const double* p,
                                  const double* mobility_n, const double* mobility_p,
                                  double* jn, double* jp, size_t n_points,
                                  double temperature = 300.0);
    
    void compute_recombination(const double* n, const double* p, const double* ni,
                              double* recombination, size_t n_points,
                              double tau_n = 1e-6, double tau_p = 1e-6);
    
    // Electric field computation
    void compute_electric_field(const double* potential, const int* elements,
                               const double* vertices, double* ex, double* ey,
                               size_t n_elements);
    
    // Charge density computation
    void compute_charge_density(const double* n, const double* p, 
                               const double* doping_nd, const double* doping_na,
                               double* rho, size_t n_points, double q = 1.602e-19);
    
private:
    GPUBackend backend_;
    bool initialized_;
    
    // Physics kernels
    void* carrier_density_kernel_;
    void* current_density_kernel_;
    void* recombination_kernel_;
    void* field_kernel_;
    void* charge_kernel_;
    
    void compile_physics_kernels();
    void cleanup_physics_kernels();
};

/**
 * @brief GPU performance profiler
 */
class GPUProfiler {
public:
    static GPUProfiler& instance();
    
    void start_event(const std::string& name);
    void end_event(const std::string& name);
    
    struct ProfileData {
        std::string name;
        double gpu_time_ms;
        double cpu_time_ms;
        size_t memory_transferred_bytes;
        double memory_bandwidth_gbps;
        size_t call_count;
    };
    
    std::vector<ProfileData> get_profile_data() const;
    void reset();
    void print_profile() const;
    
    // Memory usage tracking
    void track_memory_allocation(size_t bytes);
    void track_memory_deallocation(size_t bytes);
    size_t get_current_memory_usage() const { return current_memory_usage_; }
    size_t get_peak_memory_usage() const { return peak_memory_usage_; }
    
private:
    GPUProfiler() = default;
    
    struct EventData {
        double start_time;
        double end_time;
        size_t memory_start;
        size_t memory_end;
    };
    
    std::map<std::string, EventData> active_events_;
    std::map<std::string, ProfileData> profile_data_;
    size_t current_memory_usage_ = 0;
    size_t peak_memory_usage_ = 0;
};

/**
 * @brief Automatic GPU/CPU hybrid execution
 */
class HybridExecutor {
public:
    HybridExecutor();
    
    // Automatically choose GPU or CPU based on problem size and complexity
    template<typename Func>
    auto execute(Func&& gpu_func, Func&& cpu_func, size_t problem_size) -> decltype(gpu_func());
    
    // Performance prediction
    double predict_gpu_time(size_t problem_size, const std::string& operation) const;
    double predict_cpu_time(size_t problem_size, const std::string& operation) const;
    
    // Adaptive thresholds
    void update_thresholds(const std::string& operation, size_t size, 
                          double gpu_time, double cpu_time);
    
private:
    std::map<std::string, size_t> gpu_thresholds_;
    std::map<std::string, std::vector<std::pair<size_t, double>>> performance_history_;
    
    bool should_use_gpu(size_t problem_size, const std::string& operation) const;
};

// Utility macros for GPU profiling
#define GPU_PROFILE_START(name) gpu::GPUProfiler::instance().start_event(name)
#define GPU_PROFILE_END(name) gpu::GPUProfiler::instance().end_event(name)
#define GPU_PROFILE_SCOPE(name) struct GPUProfileScope { \
    GPUProfileScope() { gpu::GPUProfiler::instance().start_event(name); } \
    ~GPUProfileScope() { gpu::GPUProfiler::instance().end_event(name); } \
} gpu_profile_scope_##__LINE__

} // namespace gpu
} // namespace simulator
