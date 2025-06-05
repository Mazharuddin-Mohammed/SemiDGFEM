#include "gpu_acceleration.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstring>

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

// GPU Context Implementation
GPUContext& GPUContext::instance() {
    static GPUContext instance;
    return instance;
}

bool GPUContext::initialize(GPUBackend preferred_backend) {
    if (initialized_) {
        return true;
    }
    
    // Detect available devices
    auto devices = detect_devices();
    if (devices.empty()) {
        std::cerr << "No GPU devices found" << std::endl;
        return false;
    }
    
    // Select best device based on preference
    GPUDeviceInfo best_device;
    bool found = false;
    
    if (preferred_backend == GPUBackend::AUTO) {
        // Choose best available device
        for (const auto& device : devices) {
            if (!found || device.global_memory > best_device.global_memory) {
                best_device = device;
                found = true;
            }
        }
    } else {
        // Find device with preferred backend
        for (const auto& device : devices) {
            if (device.backend == preferred_backend) {
                best_device = device;
                found = true;
                break;
            }
        }
    }
    
    if (!found) {
        std::cerr << "No suitable GPU device found" << std::endl;
        return false;
    }
    
    device_info_ = best_device;
    backend_ = best_device.backend;
    
    // Initialize backend-specific context
    bool success = false;
    switch (backend_) {
#ifdef ENABLE_CUDA
        case GPUBackend::CUDA:
            success = initialize_cuda();
            break;
#endif
#ifdef ENABLE_OPENCL
        case GPUBackend::OPENCL:
            success = initialize_opencl();
            break;
#endif
        default:
            std::cerr << "Unsupported GPU backend" << std::endl;
            return false;
    }
    
    if (success) {
        initialized_ = true;
        std::cout << "GPU initialized: " << device_info_.name 
                  << " (" << device_info_.global_memory / (1024*1024) << " MB)" << std::endl;
    }
    
    return success;
}

void GPUContext::finalize() {
    if (!initialized_) return;
    
#ifdef ENABLE_CUDA
    if (backend_ == GPUBackend::CUDA) {
        if (cusolver_handle_) cusolverDnDestroy(cusolver_handle_);
        if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
        if (cublas_handle_) cublasDestroy(cublas_handle_);
        if (stream_) cudaStreamDestroy(stream_);
        cudaDeviceReset();
    }
#endif
    
#ifdef ENABLE_OPENCL
    if (backend_ == GPUBackend::OPENCL) {
        if (cl_queue_) clReleaseCommandQueue(cl_queue_);
        if (cl_context_) clReleaseContext(cl_context_);
    }
#endif
    
    initialized_ = false;
}

std::vector<GPUDeviceInfo> GPUContext::detect_devices() {
    std::vector<GPUDeviceInfo> devices;
    
#ifdef ENABLE_CUDA
    // Detect CUDA devices
    int cuda_device_count = 0;
    if (cudaGetDeviceCount(&cuda_device_count) == cudaSuccess && cuda_device_count > 0) {
        for (int i = 0; i < cuda_device_count; ++i) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                GPUDeviceInfo info;
                info.name = prop.name;
                info.global_memory = prop.totalGlobalMem;
                info.shared_memory = prop.sharedMemPerBlock;
                info.compute_capability_major = prop.major;
                info.compute_capability_minor = prop.minor;
                info.multiprocessor_count = prop.multiProcessorCount;
                info.max_threads_per_block = prop.maxThreadsPerBlock;
                info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
                info.memory_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
                info.supports_double_precision = (prop.major >= 2 || (prop.major == 1 && prop.minor >= 3));
                info.backend = GPUBackend::CUDA;
                devices.push_back(info);
            }
        }
    }
#endif
    
#ifdef ENABLE_OPENCL
    // Detect OpenCL devices
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) == CL_SUCCESS && platform_count > 0) {
        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);
        
        for (auto platform : platforms) {
            cl_uint device_count = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count) == CL_SUCCESS) {
                std::vector<cl_device_id> cl_devices(device_count);
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, cl_devices.data(), nullptr);
                
                for (auto device : cl_devices) {
                    GPUDeviceInfo info;
                    
                    char name[256];
                    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
                    info.name = name;
                    
                    cl_ulong mem_size;
                    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, nullptr);
                    info.global_memory = mem_size;
                    
                    cl_ulong local_mem_size;
                    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, nullptr);
                    info.shared_memory = local_mem_size;
                    
                    cl_uint compute_units;
                    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
                    info.multiprocessor_count = compute_units;
                    
                    size_t max_work_group_size;
                    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
                    info.max_threads_per_block = max_work_group_size;
                    
                    cl_device_fp_config fp_config;
                    clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp_config), &fp_config, nullptr);
                    info.supports_double_precision = (fp_config != 0);
                    
                    info.backend = GPUBackend::OPENCL;
                    devices.push_back(info);
                }
            }
        }
    }
#endif
    
    return devices;
}

#ifdef ENABLE_CUDA
bool GPUContext::initialize_cuda() {
    try {
        // Set device
        cudaError_t error = cudaSetDevice(0);
        if (error != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        // Create stream
        error = cudaStreamCreate(&stream_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        // Initialize cuBLAS
        cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Failed to initialize cuBLAS" << std::endl;
            return false;
        }
        cublasSetStream(cublas_handle_, stream_);
        
        // Initialize cuSPARSE
        cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle_);
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "Failed to initialize cuSPARSE" << std::endl;
            return false;
        }
        cusparseSetStream(cusparse_handle_, stream_);
        
        // Initialize cuSOLVER
        cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver_handle_);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
            std::cerr << "Failed to initialize cuSOLVER" << std::endl;
            return false;
        }
        cusolverDnSetStream(cusolver_handle_, stream_);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CUDA initialization failed: " << e.what() << std::endl;
        return false;
    }
}
#endif

#ifdef ENABLE_OPENCL
bool GPUContext::initialize_opencl() {
    try {
        // Get platform
        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);
        if (platform_count == 0) return false;
        
        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);
        
        // Get device
        cl_uint device_count = 0;
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if (device_count == 0) return false;
        
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &cl_device_, nullptr);
        
        // Create context
        cl_int error;
        cl_context_ = clCreateContext(nullptr, 1, &cl_device_, nullptr, nullptr, &error);
        if (error != CL_SUCCESS) return false;
        
        // Create command queue
        cl_queue_ = clCreateCommandQueue(cl_context_, cl_device_, CL_QUEUE_PROFILING_ENABLE, &error);
        if (error != CL_SUCCESS) return false;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "OpenCL initialization failed: " << e.what() << std::endl;
        return false;
    }
}
#endif

void GPUContext::start_timer(const std::string& name) {
    // Simple timer implementation
    (void)name; // Suppress unused parameter warning
    // In a full implementation, this would store timing data
}

void GPUContext::end_timer(const std::string& name) {
    // Simple timer implementation
    (void)name; // Suppress unused parameter warning
    // In a full implementation, this would calculate elapsed time
}

double GPUContext::get_elapsed_time(const std::string& name) const {
    // Simple timer implementation
    (void)name; // Suppress unused parameter warning
    // In a full implementation, this would return stored timing data
    return 0.0; // Default return value
}

// GPU Memory Implementation
template<typename T>
GPUMemory<T>::GPUMemory(size_t size, GPUBackend backend) 
    : device_ptr_(nullptr), size_(size), backend_(backend) {
    
    if (backend_ == GPUBackend::AUTO) {
        backend_ = GPUContext::instance().get_backend();
    }
    
    allocate();
}

template<typename T>
GPUMemory<T>::~GPUMemory() {
    deallocate();
}

template<typename T>
GPUMemory<T>::GPUMemory(GPUMemory&& other) noexcept
    : device_ptr_(other.device_ptr_), size_(other.size_), backend_(other.backend_) {
    other.device_ptr_ = nullptr;
    other.size_ = 0;
}

template<typename T>
GPUMemory<T>& GPUMemory<T>::operator=(GPUMemory&& other) noexcept {
    if (this != &other) {
        deallocate();
        device_ptr_ = other.device_ptr_;
        size_ = other.size_;
        backend_ = other.backend_;
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

template<typename T>
void GPUMemory<T>::allocate() {
    if (size_ == 0) return;
    
    size_t bytes = size_ * sizeof(T);
    
#ifdef ENABLE_CUDA
    if (backend_ == GPUBackend::CUDA) {
        cudaError_t error = cudaMalloc(&device_ptr_, bytes);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(error)));
        }
        return;
    }
#endif
    
#ifdef ENABLE_OPENCL
    if (backend_ == GPUBackend::OPENCL) {
        cl_int error;
        device_ptr_ = clCreateBuffer(GPUContext::instance().cl_context_, 
                                   CL_MEM_READ_WRITE, bytes, nullptr, &error);
        if (error != CL_SUCCESS) {
            throw std::runtime_error("OpenCL memory allocation failed");
        }
        return;
    }
#endif
    
    throw std::runtime_error("Unsupported GPU backend for memory allocation");
}

template<typename T>
void GPUMemory<T>::deallocate() {
    if (!device_ptr_) return;
    
#ifdef ENABLE_CUDA
    if (backend_ == GPUBackend::CUDA) {
        cudaFree(device_ptr_);
        device_ptr_ = nullptr;
        return;
    }
#endif
    
#ifdef ENABLE_OPENCL
    if (backend_ == GPUBackend::OPENCL) {
        clReleaseMemObject(static_cast<cl_mem>(device_ptr_));
        device_ptr_ = nullptr;
        return;
    }
#endif
}

template<typename T>
void GPUMemory<T>::copy_to_device(const T* host_data, size_t count) {
    if (!device_ptr_ || !host_data) return;
    
    size_t copy_count = (count == 0) ? size_ : std::min(count, size_);
    size_t bytes = copy_count * sizeof(T);
    
#ifdef ENABLE_CUDA
    if (backend_ == GPUBackend::CUDA) {
        cudaError_t error = cudaMemcpy(device_ptr_, host_data, bytes, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA host-to-device copy failed");
        }
        return;
    }
#endif
    
#ifdef ENABLE_OPENCL
    if (backend_ == GPUBackend::OPENCL) {
        cl_int error = clEnqueueWriteBuffer(GPUContext::instance().cl_queue_,
                                          static_cast<cl_mem>(device_ptr_),
                                          CL_TRUE, 0, bytes, host_data, 0, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            throw std::runtime_error("OpenCL host-to-device copy failed");
        }
        return;
    }
#endif
}

template<typename T>
void GPUMemory<T>::copy_to_host(T* host_data, size_t count) const {
    if (!device_ptr_ || !host_data) return;
    
    size_t copy_count = (count == 0) ? size_ : std::min(count, size_);
    size_t bytes = copy_count * sizeof(T);
    
#ifdef ENABLE_CUDA
    if (backend_ == GPUBackend::CUDA) {
        cudaError_t error = cudaMemcpy(host_data, device_ptr_, bytes, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA device-to-host copy failed");
        }
        return;
    }
#endif
    
#ifdef ENABLE_OPENCL
    if (backend_ == GPUBackend::OPENCL) {
        cl_int error = clEnqueueReadBuffer(GPUContext::instance().cl_queue_,
                                         static_cast<cl_mem>(device_ptr_),
                                         CL_TRUE, 0, bytes, host_data, 0, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            throw std::runtime_error("OpenCL device-to-host copy failed");
        }
        return;
    }
#endif
}

// Explicit template instantiations
template class GPUMemory<double>;
template class GPUMemory<float>;
template class GPUMemory<int>;

} // namespace gpu
} // namespace simulator

// C interface for GPU functions
extern "C" {
    void* gpu_context_create() {
        try {
            auto context = std::make_unique<simulator::gpu::GPUContext>();
            if (context->initialize()) {
                return context.release();
            }
            return nullptr;
        } catch (...) {
            return nullptr;
        }
    }

    void gpu_context_destroy(void* context) {
        if (context) {
            delete static_cast<simulator::gpu::GPUContext*>(context);
        }
    }

    int gpu_get_device_count() {
        try {
            auto devices = simulator::gpu::GPUContext::detect_devices();
            return static_cast<int>(devices.size());
        } catch (...) {
            return 0;
        }
    }

    int gpu_get_device_properties(int device_id, char* name, int name_size) {
        try {
            auto devices = simulator::gpu::GPUContext::detect_devices();
            if (device_id >= 0 && device_id < static_cast<int>(devices.size())) {
                if (name && name_size > 0) {
                    strncpy(name, devices[device_id].name.c_str(), name_size - 1);
                    name[name_size - 1] = '\0';
                }
                return 0;
            }
            return -1;
        } catch (...) {
            return -1;
        }
    }

    void* gpu_malloc(size_t size) {
        try {
#ifdef ENABLE_CUDA
            void* ptr = nullptr;
            cudaError_t error = cudaMalloc(&ptr, size);
            return (error == cudaSuccess) ? ptr : nullptr;
#else
            return nullptr;
#endif
        } catch (...) {
            return nullptr;
        }
    }

    void gpu_free(void* ptr) {
        if (ptr) {
#ifdef ENABLE_CUDA
            cudaFree(ptr);
#endif
        }
    }

    int gpu_memcpy_host_to_device(void* dst, void* src, size_t size) {
        try {
#ifdef ENABLE_CUDA
            cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
            return (error == cudaSuccess) ? 0 : -1;
#else
            return -1;
#endif
        } catch (...) {
            return -1;
        }
    }

    int gpu_memcpy_device_to_host(void* dst, void* src, size_t size) {
        try {
#ifdef ENABLE_CUDA
            cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
            return (error == cudaSuccess) ? 0 : -1;
#else
            return -1;
#endif
        } catch (...) {
            return -1;
        }
    }
}
