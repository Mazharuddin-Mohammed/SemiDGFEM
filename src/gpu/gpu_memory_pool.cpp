#include "gpu_linear_solvers.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

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
// GPUMemoryPool Implementation
// ============================================================================

GPUMemoryPool& GPUMemoryPool::instance() {
    static GPUMemoryPool instance;
    return instance;
}

void* GPUMemoryPool::allocate(size_t size) {
    // Align size to 256 bytes for optimal memory access
    size_t aligned_size = ((size + 255) / 256) * 256;
    
    // Try to find a free block of sufficient size
    MemoryBlock* block = find_free_block(aligned_size);
    
    if (block) {
        block->in_use = true;
        return block->ptr;
    }
    
    // No suitable free block found, allocate new one
    void* ptr = allocate_new_block(aligned_size);
    
    // Add to memory blocks list
    memory_blocks_.push_back({ptr, aligned_size, true});
    
    total_allocated_ += aligned_size;
    peak_usage_ = std::max(peak_usage_, total_allocated_);
    
    return ptr;
}

void GPUMemoryPool::deallocate(void* ptr, size_t size) {
    if (!ptr) return;
    
    // Find the memory block
    for (auto& block : memory_blocks_) {
        if (block.ptr == ptr) {
            block.in_use = false;
            return;
        }
    }
    
    // If not found in pool, it might be a direct allocation
    // In that case, just free it directly
    cudaFree(ptr);
}

void GPUMemoryPool::clear_pool() {
    for (auto& block : memory_blocks_) {
        if (block.ptr) {
            cudaFree(block.ptr);
        }
    }
    
    memory_blocks_.clear();
    total_allocated_ = 0;
}

GPUMemoryPool::~GPUMemoryPool() {
    clear_pool();
}

void* GPUMemoryPool::allocate_new_block(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

GPUMemoryPool::MemoryBlock* GPUMemoryPool::find_free_block(size_t size) {
    // Find the smallest free block that can accommodate the request
    MemoryBlock* best_block = nullptr;
    size_t best_size = SIZE_MAX;
    
    for (auto& block : memory_blocks_) {
        if (!block.in_use && block.size >= size && block.size < best_size) {
            best_block = &block;
            best_size = block.size;
        }
    }
    
    return best_block;
}

// ============================================================================
// GPU Memory Management Utilities
// ============================================================================

/**
 * @brief RAII wrapper for GPU memory
 */
template<typename T>
class GPUMemoryRAII {
public:
    explicit GPUMemoryRAII(size_t count) : size_(count * sizeof(T)) {
        ptr_ = static_cast<T*>(GPUMemoryPool::instance().allocate(size_));
    }
    
    ~GPUMemoryRAII() {
        if (ptr_) {
            GPUMemoryPool::instance().deallocate(ptr_, size_);
        }
    }
    
    // Non-copyable but movable
    GPUMemoryRAII(const GPUMemoryRAII&) = delete;
    GPUMemoryRAII& operator=(const GPUMemoryRAII&) = delete;
    
    GPUMemoryRAII(GPUMemoryRAII&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    GPUMemoryRAII& operator=(GPUMemoryRAII&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                GPUMemoryPool::instance().deallocate(ptr_, size_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    
    void copy_from_host(const T* host_ptr, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(T* host_ptr, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    void zero() {
        CUDA_CHECK(cudaMemset(ptr_, 0, size_));
    }
    
private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

// Convenience typedefs
using GPUDoubleArray = GPUMemoryRAII<double>;
using GPUIntArray = GPUMemoryRAII<int>;
using GPUFloatArray = GPUMemoryRAII<float>;

/**
 * @brief GPU memory statistics and monitoring
 */
class GPUMemoryMonitor {
public:
    static GPUMemoryMonitor& instance() {
        static GPUMemoryMonitor instance;
        return instance;
    }
    
    struct MemoryInfo {
        size_t total_memory;
        size_t free_memory;
        size_t used_memory;
        double utilization_percent;
    };
    
    MemoryInfo get_memory_info() const {
        size_t free_bytes, total_bytes;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        
        size_t used_bytes = total_bytes - free_bytes;
        double utilization = (double)used_bytes / total_bytes * 100.0;
        
        return {total_bytes, free_bytes, used_bytes, utilization};
    }
    
    void print_memory_info() const {
        auto info = get_memory_info();
        std::cout << "GPU Memory Info:\n";
        std::cout << "  Total: " << info.total_memory / (1024*1024) << " MB\n";
        std::cout << "  Used:  " << info.used_memory / (1024*1024) << " MB\n";
        std::cout << "  Free:  " << info.free_memory / (1024*1024) << " MB\n";
        std::cout << "  Utilization: " << info.utilization_percent << "%\n";
    }
    
    bool is_memory_available(size_t required_bytes) const {
        auto info = get_memory_info();
        return info.free_memory >= required_bytes;
    }
    
    void check_memory_pressure() const {
        auto info = get_memory_info();
        if (info.utilization_percent > 90.0) {
            std::cerr << "Warning: GPU memory utilization is high (" 
                      << info.utilization_percent << "%)\n";
        }
    }
    
private:
    GPUMemoryMonitor() = default;
};

/**
 * @brief Automatic GPU memory management with fallback to CPU
 */
class AdaptiveMemoryManager {
public:
    template<typename T>
    static std::unique_ptr<T[]> allocate_adaptive(size_t count, bool& use_gpu) {
        size_t required_bytes = count * sizeof(T);
        
        // Check if GPU memory is available
        if (GPUMemoryMonitor::instance().is_memory_available(required_bytes)) {
            try {
                auto gpu_ptr = std::make_unique<GPUMemoryRAII<T>>(count);
                use_gpu = true;
                return std::unique_ptr<T[]>(gpu_ptr.release()->get());
            } catch (const std::exception& e) {
                std::cerr << "GPU allocation failed, falling back to CPU: " << e.what() << "\n";
            }
        }
        
        // Fallback to CPU allocation
        use_gpu = false;
        return std::make_unique<T[]>(count);
    }
    
    template<typename T>
    static void deallocate_adaptive(T* ptr, size_t count, bool was_gpu) {
        if (was_gpu) {
            GPUMemoryPool::instance().deallocate(ptr, count * sizeof(T));
        } else {
            delete[] ptr;
        }
    }
};

} // namespace gpu
} // namespace simulator
