#include "performance_optimization.hpp"
#include <omp.h>
#include <thread>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace simulator {
namespace performance {
namespace parallel {

ThreadPool::ThreadPool(size_t num_threads) : initialized_(false) {
    if (num_threads == 0) {
        num_threads_ = std::thread::hardware_concurrency();
        if (num_threads_ == 0) num_threads_ = 4;  // Fallback
    } else {
        num_threads_ = num_threads;
    }
    
    // Initialize OpenMP
    omp_set_num_threads(static_cast<int>(num_threads_));
    initialized_ = true;
}

ThreadPool::~ThreadPool() {
    // OpenMP cleanup is automatic
}

template<typename F>
void ThreadPool::parallel_for(size_t start, size_t end, F&& func) {
    if (!initialized_) return;
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = start; i < end; ++i) {
        func(i);
    }
}

template<typename F>
void ThreadPool::parallel_reduce(size_t start, size_t end, double& result, F&& func) {
    if (!initialized_) {
        result = 0.0;
        for (size_t i = start; i < end; ++i) {
            result += func(i);
        }
        return;
    }
    
    double local_result = 0.0;
    
    #pragma omp parallel for reduction(+:local_result) schedule(dynamic)
    for (size_t i = start; i < end; ++i) {
        local_result += func(i);
    }
    
    result = local_result;
}

void OMPOps::assemble_matrix_parallel(
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices,
    const std::function<void(int, std::vector<std::vector<double>>&)>& element_func,
    std::vector<std::vector<double>>& global_matrix) {
    
    PROFILE_FUNCTION();
    
    const size_t num_elements = elements.size();
    const size_t matrix_size = global_matrix.size();
    
    // Thread-local storage for element matrices
    const int max_threads = omp_get_max_threads();
    std::vector<std::vector<std::vector<std::vector<double>>>> thread_matrices(max_threads);
    
    for (int t = 0; t < max_threads; ++t) {
        thread_matrices[t].resize(num_elements);
        for (size_t e = 0; e < num_elements; ++e) {
            thread_matrices[t][e].resize(matrix_size, std::vector<double>(matrix_size, 0.0));
        }
    }
    
    // Parallel element assembly
    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic, 10)
        for (size_t e = 0; e < num_elements; ++e) {
            element_func(static_cast<int>(e), thread_matrices[thread_id][e]);
        }
    }
    
    // Parallel assembly into global matrix
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t i = 0; i < matrix_size; ++i) {
        for (size_t j = 0; j < matrix_size; ++j) {
            double sum = 0.0;
            for (int t = 0; t < max_threads; ++t) {
                for (size_t e = 0; e < num_elements; ++e) {
                    sum += thread_matrices[t][e][i][j];
                }
            }
            global_matrix[i][j] += sum;
        }
    }
}

void OMPOps::vector_add_parallel(const double* a, const double* b, 
                                double* result, size_t n) {
    PROFILE_FUNCTION();
    
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

double OMPOps::dot_product_parallel(const double* a, const double* b, size_t n) {
    PROFILE_FUNCTION();
    
    double result = 0.0;
    
    #pragma omp parallel for reduction(+:result) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void OMPOps::compute_error_parallel(
    const std::vector<double>& solution,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices,
    std::vector<double>& error_indicators) {
    
    PROFILE_FUNCTION();
    
    const size_t num_elements = elements.size();
    error_indicators.resize(num_elements);
    
    #pragma omp parallel for schedule(dynamic, 50)
    for (size_t e = 0; e < num_elements; ++e) {
        const auto& element = elements[e];
        if (element.size() < 3) {
            error_indicators[e] = 0.0;
            continue;
        }
        
        // Get element vertices
        std::array<std::array<double, 2>, 3> elem_vertices;
        std::array<double, 3> elem_solution;
        
        bool valid = true;
        for (int i = 0; i < 3; ++i) {
            if (element[i] >= static_cast<int>(vertices.size()) || 
                element[i] >= static_cast<int>(solution.size())) {
                valid = false;
                break;
            }
            elem_vertices[i] = vertices[element[i]];
            elem_solution[i] = solution[element[i]];
        }
        
        if (!valid) {
            error_indicators[e] = 0.0;
            continue;
        }
        
        // Compute element area
        double area = 0.5 * std::abs(
            (elem_vertices[1][0] - elem_vertices[0][0]) * (elem_vertices[2][1] - elem_vertices[0][1]) -
            (elem_vertices[2][0] - elem_vertices[0][0]) * (elem_vertices[1][1] - elem_vertices[0][1])
        );
        
        if (area < 1e-12) {
            error_indicators[e] = 0.0;
            continue;
        }
        
        // Compute gradient
        double x1 = elem_vertices[1][0] - elem_vertices[0][0];
        double y1 = elem_vertices[1][1] - elem_vertices[0][1];
        double x2 = elem_vertices[2][0] - elem_vertices[0][0];
        double y2 = elem_vertices[2][1] - elem_vertices[0][1];
        
        double det = x1 * y2 - x2 * y1;
        
        if (std::abs(det) < 1e-12) {
            error_indicators[e] = 0.0;
            continue;
        }
        
        double u1 = elem_solution[1] - elem_solution[0];
        double u2 = elem_solution[2] - elem_solution[0];
        
        double grad_x = (y2 * u1 - y1 * u2) / det;
        double grad_y = (-x2 * u1 + x1 * u2) / det;
        
        double grad_magnitude = std::sqrt(grad_x * grad_x + grad_y * grad_y);
        double h = std::sqrt(area);
        
        // Error indicator: h * |∇u|
        error_indicators[e] = h * grad_magnitude;
    }
}

} // namespace parallel

namespace memory {

template<typename T>
AlignedVector<T>::AlignedVector(size_t size, size_t alignment) 
    : size_(size), alignment_(alignment) {
    
    // Allocate aligned memory
    size_t total_bytes = size * sizeof(T);
    
#ifdef _WIN32
    data_ = static_cast<T*>(_aligned_malloc(total_bytes, alignment));
#else
    if (posix_memalign(reinterpret_cast<void**>(&data_), alignment, total_bytes) != 0) {
        data_ = nullptr;
    }
#endif
    
    if (!data_) {
        throw std::bad_alloc();
    }
    
    // Initialize elements
    for (size_t i = 0; i < size_; ++i) {
        new(&data_[i]) T();
    }
}

template<typename T>
AlignedVector<T>::~AlignedVector() {
    if (data_) {
        // Destroy elements
        for (size_t i = 0; i < size_; ++i) {
            data_[i].~T();
        }
        
        // Free aligned memory
#ifdef _WIN32
        _aligned_free(data_);
#else
        free(data_);
#endif
    }
}

MemoryPool::MemoryPool(size_t block_size, size_t num_blocks) 
    : block_size_(block_size), allocated_bytes_(0), peak_usage_(0) {
    
    blocks_.reserve(num_blocks);
    free_list_.reserve(num_blocks);
    
    for (size_t i = 0; i < num_blocks; ++i) {
        char* block = new char[block_size];
        blocks_.push_back(block);
        free_list_.push_back(block);
    }
}

MemoryPool::~MemoryPool() {
    for (char* block : blocks_) {
        delete[] block;
    }
}

void* MemoryPool::allocate(size_t size) {
    if (size > block_size_ || free_list_.empty()) {
        return nullptr;  // Cannot satisfy request
    }
    
    void* ptr = free_list_.back();
    free_list_.pop_back();
    
    allocated_bytes_ += block_size_;
    peak_usage_ = std::max(peak_usage_, allocated_bytes_);
    
    return ptr;
}

void MemoryPool::deallocate(void* ptr) {
    if (ptr) {
        free_list_.push_back(ptr);
        allocated_bytes_ -= block_size_;
    }
}

void MemoryPool::reset() {
    free_list_.clear();
    for (char* block : blocks_) {
        free_list_.push_back(block);
    }
    allocated_bytes_ = 0;
}

BlockMatrix::BlockMatrix(size_t rows, size_t cols, size_t block_size)
    : rows_(rows), cols_(cols), block_size_(block_size) {
    
    blocks_per_row_ = (rows + block_size - 1) / block_size;
    blocks_per_col_ = (cols + block_size - 1) / block_size;
    
    data_.resize(blocks_per_row_ * blocks_per_col_ * block_size * block_size, 0.0);
}

double& BlockMatrix::operator()(size_t i, size_t j) {
    size_t block_row = i / block_size_;
    size_t block_col = j / block_size_;
    size_t local_row = i % block_size_;
    size_t local_col = j % block_size_;
    
    size_t index = get_block_index(block_row, block_col, local_row, local_col);
    return data_[index];
}

const double& BlockMatrix::operator()(size_t i, size_t j) const {
    size_t block_row = i / block_size_;
    size_t block_col = j / block_size_;
    size_t local_row = i % block_size_;
    size_t local_col = j % block_size_;
    
    size_t index = get_block_index(block_row, block_col, local_row, local_col);
    return data_[index];
}

size_t BlockMatrix::get_block_index(size_t block_row, size_t block_col, 
                                   size_t local_row, size_t local_col) const {
    size_t block_offset = (block_row * blocks_per_col_ + block_col) * block_size_ * block_size_;
    size_t local_offset = local_row * block_size_ + local_col;
    return block_offset + local_offset;
}

void BlockMatrix::multiply_vector(const double* vec, double* result) const {
    PROFILE_FUNCTION();
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < rows_; ++i) {
        result[i] = 0.0;
        for (size_t j = 0; j < cols_; ++j) {
            result[i] += (*this)(i, j) * vec[j];
        }
    }
}

} // namespace memory

// Explicit template instantiations
template class memory::AlignedVector<double>;
template class memory::AlignedVector<float>;
template class memory::AlignedVector<int>;

// Missing Profiler implementations
void Profiler::start_timer(const std::string& name) {
    auto now = std::chrono::high_resolution_clock::now();
    start_times_[name] = now;
}

void Profiler::end_timer(const std::string& name) {
    auto now = std::chrono::high_resolution_clock::now();
    auto it = start_times_.find(name);
    if (it != start_times_.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - it->second);
        double elapsed_ms = duration.count() / 1000.0;

        total_times_[name] += elapsed_ms;
        call_counts_[name]++;

        start_times_.erase(it);
    }
}

void Profiler::reset() {
    start_times_.clear();
    total_times_.clear();
    call_counts_.clear();
}

std::vector<Profiler::ProfileData> Profiler::get_profile_data() const {
    std::vector<ProfileData> data;
    double total_time = 0.0;

    // Calculate total time
    for (const auto& pair : total_times_) {
        total_time += pair.second;
    }

    // Create profile data
    for (const auto& pair : total_times_) {
        ProfileData pd;
        pd.name = pair.first;
        pd.total_time = pair.second;
        pd.call_count = call_counts_.at(pair.first);
        pd.average_time = pd.total_time / pd.call_count;
        pd.percentage = (total_time > 0) ? (pd.total_time / total_time * 100.0) : 0.0;
        data.push_back(pd);
    }

    return data;
}

void Profiler::print_profile() const {
    auto data = get_profile_data();

    if (data.empty()) {
        std::cout << "No profiling data available." << std::endl;
        return;
    }

    std::cout << "\n=== Performance Profile ===" << std::endl;
    std::cout << std::setw(30) << "Function"
              << std::setw(15) << "Time (ms)"
              << std::setw(12) << "Percentage"
              << std::setw(10) << "Calls" << std::endl;
    std::cout << std::string(67, '-') << std::endl;

    // Calculate total time
    double total_time_sum = 0.0;
    for (const auto& item : data) {
        total_time_sum += item.total_time;
    }

    // Sort by time (descending)
    auto sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end(),
              [](const ProfileData& a, const ProfileData& b) {
                  return a.total_time > b.total_time;
              });

    // Print sorted results
    for (const auto& item : sorted_data) {
        double percentage = (total_time_sum > 0.0) ? (item.total_time / total_time_sum * 100.0) : 0.0;
        std::cout << std::setw(30) << item.name
                  << std::setw(15) << std::fixed << std::setprecision(3) << item.total_time
                  << std::setw(11) << std::fixed << std::setprecision(1) << percentage << "%"
                  << std::setw(10) << item.call_count << std::endl;
    }

    std::cout << std::string(67, '-') << std::endl;
    std::cout << std::setw(30) << "TOTAL"
              << std::setw(15) << std::fixed << std::setprecision(3) << total_time_sum
              << std::setw(11) << "100.0%"
              << std::setw(10) << "" << std::endl;
    std::cout << std::endl;
}

} // namespace performance
} // namespace simulator
