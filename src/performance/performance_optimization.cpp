/**
 * Enhanced Performance Optimization Implementation
 * 
 * This implementation provides comprehensive performance optimization capabilities
 * including advanced threading, memory optimization, and adaptive algorithms.
 * 
 * Author: Dr. Mazharuddin Mohammed
 */

#include "performance_optimization.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <future>
#include <random>

namespace simulator {
namespace performance {

// PerformanceOptimizer implementation
PerformanceOptimizer::PerformanceOptimizer() 
    : parallelization_strategy_(ParallelizationStrategy::OPENMP),
      load_balancing_type_(LoadBalancingType::DYNAMIC),
      memory_optimization_(MemoryOptimizationType::CACHE_FRIENDLY),
      num_threads_(std::thread::hardware_concurrency()),
      cache_size_(1024 * 1024),  // 1MB default
      auto_tuning_enabled_(false) {
    
    thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
    profiler_ = std::make_unique<PerformanceProfiler>();
    
    // Initialize with default metrics
    current_metrics_ = {};
}

PerformanceOptimizer::~PerformanceOptimizer() = default;

void PerformanceOptimizer::set_parallelization_strategy(ParallelizationStrategy strategy) {
    parallelization_strategy_ = strategy;
    
    // Reconfigure thread pool if needed
    if (strategy == ParallelizationStrategy::THREAD_POOL || 
        strategy == ParallelizationStrategy::WORK_STEALING) {
        thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
    }
}

void PerformanceOptimizer::set_load_balancing_type(LoadBalancingType type) {
    load_balancing_type_ = type;
    if (thread_pool_) {
        thread_pool_->set_load_balancing(type);
    }
}

void PerformanceOptimizer::set_memory_optimization(MemoryOptimizationType type) {
    memory_optimization_ = type;
}

void PerformanceOptimizer::set_num_threads(size_t num_threads) {
    num_threads_ = num_threads;
    if (thread_pool_) {
        thread_pool_->resize(num_threads);
    }
}

void PerformanceOptimizer::set_cache_size(size_t cache_size) {
    cache_size_ = cache_size;
}

void PerformanceOptimizer::enable_auto_tuning(bool enabled) {
    auto_tuning_enabled_ = enabled;
}

void PerformanceOptimizer::optimize_for_problem_size(size_t problem_size) {
    // Adaptive optimization based on problem size
    if (problem_size < 1000) {
        // Small problems: use single thread, minimize overhead
        set_num_threads(1);
        set_parallelization_strategy(ParallelizationStrategy::STD_THREAD);
    } else if (problem_size < 100000) {
        // Medium problems: use moderate parallelization
        set_num_threads(std::min(size_t(4), std::thread::hardware_concurrency()));
        set_parallelization_strategy(ParallelizationStrategy::OPENMP);
    } else {
        // Large problems: use full parallelization
        set_num_threads(std::thread::hardware_concurrency());
        set_parallelization_strategy(ParallelizationStrategy::HYBRID);
    }
    
    // Adjust cache size based on problem size
    size_t optimal_cache = std::min(cache_size_, problem_size * sizeof(double) / 4);
    set_cache_size(optimal_cache);
}

void PerformanceOptimizer::optimize_for_memory_constraints(size_t available_memory) {
    // Adjust parameters based on available memory
    size_t memory_per_thread = available_memory / num_threads_;
    
    if (memory_per_thread < 100 * 1024 * 1024) {  // Less than 100MB per thread
        // Reduce thread count to avoid memory pressure
        size_t new_threads = std::max(size_t(1), available_memory / (100 * 1024 * 1024));
        set_num_threads(new_threads);
        set_memory_optimization(MemoryOptimizationType::COMPRESSION);
    } else {
        set_memory_optimization(MemoryOptimizationType::CACHE_FRIENDLY);
    }
    
    // Adjust cache size
    size_t max_cache = available_memory / 10;  // Use at most 10% for cache
    set_cache_size(std::min(cache_size_, max_cache));
}

void PerformanceOptimizer::optimize_for_cpu_architecture() {
    // Detect CPU features and optimize accordingly
    bool has_avx2 = PerformanceUtils::has_avx2_support();
    bool has_avx512 = PerformanceUtils::has_avx512_support();
    
    if (has_avx512) {
        set_parallelization_strategy(ParallelizationStrategy::HYBRID);
        set_memory_optimization(MemoryOptimizationType::NUMA_AWARE);
    } else if (has_avx2) {
        set_parallelization_strategy(ParallelizationStrategy::OPENMP);
        set_memory_optimization(MemoryOptimizationType::CACHE_FRIENDLY);
    } else {
        set_parallelization_strategy(ParallelizationStrategy::STD_THREAD);
        set_memory_optimization(MemoryOptimizationType::MEMORY_POOL);
    }
    
    // Adjust cache size based on CPU cache
    size_t l3_cache = PerformanceUtils::get_l3_cache_size();
    if (l3_cache > 0) {
        set_cache_size(l3_cache / 2);  // Use half of L3 cache
    }
}

void PerformanceOptimizer::auto_tune_parameters() {
    if (!auto_tuning_enabled_) return;
    
    // Run benchmark with different configurations
    std::vector<std::pair<double, std::string>> results;
    
    // Test different thread counts
    for (size_t threads = 1; threads <= std::thread::hardware_concurrency(); threads *= 2) {
        set_num_threads(threads);
        
        auto start = std::chrono::high_resolution_clock::now();
        // Run a representative workload
        std::vector<double> data(10000);
        std::iota(data.begin(), data.end(), 0.0);
        auto result = parallel_reduce(data, [](const double& a, const double& b) { return a + b; });
        auto end = std::chrono::high_resolution_clock::now();
        
        double time = std::chrono::duration<double>(end - start).count();
        results.emplace_back(time, "threads_" + std::to_string(threads));
    }
    
    // Find best configuration
    auto best = std::min_element(results.begin(), results.end());
    if (best != results.end()) {
        // Extract optimal thread count from result name
        std::string config = best->second;
        size_t pos = config.find('_');
        if (pos != std::string::npos) {
            size_t optimal_threads = std::stoul(config.substr(pos + 1));
            set_num_threads(optimal_threads);
        }
    }
}

PerformanceMetrics PerformanceOptimizer::measure_performance(std::function<void()> operation) {
    PerformanceMetrics metrics = {};
    
    // Measure execution time
    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_cpu = std::clock();
    
    operation();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto end_cpu = std::clock();
    
    metrics.execution_time = std::chrono::duration<double>(end_time - start_time).count();
    metrics.cpu_utilization = double(end_cpu - start_cpu) / CLOCKS_PER_SEC / metrics.execution_time;
    
    // Estimate other metrics
    metrics.memory_usage = 0.0;  // Would need platform-specific implementation
    metrics.cache_hit_ratio = 0.95;  // Estimated
    metrics.throughput = 1.0 / metrics.execution_time;
    metrics.efficiency = metrics.cpu_utilization;
    metrics.operations_count = 1;
    metrics.memory_allocations = 0;
    metrics.cache_misses = 0;
    metrics.parallel_efficiency = metrics.cpu_utilization / num_threads_;
    metrics.load_balance_factor = 1.0;  // Estimated
    metrics.bottleneck_analysis = "CPU-bound";
    
    return metrics;
}

void PerformanceOptimizer::start_monitoring() {
    profiler_->enable(true);
}

void PerformanceOptimizer::stop_monitoring() {
    profiler_->enable(false);
}

PerformanceMetrics PerformanceOptimizer::get_current_metrics() const {
    return current_metrics_;
}

std::string PerformanceOptimizer::generate_performance_report() const {
    std::ostringstream report;
    report << "Performance Optimization Report\n";
    report << "==============================\n\n";
    
    report << "Configuration:\n";
    report << "  Parallelization Strategy: ";
    switch (parallelization_strategy_) {
        case ParallelizationStrategy::OPENMP: report << "OpenMP"; break;
        case ParallelizationStrategy::STD_THREAD: report << "std::thread"; break;
        case ParallelizationStrategy::THREAD_POOL: report << "Thread Pool"; break;
        case ParallelizationStrategy::TASK_BASED: report << "Task-based"; break;
        case ParallelizationStrategy::WORK_STEALING: report << "Work Stealing"; break;
        case ParallelizationStrategy::HYBRID: report << "Hybrid"; break;
    }
    report << "\n";
    
    report << "  Load Balancing: ";
    switch (load_balancing_type_) {
        case LoadBalancingType::STATIC: report << "Static"; break;
        case LoadBalancingType::DYNAMIC: report << "Dynamic"; break;
        case LoadBalancingType::GUIDED: report << "Guided"; break;
        case LoadBalancingType::ADAPTIVE: report << "Adaptive"; break;
        case LoadBalancingType::WORK_STEALING: report << "Work Stealing"; break;
        case LoadBalancingType::LOCALITY_AWARE: report << "Locality Aware"; break;
    }
    report << "\n";
    
    report << "  Memory Optimization: ";
    switch (memory_optimization_) {
        case MemoryOptimizationType::CACHE_FRIENDLY: report << "Cache Friendly"; break;
        case MemoryOptimizationType::MEMORY_POOL: report << "Memory Pool"; break;
        case MemoryOptimizationType::ZERO_COPY: report << "Zero Copy"; break;
        case MemoryOptimizationType::PREFETCHING: report << "Prefetching"; break;
        case MemoryOptimizationType::NUMA_AWARE: report << "NUMA Aware"; break;
        case MemoryOptimizationType::COMPRESSION: report << "Compression"; break;
    }
    report << "\n";
    
    report << "  Number of Threads: " << num_threads_ << "\n";
    report << "  Cache Size: " << cache_size_ / 1024 << " KB\n";
    report << "  Auto-tuning: " << (auto_tuning_enabled_ ? "Enabled" : "Disabled") << "\n\n";
    
    report << "Current Metrics:\n";
    report << "  Execution Time: " << std::fixed << std::setprecision(6) 
           << current_metrics_.execution_time << " seconds\n";
    report << "  CPU Utilization: " << std::fixed << std::setprecision(2) 
           << current_metrics_.cpu_utilization * 100 << "%\n";
    report << "  Memory Usage: " << std::fixed << std::setprecision(2) 
           << current_metrics_.memory_usage / (1024*1024) << " MB\n";
    report << "  Cache Hit Ratio: " << std::fixed << std::setprecision(2) 
           << current_metrics_.cache_hit_ratio * 100 << "%\n";
    report << "  Throughput: " << std::fixed << std::setprecision(2) 
           << current_metrics_.throughput << " ops/sec\n";
    report << "  Parallel Efficiency: " << std::fixed << std::setprecision(2) 
           << current_metrics_.parallel_efficiency * 100 << "%\n";
    report << "  Load Balance Factor: " << std::fixed << std::setprecision(3) 
           << current_metrics_.load_balance_factor << "\n";
    report << "  Bottleneck: " << current_metrics_.bottleneck_analysis << "\n\n";
    
    if (profiler_) {
        report << profiler_->generate_report();
    }
    
    return report.str();
}

void PerformanceOptimizer::learn_from_execution(const PerformanceMetrics& metrics) {
    current_metrics_ = metrics;
    
    // Adaptive learning: adjust parameters based on performance
    if (metrics.parallel_efficiency < 0.5 && num_threads_ > 1) {
        // Poor parallel efficiency: reduce thread count
        set_num_threads(std::max(size_t(1), num_threads_ / 2));
    } else if (metrics.parallel_efficiency > 0.9 && metrics.cpu_utilization < 0.8) {
        // Good efficiency but low utilization: could use more threads
        set_num_threads(std::min(std::thread::hardware_concurrency(), num_threads_ * 2));
    }
    
    if (metrics.cache_hit_ratio < 0.8) {
        // Poor cache performance: adjust memory optimization
        set_memory_optimization(MemoryOptimizationType::CACHE_FRIENDLY);
    }
}

void PerformanceOptimizer::suggest_optimizations() const {
    std::cout << "Performance Optimization Suggestions:\n";
    std::cout << "=====================================\n";
    
    if (current_metrics_.parallel_efficiency < 0.5) {
        std::cout << "• Consider reducing thread count (current efficiency: " 
                  << current_metrics_.parallel_efficiency * 100 << "%)\n";
    }
    
    if (current_metrics_.cache_hit_ratio < 0.8) {
        std::cout << "• Optimize memory access patterns (cache hit ratio: " 
                  << current_metrics_.cache_hit_ratio * 100 << "%)\n";
    }
    
    if (current_metrics_.cpu_utilization < 0.7) {
        std::cout << "• Consider increasing parallelization (CPU utilization: " 
                  << current_metrics_.cpu_utilization * 100 << "%)\n";
    }
    
    if (current_metrics_.load_balance_factor < 0.8) {
        std::cout << "• Improve load balancing (balance factor: " 
                  << current_metrics_.load_balance_factor << ")\n";
    }
    
    std::cout << "\n";
}

void PerformanceOptimizer::apply_best_configuration() {
    // Apply the best known configuration based on learned metrics
    optimize_for_cpu_architecture();
    
    if (auto_tuning_enabled_) {
        auto_tune_parameters();
    }
}

size_t PerformanceOptimizer::get_optimal_thread_count() const {
    // Estimate optimal thread count based on problem characteristics
    size_t hw_threads = std::thread::hardware_concurrency();
    
    if (current_metrics_.parallel_efficiency > 0.8) {
        return hw_threads;
    } else if (current_metrics_.parallel_efficiency > 0.5) {
        return hw_threads / 2;
    } else {
        return std::max(size_t(1), hw_threads / 4);
    }
}

size_t PerformanceOptimizer::get_optimal_chunk_size(size_t total_work) const {
    // Calculate optimal chunk size for load balancing
    size_t base_chunk = total_work / (num_threads_ * 4);  // 4 chunks per thread
    
    switch (load_balancing_type_) {
        case LoadBalancingType::STATIC:
            return total_work / num_threads_;
        case LoadBalancingType::DYNAMIC:
            return std::max(size_t(1), base_chunk);
        case LoadBalancingType::GUIDED:
            return std::max(size_t(1), base_chunk / 2);
        default:
            return std::max(size_t(1), base_chunk);
    }
}

bool PerformanceOptimizer::should_parallelize(size_t work_size) const {
    // Determine if parallelization is beneficial
    const size_t min_work_per_thread = 1000;  // Minimum work to justify threading overhead
    return work_size >= min_work_per_thread * num_threads_;
}

double PerformanceOptimizer::estimate_parallel_efficiency(size_t work_size, size_t num_threads) const {
    // Estimate parallel efficiency using Amdahl's law approximation
    double serial_fraction = 0.05;  // Assume 5% serial work
    double communication_overhead = 0.01 * num_threads;  // Linear overhead model
    
    double parallel_fraction = 1.0 - serial_fraction;
    double speedup = 1.0 / (serial_fraction + parallel_fraction / num_threads + communication_overhead);
    
    return speedup / num_threads;
}

// ThreadPool implementation
ThreadPool::ThreadPool(size_t num_threads) : stop_(false), active_tasks_(0) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }

    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

                    if (stop_ && tasks_.empty()) return;

                    task = std::move(tasks_.front());
                    tasks_.pop();
                    active_tasks_++;
                }

                auto start = std::chrono::high_resolution_clock::now();
                task();
                auto end = std::chrono::high_resolution_clock::now();

                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    active_tasks_--;
                    stats_.completed_tasks++;

                    double task_time = std::chrono::duration<double>(end - start).count();
                    stats_.average_task_time = (stats_.average_task_time * (stats_.completed_tasks - 1) + task_time) / stats_.completed_tasks;
                }
            }
        });
    }

    stats_.active_threads = num_threads;
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }

    condition_.notify_all();

    for (std::thread &worker : workers_) {
        worker.join();
    }
}

void ThreadPool::resize(size_t num_threads) {
    // Simple implementation: recreate pool
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }

    condition_.notify_all();

    for (std::thread &worker : workers_) {
        worker.join();
    }

    workers_.clear();
    stop_ = false;

    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

                    if (stop_ && tasks_.empty()) return;

                    task = std::move(tasks_.front());
                    tasks_.pop();
                    active_tasks_++;
                }

                task();

                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    active_tasks_--;
                    stats_.completed_tasks++;
                }
            }
        });
    }

    stats_.active_threads = num_threads;
}

size_t ThreadPool::size() const {
    return workers_.size();
}

ThreadPoolStats ThreadPool::get_stats() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    ThreadPoolStats current_stats = stats_;
    current_stats.queued_tasks = tasks_.size();
    current_stats.thread_utilization = double(active_tasks_) / workers_.size();
    return current_stats;
}

void ThreadPool::wait_for_completion() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    condition_.wait(lock, [this] { return tasks_.empty() && active_tasks_ == 0; });
}

void ThreadPool::set_load_balancing(LoadBalancingType type) {
    // Implementation would depend on specific load balancing strategy
    // For now, just store the type
}

// PerformanceProfiler implementation
PerformanceProfiler::PerformanceProfiler() : enabled_(true) {}

void PerformanceProfiler::start_timer(const std::string& name) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);
    start_times_[name] = std::chrono::steady_clock::now();
}

void PerformanceProfiler::end_timer(const std::string& name) {
    if (!enabled_) return;

    auto end_time = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = start_times_.find(name);
    if (it != start_times_.end()) {
        double duration = std::chrono::duration<double>(end_time - it->second).count();
        accumulated_times_[name] += duration;
        call_counts_[name]++;
        start_times_.erase(it);
    }
}

double PerformanceProfiler::get_time(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = accumulated_times_.find(name);
    return (it != accumulated_times_.end()) ? it->second : 0.0;
}

size_t PerformanceProfiler::get_call_count(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = call_counts_.find(name);
    return (it != call_counts_.end()) ? it->second : 0;
}

double PerformanceProfiler::get_average_time(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto time_it = accumulated_times_.find(name);
    auto count_it = call_counts_.find(name);

    if (time_it != accumulated_times_.end() && count_it != call_counts_.end() && count_it->second > 0) {
        return time_it->second / count_it->second;
    }
    return 0.0;
}

void PerformanceProfiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    start_times_.clear();
    accumulated_times_.clear();
    call_counts_.clear();
}

void PerformanceProfiler::enable(bool enabled) {
    enabled_ = enabled;
}

std::string PerformanceProfiler::generate_report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream report;

    report << "Performance Profile Report\n";
    report << "=========================\n\n";

    if (accumulated_times_.empty()) {
        report << "No profiling data available.\n";
        return report.str();
    }

    // Calculate total time
    double total_time = 0.0;
    for (const auto& pair : accumulated_times_) {
        total_time += pair.second;
    }

    // Sort by total time
    std::vector<std::pair<std::string, double>> sorted_times(accumulated_times_.begin(), accumulated_times_.end());
    std::sort(sorted_times.begin(), sorted_times.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    report << std::left << std::setw(30) << "Function"
           << std::setw(12) << "Total (s)"
           << std::setw(12) << "Average (s)"
           << std::setw(10) << "Calls"
           << std::setw(10) << "Percent" << "\n";
    report << std::string(74, '-') << "\n";

    for (const auto& pair : sorted_times) {
        const std::string& name = pair.first;
        double time = pair.second;
        size_t calls = call_counts_.at(name);
        double average = time / calls;
        double percent = (total_time > 0) ? (time / total_time * 100) : 0;

        report << std::left << std::setw(30) << name
               << std::fixed << std::setprecision(6) << std::setw(12) << time
               << std::fixed << std::setprecision(6) << std::setw(12) << average
               << std::setw(10) << calls
               << std::fixed << std::setprecision(2) << std::setw(10) << percent << "\n";
    }

    report << std::string(74, '-') << "\n";
    report << "Total time: " << std::fixed << std::setprecision(6) << total_time << " seconds\n\n";

    return report.str();
}

void PerformanceProfiler::export_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << generate_report();
        file.close();
    }
}

// ScopedTimer implementation
ScopedTimer::ScopedTimer(PerformanceProfiler* profiler, const std::string& name)
    : profiler_(profiler), name_(name) {
    if (profiler_) {
        profiler_->start_timer(name_);
    }
}

ScopedTimer::~ScopedTimer() {
    if (profiler_) {
        profiler_->end_timer(name_);
    }
}

// Performance utility functions
namespace PerformanceUtils {

size_t get_cpu_count() {
    return std::thread::hardware_concurrency();
}

size_t get_cache_line_size() {
    return 64;  // Common cache line size
}

size_t get_l1_cache_size() {
    return 32 * 1024;  // 32KB typical L1 cache
}

size_t get_l2_cache_size() {
    return 256 * 1024;  // 256KB typical L2 cache
}

size_t get_l3_cache_size() {
    return 8 * 1024 * 1024;  // 8MB typical L3 cache
}

bool has_avx_support() {
#ifdef __AVX__
    return true;
#else
    return false;
#endif
}

bool has_avx2_support() {
#ifdef __AVX2__
    return true;
#else
    return false;
#endif
}

bool has_avx512_support() {
#ifdef __AVX512F__
    return true;
#else
    return false;
#endif
}

size_t get_total_memory() {
    // Platform-specific implementation would be needed
    return 16ULL * 1024 * 1024 * 1024;  // 16GB default
}

size_t get_available_memory() {
    // Platform-specific implementation would be needed
    return get_total_memory() / 2;  // Assume half available
}

size_t get_page_size() {
    return 4096;  // 4KB typical page size
}

bool is_numa_available() {
    return false;  // Would need platform-specific detection
}

size_t get_numa_node_count() {
    return 1;  // Single node if NUMA not available
}

void prefetch_read(const void* address) {
#ifdef __builtin_prefetch
    __builtin_prefetch(address, 0, 3);  // Read, high temporal locality
#endif
}

void prefetch_write(void* address) {
#ifdef __builtin_prefetch
    __builtin_prefetch(address, 1, 3);  // Write, high temporal locality
#endif
}

void memory_barrier() {
#ifdef __GNUC__
    __sync_synchronize();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

void cpu_pause() {
#ifdef __x86_64__
    __asm__ __volatile__("pause" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

void yield_thread() {
    std::this_thread::yield();
}

bool is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

size_t get_alignment_offset(const void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    size_t offset = addr % alignment;
    return (offset == 0) ? 0 : (alignment - offset);
}

double get_wall_time() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

double get_cpu_time() {
    return double(std::clock()) / CLOCKS_PER_SEC;
}

uint64_t get_cycle_count() {
#ifdef __x86_64__
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return (uint64_t(hi) << 32) | lo;
#else
    return 0;  // Not available on this platform
#endif
}

double cycles_to_seconds(uint64_t cycles) {
    // Assume 3GHz CPU frequency
    const double cpu_frequency = 3.0e9;
    return double(cycles) / cpu_frequency;
}

} // namespace PerformanceUtils

// LoadBalancer implementation
LoadBalancer::LoadBalancer(LoadBalancingType type, size_t num_workers)
    : type_(type), work_loads_(num_workers) {

    for (size_t i = 0; i < num_workers; ++i) {
        work_loads_[i] = 0;
    }
}

size_t LoadBalancer::assign_work(size_t work_size) {
    std::lock_guard<std::mutex> lock(balancer_mutex_);

    switch (type_) {
        case LoadBalancingType::STATIC: {
            // Round-robin assignment
            static size_t next_worker = 0;
            size_t worker = next_worker;
            next_worker = (next_worker + 1) % work_loads_.size();
            work_loads_[worker] += work_size;
            return worker;
        }

        case LoadBalancingType::DYNAMIC: {
            // Assign to worker with least load
            size_t min_worker = 0;
            size_t min_load = work_loads_[0];

            for (size_t i = 1; i < work_loads_.size(); ++i) {
                if (work_loads_[i] < min_load) {
                    min_load = work_loads_[i];
                    min_worker = i;
                }
            }

            work_loads_[min_worker] += work_size;
            return min_worker;
        }

        default:
            return 0;  // Fallback to first worker
    }
}

void LoadBalancer::update_load(size_t worker_id, size_t load_change) {
    if (worker_id < work_loads_.size()) {
        work_loads_[worker_id] -= load_change;
    }
}

void LoadBalancer::rebalance_work() {
    std::lock_guard<std::mutex> lock(balancer_mutex_);

    // Calculate average load
    size_t total_load = 0;
    for (const auto& load : work_loads_) {
        total_load += load;
    }

    size_t average_load = total_load / work_loads_.size();

    // Simple rebalancing: move work from overloaded to underloaded workers
    for (size_t i = 0; i < work_loads_.size(); ++i) {
        if (work_loads_[i] > average_load * 1.2) {  // 20% above average
            size_t excess = work_loads_[i] - average_load;

            // Find underloaded worker
            for (size_t j = 0; j < work_loads_.size(); ++j) {
                if (work_loads_[j] < average_load * 0.8) {  // 20% below average
                    size_t transfer = std::min(excess, average_load - work_loads_[j]);
                    work_loads_[i] -= transfer;
                    work_loads_[j] += transfer;
                    excess -= transfer;

                    if (excess == 0) break;
                }
            }
        }
    }
}

double LoadBalancer::get_load_balance_factor() const {
    if (work_loads_.empty()) return 1.0;

    size_t total_load = 0;
    size_t max_load = 0;
    size_t min_load = work_loads_[0];

    for (const auto& load : work_loads_) {
        total_load += load;
        max_load = std::max(max_load, load.load());
        min_load = std::min(min_load, load.load());
    }

    if (max_load == 0) return 1.0;

    double average_load = double(total_load) / work_loads_.size();
    return average_load / max_load;
}

std::vector<size_t> LoadBalancer::get_work_distribution() const {
    std::vector<size_t> distribution;
    for (const auto& load : work_loads_) {
        distribution.push_back(load);
    }
    return distribution;
}

void LoadBalancer::set_balancing_strategy(LoadBalancingType type) {
    std::lock_guard<std::mutex> lock(balancer_mutex_);
    type_ = type;
}

} // namespace performance
} // namespace simulator
