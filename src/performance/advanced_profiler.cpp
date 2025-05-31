#include "performance_optimization.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace simulator {
namespace performance {

// Advanced performance profiler with detailed analysis
class AdvancedProfiler {
private:
    struct TimingData {
        std::chrono::high_resolution_clock::time_point start_time;
        std::vector<double> durations;
        double total_time = 0.0;
        double min_time = std::numeric_limits<double>::max();
        double max_time = 0.0;
        size_t call_count = 0;
        size_t memory_usage = 0;
    };
    
    std::unordered_map<std::string, TimingData> timing_data_;
    std::mutex profiler_mutex_;
    bool enabled_ = true;
    
public:
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    
    void start_timer(const std::string& name) {
        if (!enabled_) return;
        
        std::lock_guard<std::mutex> lock(profiler_mutex_);
        timing_data_[name].start_time = std::chrono::high_resolution_clock::now();
    }
    
    void end_timer(const std::string& name) {
        if (!enabled_) return;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(profiler_mutex_);
        
        auto& data = timing_data_[name];
        if (data.start_time.time_since_epoch().count() > 0) {
            double duration = std::chrono::duration<double, std::milli>(
                end_time - data.start_time).count();
            
            data.durations.push_back(duration);
            data.total_time += duration;
            data.min_time = std::min(data.min_time, duration);
            data.max_time = std::max(data.max_time, duration);
            data.call_count++;
        }
    }
    
    void add_memory_usage(const std::string& name, size_t bytes) {
        if (!enabled_) return;
        
        std::lock_guard<std::mutex> lock(profiler_mutex_);
        timing_data_[name].memory_usage += bytes;
    }
    
    void generate_detailed_report() const {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "ADVANCED PERFORMANCE ANALYSIS REPORT\n";
        std::cout << std::string(80, '=') << "\n";
        
        // Sort by total time
        std::vector<std::pair<std::string, TimingData>> sorted_data;
        for (const auto& [name, data] : timing_data_) {
            sorted_data.emplace_back(name, data);
        }
        
        std::sort(sorted_data.begin(), sorted_data.end(),
                  [](const auto& a, const auto& b) {
                      return a.second.total_time > b.second.total_time;
                  });
        
        // Header
        std::cout << std::setw(25) << "Function"
                  << std::setw(10) << "Calls"
                  << std::setw(12) << "Total(ms)"
                  << std::setw(12) << "Avg(ms)"
                  << std::setw(12) << "Min(ms)"
                  << std::setw(12) << "Max(ms)"
                  << std::setw(12) << "StdDev"
                  << std::setw(12) << "Memory(KB)" << "\n";
        std::cout << std::string(110, '-') << "\n";
        
        double total_program_time = 0.0;
        for (const auto& [name, data] : sorted_data) {
            total_program_time += data.total_time;
        }
        
        for (const auto& [name, data] : sorted_data) {
            if (data.call_count == 0) continue;
            
            double avg_time = data.total_time / data.call_count;
            double std_dev = calculate_std_dev(data.durations, avg_time);
            double percentage = (total_program_time > 0) ? 
                               (data.total_time / total_program_time * 100.0) : 0.0;
            
            std::cout << std::setw(25) << name.substr(0, 24)
                      << std::setw(10) << data.call_count
                      << std::setw(12) << std::fixed << std::setprecision(3) << data.total_time
                      << std::setw(12) << std::fixed << std::setprecision(3) << avg_time
                      << std::setw(12) << std::fixed << std::setprecision(3) << data.min_time
                      << std::setw(12) << std::fixed << std::setprecision(3) << data.max_time
                      << std::setw(12) << std::fixed << std::setprecision(3) << std_dev
                      << std::setw(12) << data.memory_usage / 1024 << "\n";
        }
        
        std::cout << std::string(110, '-') << "\n";
        std::cout << "Total execution time: " << total_program_time << " ms\n";
        
        // Performance insights
        generate_performance_insights(sorted_data, total_program_time);
    }
    
    void export_to_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return;
        
        file << "Function,Calls,Total_ms,Average_ms,Min_ms,Max_ms,StdDev,Memory_KB\n";
        
        for (const auto& [name, data] : timing_data_) {
            if (data.call_count == 0) continue;
            
            double avg_time = data.total_time / data.call_count;
            double std_dev = calculate_std_dev(data.durations, avg_time);
            
            file << name << ","
                 << data.call_count << ","
                 << data.total_time << ","
                 << avg_time << ","
                 << data.min_time << ","
                 << data.max_time << ","
                 << std_dev << ","
                 << data.memory_usage / 1024 << "\n";
        }
    }
    
private:
    double calculate_std_dev(const std::vector<double>& values, double mean) const {
        if (values.size() <= 1) return 0.0;
        
        double sum_sq_diff = 0.0;
        for (double value : values) {
            double diff = value - mean;
            sum_sq_diff += diff * diff;
        }
        
        return std::sqrt(sum_sq_diff / (values.size() - 1));
    }
    
    void generate_performance_insights(
        const std::vector<std::pair<std::string, TimingData>>& sorted_data,
        double total_time) const {
        
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "PERFORMANCE INSIGHTS\n";
        std::cout << std::string(50, '=') << "\n";
        
        // Top hotspots
        std::cout << "🔥 TOP PERFORMANCE HOTSPOTS:\n";
        for (size_t i = 0; i < std::min(size_t(5), sorted_data.size()); ++i) {
            const auto& [name, data] = sorted_data[i];
            double percentage = (data.total_time / total_time) * 100.0;
            std::cout << "   " << (i+1) << ". " << name 
                      << " (" << std::fixed << std::setprecision(1) 
                      << percentage << "% of total time)\n";
        }
        
        // Functions with high variance
        std::cout << "\n⚠️  FUNCTIONS WITH HIGH VARIANCE:\n";
        for (const auto& [name, data] : sorted_data) {
            if (data.call_count < 2) continue;
            
            double avg_time = data.total_time / data.call_count;
            double std_dev = calculate_std_dev(data.durations, avg_time);
            double cv = (avg_time > 0) ? (std_dev / avg_time) : 0.0;
            
            if (cv > 0.5) {  // Coefficient of variation > 50%
                std::cout << "   • " << name 
                          << " (CV: " << std::fixed << std::setprecision(2) 
                          << cv * 100 << "%)\n";
            }
        }
        
        // Memory usage analysis
        std::cout << "\n💾 MEMORY USAGE ANALYSIS:\n";
        size_t total_memory = 0;
        for (const auto& [name, data] : timing_data_) {
            total_memory += data.memory_usage;
        }
        
        if (total_memory > 0) {
            std::cout << "   Total memory tracked: " 
                      << total_memory / (1024 * 1024) << " MB\n";
            
            for (const auto& [name, data] : sorted_data) {
                if (data.memory_usage > total_memory * 0.1) {  // > 10% of total
                    double percentage = (double(data.memory_usage) / total_memory) * 100.0;
                    std::cout << "   • " << name << ": " 
                              << data.memory_usage / (1024 * 1024) << " MB ("
                              << std::fixed << std::setprecision(1) 
                              << percentage << "%)\n";
                }
            }
        }
        
        // Optimization recommendations
        std::cout << "\n🚀 OPTIMIZATION RECOMMENDATIONS:\n";
        
        if (!sorted_data.empty()) {
            const auto& top_hotspot = sorted_data[0];
            double top_percentage = (top_hotspot.second.total_time / total_time) * 100.0;
            
            if (top_percentage > 50.0) {
                std::cout << "   • Focus optimization on '" << top_hotspot.first 
                          << "' (dominates " << std::fixed << std::setprecision(1)
                          << top_percentage << "% of execution time)\n";
            }
            
            if (top_percentage > 20.0) {
                std::cout << "   • Consider parallelizing '" << top_hotspot.first << "'\n";
            }
        }
        
        // Check for potential I/O bottlenecks
        for (const auto& [name, data] : sorted_data) {
            if (name.find("io") != std::string::npos || 
                name.find("file") != std::string::npos ||
                name.find("read") != std::string::npos ||
                name.find("write") != std::string::npos) {
                std::cout << "   • Potential I/O bottleneck detected in '" 
                          << name << "'\n";
            }
        }
        
        std::cout << "\n";
    }
};

// Global advanced profiler instance
static AdvancedProfiler advanced_profiler;

// Performance benchmark suite
class PerformanceBenchmark {
public:
    static void run_comprehensive_benchmarks() {
        std::cout << "\n🏃 RUNNING COMPREHENSIVE PERFORMANCE BENCHMARKS\n";
        std::cout << std::string(60, '=') << "\n";
        
        benchmark_vector_operations();
        benchmark_matrix_operations();
        benchmark_memory_operations();
        benchmark_parallel_operations();
        
        std::cout << "\n✅ Benchmarks completed!\n";
    }
    
private:
    static void benchmark_vector_operations() {
        std::cout << "\n📊 Vector Operations Benchmark:\n";
        
        const std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};
        
        for (size_t n : sizes) {
            std::vector<double> a(n, 1.0), b(n, 2.0), result(n);
            
            // Vector addition
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 100; ++i) {
                #pragma omp simd
                for (size_t j = 0; j < n; ++j) {
                    result[j] = a[j] + b[j];
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double, std::milli>(end - start).count();
            double throughput = (n * sizeof(double) * 3 * 100) / (duration * 1e-3) / 1e9;
            
            std::cout << "   Size " << std::setw(8) << n 
                      << ": " << std::setw(8) << std::fixed << std::setprecision(2) 
                      << throughput << " GB/s\n";
        }
    }
    
    static void benchmark_matrix_operations() {
        std::cout << "\n🔢 Matrix Operations Benchmark:\n";
        
        const std::vector<size_t> sizes = {100, 200, 500, 1000};
        
        for (size_t n : sizes) {
            std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));
            std::vector<double> x(n, 1.0), y(n, 0.0);
            
            // Matrix-vector multiplication
            auto start = std::chrono::high_resolution_clock::now();
            
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                y[i] = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    y[i] += A[i][j] * x[j];
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double, std::milli>(end - start).count();
            double flops = 2.0 * n * n;  // n^2 multiply-adds
            double gflops = flops / (duration * 1e-3) / 1e9;
            
            std::cout << "   " << n << "x" << n 
                      << ": " << std::setw(8) << std::fixed << std::setprecision(2) 
                      << gflops << " GFLOPS\n";
        }
    }
    
    static void benchmark_memory_operations() {
        std::cout << "\n💾 Memory Operations Benchmark:\n";
        
        const std::vector<size_t> sizes = {1024, 10240, 102400, 1024000};
        
        for (size_t n : sizes) {
            std::vector<double> data(n);
            
            // Sequential access
            auto start = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < 1000; ++iter) {
                for (size_t i = 0; i < n; ++i) {
                    data[i] = i * 1.001;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double, std::milli>(end - start).count();
            double bandwidth = (n * sizeof(double) * 1000) / (duration * 1e-3) / 1e9;
            
            std::cout << "   Sequential " << std::setw(8) << n 
                      << ": " << std::setw(8) << std::fixed << std::setprecision(2) 
                      << bandwidth << " GB/s\n";
        }
    }
    
    static void benchmark_parallel_operations() {
        std::cout << "\n🔄 Parallel Operations Benchmark:\n";
        
        const size_t n = 1000000;
        std::vector<double> a(n, 1.0), b(n, 2.0), result(n);
        
        // Test different thread counts
        const std::vector<int> thread_counts = {1, 2, 4, 8};
        
        for (int threads : thread_counts) {
            omp_set_num_threads(threads);
            
            auto start = std::chrono::high_resolution_clock::now();
            
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                result[i] = a[i] + b[i];
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            
            double duration = std::chrono::duration<double, std::milli>(end - start).count();
            double throughput = (n * sizeof(double) * 3) / (duration * 1e-3) / 1e9;
            
            std::cout << "   " << threads << " threads: " 
                      << std::setw(8) << std::fixed << std::setprecision(2) 
                      << throughput << " GB/s\n";
        }
    }
};

// Public interface functions
void start_advanced_profiling(const std::string& name) {
    advanced_profiler.start_timer(name);
}

void end_advanced_profiling(const std::string& name) {
    advanced_profiler.end_timer(name);
}

void add_memory_tracking(const std::string& name, size_t bytes) {
    advanced_profiler.add_memory_usage(name, bytes);
}

void generate_performance_report() {
    advanced_profiler.generate_detailed_report();
}

void export_performance_data(const std::string& filename) {
    advanced_profiler.export_to_csv(filename);
}

void run_performance_benchmarks() {
    PerformanceBenchmark::run_comprehensive_benchmarks();
}

void enable_profiling() {
    advanced_profiler.enable();
}

void disable_profiling() {
    advanced_profiler.disable();
}

} // namespace performance
} // namespace simulator
