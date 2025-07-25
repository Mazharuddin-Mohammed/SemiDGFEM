"""
Enhanced Performance Optimization for SemiDGFEM

This module provides comprehensive performance optimization capabilities including:
- Advanced threading and parallelization strategies
- Memory optimization and caching systems
- Computational kernel optimization
- Load balancing and work distribution
- Performance profiling and monitoring
- Adaptive algorithm selection and auto-tuning

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import time
import threading
import multiprocessing
import psutil
import gc
import sys
import os
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import weakref
from functools import wraps, lru_cache
import pickle
import json

class ParallelizationStrategy(Enum):
    """Parallelization strategies for different workloads"""
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

class LoadBalancingType(Enum):
    """Load balancing strategies"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    WORK_STEALING = "work_stealing"
    LOCALITY_AWARE = "locality_aware"
    ADAPTIVE = "adaptive"

class MemoryOptimizationType(Enum):
    """Memory optimization strategies"""
    CACHE_FRIENDLY = "cache_friendly"
    MEMORY_POOL = "memory_pool"
    ZERO_COPY = "zero_copy"
    NUMA_AWARE = "numa_aware"
    COMPRESSION = "compression"

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring and optimization"""
    execution_time: float = 0.0
    cpu_utilization: float = 0.0
    memory_usage: float = 0.0
    cache_hit_ratio: float = 0.0
    throughput: float = 0.0
    efficiency: float = 0.0
    operations_count: int = 0
    memory_allocations: int = 0
    parallel_efficiency: float = 0.0
    load_balance_factor: float = 0.0
    bottleneck_analysis: str = ""

@dataclass
class SystemInfo:
    """System hardware information"""
    cpu_count: int
    cpu_frequency: float
    total_memory: int
    available_memory: int
    cache_sizes: Dict[str, int]
    numa_nodes: int
    has_gpu: bool
    gpu_memory: int

class AdvancedMemoryPool:
    """Advanced memory pool with different allocation strategies"""
    
    def __init__(self, initial_size: int = 1000, block_size: int = 1024, 
                 strategy: str = "fixed"):
        self.strategy = strategy
        self.block_size = block_size
        self.pools = {}
        self.allocated_blocks = set()
        self.free_blocks = queue.Queue()
        self.lock = threading.Lock()
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'peak_usage': 0,
            'current_usage': 0
        }
        
        # Pre-allocate initial blocks
        self._expand_pool(initial_size)
    
    def _expand_pool(self, num_blocks: int):
        """Expand the memory pool with new blocks"""
        for _ in range(num_blocks):
            if self.strategy == "numpy":
                block = np.empty(self.block_size, dtype=np.uint8)
            else:
                block = bytearray(self.block_size)
            self.free_blocks.put(block)
    
    def allocate(self, size: int) -> Optional[Any]:
        """Allocate a memory block"""
        with self.lock:
            if size > self.block_size:
                # Large allocation: create dedicated block
                if self.strategy == "numpy":
                    block = np.empty(size, dtype=np.uint8)
                else:
                    block = bytearray(size)
                self.allocated_blocks.add(id(block))
                self.stats['allocations'] += 1
                self.stats['current_usage'] += size
                self.stats['peak_usage'] = max(self.stats['peak_usage'], 
                                             self.stats['current_usage'])
                return block
            
            try:
                block = self.free_blocks.get_nowait()
                self.allocated_blocks.add(id(block))
                self.stats['allocations'] += 1
                self.stats['current_usage'] += self.block_size
                self.stats['peak_usage'] = max(self.stats['peak_usage'], 
                                             self.stats['current_usage'])
                return block
            except queue.Empty:
                # Pool exhausted: expand
                self._expand_pool(max(10, 10))  # Expand by at least 10 blocks
                try:
                    block = self.free_blocks.get_nowait()
                    self.allocated_blocks.add(id(block))
                    self.stats['allocations'] += 1
                    self.stats['current_usage'] += self.block_size
                    self.stats['peak_usage'] = max(self.stats['peak_usage'],
                                                 self.stats['current_usage'])
                    return block
                except queue.Empty:
                    # Still empty after expansion: create new block
                    if self.strategy == "numpy":
                        block = np.empty(size, dtype=np.uint8)
                    else:
                        block = bytearray(size)
                    self.allocated_blocks.add(id(block))
                    self.stats['allocations'] += 1
                    self.stats['current_usage'] += size
                    self.stats['peak_usage'] = max(self.stats['peak_usage'],
                                                 self.stats['current_usage'])
                    return block
    
    def deallocate(self, block: Any):
        """Deallocate a memory block"""
        with self.lock:
            block_id = id(block)
            if block_id in self.allocated_blocks:
                self.allocated_blocks.remove(block_id)
                if len(block) == self.block_size:
                    self.free_blocks.put(block)
                self.stats['deallocations'] += 1
                self.stats['current_usage'] -= len(block)
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory pool statistics"""
        with self.lock:
            return self.stats.copy()

class AdaptiveCache:
    """Adaptive cache with multiple eviction strategies"""
    
    def __init__(self, max_size: int = 1000, strategy: str = "lru"):
        self.max_size = max_size
        self.strategy = strategy
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.insertion_order = []
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                self.stats['hits'] += 1
                self.access_times[key] = time.time()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return self.cache[key]
            else:
                self.stats['misses'] += 1
                return None
    
    def put(self, key: Any, value: Any):
        """Put value in cache"""
        with self.lock:
            if key in self.cache:
                self.cache[key] = value
                self.access_times[key] = time.time()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return
            
            if len(self.cache) >= self.max_size:
                self._evict()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.insertion_order.append(key)
    
    def _evict(self):
        """Evict item based on strategy"""
        if not self.cache:
            return
        
        if self.strategy == "lru":
            # Least Recently Used
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])
        elif self.strategy == "lfu":
            # Least Frequently Used
            oldest_key = min(self.access_counts.keys(), 
                           key=lambda k: self.access_counts[k])
        elif self.strategy == "fifo":
            # First In, First Out
            oldest_key = self.insertion_order[0]
            self.insertion_order.remove(oldest_key)
        else:
            # Default to LRU
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])
        
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        del self.access_counts[oldest_key]
        self.stats['evictions'] += 1
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.insertion_order.clear()
    
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio"""
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            **self.stats,
            'size': len(self.cache),
            'hit_ratio': self.get_hit_ratio()
        }

class WorkStealingQueue:
    """Work-stealing queue for load balancing"""
    
    def __init__(self):
        self.deque = []
        self.lock = threading.Lock()
    
    def push(self, item: Any):
        """Push item to bottom of deque"""
        with self.lock:
            self.deque.append(item)
    
    def pop(self) -> Optional[Any]:
        """Pop item from bottom of deque (LIFO for owner)"""
        with self.lock:
            if self.deque:
                return self.deque.pop()
            return None
    
    def steal(self) -> Optional[Any]:
        """Steal item from top of deque (FIFO for thieves)"""
        with self.lock:
            if self.deque:
                return self.deque.pop(0)
            return None
    
    def size(self) -> int:
        """Get queue size"""
        with self.lock:
            return len(self.deque)
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        with self.lock:
            return len(self.deque) == 0

class PerformanceProfiler:
    """Advanced performance profiler with detailed analysis"""
    
    def __init__(self):
        self.timers = {}
        self.call_counts = {}
        self.memory_usage = {}
        self.lock = threading.Lock()
        self.enabled = True
        self.start_memory = psutil.Process().memory_info().rss
    
    def start_timer(self, name: str):
        """Start timing a section"""
        if not self.enabled:
            return
        
        with self.lock:
            self.timers[name] = time.perf_counter()
            self.memory_usage[name] = psutil.Process().memory_info().rss
    
    def end_timer(self, name: str) -> float:
        """End timing a section and return duration"""
        if not self.enabled:
            return 0.0
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        with self.lock:
            if name in self.timers:
                duration = end_time - self.timers[name]
                memory_delta = end_memory - self.memory_usage[name]
                
                if name not in self.call_counts:
                    self.call_counts[name] = []
                
                self.call_counts[name].append({
                    'duration': duration,
                    'memory_delta': memory_delta,
                    'timestamp': time.time()
                })
                
                del self.timers[name]
                del self.memory_usage[name]
                
                return duration
        
        return 0.0
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator for profiling functions"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.start_timer(func.__name__)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.end_timer(func.__name__)
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics"""
        with self.lock:
            stats = {}
            for name, calls in self.call_counts.items():
                if calls:
                    durations = [call['duration'] for call in calls]
                    memory_deltas = [call['memory_delta'] for call in calls]
                    
                    stats[name] = {
                        'total_time': sum(durations),
                        'average_time': np.mean(durations),
                        'min_time': min(durations),
                        'max_time': max(durations),
                        'std_time': np.std(durations),
                        'call_count': len(calls),
                        'total_memory_delta': sum(memory_deltas),
                        'average_memory_delta': np.mean(memory_deltas)
                    }
            
            return stats
    
    def reset(self):
        """Reset profiler data"""
        with self.lock:
            self.timers.clear()
            self.call_counts.clear()
            self.memory_usage.clear()
    
    def enable(self, enabled: bool = True):
        """Enable or disable profiling"""
        self.enabled = enabled
    
    def generate_report(self) -> str:
        """Generate detailed profiling report"""
        stats = self.get_stats()
        
        if not stats:
            return "No profiling data available."
        
        report = ["Performance Profiling Report", "=" * 50, ""]
        
        # Sort by total time
        sorted_stats = sorted(stats.items(), 
                            key=lambda x: x[1]['total_time'], 
                            reverse=True)
        
        total_time = sum(stat['total_time'] for _, stat in sorted_stats)
        
        report.append(f"{'Function':<30} {'Total(s)':<10} {'Avg(s)':<10} {'Calls':<8} {'%':<6}")
        report.append("-" * 70)
        
        for name, stat in sorted_stats:
            percentage = (stat['total_time'] / total_time * 100) if total_time > 0 else 0
            report.append(f"{name:<30} {stat['total_time']:<10.6f} "
                         f"{stat['average_time']:<10.6f} {stat['call_count']:<8} "
                         f"{percentage:<6.2f}")
        
        report.extend(["", f"Total execution time: {total_time:.6f} seconds"])
        
        return "\n".join(report)

class EnhancedPerformanceOptimizer:
    """Enhanced performance optimizer with adaptive algorithms"""

    def __init__(self):
        self.parallelization_strategy = ParallelizationStrategy.ADAPTIVE
        self.load_balancing_type = LoadBalancingType.DYNAMIC
        self.memory_optimization = MemoryOptimizationType.CACHE_FRIENDLY
        self.num_threads = multiprocessing.cpu_count()
        self.cache_size = 1024 * 1024  # 1MB
        self.auto_tuning_enabled = False

        # Initialize components
        self.profiler = PerformanceProfiler()
        self.memory_pool = AdvancedMemoryPool()
        self.cache = AdaptiveCache(max_size=1000)
        self.work_queues = [WorkStealingQueue() for _ in range(self.num_threads)]

        # Performance history for learning
        self.performance_history = []
        self.optimization_suggestions = []

        # System information
        self.system_info = self._detect_system_info()

        # Thread pools
        self.thread_pool = None
        self.process_pool = None

        # Initialize thread pool for default strategy
        self._reconfigure_pools()

    def _detect_system_info(self) -> SystemInfo:
        """Detect system hardware information"""
        try:
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()

            # Try to detect GPU
            has_gpu = False
            gpu_memory = 0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                has_gpu = True
                gpu_memory = gpu_info.total
            except:
                pass

            return SystemInfo(
                cpu_count=multiprocessing.cpu_count(),
                cpu_frequency=cpu_freq.max if cpu_freq else 0.0,
                total_memory=memory.total,
                available_memory=memory.available,
                cache_sizes={
                    'L1': 32 * 1024,    # Typical values
                    'L2': 256 * 1024,
                    'L3': 8 * 1024 * 1024
                },
                numa_nodes=1,  # Simplified
                has_gpu=has_gpu,
                gpu_memory=gpu_memory
            )
        except Exception:
            # Fallback values
            return SystemInfo(
                cpu_count=multiprocessing.cpu_count(),
                cpu_frequency=3000.0,
                total_memory=16 * 1024**3,
                available_memory=8 * 1024**3,
                cache_sizes={'L1': 32*1024, 'L2': 256*1024, 'L3': 8*1024*1024},
                numa_nodes=1,
                has_gpu=False,
                gpu_memory=0
            )

    def set_parallelization_strategy(self, strategy: ParallelizationStrategy):
        """Set parallelization strategy"""
        self.parallelization_strategy = strategy
        self._reconfigure_pools()

    def set_load_balancing_type(self, lb_type: LoadBalancingType):
        """Set load balancing type"""
        self.load_balancing_type = lb_type

    def set_memory_optimization(self, mem_opt: MemoryOptimizationType):
        """Set memory optimization type"""
        self.memory_optimization = mem_opt
        self._reconfigure_memory()

    def set_num_threads(self, num_threads: int):
        """Set number of threads"""
        self.num_threads = min(num_threads, self.system_info.cpu_count)
        self.work_queues = [WorkStealingQueue() for _ in range(self.num_threads)]
        self._reconfigure_pools()

    def enable_auto_tuning(self, enabled: bool = True):
        """Enable or disable auto-tuning"""
        self.auto_tuning_enabled = enabled

    def _reconfigure_pools(self):
        """Reconfigure thread/process pools"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        if self.process_pool:
            self.process_pool.shutdown(wait=False)

        if self.parallelization_strategy in [ParallelizationStrategy.THREADING,
                                           ParallelizationStrategy.THREAD_POOL,
                                           ParallelizationStrategy.HYBRID]:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)

        if self.parallelization_strategy in [ParallelizationStrategy.MULTIPROCESSING,
                                           ParallelizationStrategy.PROCESS_POOL,
                                           ParallelizationStrategy.HYBRID]:
            self.process_pool = ProcessPoolExecutor(max_workers=self.num_threads)

    def _reconfigure_memory(self):
        """Reconfigure memory optimization"""
        if self.memory_optimization == MemoryOptimizationType.MEMORY_POOL:
            self.memory_pool = AdvancedMemoryPool(strategy="numpy")
        elif self.memory_optimization == MemoryOptimizationType.CACHE_FRIENDLY:
            self.cache = AdaptiveCache(strategy="lru")
        elif self.memory_optimization == MemoryOptimizationType.COMPRESSION:
            # Would implement compression-based memory management
            pass

    def optimize_for_problem_size(self, problem_size: int):
        """Optimize configuration for specific problem size"""
        if problem_size < 1000:
            # Small problems: minimize overhead
            self.set_num_threads(1)
            self.set_parallelization_strategy(ParallelizationStrategy.THREADING)
        elif problem_size < 100000:
            # Medium problems: moderate parallelization
            self.set_num_threads(min(4, self.system_info.cpu_count))
            self.set_parallelization_strategy(ParallelizationStrategy.THREAD_POOL)
        else:
            # Large problems: full parallelization
            self.set_num_threads(self.system_info.cpu_count)
            self.set_parallelization_strategy(ParallelizationStrategy.HYBRID)

        # Adjust cache size
        optimal_cache = min(self.cache_size, problem_size * 8 // 4)  # 8 bytes per double, 1/4 for cache
        self.cache = AdaptiveCache(max_size=optimal_cache // 64)  # Assume 64 bytes per cache entry

    def optimize_for_memory_constraints(self, available_memory: int):
        """Optimize for memory constraints"""
        memory_per_thread = available_memory // self.num_threads

        if memory_per_thread < 100 * 1024 * 1024:  # Less than 100MB per thread
            # Reduce threads to avoid memory pressure
            new_threads = max(1, available_memory // (100 * 1024 * 1024))
            self.set_num_threads(new_threads)
            self.set_memory_optimization(MemoryOptimizationType.COMPRESSION)

        # Adjust cache size
        max_cache = available_memory // 10  # Use at most 10% for cache
        cache_entries = min(self.cache.max_size, max_cache // 64)
        self.cache = AdaptiveCache(max_size=cache_entries)

    def parallel_map(self, func: Callable, data: List[Any],
                    chunk_size: Optional[int] = None) -> List[Any]:
        """Parallel map operation with adaptive strategy"""
        if not data:
            return []

        if len(data) < 100 or self.num_threads == 1:
            # Small data or single thread: use regular map
            return [func(item) for item in data]

        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.num_threads * 4))

        if self.parallelization_strategy == ParallelizationStrategy.MULTIPROCESSING:
            return self._parallel_map_process(func, data, chunk_size)
        else:
            return self._parallel_map_thread(func, data, chunk_size)

    def _parallel_map_thread(self, func: Callable, data: List[Any],
                           chunk_size: int) -> List[Any]:
        """Thread-based parallel map"""
        if not self.thread_pool:
            self._reconfigure_pools()

        # Fallback to sequential if thread pool creation failed
        if not self.thread_pool:
            return [func(item) for item in data]

        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        def process_chunk(chunk):
            return [func(item) for item in chunk]

        futures = [self.thread_pool.submit(process_chunk, chunk) for chunk in chunks]
        results = []

        for future in as_completed(futures):
            results.extend(future.result())

        return results

    def _parallel_map_process(self, func: Callable, data: List[Any],
                            chunk_size: int) -> List[Any]:
        """Process-based parallel map"""
        if not self.process_pool:
            self._reconfigure_pools()

        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        def process_chunk(chunk):
            return [func(item) for item in chunk]

        futures = [self.process_pool.submit(process_chunk, chunk) for chunk in chunks]
        results = []

        for future in as_completed(futures):
            results.extend(future.result())

        return results

    def parallel_reduce(self, data: List[Any], reducer: Callable[[Any, Any], Any],
                       initial: Any = None) -> Any:
        """Parallel reduce operation"""
        if not data:
            return initial

        if len(data) == 1:
            return data[0] if initial is None else reducer(initial, data[0])

        if len(data) < 100 or self.num_threads == 1:
            # Small data: use regular reduce
            if initial is None:
                result = data[0]
                start_idx = 1
            else:
                result = initial
                start_idx = 0

            for i in range(start_idx, len(data)):
                result = reducer(result, data[i])
            return result

        # Parallel reduce using divide and conquer
        chunk_size = max(1, len(data) // self.num_threads)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        def reduce_chunk(chunk):
            if not chunk:
                return initial
            result = chunk[0]
            for item in chunk[1:]:
                result = reducer(result, item)
            return result

        # Reduce chunks in parallel
        chunk_results = self.parallel_map(reduce_chunk, chunks)

        # Final reduction
        if not chunk_results:
            return initial

        # Start with initial value if provided, otherwise first chunk result
        if initial is not None:
            final_result = initial
            start_idx = 0
        else:
            final_result = chunk_results[0]
            start_idx = 1

        for i in range(start_idx, len(chunk_results)):
            if chunk_results[i] is not None:
                final_result = reducer(final_result, chunk_results[i])

        return final_result

    def measure_performance(self, operation: Callable, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of an operation"""
        # Clear caches and force garbage collection
        gc.collect()

        # Get initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Measure execution
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        try:
            operation(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            end_time = time.perf_counter()
            end_cpu = time.process_time()

            # Get final system state
            final_memory = process.memory_info().rss

        # Calculate metrics
        execution_time = end_time - start_time
        cpu_time = end_cpu - start_cpu
        cpu_utilization = cpu_time / execution_time if execution_time > 0 else 0
        memory_usage = final_memory - initial_memory

        # Estimate other metrics
        cache_hit_ratio = self.cache.get_hit_ratio()
        throughput = 1.0 / execution_time if execution_time > 0 else 0
        parallel_efficiency = cpu_utilization / self.num_threads if self.num_threads > 0 else 0

        metrics = PerformanceMetrics(
            execution_time=execution_time,
            cpu_utilization=cpu_utilization,
            memory_usage=memory_usage / (1024 * 1024),  # Convert to MB
            cache_hit_ratio=cache_hit_ratio,
            throughput=throughput,
            efficiency=cpu_utilization,
            operations_count=1,
            memory_allocations=0,  # Would need instrumentation
            parallel_efficiency=parallel_efficiency,
            load_balance_factor=1.0,  # Simplified
            bottleneck_analysis=self._analyze_bottleneck(cpu_utilization, memory_usage)
        )

        # Store for learning
        self.performance_history.append(metrics)

        return metrics

    def _analyze_bottleneck(self, cpu_utilization: float, memory_usage: int) -> str:
        """Analyze performance bottleneck"""
        if cpu_utilization > 0.9:
            return "CPU-bound"
        elif memory_usage > self.system_info.available_memory * 0.8:
            return "Memory-bound"
        elif cpu_utilization < 0.3:
            return "I/O-bound or synchronization overhead"
        else:
            return "Balanced"

    def auto_tune(self, benchmark_func: Callable, benchmark_args: tuple = (),
                  benchmark_kwargs: dict = None) -> Dict[str, Any]:
        """Auto-tune performance parameters"""
        if not self.auto_tuning_enabled:
            return {}

        if benchmark_kwargs is None:
            benchmark_kwargs = {}

        best_config = None
        best_performance = float('inf')

        # Test different configurations
        configs_to_test = [
            {'strategy': ParallelizationStrategy.THREADING, 'threads': 1},
            {'strategy': ParallelizationStrategy.THREADING, 'threads': 2},
            {'strategy': ParallelizationStrategy.THREADING, 'threads': 4},
            {'strategy': ParallelizationStrategy.THREAD_POOL, 'threads': self.system_info.cpu_count},
            {'strategy': ParallelizationStrategy.MULTIPROCESSING, 'threads': 2},
            {'strategy': ParallelizationStrategy.HYBRID, 'threads': self.system_info.cpu_count},
        ]

        for config in configs_to_test:
            # Apply configuration
            original_strategy = self.parallelization_strategy
            original_threads = self.num_threads

            self.set_parallelization_strategy(config['strategy'])
            self.set_num_threads(config['threads'])

            try:
                # Benchmark
                metrics = self.measure_performance(benchmark_func, *benchmark_args, **benchmark_kwargs)

                # Use execution time as primary metric
                if metrics.execution_time < best_performance:
                    best_performance = metrics.execution_time
                    best_config = config.copy()
                    best_config['metrics'] = metrics

            except Exception as e:
                print(f"Configuration {config} failed: {e}")
            finally:
                # Restore original configuration
                self.set_parallelization_strategy(original_strategy)
                self.set_num_threads(original_threads)

        # Apply best configuration
        if best_config:
            self.set_parallelization_strategy(best_config['strategy'])
            self.set_num_threads(best_config['threads'])

            print(f"Auto-tuning complete. Best configuration: {best_config}")
            return best_config

        return {}

    def suggest_optimizations(self) -> List[str]:
        """Generate optimization suggestions based on performance history"""
        if not self.performance_history:
            return ["No performance data available for analysis"]

        suggestions = []
        recent_metrics = self.performance_history[-10:]  # Last 10 measurements

        avg_cpu_util = np.mean([m.cpu_utilization for m in recent_metrics])
        avg_parallel_eff = np.mean([m.parallel_efficiency for m in recent_metrics])
        avg_cache_hit = np.mean([m.cache_hit_ratio for m in recent_metrics])

        if avg_parallel_eff < 0.5 and self.num_threads > 1:
            suggestions.append(f"Consider reducing thread count (current efficiency: {avg_parallel_eff:.2%})")

        if avg_cpu_util < 0.7 and self.num_threads < self.system_info.cpu_count:
            suggestions.append(f"Consider increasing parallelization (CPU utilization: {avg_cpu_util:.2%})")

        if avg_cache_hit < 0.8:
            suggestions.append(f"Optimize memory access patterns (cache hit ratio: {avg_cache_hit:.2%})")

        if all(m.bottleneck_analysis == "Memory-bound" for m in recent_metrics[-3:]):
            suggestions.append("Consider memory optimization strategies or reducing memory usage")

        if all(m.bottleneck_analysis == "I/O-bound or synchronization overhead" for m in recent_metrics[-3:]):
            suggestions.append("Consider reducing synchronization overhead or optimizing I/O operations")

        return suggestions if suggestions else ["Performance appears optimal"]

    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        report = ["Enhanced Performance Optimization Report", "=" * 50, ""]

        # System information
        report.extend([
            "System Information:",
            f"  CPU Cores: {self.system_info.cpu_count}",
            f"  CPU Frequency: {self.system_info.cpu_frequency:.1f} MHz",
            f"  Total Memory: {self.system_info.total_memory / (1024**3):.1f} GB",
            f"  Available Memory: {self.system_info.available_memory / (1024**3):.1f} GB",
            f"  GPU Available: {self.system_info.has_gpu}",
            ""
        ])

        # Current configuration
        report.extend([
            "Current Configuration:",
            f"  Parallelization Strategy: {self.parallelization_strategy.value}",
            f"  Load Balancing: {self.load_balancing_type.value}",
            f"  Memory Optimization: {self.memory_optimization.value}",
            f"  Number of Threads: {self.num_threads}",
            f"  Auto-tuning: {'Enabled' if self.auto_tuning_enabled else 'Disabled'}",
            ""
        ])

        # Performance history
        if self.performance_history:
            recent = self.performance_history[-10:]
            report.extend([
                "Recent Performance (last 10 measurements):",
                f"  Average Execution Time: {np.mean([m.execution_time for m in recent]):.6f} seconds",
                f"  Average CPU Utilization: {np.mean([m.cpu_utilization for m in recent]):.2%}",
                f"  Average Memory Usage: {np.mean([m.memory_usage for m in recent]):.2f} MB",
                f"  Average Cache Hit Ratio: {np.mean([m.cache_hit_ratio for m in recent]):.2%}",
                f"  Average Parallel Efficiency: {np.mean([m.parallel_efficiency for m in recent]):.2%}",
                ""
            ])

        # Component statistics
        cache_stats = self.cache.get_stats()
        memory_stats = self.memory_pool.get_stats()

        report.extend([
            "Component Statistics:",
            f"  Cache Hit Ratio: {cache_stats['hit_ratio']:.2%}",
            f"  Cache Size: {cache_stats['size']} entries",
            f"  Memory Pool Allocations: {memory_stats['allocations']}",
            f"  Memory Pool Peak Usage: {memory_stats['peak_usage']} bytes",
            ""
        ])

        # Optimization suggestions
        suggestions = self.suggest_optimizations()
        report.extend([
            "Optimization Suggestions:",
            *[f"  • {suggestion}" for suggestion in suggestions],
            ""
        ])

        # Profiler report
        if self.profiler.call_counts:
            report.extend(["", "Detailed Profiling:", self.profiler.generate_report()])

        return "\n".join(report)

    def cleanup(self):
        """Cleanup resources"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)

        self.profiler.reset()
        self.cache.clear()

# Utility functions for performance optimization
def optimize_numpy_operations():
    """Optimize NumPy operations for better performance"""
    # Set optimal BLAS threads
    try:
        import os
        os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
        os.environ['NUMEXPR_NUM_THREADS'] = str(multiprocessing.cpu_count())
    except:
        pass

def profile_decorator(profiler: PerformanceProfiler):
    """Decorator factory for profiling functions"""
    def decorator(func):
        return profiler.profile_function(func)
    return decorator

# Context manager for performance measurement
class PerformanceContext:
    """Context manager for measuring performance of code blocks"""

    def __init__(self, optimizer: EnhancedPerformanceOptimizer, name: str):
        self.optimizer = optimizer
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.optimizer.profiler.start_timer(self.name)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.optimizer.profiler.end_timer(self.name)
        if exc_type is None:
            print(f"{self.name} completed in {duration:.6f} seconds")
        else:
            print(f"{self.name} failed after {duration:.6f} seconds")

# Example usage and testing functions
def example_cpu_intensive_task(n: int) -> float:
    """Example CPU-intensive task for testing"""
    return sum(i * i for i in range(n))

def example_memory_intensive_task(size: int) -> np.ndarray:
    """Example memory-intensive task for testing"""
    data = np.random.random((size, size))
    return np.dot(data, data.T)

def run_performance_demo():
    """Demonstrate performance optimization capabilities"""
    print("Enhanced Performance Optimization Demo")
    print("=" * 50)

    optimizer = EnhancedPerformanceOptimizer()
    optimizer.enable_auto_tuning(True)

    # Test CPU-intensive task
    print("\nTesting CPU-intensive task...")
    with PerformanceContext(optimizer, "cpu_task"):
        result = optimizer.parallel_map(example_cpu_intensive_task, [10000] * 100)

    # Test memory-intensive task
    print("\nTesting memory-intensive task...")
    with PerformanceContext(optimizer, "memory_task"):
        result = example_memory_intensive_task(1000)

    # Auto-tune for the CPU task
    print("\nAuto-tuning for CPU task...")
    best_config = optimizer.auto_tune(
        lambda: optimizer.parallel_map(example_cpu_intensive_task, [5000] * 50)
    )

    # Generate report
    print("\n" + optimizer.generate_report())

    # Cleanup
    optimizer.cleanup()

if __name__ == "__main__":
    run_performance_demo()
