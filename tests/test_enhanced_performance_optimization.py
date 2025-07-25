"""
Comprehensive tests for enhanced performance optimization

This test suite validates all components of the enhanced performance optimization
system including adaptive algorithms, memory management, and auto-tuning.

Author: Dr. Mazharuddin Mohammed
"""

import unittest
import numpy as np
import time
import threading
import multiprocessing
import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from enhanced_performance_optimization import (
    EnhancedPerformanceOptimizer,
    PerformanceProfiler,
    AdvancedMemoryPool,
    AdaptiveCache,
    WorkStealingQueue,
    PerformanceContext,
    ParallelizationStrategy,
    LoadBalancingType,
    MemoryOptimizationType,
    PerformanceMetrics,
    SystemInfo,
    optimize_numpy_operations,
    profile_decorator,
    example_cpu_intensive_task,
    example_memory_intensive_task
)

class TestAdvancedMemoryPool(unittest.TestCase):
    """Test advanced memory pool functionality"""
    
    def setUp(self):
        self.pool = AdvancedMemoryPool(initial_size=10, block_size=1024)
    
    def test_allocation_and_deallocation(self):
        """Test basic allocation and deallocation"""
        # Allocate a block
        block = self.pool.allocate(512)
        self.assertIsNotNone(block)
        self.assertGreaterEqual(len(block), 512)
        
        # Check stats
        stats = self.pool.get_stats()
        self.assertEqual(stats['allocations'], 1)
        self.assertGreater(stats['current_usage'], 0)
        
        # Deallocate
        self.pool.deallocate(block)
        stats = self.pool.get_stats()
        self.assertEqual(stats['deallocations'], 1)
    
    def test_large_allocation(self):
        """Test allocation larger than block size"""
        large_block = self.pool.allocate(2048)  # Larger than default block size
        self.assertIsNotNone(large_block)
        self.assertGreaterEqual(len(large_block), 2048)
        
        self.pool.deallocate(large_block)
    
    def test_pool_expansion(self):
        """Test automatic pool expansion"""
        blocks = []
        # Allocate more blocks than initial pool size
        for _ in range(15):
            block = self.pool.allocate(512)
            blocks.append(block)
        
        # All allocations should succeed
        self.assertEqual(len(blocks), 15)
        
        # Clean up
        for block in blocks:
            self.pool.deallocate(block)
    
    def test_numpy_strategy(self):
        """Test numpy allocation strategy"""
        numpy_pool = AdvancedMemoryPool(strategy="numpy")
        block = numpy_pool.allocate(1024)
        self.assertIsInstance(block, np.ndarray)
        numpy_pool.deallocate(block)

class TestAdaptiveCache(unittest.TestCase):
    """Test adaptive cache functionality"""
    
    def setUp(self):
        self.cache = AdaptiveCache(max_size=5, strategy="lru")
    
    def test_basic_operations(self):
        """Test basic cache operations"""
        # Test miss
        result = self.cache.get("key1")
        self.assertIsNone(result)
        
        # Test put and hit
        self.cache.put("key1", "value1")
        result = self.cache.get("key1")
        self.assertEqual(result, "value1")
        
        # Check stats
        stats = self.cache.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
    
    def test_lru_eviction(self):
        """Test LRU eviction policy"""
        # Fill cache to capacity
        for i in range(5):
            self.cache.put(f"key{i}", f"value{i}")
        
        # Access key0 to make it recently used
        self.cache.get("key0")
        
        # Add one more item to trigger eviction
        self.cache.put("key5", "value5")
        
        # key1 should be evicted (least recently used)
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNotNone(self.cache.get("key0"))  # Should still be there
    
    def test_lfu_eviction(self):
        """Test LFU eviction policy"""
        lfu_cache = AdaptiveCache(max_size=3, strategy="lfu")
        
        # Add items with different access frequencies
        lfu_cache.put("key1", "value1")
        lfu_cache.put("key2", "value2")
        lfu_cache.put("key3", "value3")
        
        # Access key1 multiple times
        for _ in range(5):
            lfu_cache.get("key1")
        
        # Access key2 once
        lfu_cache.get("key2")
        
        # Add new item to trigger eviction
        lfu_cache.put("key4", "value4")
        
        # key3 should be evicted (least frequently used)
        self.assertIsNone(lfu_cache.get("key3"))
        self.assertIsNotNone(lfu_cache.get("key1"))
    
    def test_hit_ratio_calculation(self):
        """Test hit ratio calculation"""
        # Start with empty cache
        self.assertEqual(self.cache.get_hit_ratio(), 0.0)
        
        # Add some items and access them
        self.cache.put("key1", "value1")
        self.cache.get("key1")  # Hit
        self.cache.get("key2")  # Miss
        
        # Should be 50% hit ratio
        self.assertAlmostEqual(self.cache.get_hit_ratio(), 0.5, places=2)

class TestWorkStealingQueue(unittest.TestCase):
    """Test work-stealing queue functionality"""
    
    def setUp(self):
        self.queue = WorkStealingQueue()
    
    def test_basic_operations(self):
        """Test basic queue operations"""
        # Test empty queue
        self.assertTrue(self.queue.empty())
        self.assertEqual(self.queue.size(), 0)
        self.assertIsNone(self.queue.pop())
        self.assertIsNone(self.queue.steal())
        
        # Add items
        self.queue.push("item1")
        self.queue.push("item2")
        self.assertFalse(self.queue.empty())
        self.assertEqual(self.queue.size(), 2)
        
        # Pop (LIFO for owner)
        item = self.queue.pop()
        self.assertEqual(item, "item2")
        
        # Steal (FIFO for thieves)
        item = self.queue.steal()
        self.assertEqual(item, "item1")
        
        self.assertTrue(self.queue.empty())
    
    def test_concurrent_access(self):
        """Test concurrent access to work-stealing queue"""
        items_pushed = []
        items_popped = []
        items_stolen = []
        
        def pusher():
            for i in range(100):
                item = f"item_{i}"
                self.queue.push(item)
                items_pushed.append(item)
                time.sleep(0.001)
        
        def popper():
            while len(items_popped) < 50:
                item = self.queue.pop()
                if item:
                    items_popped.append(item)
                time.sleep(0.001)
        
        def stealer():
            while len(items_stolen) < 50:
                item = self.queue.steal()
                if item:
                    items_stolen.append(item)
                time.sleep(0.001)
        
        # Start threads
        threads = [
            threading.Thread(target=pusher),
            threading.Thread(target=popper),
            threading.Thread(target=stealer)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all items were processed
        total_processed = len(items_popped) + len(items_stolen)
        self.assertGreaterEqual(total_processed, 90)  # Allow some items to remain

class TestPerformanceProfiler(unittest.TestCase):
    """Test performance profiler functionality"""
    
    def setUp(self):
        self.profiler = PerformanceProfiler()
    
    def test_timer_operations(self):
        """Test timer start/end operations"""
        # Test basic timing
        self.profiler.start_timer("test_function")
        time.sleep(0.1)
        duration = self.profiler.end_timer("test_function")
        
        self.assertGreater(duration, 0.09)
        self.assertLess(duration, 0.2)
    
    def test_function_decorator(self):
        """Test function profiling decorator"""
        @self.profiler.profile_function
        def test_function():
            time.sleep(0.05)
            return "result"
        
        result = test_function()
        self.assertEqual(result, "result")
        
        stats = self.profiler.get_stats()
        self.assertIn("test_function", stats)
        self.assertGreater(stats["test_function"]["total_time"], 0.04)
    
    def test_multiple_calls(self):
        """Test profiling multiple function calls"""
        @self.profiler.profile_function
        def fast_function():
            time.sleep(0.01)
        
        # Call function multiple times
        for _ in range(5):
            fast_function()
        
        stats = self.profiler.get_stats()
        self.assertEqual(stats["fast_function"]["call_count"], 5)
        self.assertGreater(stats["fast_function"]["total_time"], 0.04)
    
    def test_report_generation(self):
        """Test report generation"""
        @self.profiler.profile_function
        def sample_function():
            time.sleep(0.02)
        
        sample_function()
        
        report = self.profiler.generate_report()
        self.assertIn("Performance Profiling Report", report)
        self.assertIn("sample_function", report)
    
    def test_enable_disable(self):
        """Test enabling/disabling profiler"""
        self.profiler.enable(False)
        
        self.profiler.start_timer("disabled_test")
        time.sleep(0.05)
        duration = self.profiler.end_timer("disabled_test")
        
        # Should return 0 when disabled
        self.assertEqual(duration, 0.0)
        
        # Re-enable
        self.profiler.enable(True)
        
        self.profiler.start_timer("enabled_test")
        time.sleep(0.05)
        duration = self.profiler.end_timer("enabled_test")
        
        self.assertGreater(duration, 0.04)

class TestEnhancedPerformanceOptimizer(unittest.TestCase):
    """Test enhanced performance optimizer"""
    
    def setUp(self):
        self.optimizer = EnhancedPerformanceOptimizer()
    
    def tearDown(self):
        self.optimizer.cleanup()
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertIsInstance(self.optimizer.system_info, SystemInfo)
        self.assertGreater(self.optimizer.system_info.cpu_count, 0)
        self.assertGreater(self.optimizer.system_info.total_memory, 0)
    
    def test_configuration_changes(self):
        """Test configuration changes"""
        # Test parallelization strategy
        self.optimizer.set_parallelization_strategy(ParallelizationStrategy.THREADING)
        self.assertEqual(self.optimizer.parallelization_strategy, ParallelizationStrategy.THREADING)
        
        # Test thread count
        self.optimizer.set_num_threads(2)
        self.assertEqual(self.optimizer.num_threads, 2)
        
        # Test load balancing
        self.optimizer.set_load_balancing_type(LoadBalancingType.STATIC)
        self.assertEqual(self.optimizer.load_balancing_type, LoadBalancingType.STATIC)
        
        # Test memory optimization
        self.optimizer.set_memory_optimization(MemoryOptimizationType.MEMORY_POOL)
        self.assertEqual(self.optimizer.memory_optimization, MemoryOptimizationType.MEMORY_POOL)
    
    def test_parallel_map(self):
        """Test parallel map operation"""
        data = list(range(100))
        
        def square(x):
            return x * x
        
        # Test with small data (should use sequential)
        small_data = list(range(10))
        result = self.optimizer.parallel_map(square, small_data)
        expected = [x * x for x in small_data]
        self.assertEqual(result, expected)
        
        # Test with larger data (should use parallel)
        result = self.optimizer.parallel_map(square, data)
        expected = [x * x for x in data]
        self.assertEqual(sorted(result), sorted(expected))
    
    def test_parallel_reduce(self):
        """Test parallel reduce operation"""
        data = list(range(1, 101))  # 1 to 100
        
        def add(a, b):
            return a + b
        
        result = self.optimizer.parallel_reduce(data, add)
        expected = sum(data)
        self.assertEqual(result, expected)
        
        # Test with initial value
        result = self.optimizer.parallel_reduce(data, add, initial=1000)
        expected = 1000 + sum(data)
        self.assertEqual(result, expected)
    
    def test_performance_measurement(self):
        """Test performance measurement"""
        def test_operation():
            time.sleep(0.1)
            return sum(range(1000))
        
        metrics = self.optimizer.measure_performance(test_operation)
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.execution_time, 0.09)
        self.assertLess(metrics.execution_time, 0.2)
        self.assertGreaterEqual(metrics.cpu_utilization, 0.0)
        self.assertLessEqual(metrics.cpu_utilization, 1.0)
    
    def test_optimization_for_problem_size(self):
        """Test optimization for different problem sizes"""
        # Small problem
        self.optimizer.optimize_for_problem_size(100)
        self.assertEqual(self.optimizer.num_threads, 1)
        
        # Medium problem
        self.optimizer.optimize_for_problem_size(50000)
        self.assertLessEqual(self.optimizer.num_threads, 4)
        
        # Large problem
        self.optimizer.optimize_for_problem_size(500000)
        self.assertGreater(self.optimizer.num_threads, 1)
    
    def test_memory_constraint_optimization(self):
        """Test optimization for memory constraints"""
        # Low memory scenario
        low_memory = 100 * 1024 * 1024  # 100MB
        original_threads = self.optimizer.num_threads
        
        self.optimizer.optimize_for_memory_constraints(low_memory)
        
        # Should reduce threads or change memory optimization
        self.assertTrue(
            self.optimizer.num_threads <= original_threads or
            self.optimizer.memory_optimization == MemoryOptimizationType.COMPRESSION
        )
    
    def test_suggestion_generation(self):
        """Test optimization suggestion generation"""
        # Generate some performance history
        def dummy_operation():
            time.sleep(0.01)
        
        for _ in range(5):
            self.optimizer.measure_performance(dummy_operation)
        
        suggestions = self.optimizer.suggest_optimizations()
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
    
    def test_report_generation(self):
        """Test comprehensive report generation"""
        # Generate some performance data
        def test_operation():
            return sum(range(1000))
        
        self.optimizer.measure_performance(test_operation)
        
        report = self.optimizer.generate_report()
        self.assertIn("Enhanced Performance Optimization Report", report)
        self.assertIn("System Information", report)
        self.assertIn("Current Configuration", report)

class TestPerformanceContext(unittest.TestCase):
    """Test performance context manager"""
    
    def setUp(self):
        self.optimizer = EnhancedPerformanceOptimizer()
    
    def tearDown(self):
        self.optimizer.cleanup()
    
    def test_context_manager(self):
        """Test context manager functionality"""
        with PerformanceContext(self.optimizer, "test_context"):
            time.sleep(0.05)
        
        stats = self.optimizer.profiler.get_stats()
        self.assertIn("test_context", stats)
        self.assertGreater(stats["test_context"]["total_time"], 0.04)

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_example_tasks(self):
        """Test example CPU and memory intensive tasks"""
        # Test CPU intensive task
        result = example_cpu_intensive_task(1000)
        expected = sum(i * i for i in range(1000))
        self.assertEqual(result, expected)
        
        # Test memory intensive task
        result = example_memory_intensive_task(10)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 10))
    
    def test_numpy_optimization(self):
        """Test NumPy optimization function"""
        # Should not raise any exceptions
        optimize_numpy_operations()
    
    def test_profile_decorator_factory(self):
        """Test profile decorator factory"""
        profiler = PerformanceProfiler()
        decorator = profile_decorator(profiler)
        
        @decorator
        def test_function():
            time.sleep(0.02)
            return "decorated"
        
        result = test_function()
        self.assertEqual(result, "decorated")
        
        stats = profiler.get_stats()
        self.assertIn("test_function", stats)

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
