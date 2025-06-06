#!/usr/bin/env python3
"""
Performance Profiler and Optimization Framework
Comprehensive benchmarking and optimization for semiconductor device simulation

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import time
import psutil
import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import queue

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    operation: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: float
    throughput: float
    efficiency: float
    problem_size: int
    backend: str
    timestamp: float

@dataclass
class SystemInfo:
    """System information for benchmarking"""
    cpu_model: str
    cpu_cores: int
    cpu_frequency: float
    memory_total: float
    gpu_model: str
    gpu_memory: float
    python_version: str
    numpy_version: str

class PerformanceProfiler:
    """Advanced performance profiler with real-time monitoring"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.system_info = self._get_system_info()
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_queue = queue.Queue()
        
    def _get_system_info(self) -> SystemInfo:
        """Get comprehensive system information"""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            cpu_model = cpu_info.get('brand_raw', 'Unknown CPU')
        except:
            cpu_model = 'Unknown CPU'
        
        try:
            # Try to get GPU info
            from gpu_acceleration import GPUContext
            gpu_context = GPUContext()
            if gpu_context.initialize():
                device_info = gpu_context.get_device_info()
                gpu_model = device_info.name
                gpu_memory = device_info.memory / (1024**3)  # GB
                gpu_context.finalize()
            else:
                gpu_model = 'No GPU'
                gpu_memory = 0.0
        except:
            gpu_model = 'No GPU'
            gpu_memory = 0.0
        
        return SystemInfo(
            cpu_model=cpu_model,
            cpu_cores=psutil.cpu_count(),
            cpu_frequency=psutil.cpu_freq().max if psutil.cpu_freq() else 0.0,
            memory_total=psutil.virtual_memory().total / (1024**3),  # GB
            gpu_model=gpu_model,
            gpu_memory=gpu_memory,
            python_version=sys.version.split()[0],
            numpy_version=np.__version__
        )
    
    @contextmanager
    def profile_operation(self, operation: str, problem_size: int = 0, backend: str = "CPU"):
        """Context manager for profiling operations"""
        
        # Start monitoring
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)  # MB
        start_cpu = psutil.cpu_percent()
        
        # Start GPU monitoring if available
        gpu_usage = 0.0
        try:
            # Placeholder for GPU monitoring
            # In real implementation, would use nvidia-ml-py or similar
            pass
        except:
            pass
        
        try:
            yield
        finally:
            # End monitoring
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**2)  # MB
            end_cpu = psutil.cpu_percent()
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = (start_cpu + end_cpu) / 2
            
            # Calculate throughput and efficiency
            throughput = problem_size / execution_time if execution_time > 0 else 0
            efficiency = throughput / (cpu_usage / 100) if cpu_usage > 0 else 0
            
            # Store metrics
            metrics = PerformanceMetrics(
                operation=operation,
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                throughput=throughput,
                efficiency=efficiency,
                problem_size=problem_size,
                backend=backend,
                timestamp=end_time
            )
            
            self.metrics_history.append(metrics)
    
    def start_continuous_monitoring(self, interval: float = 0.1):
        """Start continuous system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_system, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_continuous_monitoring(self):
        """Stop continuous system monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_system(self, interval: float):
        """Continuous system monitoring thread"""
        while self.monitoring_active:
            try:
                timestamp = time.time()
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                
                self.monitor_queue.put({
                    'timestamp': timestamp,
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage
                })
                
                time.sleep(interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def get_metrics_summary(self, operation: str = None) -> Dict[str, Any]:
        """Get summary statistics for metrics"""
        if operation:
            metrics = [m for m in self.metrics_history if m.operation == operation]
        else:
            metrics = self.metrics_history
        
        if not metrics:
            return {}
        
        execution_times = [m.execution_time for m in metrics]
        memory_usages = [m.memory_usage for m in metrics]
        throughputs = [m.throughput for m in metrics]
        
        return {
            'count': len(metrics),
            'execution_time': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times),
                'median': np.median(execution_times)
            },
            'memory_usage': {
                'mean': np.mean(memory_usages),
                'std': np.std(memory_usages),
                'min': np.min(memory_usages),
                'max': np.max(memory_usages)
            },
            'throughput': {
                'mean': np.mean(throughputs),
                'std': np.std(throughputs),
                'min': np.min(throughputs),
                'max': np.max(throughputs)
            }
        }
    
    def export_metrics(self, filename: str):
        """Export metrics to JSON file"""
        data = {
            'system_info': asdict(self.system_info),
            'metrics': [asdict(m) for m in self.metrics_history]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_metrics(self, filename: str):
        """Import metrics from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.metrics_history = [
            PerformanceMetrics(**m) for m in data['metrics']
        ]
    
    def clear_metrics(self):
        """Clear all stored metrics"""
        self.metrics_history.clear()
    
    def plot_performance_trends(self, operation: str = None, save_path: str = None):
        """Plot performance trends over time"""
        if operation:
            metrics = [m for m in self.metrics_history if m.operation == operation]
        else:
            metrics = self.metrics_history
        
        if not metrics:
            print("No metrics available for plotting")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        timestamps = [m.timestamp for m in metrics]
        execution_times = [m.execution_time for m in metrics]
        memory_usages = [m.memory_usage for m in metrics]
        throughputs = [m.throughput for m in metrics]
        cpu_usages = [m.cpu_usage for m in metrics]
        
        # Execution time trend
        ax1.plot(timestamps, execution_times, 'b-o', markersize=3)
        ax1.set_title('Execution Time Trend')
        ax1.set_ylabel('Time (s)')
        ax1.grid(True)
        
        # Memory usage trend
        ax2.plot(timestamps, memory_usages, 'r-o', markersize=3)
        ax2.set_title('Memory Usage Trend')
        ax2.set_ylabel('Memory (MB)')
        ax2.grid(True)
        
        # Throughput trend
        ax3.plot(timestamps, throughputs, 'g-o', markersize=3)
        ax3.set_title('Throughput Trend')
        ax3.set_ylabel('Operations/sec')
        ax3.grid(True)
        
        # CPU usage trend
        ax4.plot(timestamps, cpu_usages, 'm-o', markersize=3)
        ax4.set_title('CPU Usage Trend')
        ax4.set_ylabel('CPU %')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("🏁 PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 50)
        
        # System information
        report.append(f"\n📊 SYSTEM INFORMATION:")
        report.append(f"   CPU: {self.system_info.cpu_model}")
        report.append(f"   Cores: {self.system_info.cpu_cores}")
        report.append(f"   Frequency: {self.system_info.cpu_frequency:.1f} MHz")
        report.append(f"   Memory: {self.system_info.memory_total:.1f} GB")
        report.append(f"   GPU: {self.system_info.gpu_model}")
        report.append(f"   GPU Memory: {self.system_info.gpu_memory:.1f} GB")
        
        # Overall statistics
        if self.metrics_history:
            report.append(f"\n📈 OVERALL STATISTICS:")
            report.append(f"   Total Operations: {len(self.metrics_history)}")
            
            all_times = [m.execution_time for m in self.metrics_history]
            all_throughputs = [m.throughput for m in self.metrics_history]
            
            report.append(f"   Average Execution Time: {np.mean(all_times):.6f}s")
            report.append(f"   Average Throughput: {np.mean(all_throughputs):.0f} ops/sec")
            
            # Per-operation statistics
            operations = set(m.operation for m in self.metrics_history)
            for op in sorted(operations):
                summary = self.get_metrics_summary(op)
                if summary:
                    report.append(f"\n   {op}:")
                    report.append(f"      Count: {summary['count']}")
                    report.append(f"      Avg Time: {summary['execution_time']['mean']:.6f}s")
                    report.append(f"      Avg Throughput: {summary['throughput']['mean']:.0f} ops/sec")
        
        return "\n".join(report)
    
    def print_system_info(self):
        """Print detailed system information"""
        print("🖥️  SYSTEM INFORMATION")
        print("=" * 30)
        print(f"CPU: {self.system_info.cpu_model}")
        print(f"Cores: {self.system_info.cpu_cores}")
        print(f"Frequency: {self.system_info.cpu_frequency:.1f} MHz")
        print(f"Memory: {self.system_info.memory_total:.1f} GB")
        print(f"GPU: {self.system_info.gpu_model}")
        print(f"GPU Memory: {self.system_info.gpu_memory:.1f} GB")
        print(f"Python: {self.system_info.python_version}")
        print(f"NumPy: {self.system_info.numpy_version}")

# Global profiler instance
profiler = PerformanceProfiler()
