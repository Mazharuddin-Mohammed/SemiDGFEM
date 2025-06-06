#!/usr/bin/env python3
"""
CUDA/OpenCL Integration Examples
Demonstrates practical GPU acceleration for semiconductor device simulation

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def example_cuda_transport_solver():
    """Example: CUDA-accelerated transport solver"""
    print("🚀 CUDA TRANSPORT SOLVER EXAMPLE")
    print("=" * 50)
    
    try:
        from gpu_acceleration import GPUAcceleratedSolver, GPUBackend
        
        # Initialize CUDA solver
        solver = GPUAcceleratedSolver(GPUBackend.CUDA)
        
        print(f"GPU Backend: {solver.context.get_backend()}")
        print(f"GPU Available: {solver.is_gpu_available()}")
        
        if solver.is_gpu_available():
            device_info = solver.context.get_device_info()
            print(f"Device: {device_info}")
        
        # Create 2D device simulation
        nx, ny = 100, 50
        Lx, Ly = 2e-6, 1e-6
        
        # Generate mesh
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x, y)
        
        # Create potential distribution (simplified)
        potential = np.zeros((ny, nx))
        potential[:, -1] = 1.0  # Drain voltage
        
        # Doping profile
        doping_nd = np.full((ny, nx), 1e17)
        doping_na = np.full((ny, nx), 1e16)
        
        # Add channel region
        channel_start = ny // 3
        channel_end = 2 * ny // 3
        doping_nd[channel_start:channel_end, :] = 1e15
        doping_na[channel_start:channel_end, :] = 1e17
        
        print(f"Mesh: {nx}×{ny} = {nx*ny} points")
        print(f"Device: {Lx*1e6:.1f}μm × {Ly*1e6:.1f}μm")
        
        # Flatten for 1D solver
        potential_1d = potential.flatten()
        doping_nd_1d = doping_nd.flatten()
        doping_na_1d = doping_na.flatten()
        
        # Solve transport equations
        print("\n🔧 Solving transport equations...")
        start = time.time()
        
        results = solver.solve_transport_gpu(
            potential_1d, doping_nd_1d, doping_na_1d, temperature=300.0
        )
        
        solve_time = time.time() - start
        
        print(f"   Solution time: {solve_time:.4f}s")
        print(f"   Fields computed: {list(results.keys())}")
        
        # Reshape results back to 2D
        n_2d = results['n'].reshape((ny, nx))
        p_2d = results['p'].reshape((ny, nx))
        Jn_2d = results['Jn'].reshape((ny, nx))
        Jp_2d = results['Jp'].reshape((ny, nx))
        
        # Print statistics
        print(f"\n📊 Solution Statistics:")
        print(f"   Electron density: {np.mean(n_2d):.2e} ± {np.std(n_2d):.2e} /m³")
        print(f"   Hole density: {np.mean(p_2d):.2e} ± {np.std(p_2d):.2e} /m³")
        print(f"   Electron current: {np.mean(Jn_2d):.2e} ± {np.std(Jn_2d):.2e} A/m²")
        print(f"   Hole current: {np.mean(Jp_2d):.2e} ± {np.std(Jp_2d):.2e} A/m²")
        
        # Calculate total current
        total_current = np.sum(Jn_2d + Jp_2d) * (Lx/nx) * (Ly/ny)
        print(f"   Total current: {total_current:.2e} A")
        
        print("   ✅ CUDA transport solver example completed")
        return True
        
    except Exception as e:
        print(f"   ❌ CUDA example failed: {e}")
        return False

def example_opencl_matrix_operations():
    """Example: OpenCL matrix operations"""
    print("\n🌐 OPENCL MATRIX OPERATIONS EXAMPLE")
    print("=" * 50)
    
    try:
        from gpu_acceleration import GPUKernels, GPUContext, GPUBackend
        
        # Initialize OpenCL context
        context = GPUContext()
        success = context.initialize(GPUBackend.OPENCL)
        
        if not success:
            print("   ⚠️  OpenCL not available, using CPU simulation")
        
        kernels = GPUKernels(context)
        
        # Large matrix operations for finite element method
        print("🔧 Setting up finite element matrices...")
        
        # System size (degrees of freedom)
        n_dof = 1000
        
        # Generate stiffness matrix (sparse, but using dense for demo)
        print(f"   Generating {n_dof}×{n_dof} stiffness matrix...")
        K = np.random.random((n_dof, n_dof))
        K = 0.5 * (K + K.T)  # Make symmetric
        K += n_dof * np.eye(n_dof)  # Make positive definite
        
        # Generate mass matrix
        print(f"   Generating {n_dof}×{n_dof} mass matrix...")
        M = np.random.random((n_dof, n_dof))
        M = 0.5 * (M + M.T)  # Make symmetric
        M += 0.1 * np.eye(n_dof)  # Make positive definite
        
        # Right-hand side vector
        b = np.random.random(n_dof)
        
        # Test matrix-vector multiplication
        print("\n🚀 Testing matrix-vector operations...")
        
        start = time.time()
        if context.is_available():
            result_gpu = kernels.matrix_multiply(K[:100, :100], M[:100, :100])
        else:
            result_gpu = np.dot(K[:100, :100], M[:100, :100])
        gpu_time = time.time() - start
        
        start = time.time()
        result_cpu = np.dot(K[:100, :100], M[:100, :100])
        cpu_time = time.time() - start
        
        print(f"   Matrix multiply (100×100):")
        print(f"      GPU time: {gpu_time:.6f}s")
        print(f"      CPU time: {cpu_time:.6f}s")
        print(f"      Speedup: {cpu_time/gpu_time:.2f}x")
        print(f"      Accuracy: {np.allclose(result_gpu, result_cpu)}")
        
        # Test linear system solving
        print("\n🔧 Testing linear system solver...")
        
        A_small = K[:100, :100]
        b_small = b[:100]
        
        start = time.time()
        if context.is_available():
            x_gpu = kernels.solve_linear_system(A_small, b_small)
        else:
            x_gpu = np.linalg.solve(A_small, b_small)
        gpu_solve_time = time.time() - start
        
        start = time.time()
        x_cpu = np.linalg.solve(A_small, b_small)
        cpu_solve_time = time.time() - start
        
        print(f"   Linear solve (100×100):")
        print(f"      GPU time: {gpu_solve_time:.6f}s")
        print(f"      CPU time: {cpu_solve_time:.6f}s")
        print(f"      Speedup: {cpu_solve_time/gpu_solve_time:.2f}x")
        print(f"      Accuracy: {np.allclose(x_gpu, x_cpu)}")
        
        # Verify solution
        residual = np.linalg.norm(A_small @ x_gpu - b_small)
        print(f"      Residual: {residual:.2e}")
        
        print("   ✅ OpenCL matrix operations example completed")
        return True
        
    except Exception as e:
        print(f"   ❌ OpenCL example failed: {e}")
        return False

def example_hybrid_simd_gpu():
    """Example: Hybrid SIMD+GPU acceleration"""
    print("\n⚡ HYBRID SIMD+GPU ACCELERATION EXAMPLE")
    print("=" * 50)
    
    try:
        from performance_bindings import PerformanceOptimizer, PhysicsAcceleration
        from gpu_acceleration import GPUAcceleratedSolver
        
        # Initialize hybrid system
        optimizer = PerformanceOptimizer()
        physics = PhysicsAcceleration(optimizer)
        gpu_solver = GPUAcceleratedSolver()
        
        print("🔧 Hybrid acceleration system initialized")
        optimizer.print_performance_info()
        
        # Large-scale device simulation
        print("\n🚀 Large-scale device simulation...")
        
        # 3D-like simulation (flattened to 1D for simplicity)
        nx, ny, nz = 50, 50, 20
        total_points = nx * ny * nz
        
        print(f"   Simulation size: {nx}×{ny}×{nz} = {total_points} points")
        
        # Generate device structure
        potential = np.random.random(total_points) * 0.1
        doping_nd = np.full(total_points, 1e17)
        doping_na = np.full(total_points, 1e16)
        
        # Add device regions
        # Source region
        source_region = slice(0, total_points//4)
        doping_nd[source_region] = 1e19
        
        # Drain region  
        drain_region = slice(3*total_points//4, total_points)
        doping_nd[drain_region] = 1e19
        
        # Channel region
        channel_region = slice(total_points//4, 3*total_points//4)
        doping_nd[channel_region] = 1e15
        doping_na[channel_region] = 1e17
        
        print("   Device structure created")
        
        # Physics computation with hybrid acceleration
        print("\n⚡ Running hybrid physics computation...")
        
        start = time.time()
        
        # Carrier densities (SIMD accelerated)
        n, p = physics.compute_carrier_densities(potential, doping_nd, doping_na)
        carrier_time = time.time() - start
        
        # Current densities (GPU accelerated if available)
        start = time.time()
        Jn, Jp = physics.compute_current_densities(n, p, potential)
        current_time = time.time() - start
        
        # Recombination (SIMD accelerated)
        start = time.time()
        R = physics.compute_recombination(n, p)
        recomb_time = time.time() - start
        
        # Energy transport (hybrid)
        T_n = np.full(total_points, 300.0) + 50 * np.random.random(total_points)
        T_p = np.full(total_points, 300.0) + 30 * np.random.random(total_points)
        
        start = time.time()
        Wn, Wp = physics.compute_energy_densities(n, p, T_n, T_p)
        energy_time = time.time() - start
        
        total_physics_time = carrier_time + current_time + recomb_time + energy_time
        
        print(f"   Physics computation times:")
        print(f"      Carrier densities: {carrier_time:.4f}s")
        print(f"      Current densities: {current_time:.4f}s")
        print(f"      Recombination: {recomb_time:.4f}s")
        print(f"      Energy transport: {energy_time:.4f}s")
        print(f"      Total: {total_physics_time:.4f}s")
        
        # Performance analysis
        points_per_second = total_points / total_physics_time
        print(f"   Performance: {points_per_second:.0f} points/second")
        
        # Memory usage estimate
        memory_per_point = 8 * 10  # 10 double values per point
        total_memory = total_points * memory_per_point / (1024**2)  # MB
        print(f"   Memory usage: {total_memory:.1f} MB")
        
        # Results analysis
        print(f"\n📊 Physics Results:")
        print(f"   Electron density: {np.mean(n):.2e} ± {np.std(n):.2e} /m³")
        print(f"   Hole density: {np.mean(p):.2e} ± {np.std(p):.2e} /m³")
        print(f"   Recombination rate: {np.mean(R):.2e} ± {np.std(R):.2e} /m³/s")
        print(f"   Electron temperature: {np.mean(T_n):.1f} ± {np.std(T_n):.1f} K")
        print(f"   Hole temperature: {np.mean(T_p):.1f} ± {np.std(T_p):.1f} K")
        
        print("   ✅ Hybrid SIMD+GPU example completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Hybrid example failed: {e}")
        return False

def example_performance_comparison():
    """Example: Performance comparison across backends"""
    print("\n🏁 PERFORMANCE COMPARISON EXAMPLE")
    print("=" * 50)
    
    try:
        from performance_bindings import SIMDKernels, PerformanceOptimizer
        from gpu_acceleration import GPUAcceleratedSolver
        
        # Test different problem sizes
        sizes = [1000, 5000, 10000, 50000]
        
        print("🔧 Running performance comparison...")
        print(f"{'Size':<8} {'SIMD (ms)':<12} {'GPU (ms)':<12} {'NumPy (ms)':<12} {'Best':<8}")
        print("-" * 60)
        
        simd = SIMDKernels()
        gpu_solver = GPUAcceleratedSolver()
        
        for size in sizes:
            # Generate test data
            a = np.random.random(size)
            b = np.random.random(size)
            
            # SIMD timing
            start = time.time()
            for _ in range(10):
                result_simd = simd.vector_add(a, b)
            simd_time = (time.time() - start) * 100  # Convert to ms
            
            # GPU timing
            start = time.time()
            for _ in range(10):
                if gpu_solver.kernels:
                    result_gpu = gpu_solver.kernels.vector_add(a, b)
                else:
                    result_gpu = a + b
            gpu_time = (time.time() - start) * 100  # Convert to ms
            
            # NumPy timing
            start = time.time()
            for _ in range(10):
                result_numpy = a + b
            numpy_time = (time.time() - start) * 100  # Convert to ms
            
            # Determine best
            times = {'SIMD': simd_time, 'GPU': gpu_time, 'NumPy': numpy_time}
            best = min(times, key=times.get)
            
            print(f"{size:<8} {simd_time:<12.2f} {gpu_time:<12.2f} {numpy_time:<12.2f} {best:<8}")
        
        print("\n📊 Performance Summary:")
        print("   - SIMD: Optimized CPU vectorization")
        print("   - GPU: GPU acceleration (if available)")
        print("   - NumPy: Standard NumPy operations")
        
        print("   ✅ Performance comparison completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Performance comparison failed: {e}")
        return False

def main():
    """Main example runner"""
    print("🚀 CUDA/OPENCL INTEGRATION EXAMPLES")
    print("=" * 70)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    examples_passed = 0
    total_examples = 4
    
    # Run all examples
    if example_cuda_transport_solver():
        examples_passed += 1
    
    if example_opencl_matrix_operations():
        examples_passed += 1
    
    if example_hybrid_simd_gpu():
        examples_passed += 1
    
    if example_performance_comparison():
        examples_passed += 1
    
    # Print summary
    print(f"\n📊 EXAMPLES SUMMARY")
    print("=" * 30)
    print(f"Examples completed: {examples_passed}/{total_examples}")
    print(f"Success rate: {examples_passed/total_examples*100:.1f}%")
    
    if examples_passed == total_examples:
        print("\n🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("   CUDA/OpenCL integration is working")
        return 0
    else:
        print(f"\n⚠️  {total_examples - examples_passed} EXAMPLES FAILED")
        print("   Some GPU features may not be available")
        return 1

if __name__ == "__main__":
    sys.exit(main())
