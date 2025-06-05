# SemiDGFEM - Advanced Semiconductor Device Simulator
# Author: Dr. Mazharuddin Mohammed

cmake_minimum_required(VERSION 3.10)
project(SemiconductorSimulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Performance optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -DDEBUG")

# Enable SIMD and vectorization
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
endif()

find_package(PkgConfig REQUIRED)
find_package(MPI REQUIRED)
pkg_check_modules(PETSC REQUIRED petsc)
find_package(Boost REQUIRED)
find_library(GMSH_LIBRARY gmsh)
find_package(Vulkan QUIET)
find_package(OpenMP REQUIRED)

# Optional GPU support
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_OPENCL "Enable OpenCL support" OFF)

if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DENABLE_CUDA)
endif()

if(ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)
    add_definitions(-DENABLE_OPENCL)
endif()

include_directories(${PETSC_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS} include)

# Add Vulkan includes if available
if(Vulkan_FOUND)
    include_directories(${Vulkan_INCLUDE_DIRS})
endif()
link_directories(${PETSC_LIBRARY_DIRS})

# Collect source files
set(SIMULATOR_SOURCES
    src/device.cpp
    src/mesh.cpp
    src/structured/poisson_struct_2d.cpp
    src/structured/driftdiffusion_struct_2d.cpp
    src/unstructured/poisson_unstruct_2d.cpp
    src/unstructured/driftdiffusion_unstruct_2d.cpp
    src/dg_math/dg_basis_functions.cpp
    src/dg_math/dg_assembly.cpp
    src/amr/amr_algorithms.cpp
    src/performance/simd_kernels.cpp
    src/performance/parallel_computing.cpp
    src/performance/advanced_profiler.cpp
    src/gpu/gpu_context.cpp
    src/physics/advanced_physics.cpp
    src/physics/advanced_transport.cpp
)

# Add CUDA sources if enabled
if(ENABLE_CUDA)
    list(APPEND SIMULATOR_SOURCES src/gpu/cuda_kernels.cu)
endif()

add_library(simulator SHARED ${SIMULATOR_SOURCES})

target_link_libraries(simulator
    ${PETSC_LIBRARIES}
    ${Boost_LIBRARIES}
    ${GMSH_LIBRARY}
    ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Add Vulkan if available
if(Vulkan_FOUND)
    target_link_libraries(simulator ${Vulkan_LIBRARIES})
    target_compile_definitions(simulator PRIVATE VULKAN_AVAILABLE)
endif()

# Add GPU libraries if enabled
if(ENABLE_CUDA)
    target_link_libraries(simulator
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::cusolver
    )
endif()

if(ENABLE_OPENCL)
    target_link_libraries(simulator OpenCL::OpenCL)
endif()

# Set target properties for optimization
set_target_properties(simulator PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS simulator DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)