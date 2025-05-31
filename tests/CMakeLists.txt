cmake_minimum_required(VERSION 3.10)
project(SemiconductorSimulatorSimpleTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(MPI REQUIRED)
pkg_check_modules(PETSC REQUIRED petsc)
find_package(Boost REQUIRED)
find_library(GMSH_LIBRARY gmsh)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(
    ${PETSC_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS} 
    ${MPI_CXX_INCLUDE_DIRS}
    ../include
)

# Link directories
link_directories(${PETSC_LIBRARY_DIRS})

# Create test executable
add_executable(simple_backend_test
    simple_backend_test.cpp
)

# Link with the simulator library
target_link_libraries(simple_backend_test
    simulator
    ${PETSC_LIBRARIES}
    ${Boost_LIBRARIES}
    ${GMSH_LIBRARY}
    ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Set library path
target_link_directories(simple_backend_test PRIVATE ../build)

# Compiler flags
target_compile_options(simple_backend_test PRIVATE
    -Wall -Wextra
    -Wno-unused-parameter
    -O2 -g
)

# Add test
enable_testing()
add_test(NAME SimpleBackendTest COMMAND simple_backend_test)
