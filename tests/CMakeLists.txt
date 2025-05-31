cmake_minimum_required(VERSION 3.10)
project(SemiconductorSimulatorTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED petsc)
find_package(Boost REQUIRED COMPONENTS numeric_ublas)
find_library(GMSH_LIBRARY gmsh)
find_package(Vulkan REQUIRED)

# Find Google Test
find_package(GTest REQUIRED)
find_package(GMock REQUIRED)

# Include directories
include_directories(
    ${PETSC_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS} 
    ${Vulkan_INCLUDE_DIRS}
    ../include
    ${GTEST_INCLUDE_DIRS}
    ${GMOCK_INCLUDE_DIRS}
)

# Link directories
link_directories(${PETSC_LIBRARY_DIRS})

# Create test executable
add_executable(run_tests
    test_device.cpp
    test_mesh.cpp
    test_poisson.cpp
    test_driftdiffusion.cpp
    test_integration.cpp
    ../src/device.cpp
    ../src/mesh.cpp
    ../src/structured/poisson_struct_2d.cpp
    ../src/structured/driftdiffusion_struct_2d.cpp
    ../src/unstructured/poisson_unstruct_2d.cpp
    ../src/unstructured/driftdiffusion_unstruct_2d.cpp
)

# Link libraries
target_link_libraries(run_tests
    ${PETSC_LIBRARIES}
    ${Boost_LIBRARIES}
    ${GMSH_LIBRARY}
    ${Vulkan_LIBRARIES}
    ${GTEST_LIBRARIES}
    ${GMOCK_LIBRARIES}
    pthread
)

# Compiler flags
target_compile_options(run_tests PRIVATE
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -O2 -g
)

# Add test definitions
add_definitions(-DTEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# Enable testing
enable_testing()

# Add individual tests
add_test(NAME DeviceTests COMMAND run_tests --gtest_filter=DeviceTest.*)
add_test(NAME MeshTests COMMAND run_tests --gtest_filter=MeshTest.*)
add_test(NAME PoissonTests COMMAND run_tests --gtest_filter=PoissonTest.*)
add_test(NAME DriftDiffusionTests COMMAND run_tests --gtest_filter=DriftDiffusionTest.*)
add_test(NAME IntegrationTests COMMAND run_tests --gtest_filter=IntegrationTest.*)

# Add custom target for running tests
add_custom_target(test_all
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS run_tests
)

# Add memory check target (if valgrind is available)
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_custom_target(test_memcheck
        COMMAND ${VALGRIND_EXECUTABLE} 
            --tool=memcheck 
            --leak-check=full 
            --show-leak-kinds=all 
            --track-origins=yes 
            --verbose 
            --error-exitcode=1
            $<TARGET_FILE:run_tests>
        DEPENDS run_tests
    )
endif()

# Add coverage target (if gcov is available)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
if(ENABLE_COVERAGE)
    target_compile_options(run_tests PRIVATE --coverage)
    target_link_libraries(run_tests --coverage)
    
    find_program(GCOV_EXECUTABLE gcov)
    find_program(LCOV_EXECUTABLE lcov)
    find_program(GENHTML_EXECUTABLE genhtml)
    
    if(GCOV_EXECUTABLE AND LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
        add_custom_target(coverage
            COMMAND ${CMAKE_CTEST_COMMAND}
            COMMAND ${LCOV_EXECUTABLE} --directory . --capture --output-file coverage.info
            COMMAND ${LCOV_EXECUTABLE} --remove coverage.info '/usr/*' --output-file coverage.info
            COMMAND ${LCOV_EXECUTABLE} --list coverage.info
            COMMAND ${GENHTML_EXECUTABLE} coverage.info --output-directory coverage_html
            DEPENDS run_tests
        )
    endif()
endif()

# Add benchmark target
add_executable(benchmark
    benchmark_main.cpp
    ../src/device.cpp
    ../src/mesh.cpp
    ../src/structured/poisson_struct_2d.cpp
    ../src/structured/driftdiffusion_struct_2d.cpp
    ../src/unstructured/poisson_unstruct_2d.cpp
    ../src/unstructured/driftdiffusion_unstruct_2d.cpp
)

target_link_libraries(benchmark
    ${PETSC_LIBRARIES}
    ${Boost_LIBRARIES}
    ${GMSH_LIBRARY}
    ${Vulkan_LIBRARIES}
)

# Install test executable
install(TARGETS run_tests DESTINATION bin)
install(TARGETS benchmark DESTINATION bin)
