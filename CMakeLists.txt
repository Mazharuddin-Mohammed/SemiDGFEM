cmake_minimum_required(VERSION 3.10)
project(SemiconductorSimulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED petsc)
find_package(Boost REQUIRED COMPONENTS numeric_ublas)
find_library(GMSH_LIBRARY gmsh)
find_package(Vulkan REQUIRED)

include_directories(${PETSC_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} include ${Vulkan_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

add_library(simulator SHARED
    src/device.cpp
    src/mesh.cpp
    src/structured/poisson_struct_2d.cpp
    src/structured/driftdiffusion_struct_2d.cpp
    src/unstructured/poisson_unstruct_2d.cpp
    src/unstructured/driftdiffusion_unstruct_2d.cpp
)

target_link_libraries(simulator ${PETSC_LIBRARIES} ${Boost_LIBRARIES} ${GMSH_LIBRARY} ${Vulkan_LIBRARIES})

install(TARGETS simulator DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)