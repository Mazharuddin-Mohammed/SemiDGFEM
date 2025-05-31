cmake_minimum_required(VERSION 3.10)
project(SemiconductorSimulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED petsc)
find_package(Boost REQUIRED COMPONENTS numeric_ublas)
find_library(GMSH_LIBRARY gmsh)

include_directories(${PETSC_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} include)
link_directories(${PETSC_LIBRARY_DIRS})

add_library(simulator SHARED
    src/device.cpp
    src/mesh.cpp
    src/2d/dg/structured/poisson_dg_struct_2d.cpp
    src/2d/dg/structured/driftdiffusion_dg_struct_2d.cpp
    src/2d/dg/unstructured/poisson_dg_unstruct_2d.cpp
    src/2d/dg/unstructured/driftdiffusion_dg_unstruct_2d.cpp
)

target_link_libraries(simulator ${PETSC_LIBRARIES} ${Boost_LIBRARIES} ${GMSH_LIBRARY})

install(TARGETS simulator DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)