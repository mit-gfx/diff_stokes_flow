cmake_minimum_required(VERSION 3.10)

project(diff_stokes_flow)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions("-Wall")
set(GCC_COVERAGE_COMPILE_FLAGS "-Wl,--no-as-needed")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

# OpenMP.
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    add_definitions(-DEIGEN_DONT_PARALLELIZE)
endif()

# Include directories.
include_directories("../external/eigen")
include_directories("./core/include")
execute_process(COMMAND bash -c "python3-config --includes|sed 's/-I//g'|awk '{print $1;}'" OUTPUT_VARIABLE PYTHON_INCLUDES)
include_directories(${PYTHON_INCLUDES})

# The core library.
file(GLOB_RECURSE CORE_CPP "./core/src/**/*.cpp")
add_library(diff_stokes_flow_core SHARED ${CORE_CPP})
target_link_libraries(diff_stokes_flow_core OpenMP::OpenMP_CXX)

# Python wrapper library.
file(GLOB_RECURSE PYTHON_BINDING "./core/src/**/*.cpp" "./core/src/py_diff_stokes_flow_core_wrap.cxx")
add_library(py_diff_stokes_flow_core SHARED ${CORE_HEADER} ${PYTHON_BINDING})
target_link_libraries(py_diff_stokes_flow_core OpenMP::OpenMP_CXX)

# Demo.
file(GLOB DEMO_CPP "./main.cpp")
add_executable(diff_stokes_flow_demo ${DEMO_CPP})
target_link_libraries(diff_stokes_flow_demo diff_stokes_flow_core)