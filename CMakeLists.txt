cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8)

project(IPM)

# Specify desired GCC version
set(GCC_VERSION "gcc-8")
set(GXX_VERSION "g++-8")

# Find specified GCC version
find_program(GCC_PATH ${GCC_VERSION})
find_program(GXX_PATH ${GXX_VERSION})

if(NOT GCC_PATH)
    message(FATAL_ERROR "GCC version not found: ${GCC_VERSION}")
endif()

if(NOT GXX_PATH)
    message(FATAL_ERROR "G++ version not found: ${GXX_VERSION}")
endif()

# Set C and C++ compilers
set(CMAKE_C_COMPILER ${GCC_PATH})
set(CMAKE_CXX_COMPILER ${GXX_PATH})

# CUDA setup
enable_language(CUDA)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")

# Eigen setup
find_package(Eigen3 REQUIRED NO_MODULE)

# OpenCV setup
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#link_directories(/usr/lib/aarch64-linux-gnu/tegra)

# Executable
cuda_add_executable(IPM main.cpp ipm.cu calc_arrays.cpp util.cpp)

# Linking libraries
target_link_libraries(IPM ${OpenCV_LIBS} Eigen3::Eigen)
