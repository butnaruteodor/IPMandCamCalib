cmake_minimum_required(VERSION 2.8)
project(IPM)

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
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

# Executable
cuda_add_executable(IPM main.cpp ipm.cu calc_arrays.cpp)

# Linking libraries
target_link_libraries(IPM ${OpenCV_LIBS} Eigen3::Eigen)
