cmake_minimum_required(VERSION 3.10)
project(MM LANGUAGES CXX CUDA)

# src libraries
include_directories(tool images)
set(SOURCE_FILE tool/CImg.h)
add_library(lib STATIC ${SOURCE_FILE})

# C++
set(CMAKE_CXX_COMPILER)
find_package(X11 REQUIRED)
find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
add_executable(mm_sequential sequential_CPP/main.cpp)
target_link_libraries(mm_sequential lib X11 -lpng)

# CUDA
find_package(CUDA QUIET REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
add_executable(mm_parallel CUDA/main.cu)
target_link_libraries(mm_parallel lib)
