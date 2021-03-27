cmake_minimum_required(VERSION 3.10)
project(MM LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)


## libraries
include_directories(tool images include results_CUDA)
set(TOOL_FILES tool/CImg.h
    tool/Image.h tool/Image.cpp
    tool/utils.cuh
)

set(SRC_FILES include/Probe.h include/Probe.cpp
    include/MathematicalMorphology.h
    include/MathematicalMorphology.cpp
    include/MM.cuh include/MM.cu
)

add_library(libTool STATIC ${TOOL_FILES})
add_library(libSrc STATIC ${SRC_FILES})


## C++
set(CMAKE_CXX_COMPILER)
find_package(X11 REQUIRED)
find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
add_executable(mm_sequential sequential_CPP/main.cpp)
target_link_libraries(mm_sequential libTool libSrc
    Threads::Threads X11 -lpng -lstdc++fs -std=c++17
)


## CUDA
find_package(CUDA QUIET REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# naive version
add_executable(mm_naive CUDA/naive.cu)
target_link_libraries(mm_naive libTool libSrc -lpng -lstdc++fs)

# # sharedOpt version
# add_executable(mm_sharedOpt CUDA/sharedOpt.cu)
# target_link_libraries(mm_sharedOpt libTool libSrc -lpng -lstdc++fs)
