cmake_minimum_required(VERSION 3.18)
project(CudaInferenceEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 CACHE STRING "CUDA architectures")

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

add_library(inference_lib STATIC
    src/inference.cu
    src/cpu_reference.cpp
)
target_include_directories(inference_lib PUBLIC src)
target_link_libraries(inference_lib PUBLIC CUDA::cublas CUDA::cudart)
set_target_properties(inference_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# For Nsight profiling
target_compile_options(inference_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

pybind11_add_module(cuda_lstm_ops
    src/bindings.cpp
)
target_link_libraries(cuda_lstm_ops PRIVATE inference_lib)
