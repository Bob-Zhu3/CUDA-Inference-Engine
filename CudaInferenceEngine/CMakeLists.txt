cmake_minimum_required(VERSION 3.18)
project(CudaInferenceEngine LANGUAGES CXX CUDA)

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
target_link_libraries(inference_lib PUBLIC CUDA::cublas)
set_target_properties(inference_lib PROPERTIES CXX_STANDARD 17)
set_property(TARGET inference_lib PROPERTY POSITION_INDEPENDENT_CODE ON)



pybind11_add_module(cuda_lstm_ops
    src/bindings.cpp
)
target_link_libraries(cuda_lstm_ops PRIVATE inference_lib)
set_target_properties(cuda_lstm_ops PROPERTIES CXX_STANDARD 17)