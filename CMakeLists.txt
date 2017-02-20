cmake_minimum_required(VERSION 2.8.4) 
set(CMAKE_CXX_STANDARD 11)
project(cpu)

set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG};-Xptxas -v")
find_package(CUDA REQUIRED) # This is needed, to then use cuda_add_executable
set_source_files_properties(cpu.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
cuda_add_executable(gpu gpu.cu)

add_executable(cpu cpu.cpp)
