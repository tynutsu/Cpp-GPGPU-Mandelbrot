cmake_minimum_required(VERSION 2.8.4) 
set(CMAKE_CXX_STANDARD 11)
project(cpu)

# find_package(CUDA REQUIRED) # This is needed, to then use cuda_add_executable
# set_source_files_properties(mandelbrot.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
# cuda_add_executable(mandelbrot mandelbrot.cu)

add_executable(cpu cpu.cpp)
