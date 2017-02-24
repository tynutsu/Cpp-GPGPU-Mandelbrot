project (gpu VERSION 1.0.0)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xptxas -v -fmad=false")

find_package(CUDA REQUIRED)

cuda_include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/include")

file (GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file (GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

source_group ("Include\\" FILES ${HEADER_FILES})
source_group ("Source\\" FILES ${SOURCE_FILES})

cuda_add_executable (${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories (${PROJECT_NAME} PUBLIC  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)