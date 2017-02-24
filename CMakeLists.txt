cmake_minimum_required(VERSION 3.2) 

project (B00243868 VERSION 1.0.0)

if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
	message(FATAL_ERROR "In-tree builds are not supported; please run cmake in a separate build directory.")
endif ()

add_subdirectory ("${CMAKE_SOURCE_DIR}/cpu")
add_subdirectory ("${CMAKE_SOURCE_DIR}/gpu")
