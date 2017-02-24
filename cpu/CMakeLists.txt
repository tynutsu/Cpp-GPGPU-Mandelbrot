project (cpu VERSION 1.0.0)

file (GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
file (GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

source_group ("Include\\" FILES ${HEADER_FILES})
source_group ("Source\\" FILES ${SOURCE_FILES})

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories (${PROJECT_NAME} PUBLIC  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)