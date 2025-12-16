cmake_minimum_required(VERSION 3.15)
project(MRPT_JacobiansPoseComposition_tests)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# eigen-5.0.0
add_library(Eigen INTERFACE) # Header only

target_include_directories(Eigen INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen-5.0.0
)

# MRPT_JacobiansPoseComposition_tests
add_executable(${PROJECT_NAME}
    main.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    Eigen
)
