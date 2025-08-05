cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(stage18 LANGUAGES CUDA CXX)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add CUDA executable
add_executable(stage18 main.cu)

# CUDA compiler flags (optional but recommended)
set_target_properties(stage18 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)

# Link OpenCV
target_link_libraries(stage18 ${OpenCV_LIBS})
