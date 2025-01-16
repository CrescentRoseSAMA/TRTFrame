cmake_minimum_required(VERSION 3.6)

project(TRT_Frame)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

find_package(Eigen3 REQUIRED)

find_package(TensorRT REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${Eigen_INCLUDE_DIRS})

include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${TensorRT_INCLUDE_DIRS})

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(TRTFrame main.cpp TRTFrame.cpp ${TensorRT_SOURCE})

target_link_libraries(TRTFrame ${TensorRT_LIBS} ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvonnxparser cuda pthread) 

