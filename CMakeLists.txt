cmake_minimum_required(VERSION 3.10)
project(imageproc VERSION 0.0.1 LANGUAGES CXX CUDA)

# add gdb symbol table (-g)
set(CMAKE_BUILD_TYPE Debug)
# add gdb symbol table optimized
# set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_definitions(-DLITTLEBUG)
add_definitions(-DLITTLEBENCH)

file(GLOB_RECURSE srcFiles src/*)
add_executable(main ${srcFiles})

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(main ${OpenCV_LIBS})