cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch-basic-introduction)

find_package(Torch REQUIRED PATHS "../libtorch")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(libtorch-basics Libtorch_Introduction.cpp)
target_link_libraries(libtorch-basics "${TORCH_LIBRARIES}")
set_property(TARGET libtorch-basics PROPERTY CXX_STANDARD 14)
