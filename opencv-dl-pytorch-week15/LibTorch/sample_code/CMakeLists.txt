cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sample-code)

find_package(Torch REQUIRED PATHS "../libtorch" )

add_executable(sample sample.cpp)
target_link_libraries(sample "${TORCH_LIBRARIES}")
set_property(TARGET sample PROPERTY CXX_STANDARD 14)
