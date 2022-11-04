cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(from-pytorch-to-libtorch)


find_package(Torch REQUIRED PATHS "../libtorch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



add_executable(run-traced-model load_traced_LeNet.cpp)
target_link_libraries(run-traced-model "${TORCH_LIBRARIES}")
set_property(TARGET run-traced-model PROPERTY CXX_STANDARD 14)
