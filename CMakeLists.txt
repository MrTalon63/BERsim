cmake_minimum_required(VERSION 3.10)
project(bersim_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops -flto=auto -ffast-math")

find_package(Threads REQUIRED)

find_package(AFF3CT REQUIRED)

add_executable(simulator simulator.cpp)

target_link_libraries(simulator PRIVATE aff3ct::aff3ct-shared-lib)