cmake_minimum_required(VERSION 3.4)

project(reFFT VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -Wall -Wextra -Weffc++ -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION=1")


if(NOT TARGET gsl)
   add_subdirectory(external/GSL)
endif()

add_library(re_fft INTERFACE)
target_include_directories(re_fft INTERFACE include)
target_link_libraries(re_fft INTERFACE gsl)

add_subdirectory(tests)
