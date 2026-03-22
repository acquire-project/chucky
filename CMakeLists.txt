cmake_minimum_required(VERSION 3.18)
project(chucky LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 100)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(Msvc)
include(Warnings)
include(Nix)
include(OpenMP)

find_package(CUDAToolkit 12.8 REQUIRED)
find_package(Nvcomp REQUIRED)
find_package(Threads REQUIRED)
find_package(lz4 CONFIG QUIET)
if(lz4_FOUND)
    add_library(Lz4::Lz4 ALIAS lz4::lz4)
else()
    find_package(Lz4 REQUIRED)
endif()

find_package(zstd CONFIG QUIET)
if(zstd_FOUND)
    add_library(Zstd::Zstd ALIAS zstd::libzstd)
else()
    find_package(Zstd REQUIRED)
endif()
find_package(aws-c-s3 REQUIRED CONFIG)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
add_subdirectory(bench)
