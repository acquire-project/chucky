cmake_minimum_required(VERSION 3.18)
project(chucky LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(Msvc)
include(Warnings)
include(Nix)
include(OpenMP)
include(Coverage)
include(Gpu)
include(Lz4)
include(Zstd)

find_package(Threads REQUIRED)
find_package(aws-c-s3 REQUIRED CONFIG)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)

if(CHUCKY_ENABLE_GPU)
    add_subdirectory(bench)
endif()
