# Gpu.cmake — GPU (CUDA) auto-detection and setup
#
# Creates: CHUCKY_ENABLE_GPU option (auto-detected, overridable)
# When ON: enables CUDA language, sets standards/architectures, finds CUDAToolkit + Nvcomp

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(_GPU_DEFAULT ON)
else()
    set(_GPU_DEFAULT OFF)
endif()
option(CHUCKY_ENABLE_GPU "Build GPU (CUDA) backends and tests" ${_GPU_DEFAULT})

if(CHUCKY_ENABLE_GPU)
    # Before enable_language(CUDA), which caches the compiler's own default and
    # leaves a later set() with nothing to do. Each entry also emits PTX, so a
    # newer card than any listed here still runs, after a wait at load.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89 100
            CACHE STRING "CUDA architectures to generate code for")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit 12.8 REQUIRED)
    find_package(Nvcomp REQUIRED)
endif()
