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
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 100)
    find_package(CUDAToolkit 12.8 REQUIRED)
    find_package(Nvcomp REQUIRED)
endif()
