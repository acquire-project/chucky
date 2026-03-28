# add_experiment(NAME source.c [LIBS ...])
#   Creates an experiment executable with warnings and nix rpath.
#   NOT registered with ctest.
function(add_experiment TGT SOURCE_FILE)
    add_executable(${TGT} ${SOURCE_FILE})
    target_link_libraries(${TGT} PRIVATE warnings ${ARGN})
    set_nix_rpath(${TGT})
endfunction()

# GPU-only wrapper: no-op when CHUCKY_ENABLE_GPU is OFF.
function(add_gpu_experiment TGT)
    if(CHUCKY_ENABLE_GPU)
        add_experiment(${TGT} ${ARGN})
    endif()
endfunction()

add_gpu_experiment(index.ops index.ops.c CUDA::cuda_driver index_ops_util stream)
if(CHUCKY_ENABLE_GPU)
    enable_openmp(index.ops)
endif()

add_gpu_experiment(index.ops.cuda index.ops.cuda.c CUDA::cuda_driver CUDA::cudart_static transpose index_ops_util chucky_log)

add_gpu_experiment(test_morton morton.c lod_plan lod index_ops chucky_log)
if(CHUCKY_ENABLE_GPU)
    target_include_directories(test_morton PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()
