# add_experiment(NAME source.c [LIBS ...])
#   Creates an experiment executable with warnings and nix rpath.
#   NOT registered with ctest.
function(add_experiment TGT SOURCE_FILE)
    add_executable(${TGT} ${SOURCE_FILE})
    target_link_libraries(${TGT} PRIVATE warnings ${ARGN})
    set_nix_rpath(${TGT})
endfunction()

add_experiment(index.ops index.ops.c CUDA::cuda_driver index_ops_util stream)
enable_openmp(index.ops)

add_experiment(index.ops.cuda index.ops.cuda.c CUDA::cuda_driver CUDA::cudart_static transpose index_ops_util chucky_log)

add_experiment(test_morton morton.c lod_plan index_ops chucky_log)
target_include_directories(test_morton PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
