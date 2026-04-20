# OpenMP.cmake — add OpenMP support to a target
#
# enable_openmp(target ...)
#   Adds OpenMP compile flags and link libraries to each target.

if(MSVC)
    set(_OPENMP_FLAGS /openmp:llvm)
else()
    find_package(OpenMP REQUIRED COMPONENTS C)
endif()

function(enable_openmp)
    foreach(tgt IN LISTS ARGN)
        if(MSVC)
            target_compile_options(${tgt} PRIVATE ${_OPENMP_FLAGS})
        else()
            target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_C)
        endif()
    endforeach()
endfunction()
