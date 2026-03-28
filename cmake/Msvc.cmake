# Msvc.cmake — MSVC-specific workarounds
#
# - Enable C11 atomics (experimental)
# - Use /MD runtime to match nvcomp_static and zstd_static
# - Suppress harmless LNK4098 from cudart_static's LIBCMT reference
# - Disable CRT secure warnings project-wide

if(NOT MSVC)
    return()
endif()

add_compile_options($<$<COMPILE_LANGUAGE:C>:/experimental:c11atomics>)
add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>)
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=/Zc:preprocessor>
)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
add_link_options(/NODEFAULTLIB:libcmt)
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
