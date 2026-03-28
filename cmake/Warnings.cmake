# Warnings.cmake — shared warning flags via INTERFACE target
#
# Creates: warnings (INTERFACE library)
# Provides: add_platform_sources(TARGET BASENAME)

add_library(warnings INTERFACE)
if(MSVC)
    target_compile_options(
        warnings
        INTERFACE
            $<$<COMPILE_LANGUAGE:C>:/W3>
            $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=/W3>
    )
else()
    target_compile_options(
        warnings
        INTERFACE
            $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
    )
endif()

# add_platform_sources(TARGET [BASENAME])
#   Appends basename.win32.c or basename.posix.c to the target's sources.
#   BASENAME defaults to TARGET.
function(add_platform_sources TARGET)
    if(ARGC GREATER 1)
        set(BASE ${ARGV1})
    else()
        set(BASE ${TARGET})
    endif()
    if(WIN32)
        target_sources(${TARGET} PRIVATE ${BASE}.win32.c)
    else()
        target_sources(${TARGET} PRIVATE ${BASE}.posix.c)
    endif()
endfunction()
