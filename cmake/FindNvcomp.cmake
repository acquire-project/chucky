# FindNvcomp.cmake — locate nvcomp headers and library
#
# find_package(nvcomp) is broken on Nix (split outputs),
# so we locate headers and library manually.
#
# Creates imported target: nvcomp::nvcomp

include(FindPackageHandleStandardArgs)

find_path(NVCOMP_INCLUDE_DIR NAMES nvcomp.h PATH_SUFFIXES include)

find_library(
    NVCOMP_LIBRARY
    NAMES nvcomp_static nvcomp
    PATH_SUFFIXES lib lib/${CUDAToolkit_VERSION_MAJOR}
)

find_package_handle_standard_args(
    Nvcomp
    REQUIRED_VARS NVCOMP_LIBRARY NVCOMP_INCLUDE_DIR
)

if(Nvcomp_FOUND AND NOT TARGET nvcomp::nvcomp)
    add_library(nvcomp::nvcomp UNKNOWN IMPORTED)
    set_target_properties(
        nvcomp::nvcomp
        PROPERTIES
            IMPORTED_LOCATION "${NVCOMP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NVCOMP_INCLUDE_DIR}"
    )
    if(NVCOMP_LIBRARY MATCHES "nvcomp_static")
        set_property(
            TARGET nvcomp::nvcomp
            PROPERTY INTERFACE_COMPILE_DEFINITIONS NVCOMP_STATIC_DEFINE
        )
    endif()
endif()
