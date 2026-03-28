# FindLz4.cmake — locate lz4 headers and library
#
# Creates imported target: Lz4::Lz4

include(FindPackageHandleStandardArgs)

find_path(LZ4_INCLUDE_DIR NAMES lz4.h PATH_SUFFIXES include)

# Prefer static libraries (.a before .so)
set(_lz4_save_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 .a)

find_library(
    LZ4_LIBRARY
    NAMES lz4_static liblz4_static lz4 liblz4
    PATH_SUFFIXES lib static
)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_lz4_save_suffixes})

find_package_handle_standard_args(Lz4 REQUIRED_VARS LZ4_LIBRARY LZ4_INCLUDE_DIR)

if(Lz4_FOUND AND NOT TARGET Lz4::Lz4)
    add_library(Lz4::Lz4 UNKNOWN IMPORTED)
    set_target_properties(
        Lz4::Lz4
        PROPERTIES
            IMPORTED_LOCATION "${LZ4_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}"
    )
endif()
