# FindZstd.cmake — locate zstd headers and library
#
# Creates imported target: Zstd::Zstd

include(FindPackageHandleStandardArgs)

find_path(ZSTD_INCLUDE_DIR NAMES zstd.h
    PATH_SUFFIXES include)

# Prefer static libraries (.a before .so)
set(_zstd_save_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 .a)

find_library(ZSTD_LIBRARY NAMES zstd_static libzstd_static zstd libzstd
    PATH_SUFFIXES lib static)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_zstd_save_suffixes})

find_package_handle_standard_args(Zstd
    REQUIRED_VARS ZSTD_LIBRARY ZSTD_INCLUDE_DIR)

if(Zstd_FOUND AND NOT TARGET Zstd::Zstd)
    add_library(Zstd::Zstd UNKNOWN IMPORTED)
    set_target_properties(Zstd::Zstd PROPERTIES
        IMPORTED_LOCATION "${ZSTD_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}"
    )
endif()
