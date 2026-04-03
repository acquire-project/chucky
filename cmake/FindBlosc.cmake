# FindBlosc.cmake — locate blosc (c-blosc) headers and library
#
# Creates imported target: Blosc::Blosc
# Assumes Lz4::Lz4 and Zstd::Zstd targets already exist (for static link).

include(FindPackageHandleStandardArgs)

find_path(BLOSC_INCLUDE_DIR NAMES blosc.h PATH_SUFFIXES include)

# Prefer static libraries (.a before .so)
set(_blosc_save_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 .a)

find_library(
    BLOSC_LIBRARY
    NAMES blosc_static libblosc_static blosc libblosc
    PATH_SUFFIXES lib static
)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_blosc_save_suffixes})

find_package_handle_standard_args(
    Blosc
    REQUIRED_VARS BLOSC_LIBRARY BLOSC_INCLUDE_DIR
)

if(Blosc_FOUND AND NOT TARGET Blosc::Blosc)
    add_library(Blosc::Blosc UNKNOWN IMPORTED)
    set_target_properties(
        Blosc::Blosc
        PROPERTIES
            IMPORTED_LOCATION "${BLOSC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${BLOSC_INCLUDE_DIR}"
    )

    # Static blosc bundles lz4/zstd/zlib/snappy — re-link them so they
    # appear after libblosc.a and satisfy its undefined references.
    if(BLOSC_LIBRARY MATCHES "\\.(a|lib)$")
        set(_blosc_deps Lz4::Lz4 Zstd::Zstd)

        find_package(ZLIB QUIET)
        if(TARGET ZLIB::ZLIB)
            list(APPEND _blosc_deps ZLIB::ZLIB)
        endif()

        find_library(_SNAPPY_LIB NAMES snappy)
        if(_SNAPPY_LIB)
            list(APPEND _blosc_deps "${_SNAPPY_LIB}")
        endif()

        set_property(
            TARGET Blosc::Blosc
            PROPERTY INTERFACE_LINK_LIBRARIES ${_blosc_deps}
        )
    endif()
endif()
