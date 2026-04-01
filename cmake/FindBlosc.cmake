# FindBlosc.cmake — locate blosc (c-blosc) headers and library
#
# Creates imported target: Blosc::Blosc

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

    # Static blosc pulls in zlib and snappy as transitive dependencies
    if(BLOSC_LIBRARY MATCHES "\\.a$")
        find_package(ZLIB QUIET)
        if(TARGET ZLIB::ZLIB)
            set_property(
                TARGET Blosc::Blosc
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES ZLIB::ZLIB
            )
        endif()
        find_library(_SNAPPY_LIB NAMES snappy)
        if(_SNAPPY_LIB)
            set_property(
                TARGET Blosc::Blosc
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES "${_SNAPPY_LIB}"
            )
        endif()
    endif()
endif()
