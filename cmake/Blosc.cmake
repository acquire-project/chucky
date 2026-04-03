# Blosc.cmake — find blosc (c-blosc) and provide a unified Blosc::Blosc target
#
# Tries CONFIG mode first (covers vcpkg, Conan, system CMake configs),
# then falls back to the bundled FindBlosc.cmake module.
# If neither finds blosc, HAVE_BLOSC is OFF and blosc codecs are unavailable.

find_package(blosc CONFIG QUIET)
if(blosc_FOUND)
    if(TARGET blosc::blosc_static)
        set(_blosc_upstream blosc::blosc_static)
    elseif(TARGET blosc_static)
        set(_blosc_upstream blosc_static)
    elseif(TARGET blosc::blosc_shared)
        set(_blosc_upstream blosc::blosc_shared)
    elseif(TARGET blosc_shared)
        set(_blosc_upstream blosc_shared)
    elseif(TARGET Blosc::blosc)
        set(_blosc_upstream Blosc::blosc)
    endif()
    if(_blosc_upstream AND NOT TARGET Blosc::Blosc)
        add_library(Blosc::Blosc INTERFACE IMPORTED)
        target_link_libraries(Blosc::Blosc INTERFACE ${_blosc_upstream})
        # Static blosc may need zlib explicitly on some platforms.
        find_package(ZLIB QUIET)
        if(TARGET ZLIB::ZLIB)
            target_link_libraries(Blosc::Blosc INTERFACE ZLIB::ZLIB)
        endif()
    endif()
endif()

if(NOT TARGET Blosc::Blosc)
    find_package(Blosc QUIET)
endif()

if(TARGET Blosc::Blosc)
    set(HAVE_BLOSC ON)
else()
    set(HAVE_BLOSC OFF)
    message(STATUS "Blosc not found — blosc codecs disabled")
endif()
