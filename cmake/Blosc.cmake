# Blosc.cmake — find blosc (c-blosc) and provide a unified Blosc::Blosc target
#
# Tries CONFIG mode first (covers vcpkg, Conan, system CMake configs),
# then falls back to the bundled FindBlosc.cmake module.

find_package(blosc CONFIG QUIET)
if(blosc_FOUND)
    if(TARGET blosc::blosc_static)
        add_library(Blosc::Blosc ALIAS blosc::blosc_static)
    elseif(TARGET blosc::blosc_shared)
        add_library(Blosc::Blosc ALIAS blosc::blosc_shared)
    elseif(TARGET Blosc::blosc)
        add_library(Blosc::Blosc ALIAS Blosc::blosc)
    endif()
endif()

if(NOT TARGET Blosc::Blosc)
    find_package(Blosc REQUIRED)
endif()
