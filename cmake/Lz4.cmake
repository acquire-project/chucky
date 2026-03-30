# Lz4.cmake — find lz4 and provide a unified Lz4::Lz4 target
#
# Tries CONFIG mode first (covers vcpkg, Conan, system CMake configs),
# then falls back to the bundled FindLz4.cmake module.

find_package(lz4 CONFIG QUIET)
if(lz4_FOUND)
    if(TARGET LZ4::lz4_static)
        add_library(Lz4::Lz4 ALIAS LZ4::lz4_static)
    elseif(TARGET lz4::lz4)
        add_library(Lz4::Lz4 ALIAS lz4::lz4)
    elseif(TARGET LZ4::lz4)
        add_library(Lz4::Lz4 ALIAS LZ4::lz4)
    endif()
else()
    find_package(Lz4 REQUIRED)
endif()
