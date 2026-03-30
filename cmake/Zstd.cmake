# Zstd.cmake — find zstd and provide a unified Zstd::Zstd target
#
# Tries CONFIG mode first (covers vcpkg, Conan, system CMake configs),
# then falls back to the bundled FindZstd.cmake module.

find_package(zstd CONFIG QUIET)
if(zstd_FOUND)
    if(TARGET zstd::libzstd_static)
        add_library(Zstd::Zstd ALIAS zstd::libzstd_static)
    else()
        add_library(Zstd::Zstd ALIAS zstd::libzstd)
    endif()
else()
    find_package(Zstd REQUIRED)
endif()
