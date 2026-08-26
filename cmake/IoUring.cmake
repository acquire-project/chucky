# IoUring.cmake — find liburing and provide a unified IoUring::IoUring target
#
# Tries pkg-config first, then a plain header/library search.
# Linux only. If liburing is not found, HAVE_IO_URING is OFF and the io_uring
# write backend is left out of the build.

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT TARGET IoUring::IoUring)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(liburing QUIET IMPORTED_TARGET liburing)
    endif()

    if(TARGET PkgConfig::liburing)
        add_library(IoUring::IoUring INTERFACE IMPORTED)
        target_link_libraries(IoUring::IoUring INTERFACE PkgConfig::liburing)
    else()
        find_path(IoUring_INCLUDE_DIR liburing.h)
        find_library(IoUring_LIBRARY NAMES uring)
        if(IoUring_INCLUDE_DIR AND IoUring_LIBRARY)
            add_library(IoUring::IoUring INTERFACE IMPORTED)
            target_include_directories(
                IoUring::IoUring
                INTERFACE ${IoUring_INCLUDE_DIR}
            )
            target_link_libraries(IoUring::IoUring INTERFACE ${IoUring_LIBRARY})
        endif()
    endif()
endif()

if(TARGET IoUring::IoUring)
    set(HAVE_IO_URING ON)
else()
    set(HAVE_IO_URING OFF)
    message(STATUS "liburing not found — the io_uring write backend is off")
endif()
