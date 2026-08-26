cmake_minimum_required(VERSION 3.24)
project(chucky VERSION 0.1.0 LANGUAGES C CXX)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE)
    add_link_options("LINKER:-no_warn_duplicate_libraries")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(Msvc)
include(Warnings)
include(Nix)
include(Gpu)
include(Lz4)
include(Zstd)
include(Blosc)
include(IoUring)

find_package(Threads REQUIRED)
find_package(aws-c-s3 REQUIRED CONFIG)

# Coverage flags must come after find_package calls — the instrumentation
# flags break CMake's try_compile checks (e.g. FindThreads).
include(Coverage)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)

add_subdirectory(bench)

# --- Install rules: aggregate library, public headers, CMake/pkg-config ---

include(CMakePackageConfigHelpers)
set(_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/${CHUCKY_AGGREGATE_NAME})

set(_chucky_install_targets ${CHUCKY_AGGREGATE_NAME}-static)
if(CHUCKY_BUILD_SHARED)
    list(APPEND _chucky_install_targets ${CHUCKY_AGGREGATE_NAME}-shared)
endif()
install(
    TARGETS ${_chucky_install_targets}
    EXPORT ${CHUCKY_AGGREGATE_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

file(GLOB _public_headers "${PROJECT_SOURCE_DIR}/src/*.h")
install(FILES ${_public_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/chucky)
install(
    FILES "${PROJECT_SOURCE_DIR}/src/multiarray/multiarray.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/chucky/multiarray
)

# Bundled Find*.cmake modules so consumers can resolve transitive deps even
# without native CMake configs (e.g. nixpkgs c-blosc). FindNvcomp ships in
# both variants — harmless in chucky-cpu, never invoked there.
set(_find_modules
    cmake/FindLz4.cmake
    cmake/FindZstd.cmake
    cmake/FindBlosc.cmake
    cmake/FindNvcomp.cmake
)
list(TRANSFORM _find_modules PREPEND "${PROJECT_SOURCE_DIR}/")
install(FILES ${_find_modules} DESTINATION ${_cmake_dir})

install(
    EXPORT ${CHUCKY_AGGREGATE_NAME}Targets
    NAMESPACE chucky::
    DESTINATION ${_cmake_dir}
    FILE ${CHUCKY_AGGREGATE_NAME}Targets.cmake
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/chuckyConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${CHUCKY_AGGREGATE_NAME}Config.cmake"
    INSTALL_DESTINATION ${_cmake_dir}
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${CHUCKY_AGGREGATE_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${CHUCKY_AGGREGATE_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${CHUCKY_AGGREGATE_NAME}ConfigVersion.cmake"
    DESTINATION ${_cmake_dir}
)

# pkg-config — Libs.private is best-effort; CMake config is the supported
# integration path. cudart_static / nvcomp are baked into the chucky archive
# but a bare -l flag cannot locate them without -L into the toolkit.
set(CHUCKY_VARIANT_DESCRIPTION "CPU only")
set(CHUCKY_PC_LIBS_PRIVATE "-lm -llz4 -lzstd -laws-c-s3 -ldl -lrt -lpthread")
if(CHUCKY_ENABLE_GPU)
    set(CHUCKY_VARIANT_DESCRIPTION "GPU + CPU")
    string(APPEND CHUCKY_PC_LIBS_PRIVATE " -lcudart_static -lnvcomp")
endif()
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/chucky.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${CHUCKY_AGGREGATE_NAME}.pc"
    @ONLY
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${CHUCKY_AGGREGATE_NAME}.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
