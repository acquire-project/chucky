
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was chuckyConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

# Make bundled Find*.cmake modules visible to find_dependency for the duration
# of this file. Restore the consumer's CMAKE_MODULE_PATH at the end.
set(_chucky_orig_module_path "${CMAKE_MODULE_PATH}")
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

find_dependency(Threads)

# CONFIG-first / MODULE-fallback for libs that may be packaged either way.
# This matches the project's own Lz4.cmake/Zstd.cmake/Blosc.cmake so the
# imported target names baked into the targets export (e.g. LZ4::lz4_static)
# resolve regardless of the consumer's environment.
macro(_chucky_find_dep_config_or_module config_name module_name)
    find_package(${config_name} CONFIG QUIET)
    if(NOT ${config_name}_FOUND)
        find_dependency(${module_name})
    endif()
endmacro()

_chucky_find_dep_config_or_module(lz4 Lz4)
_chucky_find_dep_config_or_module(zstd Zstd)
if(ON)
    _chucky_find_dep_config_or_module(blosc Blosc)
endif()

find_dependency(aws-c-s3 CONFIG)

if(ON)
    find_dependency(CUDAToolkit 12.8)
    find_dependency(Nvcomp)
endif()

set(CMAKE_MODULE_PATH "${_chucky_orig_module_path}")
unset(_chucky_orig_module_path)

include("${CMAKE_CURRENT_LIST_DIR}/chuckyTargets.cmake")

# Default `chucky::chucky[-cpu]` to the shared variant. Guard against being
# re-included on a transitive find_package call.
if(NOT TARGET chucky::chucky)
    add_library(chucky::chucky ALIAS chucky::chucky-shared)
endif()

check_required_components(chucky)
