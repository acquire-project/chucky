#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "chucky::chucky-static" for configuration "RelWithDebInfo"
set_property(TARGET chucky::chucky-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(chucky::chucky-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C;CUDA;CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libchucky.a"
  )

list(APPEND _cmake_import_check_targets chucky::chucky-static )
list(APPEND _cmake_import_check_files_for_chucky::chucky-static "${_IMPORT_PREFIX}/lib/libchucky.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
