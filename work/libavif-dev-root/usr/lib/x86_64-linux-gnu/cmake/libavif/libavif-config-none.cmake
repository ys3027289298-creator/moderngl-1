#----------------------------------------------------------------
# Generated CMake target import file for configuration "None".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "avif" for configuration "None"
set_property(TARGET avif APPEND PROPERTY IMPORTED_CONFIGURATIONS NONE)
set_target_properties(avif PROPERTIES
  IMPORTED_LOCATION_NONE "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libavif.so.16.0.4"
  IMPORTED_SONAME_NONE "libavif.so.16"
  )

list(APPEND _cmake_import_check_targets avif )
list(APPEND _cmake_import_check_files_for_avif "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libavif.so.16.0.4" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
