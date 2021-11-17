#.rst:
# Findcutensor
# --------
# Finds the cutensor library
#
# This will will define the following variables::
#
# cutensor_FOUND - system has cutensor
# cutensor_INCLUDE_DIRS - the cutensor include directory
# cutensor_LIBRARIES - the cutensor libraries
#
# and the following imported targets::
#
#   cutensor::cutensor   - The libcutensor library

if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_cutensor cutensor QUIET)
endif()

find_path(cutensor_INCLUDE_DIR cutensor.h
                           PATHS ${PC_cutensor_INCLUDEDIR})
find_library(cutensor_LIBRARY cutensor
                          PATHS ${PC_cutensor_LIBRARY})
set(cutensor_VERSION ${PC_cutensor_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cutensor
                                  REQUIRED_VARS cutensor_LIBRARY cutensor_INCLUDE_DIR
                                  VERSION_VAR cutensor_VERSION)

if(cutensor_FOUND)
  set(cutensor_LIBRARIES ${cutensor_LIBRARY})
  set(cutensor_INCLUDE_DIRS ${cutensor_INCLUDE_DIR})

  if(NOT TARGET cutensor::cutensor)
    add_library(cutensor::cutensor UNKNOWN IMPORTED)
    set_target_properties(cutensor::cutensor PROPERTIES
                                     IMPORTED_LOCATION "${cutensor_LIBRARY}"
                                     INTERFACE_INCLUDE_DIRECTORIES "${cutensor_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(cutensor_INCLUDE_DIR cutensor_LIBRARY)