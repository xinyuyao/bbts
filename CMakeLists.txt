cmake_minimum_required(VERSION 3.17)
project(tra_sub_sys_barbatos)

# set the c++ version
set(CMAKE_CXX_STANDARD 20)

# include the third party stuff
include_directories(third-party)

# set the location for cmake scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/tools/cmake")

# the tensors
file(GLOB TENSOR_SOURCE "src/tensor/*.cc")
add_library(tensor OBJECT ${TENSOR_SOURCE})

file(GLOB UD_SOURCE "src/ud_functions/*.cc")
add_library(ud_functions OBJECT ${UD_SOURCE})

# link all modules up
add_library(bbts-common $<TARGET_OBJECTS:tensor>
                        $<TARGET_OBJECTS:ud_functions>)

# add the tests
include(tests/Tests.cmake)
include(integration_tests/Integration.cmake)

# the main executable
add_executable(tra_sub_sys_barbatos src/server/node_main.cc)