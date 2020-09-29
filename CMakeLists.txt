cmake_minimum_required(VERSION 3.17)
project(tra_sub_sys_barbatos)

# set the c++ version
set(CMAKE_CXX_STANDARD 20)

# set the location for cmake scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/tools/cmake")

# the tensors
file(GLOB TENSOR_SOURCE "${PROJECT_SOURCE_DIR}/tensor/*.cc")
add_library(tensor OBJECT ${TENSOR_SOURCE})

#file(GLOB UD_SOURCE "${PROJECT_SOURCE_DIR}/ud_functions/*.cc")
#add_library(ud_functions OBJECT ${UD_SOURCE})

# link all modules up
add_library(bbts-common $<TARGET_OBJECTS:tensor>)

# add the tests
include(tests/Tests.cmake)

add_executable(tra_sub_sys_barbatos server/node_main.cc)