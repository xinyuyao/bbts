cmake_minimum_required(VERSION 3.15)
project(tra_sub_sys_barbatos LANGUAGES CXX CUDA)

# set the c++ and cuda version
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

# set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${PROJECT_SOURCE_DIR}/bin")

# set the location for cmake scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# if we are compiling for debug add the debug flags
if(CMAKE_BUILD_TYPE MATCHES Debug)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "-D_GLIBCXX_DEBUG")
    endif()
    set(CMAKE_CXX_FLAGS "-g3")

    #set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    #set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

# if we are compiling for release
elseif(CMAKE_BUILD_TYPE MATCHES Release)

    # do -O3 and set the instruction set to the native one for this machine
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

endif()

# forward the definitions ENABLE_HOOKS=ON/OFF
if(${ENABLE_HOOKS})
    add_compile_definitions(ENABLE_HOOKS)
endif()

# try to find mkl
set(BLA_VENDOR Intel10_64lp_seq)
find_package(BLAS)

# try to find the mpi library
find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif (MPI_FOUND)

# include the third party stuff
include_directories(third-party)

# the commands
file(GLOB COMMANDS_SOURCE "src/commands/*.cc")
add_library(commands OBJECT ${COMMANDS_SOURCE})

# the operations
file(GLOB OPERATIONS_SOURCE "src/operations/*.cc")
add_library(operations OBJECT ${OPERATIONS_SOURCE})

# the tensors
file(GLOB TENSOR_SOURCE "src/tensor/*.cc")
add_library(tensor OBJECT ${TENSOR_SOURCE})

# the storage
file(GLOB STORAGE_SOURCE "src/storage/*.cc")
add_library(storage OBJECT ${STORAGE_SOURCE})

# the server
file(GLOB SERVER_SOURCE "src/server/*.cc")
add_library(server OBJECT ${SERVER_SOURCE})

# the communication
file(GLOB COMMUNICATION_SOURCE "src/communication/*.cc")
add_library(communication OBJECT ${COMMUNICATION_SOURCE})

# the ud functions
file(GLOB UD_SOURCE "src/ud_functions/*.cc" "src/ud_functions/builtin/*.cc")
add_library(ud_functions OBJECT ${UD_SOURCE})

# link all modules up
add_library(bbts-common $<TARGET_OBJECTS:tensor>
                        $<TARGET_OBJECTS:commands>
                        $<TARGET_OBJECTS:operations>
                        $<TARGET_OBJECTS:storage>
                        $<TARGET_OBJECTS:communication>
                        $<TARGET_OBJECTS:server>
                        $<TARGET_OBJECTS:ud_functions>)

# link the mkl library
target_link_libraries(bbts-common ${BLAS_LIBRARIES})

# add the tests
include(tests/Tests.cmake)
include(integration_tests/Integration.cmake)
include(applications/Applications.cmake)