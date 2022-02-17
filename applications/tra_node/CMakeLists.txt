# create the application executable
get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

set_source_files_properties(${app_dir}/sqlite3.c PROPERTIES LANGUAGE C)
add_executable(tra_node_cli ${app_dir}/tra_node_cli.cc ${app_dir}/sqlite3.c)
# set_target_properties(tra_node_cli PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(tra_node_cli ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(tra_node_cli ${MPI_LIBRARIES})
target_link_libraries(tra_node_cli bbts-common)

# link to cnpy library for reading and writing numpy file in c++
add_subdirectory(cnpy)
target_link_libraries(tra_node_cli cnpy)
target_include_directories(tra_node_cli PRIVATE cnpy)

# add the dependencies
add_dependencies(applications tra_node_cli)
