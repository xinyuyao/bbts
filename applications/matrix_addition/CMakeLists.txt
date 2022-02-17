# create the application executable
get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

add_executable(generate_matrix_addition ${app_dir}/generate_matrix_addition.cc)
set_target_properties(generate_matrix_addition PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(generate_matrix_addition ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(generate_matrix_addition ${MPI_LIBRARIES})
target_link_libraries(generate_matrix_addition bbts-common)

# add the dependencies
add_dependencies(applications generate_matrix_addition)
