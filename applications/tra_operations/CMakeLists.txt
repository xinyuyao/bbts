# create the application executable
get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

add_executable(generate_tra_operations ${app_dir}/generate_tra_operations.cc)
set_target_properties(generate_tra_operations PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(generate_tra_operations ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(generate_tra_operations ${MPI_LIBRARIES})
target_link_libraries(generate_tra_operations bbts-common)



# add the dependencies
add_dependencies(applications generate_tra_operations)
