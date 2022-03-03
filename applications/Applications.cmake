# get the current directory
get_filename_component(app_path ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# add the applications target
add_custom_target(applications)

# this function is used to add an aplication
function(add_aplication folder_path)

# compile all the objects
include("${app_path}/${folder_path}/CMakeLists.txt")

endfunction()

# add the applications
add_aplication("bmm")
add_aplication("cpmm")
add_aplication("ffnn")
add_aplication("node")
add_aplication("tra_node")
add_aplication("tra_operations")
add_aplication("matrix_addition") 
add_aplication("multiplication_chain")


add_aplication("multiplication_chain")
add_aplication("mm_single_node")

if(${ENABLE_GPU})
    add_aplication("ffnn-gpu")
endif()


