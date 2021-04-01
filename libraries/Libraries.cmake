# get the current directory
get_filename_component(integration-test-path ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# compile each .cc file into a shared library
file(GLOB files "${integration-test-path}/*.cc")

add_custom_target(libraries)
foreach(file ${files})
    # grab the name of the test without the extension
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} SHARED ${file})
  
    target_link_libraries(${fileName} bbts-common)
  
    add_dependencies(libraries ${fileName})
endforeach()
