file(GLOB_RECURSE TENSORFROST_SOURCE_LIST CONFIGURE_DEPENDS *.cpp)
file(GLOB_RECURSE TENSORFROST_HEADER_LIST CONFIGURE_DEPENDS *.h *.hpp)

pybind11_add_module(TensorFrost ${TENSORFROST_SOURCE_LIST} ${TENSORFROST_HEADER_LIST})

# Add GLFW
target_link_libraries(TensorFrost PRIVATE glfw)


# Add GLAD
# Check if jinja2 is installed
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import jinja2"
    RESULT_VARIABLE JINJA2_NOT_FOUND
)

# If jinja2 is not found, install it
if(JINJA2_NOT_FOUND)
    message(STATUS "Installing jinja2 using pip")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pip install jinja2
        RESULT_VARIABLE PIP_INSTALL_RESULT
    )
    if(NOT PIP_INSTALL_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to install jinja2")
    endif()
endif()

glad_add_library(glad_gl_core_46 SHARED API gl:core=4.6)
target_link_libraries(TensorFrost PRIVATE glad_gl_core_46)

target_include_directories(TensorFrost PRIVATE ${Python3_INCLUDE_DIRS})

target_include_directories(TensorFrost PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(
    TARGET TensorFrost
    POST_BUILD
    COMMAND ${Python3_EXECUTABLE} -m pip install -e PythonBuild/
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(install_python_package ALL
    DEPENDS TensorFrost
)

set(DEBUG_PYTHON_SCRIPT "${CMAKE_SOURCE_DIR}/examples/debug.py")

set_target_properties(TensorFrost PROPERTIES
    VS_DEBUGGER_COMMAND "${Python3_EXECUTABLE}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "${DEBUG_PYTHON_SCRIPT}"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Source Files" FILES ${TENSORFROST_SOURCE_LIST} ${TENSORFROST_HEADER_LIST})

