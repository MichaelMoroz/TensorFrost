file(GLOB_RECURSE TENSORFROST_SOURCE_LIST CONFIGURE_DEPENDS *.cpp)
file(GLOB_RECURSE TENSORFROST_HEADER_LIST CONFIGURE_DEPENDS *.h *.hpp)

pybind11_add_module(TensorFrost ${TENSORFROST_SOURCE_LIST} ${TENSORFROST_HEADER_LIST})

# Add GLFW
target_link_libraries(TensorFrost PRIVATE glfw)

glad_add_library(glad_gl_core_46 SHARED API gl:core=4.6)
target_link_libraries(TensorFrost PRIVATE glad_gl_core_46)

glad_add_library(glad_vulkan_12 REPRODUCIBLE LOADER API vulkan=1.2)
target_link_libraries(TensorFrost PRIVATE glad_vulkan_12)

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

