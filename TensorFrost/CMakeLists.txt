file(GLOB_RECURSE TENSORFROST_SOURCE_LIST CONFIGURE_DEPENDS *.cpp)
file(GLOB_RECURSE TENSORFROST_HEADER_LIST CONFIGURE_DEPENDS *.h *.hpp)

pybind11_add_module(TensorFrost ${TENSORFROST_SOURCE_LIST} ${TENSORFROST_HEADER_LIST})

# Set a default Python version
set(PYTHON_VERSION "3.8" CACHE STRING "Default Python version")

# Allow the user to override the default Python version
# Example usage: cmake -DPYTHON_VERSION=3.7 ..
message(STATUS "Using Python version: ${PYTHON_VERSION}")

find_package (Python3 3.10 REQUIRED COMPONENTS Interpreter)

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

