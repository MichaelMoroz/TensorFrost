cmake_minimum_required(VERSION 3.12)
project(TensorFrost)

set(CMAKE_CXX_STANDARD 14)

# Set the output directory for the .pyd file
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/python_module)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/python_module)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/python_module)
set(CMAKE_PDB_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/python_module)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/python_module)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/python_module)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/python_module)
set(CMAKE_PDB_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/python_module)

# List all your source files here
file(GLOB_RECURSE TENSORFROST_SOURCES "TensorFrost/*.cpp" "TensorFrost/*.h")

# Organize files according to the directory structure
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/TensorFrost" PREFIX "Source Files" FILES ${TENSORFROST_SOURCES})

# pybind11
add_subdirectory(pybind11)
pybind11_add_module(TensorFrost ${TENSORFROST_SOURCES})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

# Custom command to run pip install .
add_custom_command(
    TARGET TensorFrost
    POST_BUILD
    COMMAND ${PYTHON_EXECUTABLE} -m pip install -e .
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Custom target that depends on test_pybind and runs the custom command
add_custom_target(install_python_package ALL
    DEPENDS TensorFrost
)

set(CMAKE_BUILD_TYPE Debug)

if(MSVC)
    # Path to your script that you want to run
    set(DEBUG_PYTHON_SCRIPT "${CMAKE_SOURCE_DIR}/examples/debug.py")

    set_target_properties(TensorFrost PROPERTIES
        VS_DEBUGGER_COMMAND "${PYTHON_EXECUTABLE}"
        VS_DEBUGGER_COMMAND_ARGUMENTS "${DEBUG_PYTHON_SCRIPT}"
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endif()

# List all your Python example files here
file(GLOB_RECURSE PYTHON_EXAMPLE_FILES "examples/*.py")

# Organize Python files according to the directory structure
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/examples" PREFIX "Python Examples" FILES ${PYTHON_EXAMPLE_FILES})

# Create a custom target just for visibility in VS
add_custom_target(PythonExamples ALL SOURCES ${PYTHON_EXAMPLE_FILES})