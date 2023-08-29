cmake_minimum_required(VERSION 3.12)
project(TensorFrost)

set(CMAKE_CXX_STANDARD 14)

# Set the output directory for the .pyd file
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/python_module)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/python_module)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/python_module)

# pybind11
add_subdirectory(pybind11)
pybind11_add_module(test_pybind "TensorFrost/Frontend/Python/test_pybind.cpp")

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

# Custom command to run pip install .
add_custom_command(
    TARGET test_pybind
    POST_BUILD
    COMMAND ${PYTHON_EXECUTABLE} -m pip install .
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Custom target that depends on test_pybind and runs the custom command
add_custom_target(install_python_package ALL
    DEPENDS test_pybind
)