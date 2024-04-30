cmake_minimum_required(VERSION 3.12)
project(TensorFrost)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# Set the output directory for the .pyd file
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/PythonBuild/TensorFrost)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/PythonBuild/TensorFrost)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/PythonBuild/TensorFrost)
set(CMAKE_PDB_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/PythonBuild/TensorFrost)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/PythonBuild/TensorFrost)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/PythonBuild/TensorFrost)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/PythonBuild/TensorFrost)
set(CMAKE_PDB_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/PythonBuild/TensorFrost)

# If PYBIND11_PYTHON_VERSION is not set just use the default python version
if(NOT DEFINED PYBIND11_PYTHON_VERSION)
  find_package(Python3 COMPONENTS Interpreter Development)
else()
  find_package(Python3 ${PYBIND11_PYTHON_VERSION} EXACT COMPONENTS Interpreter Development)
endif()

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(external/pybind11)
add_subdirectory(external/glfw)
add_subdirectory(external/glad/cmake)
add_subdirectory(TensorFrost)
add_subdirectory(examples)

set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT TensorFrost)
