cmake_minimum_required(VERSION 3.15)
project(TensorFrost)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(_RELWITHDEBINFO)
endif()

# Set the output directory for the .pyd file for all configurations and types
foreach(TYPE ARCHIVE LIBRARY RUNTIME PDB)
  foreach(CONFIG RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_${TYPE}_OUTPUT_DIRECTORY_${CONFIG} ${CMAKE_SOURCE_DIR}/Python/TensorFrost)
  endforeach()
endforeach()

# If PYBIND11_PYTHON_VERSION is not set just use the default python version
if(NOT DEFINED PYBIND11_PYTHON_VERSION)
  message("Looking for LAST Python version ${PYBIND11_PYTHON_VERSION}")
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
else()
  message("Looking for EXACT Python version ${PYBIND11_PYTHON_VERSION}")
  find_package(Python3 ${PYBIND11_PYTHON_VERSION} EXACT COMPONENTS Interpreter Development.Module REQUIRED)
endif()
message("TensorFrost found Python: ${Python3_VERSION}")

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(PYBIND11_FINDPYTHON ON)

add_subdirectory(external/pybind11)
add_subdirectory(external/glfw)
add_subdirectory(external/glad/cmake)
add_subdirectory(TensorFrost)
add_subdirectory(examples)

set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT TensorFrost)