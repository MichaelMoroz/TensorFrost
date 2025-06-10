set(TF_INC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TF_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB_RECURSE TENSORFROST_SOURCE_LIST  CONFIGURE_DEPENDS
        ${TF_SRC_DIR}/*.cpp)

file(GLOB_RECURSE TENSORFROST_HEADER_LIST  CONFIGURE_DEPENDS
        ${TF_INC_DIR}/*.h  ${TF_INC_DIR}/*.hpp)

set(PYBIND_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/PybindModule.cpp)

# ---- Build pybind11 module ----
pybind11_add_module(TensorFrost
        ${PYBIND_MODULE}
        ${TENSORFROST_SOURCE_LIST}
        ${TENSORFROST_HEADER_LIST}
)

# ---- Include directories ----
target_include_directories(TensorFrost
        PRIVATE
        ${TF_INC_DIR}
        ${Python3_INCLUDE_DIRS}
)

# ---- Fix for macOS RPATH issues ----
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set_target_properties(TensorFrost PROPERTIES
            BUILD_WITH_INSTALL_RPATH ON
            INSTALL_RPATH "@loader_path/."
    )
endif()

# ---- libraries ----
target_link_libraries(TensorFrost PRIVATE glfw)

glad_add_library(glad_gl_core_46 SHARED  API gl:core=4.6)
target_link_libraries(TensorFrost PRIVATE glad_gl_core_46)

glad_add_library(glad_vulkan_12 REPRODUCIBLE LOADER API vulkan=1.2)
target_link_libraries(TensorFrost PRIVATE glad_vulkan_12)

# ---- ImGui ----
target_include_directories(TensorFrost PRIVATE
        ${CMAKE_SOURCE_DIR}/external/imgui
        ${CMAKE_SOURCE_DIR}/external/imgui/backends
)
file(GLOB  IMGUI_SOURCE_LIST
        ${CMAKE_SOURCE_DIR}/external/imgui/*.cpp)
file(GLOB  IMGUI_BACKEND_SOURCE_LIST
        ${CMAKE_SOURCE_DIR}/external/imgui/backends/imgui_impl_glfw.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/backends/imgui_impl_opengl3.cpp)
target_sources(TensorFrost PRIVATE ${IMGUI_SOURCE_LIST} ${IMGUI_BACKEND_SOURCE_LIST})

# ---- RenderDoc headers ----
target_include_directories(TensorFrost PRIVATE ${CMAKE_SOURCE_DIR}/external/renderdoc)

# ---- convenience / IDE niceties ----
source_group(TREE ${TF_SRC_DIR} PREFIX "Source Files"
        FILES ${TENSORFROST_SOURCE_LIST})

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Source Files"
        FILES ${PYBIND_MODULE})

source_group(TREE ${TF_INC_DIR} PREFIX "Header Files"
        FILES ${TENSORFROST_HEADER_LIST})

# ---- VS/debug helpers, profiling flags, etc. (unchanged) ----
add_custom_target(install_python_package ALL DEPENDS TensorFrost)

set(DEBUG_PYTHON_SCRIPT "${CMAKE_SOURCE_DIR}/examples/debug.py")
set_target_properties(TensorFrost PROPERTIES
        VS_DEBUGGER_COMMAND            "${Python3_EXECUTABLE}"
        VS_DEBUGGER_COMMAND_ARGUMENTS  "${DEBUG_PYTHON_SCRIPT}"
        VS_DEBUGGER_WORKING_DIRECTORY  "${CMAKE_SOURCE_DIR}"
        LINK_FLAGS_RELWITHDEBINFO      "/PROFILE"
)