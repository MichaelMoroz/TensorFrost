set(TF_INC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TF_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_library(SLANG_LIB_RELEASE NAMES slang        PATHS $ENV{VULKAN_SDK}/Lib)
find_library(SLANG_LIB_DEBUG   NAMES slangd slang PATHS $ENV{VULKAN_SDK}/Lib)

if (NOT SLANG_LIB_RELEASE)
        message(FATAL_ERROR "slang.lib not found in %VULKAN_SDK%/Lib")
endif()

add_subdirectory(Compiler)
add_subdirectory(Backend)

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

target_link_libraries(TensorFrost PRIVATE
        TensorFrostCompiler
        TensorFrostBackend)

if (MSVC)
    target_compile_options(TensorFrost PRIVATE /wd4804 /wd4805 /wd4018)
endif()

# ---- ImGui ----
target_include_directories(TensorFrost PRIVATE
        ${CMAKE_SOURCE_DIR}/external/imgui
        ${CMAKE_SOURCE_DIR}/external/imgui/backends
)
file(GLOB  IMGUI_SOURCE_LIST
        ${CMAKE_SOURCE_DIR}/external/imgui/*.cpp)
file(GLOB  IMGUI_BACKEND_SOURCE_LIST
        ${CMAKE_SOURCE_DIR}/external/imgui/backends/imgui_impl_glfw.cpp
        ${CMAKE_SOURCE_DIR}/external/imgui/backends/imgui_impl_vulkan.cpp)
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

set(VKSDK_BIN "$ENV{VULKAN_SDK}/Bin")

add_custom_command(TARGET TensorFrost POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:TensorFrost>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VKSDK_BIN}/slang$<$<CONFIG:Debug>:d>.dll"                         "$<TARGET_FILE_DIR:TensorFrost>/slang$<$<CONFIG:Debug>:d>.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VKSDK_BIN}/slang-glslang$<$<CONFIG:Debug>:d>.dll"                 "$<TARGET_FILE_DIR:TensorFrost>/slang-glslang$<$<CONFIG:Debug>:d>.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VKSDK_BIN}/slang-glsl-module$<$<CONFIG:Debug>:d>.dll"             "$<TARGET_FILE_DIR:TensorFrost>/slang-glsl-module$<$<CONFIG:Debug>:d>.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VKSDK_BIN}/glslang$<$<CONFIG:Debug>:d>.dll"                       "$<TARGET_FILE_DIR:TensorFrost>/glslang$<$<CONFIG:Debug>:d>.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VKSDK_BIN}/glslang-default-resource-limits$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:TensorFrost>/glslang-default-resource-limits$<$<CONFIG:Debug>:d>.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VKSDK_BIN}/SPIRV-Tools-shared$<$<CONFIG:Debug>:d>.dll"            "$<TARGET_FILE_DIR:TensorFrost>/SPIRV-Tools-shared$<$<CONFIG:Debug>:d>.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VKSDK_BIN}/SPIRV$<$<CONFIG:Debug>:d>.dll"                         "$<TARGET_FILE_DIR:TensorFrost>/SPIRV$<$<CONFIG:Debug>:d>.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VKSDK_BIN}/SPVRemapper$<$<CONFIG:Debug>:d>.dll"                   "$<TARGET_FILE_DIR:TensorFrost>/SPVRemapper$<$<CONFIG:Debug>:d>.dll"
)