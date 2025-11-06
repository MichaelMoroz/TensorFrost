set(TF_BACKEND_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TF_BACKEND_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB_RECURSE TENSORFROST_BACKEND_SOURCE_LIST CONFIGURE_DEPENDS
        ${TF_BACKEND_SRC_DIR}/*.cpp)

file(GLOB_RECURSE TENSORFROST_BACKEND_HEADER_LIST CONFIGURE_DEPENDS
        ${TF_BACKEND_INC_DIR}/*.h
        ${TF_BACKEND_INC_DIR}/*.hpp)

add_library(TensorFrostBackend STATIC
        ${TENSORFROST_BACKEND_SOURCE_LIST}
        ${TENSORFROST_BACKEND_HEADER_LIST})

target_include_directories(TensorFrostBackend
        PUBLIC
        ${TF_BACKEND_INC_DIR}
        $ENV{VULKAN_SDK}/Include)

target_link_libraries(TensorFrostBackend
        PUBLIC
        Vulkan::Vulkan
        Vulkan::shaderc_combined
        glfw
        $<$<CONFIG:Debug>:${SLANG_LIB_DEBUG}>
        $<$<NOT:$<CONFIG:Debug>>:${SLANG_LIB_RELEASE}>)

target_compile_definitions(TensorFrostBackend PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

target_compile_features(TensorFrostBackend PUBLIC cxx_std_20)

if (MSVC)
    target_compile_options(TensorFrostBackend PRIVATE /wd4804 /wd4805 /wd4018)
endif()

source_group(TREE ${TF_BACKEND_SRC_DIR} PREFIX "Source Files"
        FILES ${TENSORFROST_BACKEND_SOURCE_LIST})

source_group(TREE ${TF_BACKEND_INC_DIR} PREFIX "Header Files"
        FILES ${TENSORFROST_BACKEND_HEADER_LIST})
