set(TF_COMPILER_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TF_COMPILER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB_RECURSE TENSORFROST_COMPILER_SOURCE_LIST CONFIGURE_DEPENDS
        ${TF_COMPILER_SRC_DIR}/*.cpp)

file(GLOB_RECURSE TENSORFROST_COMPILER_HEADER_LIST CONFIGURE_DEPENDS
        ${TF_COMPILER_INC_DIR}/*.h
        ${TF_COMPILER_INC_DIR}/*.hpp)

add_library(TensorFrostCompiler STATIC
        ${TENSORFROST_COMPILER_SOURCE_LIST}
        ${TENSORFROST_COMPILER_HEADER_LIST})

target_include_directories(TensorFrostCompiler
        PUBLIC
        ${TF_COMPILER_INC_DIR})

target_compile_features(TensorFrostCompiler PUBLIC cxx_std_20)

if (MSVC)
        target_compile_options(TensorFrostCompiler PRIVATE /wd4804 /wd4805 /wd4018)
        target_compile_definitions(TensorFrostCompiler PRIVATE
                $<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=2>
                $<$<CONFIG:Debug>:_HAS_ITERATOR_DEBUGGING=1>
        )
        set_property(TARGET TensorFrostCompiler PROPERTY
                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

source_group(TREE ${TF_COMPILER_SRC_DIR} PREFIX "Source Files"
        FILES ${TENSORFROST_COMPILER_SOURCE_LIST})

source_group(TREE ${TF_COMPILER_INC_DIR} PREFIX "Header Files"
        FILES ${TENSORFROST_COMPILER_HEADER_LIST})
