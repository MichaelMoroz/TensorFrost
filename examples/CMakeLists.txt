file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS *.py)

# Organize Python files according to the directory structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Python Examples" FILES ${SOURCE_FILES})

# Create a custom target just for visibility in VS
add_custom_target(PythonExamples ALL SOURCES ${SOURCE_FILES})