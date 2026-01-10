cmake_minimum_required(VERSION 3.15)
project(VisCore)

# Force static build for this internal library to avoid installation issues
set(BUILD_SHARED_LIBS OFF)

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Include VisCore macros
include(VisCoreModule)
include(VisCoreCompilerOptions)

# Add sub-modules
add_subdirectory(common)
add_subdirectory(modules)
