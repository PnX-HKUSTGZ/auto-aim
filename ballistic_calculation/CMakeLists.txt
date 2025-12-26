cmake_minimum_required(VERSION 3.8)
project(ballistic_calculation)

# Set C++ standard to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

## Add optimization flags
add_compile_options(-O2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add third_party_install to CMAKE_PREFIX_PATH
get_filename_component(ABS_PREFIX_PREFIX "../../../third_party_install" ABSOLUTE)
set(CMAKE_PREFIX_PREFIX ${ABS_PREFIX_PREFIX})
message(STATUS "ABS_PREFIX_PREFIX: ${ABS_PREFIX_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH ${ABS_PREFIX_PREFIX})

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find packages
find_package(ament_cmake_auto REQUIRED)
find_package(Ceres REQUIRED)
find_package(tf2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)

ament_auto_find_build_dependencies()
# Include directories
include_directories(
  include 
  ${OpenCV_INCLUDE_DIRS} 
  ${CERES_INCLUDE_DIRS}
  ${tf2_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/tinympc
  ${YAML_CPP_INCLUDE_DIR}
)

add_subdirectory(tinympc)

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_link_libraries(${PROJECT_NAME}
  ${CERES_LIBRARIES}
  ${OpenCV_LIBS}
  tinympc
  yaml-cpp
)

# Add executable
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_aim::BallisticCalculateNode
  EXECUTABLE ${PROJECT_NAME}_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()



