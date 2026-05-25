cmake_minimum_required(VERSION 3.10)
project(rune_detector)

## Use C++20 (Tiger Core requires concepts/remove_cvref)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
ament_auto_find_build_dependencies()

# Tiger Core: reuse third_party from rune_detector_for_all
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/rm_vision_core ${CMAKE_CURRENT_BINARY_DIR}/rm_vision_core)

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  Ceres::ceres
  VisCore_rune_detector
  VisCore_core
  VisCore_rune_group
  ament_index_cpp::ament_index_cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} )
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_aim::RuneDetectorNode
  EXECUTABLE rune_detector_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  docs
)

