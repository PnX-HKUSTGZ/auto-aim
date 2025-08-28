cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)

## Add optimization flags
add_compile_options(-O2)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add third_party_install to CMAKE_PREFIX_PATH
get_filename_component(ABS_PREFIX_PREFIX "../../../third_party_install" ABSOLUTE)
set(CMAKE_PREFIX_PREFIX ${ABS_PREFIX_PREFIX})
message(STATUS "ABS_PREFIX_PREFIX: ${ABS_PREFIX_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH ${ABS_PREFIX_PREFIX})

# 设置 OpenGL 策略以消除警告
cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE GLVND)

#######################
## Find dependencies ##
#######################
find_package(ament_cmake_auto REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)

# 查找本地安装的 OpenVINO
set(OpenVINO_DIR "${ABS_PREFIX_PREFIX}/lib/cmake/openvino2024.3.0")
find_package(OpenVINO QUIET)

if(NOT OpenVINO_FOUND)
    message(ERROR "OpenVINO not found via CMake")
endif()

#set(CMAKE_FIND_DEBUG_MODE TRUE)


ament_auto_find_build_dependencies()
if(NOT g2o_FOUND OR NOT OpenCV_FOUND OR NOT fmt_FOUND OR NOT OpenVINO_FOUND)
  message(ERROR "g2o, OpenCV, fmt or OpenVINO not found, skipping ${PROJECT_NAME}")
  return()
else()
  message(STATUS "g2o, OpenCV, fmt and OpenVINO found, building ${PROJECT_NAME}")
endif()

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS} ${g2o_INCLUDE_DIRS}
  ${ABS_PREFIX_PREFIX}/include
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  g2o::core
  g2o::stuff
  g2o::solver_csparse
  g2o::types_sba
  g2o::types_slam3d
  g2o::solver_dense
  fmt::fmt
  tbb
  openvino::runtime
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_aim::ArmorDetectorNode
  EXECUTABLE armor_detector_node
)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_uncrustify
    ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest)
  ament_add_gtest(test_node_startup test/test_node_startup.cpp)
  target_link_libraries(test_node_startup ${PROJECT_NAME})

  ament_add_gtest(test_number_cls test/test_number_cls.cpp)
  target_link_libraries(test_number_cls ${PROJECT_NAME})

endif()

#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
  model
)
