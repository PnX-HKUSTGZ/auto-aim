cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################
get_filename_component(ABS_PREFIX_PREFIX "../../../third_party_install" ABSOLUTE)
set(CMAKE_PREFIX_PREFIX ${ABS_PREFIX_PREFIX})
message(STATUS "ABS_PREFIX_PREFIX: ${ABS_PREFIX_PREFIX}")

# Add third_party_install to CMAKE_PREFIX_PATH
# list(APPEND CMAKE_PREFIX_PATH ${ABS_INSTALL_PREFIX}/lib/cmake/g2o)
# list(APPEND CMAKE_PREFIX_PATH ${ABS_INSTALL_PREFIX}/share/sophus/cmake)
# list(APPEND CMAKE_PREFIX_PATH ${ABS_INSTALL_PREFIX}/lib/cmake/Ceres)
# set(EXTERNAL_INCLUDE_DIRS ${G2O_INCLUDE_DIR})
# set(EXTERNAL_LIBS ${G2O_LIBRARIES})
list(APPEND CMAKE_PREFIX_PATH ${ABS_PREFIX_PREFIX})

find_package(ament_cmake_auto REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)

#set(CMAKE_FIND_DEBUG_MODE TRUE)


ament_auto_find_build_dependencies()
if(NOT g2o_FOUND OR NOT OpenCV_FOUND OR NOT fmt_FOUND)
  message(ERROR "g2o, OpenCV or fmt not found, skipping ${PROJECT_NAME}")
  return()
else ()
  message(STATUS "g2o, OpenCV and fmt found, building ${PROJECT_NAME}")
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
