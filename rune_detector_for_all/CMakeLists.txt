cmake_minimum_required(VERSION 3.8)
project(rune_detector_for_all)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Tiger Core 静态库要链接成组件共享库，需开启全局 PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ================= 查找依赖 =================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(auto_aim_interfaces REQUIRED)
find_package(angles REQUIRED)

# 第三方依赖 (透传给 Tiger Core)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

# ================= 集成 Tiger Core =================
# 使用 add_subdirectory 而非 GLOB，保留其模块化结构
# 设置输出路径以避免污染源码或与其他构建冲突
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# 禁用 HNT 的 Example 和测试
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/rm_vision_core)

# ================= 构建节点 =================
# rclcpp_components 需要可加载的共享库
add_library(rune_detector_for_all_node SHARED
  src/rune_detector_node.cpp 
  src/curve_fitter.cpp
)

target_include_directories(rune_detector_for_all_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 链接库
# VisCore_detector_rune_detector 是 HNT 生成的目标
# 链接它会自动带上它所依赖的 common, math, core 等所有头文件路径和库
target_link_libraries(rune_detector_for_all_node
  VisCore_rune_detector
  # 如果 HNT 的依赖关系不是完全闭环的，可能还需要显式链接 core/feature 模块
  # 通常 VisCore 的模块宏会处理好 FIND_DEPENDENCIES ADD_DEPENDENCIES
  # 但保险起见，如果报错找不到符号，可以把下面的注释打开
  VisCore_core
  VisCore_rune_group
  
  ${OpenCV_LIBRARIES}
  Ceres::ceres
  Eigen3::Eigen
)

# 链接 ROS 依赖
ament_target_dependencies(rune_detector_for_all_node
  rclcpp
  rclcpp_components
  sensor_msgs
  image_transport
  cv_bridge
  tf2_ros
  tf2_geometry_msgs
  auto_aim_interfaces
  angles
)

# ================= 注册与安装 =================
rclcpp_components_register_node(rune_detector_for_all_node
  PLUGIN "rune_detector_for_all::RuneDetectorNode"
  EXECUTABLE rune_detector_node
)

# 只需要安装我们自己的节点，Tiger Core 已经静态链接进去了
install(TARGETS rune_detector_for_all_node rune_detector_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
