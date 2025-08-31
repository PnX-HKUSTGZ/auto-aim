cmake_minimum_required(VERSION 3.8)
project(auto_aim_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Armor.msg"
  "msg/Armors.msg"
  "msg/Target.msg"
  "msg/Firecontrol.msg"

  "msg/TrackerInfo.msg"
  "msg/Point2d.msg"
  "msg/Point3d.msg"
  "msg/Rune.msg"
  "msg/RuneTarget.msg"
  "msg/DebugRuneAngle.msg"

  "srv/SetMode.srv"
  

  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
)


ament_package()
