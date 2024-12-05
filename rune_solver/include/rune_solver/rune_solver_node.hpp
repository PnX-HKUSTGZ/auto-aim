// Maintained by Chengfu Zou, Labor
// Copyright (C) FYT Vision Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef RUNE_SOLVER_RUNE_SOLVER_NODE_HPP_
#define RUNE_SOLVER_RUNE_SOLVER_NODE_HPP_

// std
#include <algorithm>
#include <auto_aim_interfaces/msg/detail/rune_target__struct.hpp>
#include <deque>
#include <iostream>
#include <auto_aim_interfaces/msg/detail/debug_rune_angle__struct.hpp>
#include <string>
#include <tuple>
#include <vector>
// ros2
#include <geometry_msgs/msg/point_stamped.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
// third party
#include <opencv2/opencv.hpp>
// project
#include "auto_aim_interfaces/msg/debug_rune_angle.hpp"
#include "auto_aim_interfaces/msg/rune_target.hpp"
#include "auto_aim_interfaces/msg/rune.hpp"
#include "auto_aim_interfaces/srv/set_mode.hpp"
#include "rune_solver/rune_solver.hpp"

namespace rm_auto_aim {
class RuneSolverNode : public rclcpp::Node {
public:
  RuneSolverNode(const rclcpp::NodeOptions &options);

private:
  void runeTargetCallback(const auto_aim_interfaces::msg::Rune::SharedPtr rune_target_msg);

  void setModeCallback(const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
                       std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response);

  // Rune solver
  std::unique_ptr<RuneSolver> rune_solver_;
  double predict_offset_;

  // Tf message
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;

  // Target Subscriber
  rclcpp::Subscription<auto_aim_interfaces::msg::Rune>::SharedPtr rune_target_sub_;
  auto_aim_interfaces::msg::Rune last_rune_target_;

  // Target publisher
  rclcpp::Publisher<auto_aim_interfaces::msg::RuneTarget>::SharedPtr rune_target_pub_;

  // Predict Target publisher
  rclcpp::TimerBase::SharedPtr pub_timer_;
  void timerCallback();

  // Enable/Disable Rune Solver
  bool enable_;
  rclcpp::Service<auto_aim_interfaces::srv::SetMode>::SharedPtr set_mode_srv_;

  // Dynamic Parameter
  rcl_interfaces::msg::SetParametersResult onSetParameters(
    std::vector<rclcpp::Parameter> parameters);
  rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr on_set_parameters_callback_handle_;

  // Camera info part
  std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;

  // Debug info
  bool debug_;
  rclcpp::Publisher<auto_aim_interfaces::msg::DebugRuneAngle>::SharedPtr observed_angle_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fitter_text_pub_;
  visualization_msgs::msg::Marker obs_pos_marker_;
  visualization_msgs::msg::Marker r_tag_pos_marker_;
  visualization_msgs::msg::Marker pred_pos_marker_;
  visualization_msgs::msg::Marker aimming_line_marker_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_; 
  rclcpp::Time stamp;
};
}  // namespace rm_auto_aim
#endif
