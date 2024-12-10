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

#include "rune_solver/rune_solver_node.hpp"
// ros2
#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <rclcpp/qos.hpp>
// third party
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
// project
#include "rune_solver/pnp_solver.hpp"
#include "rune_solver/motion_model.hpp"

namespace rm_auto_aim {

// 构造函数
RuneSolverNode::RuneSolverNode(const rclcpp::NodeOptions &options) : Node("rune_solver", options) {
  RCLCPP_INFO(this->get_logger(), "Starting RuneSolverNode!");

  // 初始化 TF2 缓冲区和监听器
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  // 初始化 RuneSolver 参数
  auto rune_solver_params = RuneSolver::RuneSolverParams{
    .compensator_type = declare_parameter("compensator_type", "ideal"),
    .gravity = declare_parameter("gravity", 9.8),
    .bullet_speed = declare_parameter("bullet_speet", 28.0),
    .angle_offset_thres = declare_parameter("angle_offset_thres", 0.78), 
    .lost_time_thres = declare_parameter("lost_time_thres", 0.5),
    .auto_type_determined = declare_parameter("auto_type_determined", true),
  };
  rune_solver_ = std::make_unique<RuneSolver>(rune_solver_params, tf2_buffer_);

  // 初始化 EKF（扩展卡尔曼滤波器）用于滤波 R 标记的位置
  // 状态：x, y, z, yaw
  // 测量：x, y, z, yaw
  // f - 过程函数
  auto f = Predict();
  // h - 观测函数
  auto h = Measure();
  // update_Q - 过程噪声协方差矩阵
  std::vector<double> q_vec =
    declare_parameter("ekf.q", std::vector<double>{0.001, 0.001, 0.001, 0.001});
  auto u_q = [q_vec]() {
    Eigen::Matrix<double, X_N, X_N> q = Eigen::MatrixXd::Zero(4, 4);
    q.diagonal() << q_vec[0], q_vec[1], q_vec[2], q_vec[3];
    return q;
  };
  // update_R - 测量噪声协方差矩阵
  std::vector<double> r_vec = declare_parameter("ekf.r", std::vector<double>{0.1, 0.1, 0.1, 0.1});
  auto u_r = [r_vec](const Eigen::Matrix<double, Z_N, 1> &z) {
    Eigen::Matrix<double, Z_N, Z_N> r = Eigen::MatrixXd::Zero(4, 4);
    r.diagonal() << r_vec[0], r_vec[1], r_vec[2], r_vec[3];
    return r;
  };
  // P - 误差估计协方差矩阵
  Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(4, 4);
  rune_solver_->ekf = std::make_unique<RuneCenterEKF>(f, h, u_q, u_r, p0);

  // 订阅目标话题
  rune_target_sub_ = this->create_subscription<auto_aim_interfaces::msg::Rune>(
    "rune_detector/rune",
    rclcpp::SensorDataQoS(),
    std::bind(&RuneSolverNode::runeTargetCallback, this, std::placeholders::_1));

  // 设置动态参数回调
  on_set_parameters_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&RuneSolverNode::onSetParameters, this, std::placeholders::_1));

  // 订阅相机信息
  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "camera_info",
    rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
      cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
      rune_solver_->pnp_solver = std::make_unique<PnPSolver>(camera_info->k, camera_info->d);
      rune_solver_->pnp_solver->setObjectPoints("rune", RUNE_OBJECT_POINTS);
      cam_info_sub_.reset();
    });

  // 启用/禁用 Rune Solver 服务
  set_mode_srv_ = this->create_service<auto_aim_interfaces::srv::SetMode>(
    "rune_solver/set_mode",
    std::bind(
      &RuneSolverNode::setModeCallback, this, std::placeholders::_1, std::placeholders::_2));

  // Debug 信息
  debug_ = this->declare_parameter("debug", true);
  rune_target_pub_ = this->create_publisher<auto_aim_interfaces::msg::RuneTarget>(
    "rune_solver/rune_target", 10);
  if (debug_) {
    observed_angle_pub_ = this->create_publisher<auto_aim_interfaces::msg::DebugRuneAngle>(
      "rune_solver/observed_angle", rclcpp::SensorDataQoS());
    fitter_text_pub_ = this->create_publisher<std_msgs::msg::String>("rune_solver/fitting_info",
                                                                     rclcpp::SensorDataQoS());
    // Marker
    r_tag_pos_marker_.ns = "r_tag_position";
    r_tag_pos_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    r_tag_pos_marker_.scale.x = r_tag_pos_marker_.scale.y = r_tag_pos_marker_.scale.z = 0.15;
    r_tag_pos_marker_.text = "R";
    r_tag_pos_marker_.color.a = 1.0;
    r_tag_pos_marker_.color.r = 1.0;
    r_tag_pos_marker_.color.g = 1.0;
    obs_pos_marker_.ns = "observed_position";
    obs_pos_marker_.type = visualization_msgs::msg::Marker::SPHERE;
    obs_pos_marker_.scale.x = obs_pos_marker_.scale.y = obs_pos_marker_.scale.z = 0.308;
    obs_pos_marker_.color.a = 1.0;
    obs_pos_marker_.color.r = 1.0;
    pred_pos_marker_.ns = "predicted_position";
    pred_pos_marker_.type = visualization_msgs::msg::Marker::SPHERE;
    pred_pos_marker_.scale.x = pred_pos_marker_.scale.y = pred_pos_marker_.scale.z = 0.308;
    pred_pos_marker_.color.a = 1.0;
    pred_pos_marker_.color.g = 1.0;
    aimming_line_marker_.ns = "aimming_line";
    aimming_line_marker_.type = visualization_msgs::msg::Marker::ARROW;
    aimming_line_marker_.scale.x = 0.03;
    aimming_line_marker_.scale.y = 0.05;
    aimming_line_marker_.color.a = 0.5;
    aimming_line_marker_.color.r = 1.0;
    aimming_line_marker_.color.b = 1.0;
    aimming_line_marker_.color.g = 1.0;
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "rune_solver/marker", rclcpp::SensorDataQoS());
  }
  last_rune_target_.header.frame_id = "";
  // 定时器 250 Hz
  pub_timer_ = this->create_wall_timer(std::chrono::milliseconds(4),
                                       std::bind(&RuneSolverNode::timerCallback, this));
}

// 定时器回调函数
void RuneSolverNode::timerCallback() {
  // 如果未接收到相机信息，返回
  if (rune_solver_->pnp_solver == nullptr) {
    return;
  }

  // 如果未启用，返回
  if (!enable_) {
    return;
  }

  // 初始化消息
  geometry_msgs::msg::PointStamped target_msg;
  target_msg.header.frame_id = "odom";
  Eigen::Vector3d cur_pos = rune_solver_->getTargetPosition(0);
  auto_aim_interfaces::msg::RuneTarget rune_target; 
  rune_target.header.stamp = stamp; 
  rune_target.header.frame_id = "odom";
  rune_solver_->pubTargetPosition(rune_target);

  if (debug_) {
    // 发布拟合信息
    std_msgs::msg::String fitter_text_msg;
    fitter_text_pub_->publish(fitter_text_msg);

    // 发布可视化标记
    visualization_msgs::msg::MarkerArray marker_array;
    if (rune_solver_->tracker_state == RuneSolver::LOST) {
      obs_pos_marker_.action = visualization_msgs::msg::Marker::DELETEALL;
      pred_pos_marker_.action = visualization_msgs::msg::Marker::DELETEALL;
      r_tag_pos_marker_.action = visualization_msgs::msg::Marker::DELETEALL;
      aimming_line_marker_.action = visualization_msgs::msg::Marker::DELETEALL;
      marker_array.markers.push_back(obs_pos_marker_);
      marker_array.markers.push_back(pred_pos_marker_);
      marker_array.markers.push_back(r_tag_pos_marker_);
      marker_array.markers.push_back(aimming_line_marker_);
      marker_pub_->publish(marker_array);
    } else {
      obs_pos_marker_.header.frame_id = "odom";
      obs_pos_marker_.header.stamp = last_rune_target_.header.stamp;
      obs_pos_marker_.action = visualization_msgs::msg::Marker::ADD;
      obs_pos_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);
      obs_pos_marker_.pose.position.x = cur_pos.x();
      obs_pos_marker_.pose.position.y = cur_pos.y();
      obs_pos_marker_.pose.position.z = cur_pos.z();

      Eigen::Vector3d r_tag_pos = rune_solver_->getCenterPosition();
      r_tag_pos_marker_.header.frame_id = "odom";
      r_tag_pos_marker_.header.stamp = last_rune_target_.header.stamp;
      r_tag_pos_marker_.action = visualization_msgs::msg::Marker::ADD;
      r_tag_pos_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);
      r_tag_pos_marker_.pose.position.x = r_tag_pos.x();
      r_tag_pos_marker_.pose.position.y = r_tag_pos.y();
      r_tag_pos_marker_.pose.position.z = r_tag_pos.z();

      marker_array.markers.push_back(obs_pos_marker_);
      marker_array.markers.push_back(r_tag_pos_marker_);
      marker_pub_->publish(marker_array);
    }
  }
}

// 目标话题回调函数
void RuneSolverNode::runeTargetCallback(
  const auto_aim_interfaces::msg::Rune::SharedPtr rune_target_msg) {
  stamp = rune_target_msg->header.stamp;
  // 如果未接收到相机信息，返回
  if (rune_solver_->pnp_solver == nullptr) {
    return;
  }

  // 保留最后检测到的目标
  if (!rune_target_msg->is_lost) {
    last_rune_target_ = *rune_target_msg;
  }
  double observed_angle = 0;
  if (rune_solver_->tracker_state == RuneSolver::LOST) {
    observed_angle = rune_solver_->init(rune_target_msg);
  } else {
    observed_angle = rune_solver_->update(rune_target_msg);

    if (debug_) {
      auto_aim_interfaces::msg::DebugRuneAngle observed_angle_msg;
      observed_angle_msg.header = rune_target_msg->header;
      observed_angle_msg.data = observed_angle;
      observed_angle_pub_->publish(observed_angle_msg);
    }
  }
}

// 动态参数回调函数
rcl_interfaces::msg::SetParametersResult RuneSolverNode::onSetParameters(
  std::vector<rclcpp::Parameter> parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto &param : parameters) {
    if (param.get_name() == "predict_time") {
      predict_offset_ = param.as_double();
    } else if (param.get_name() == "debug") {
      debug_ = param.as_bool();
    } else if (param.get_name() == "gravity") {
      rune_solver_->rune_solver_params.gravity = param.as_double();
    } else if (param.get_name() == "bullet_speed") {
      rune_solver_->rune_solver_params.bullet_speed = param.as_double();
    } else if (param.get_name() == "angle_offset_thres") {
      rune_solver_->rune_solver_params.angle_offset_thres = param.as_double();
    } else if (param.get_name() == "lost_time_thres") {
      rune_solver_->rune_solver_params.lost_time_thres = param.as_double();
    }
  }
  return result;
}

// 设置模式服务回调函数
void RuneSolverNode::setModeCallback(
  const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
  std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response) {
  response->success = true;

  VisionMode mode = static_cast<VisionMode>(request->mode);
  std::string mode_name = visionModeToString(mode);
  if (mode_name == "UNKNOWN") {
    RCLCPP_ERROR(this->get_logger(), "Invalid mode: %d", request->mode);
    return;
  }

  switch (mode) {
    case VisionMode::SMALL_RUNE:
    case VisionMode::BIG_RUNE: {
      enable_ = true;
      break;
    }
    default: {
      enable_ = false;
      break;
    }
  }

  RCLCPP_WARN(this->get_logger(), "Set Rune Mode: %s", visionModeToString(mode).c_str());
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// 注册组件
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::RuneSolverNode)