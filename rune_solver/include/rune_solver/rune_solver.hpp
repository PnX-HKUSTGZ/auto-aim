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

#ifndef RUNE_SOLVER_RUNE_SOLVER_HPP_
#define RUNE_SOLVER_RUNE_SOLVER_HPP_

// std
#include <algorithm>
#include <array>
#include <deque>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>
// ros2
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/clock.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <std_msgs/msg/float32.hpp>
// project
#include "auto_aim_interfaces/msg/rune.hpp"
#include "auto_aim_interfaces/msg/rune_target.hpp"
#include "rune_solver/extended_kalman_filter.hpp"
#include "rune_solver/pnp_solver.hpp"
#include "rune_solver/curve_fitter.hpp"
#include "rune_solver/motion_model.hpp"
#include "rune_solver/types.hpp"

namespace rm_auto_aim {

// Usage: 
//   1. init(msg), if tracker_state == LOST
//   2. update(msg), if tracker_state == DETECTING or TRACKING
//   3. p = predictTarget(timestamp), to get the predicted position
//   4. cmd = solveGimbalCmd(p), to get the gimbal command
class RuneSolver {
public:
  struct RuneSolverParams {
    std::string compensator_type;
    double gravity;
    double bullet_speed;
    double angle_offset_thres;
    double lost_time_thres;
    bool auto_type_determined;
  };

  enum State {
    LOST,
    DETECTING,
    TRACKING,
  } tracker_state;

  RuneSolver(const RuneSolverParams &sr_params, std::shared_ptr<tf2_ros::Buffer> tf2_buffer);

  // Return: initial angle
  double init(const auto_aim_interfaces::msg::Rune::SharedPtr received_target);

  // Return: normalized angle
  double update(const auto_aim_interfaces::msg::Rune::SharedPtr receive_target);

  // Return: transormation matrix from rune to odom
  // Throws: tf2::TransformException or std::runtime_error
  Eigen::Matrix4d solvePose(const auto_aim_interfaces::msg::Rune &target);

  // Return: 3d position of R tag
  Eigen::Vector3d getCenterPosition() const;

  // Param: angle_diff: how much the angle target should prerotate, 0 for no prediction
  // Return: 3d position of target to be aimed at
  Eigen::Vector3d getTargetPosition(double angle_diff) const; 
  void pubTargetPosition(auto_aim_interfaces::msg::RuneTarget &rune_target) const;

  double getCurAngle() const;

  // Solvers
  std::unique_ptr<PnPSolver> pnp_solver;
  std::unique_ptr<CurveFitter> curve_fitter;
  std::unique_ptr<RuneCenterEKF> ekf;

  RuneSolverParams rune_solver_params;

private:
  double getNormalAngle(const auto_aim_interfaces::msg::Rune::SharedPtr received_target);

  double getObservedAngle(double normal_angle);

  // Return the centroid of the input armor points
  cv::Point2f getCenterPoint(const std::array<cv::Point2f, ARMOR_KEYPOINTS_NUM> &armor_points);

  // Return ekf state
  Eigen::Vector4d getStateFromTransform(const Eigen::Matrix4d &transform) const;

  // Observation data

  // last_observed_angle_ is continuously increasing (or decreasing)
  // from the first detection (call init()) of the target without
  // any abrupt change in between.
  double last_observed_angle_;

  // last_angle_ would change (N * DEG_72) when the target jumps
  double last_angle_;
  double start_time_;
  double last_time_;

  Eigen::Vector4d ekf_state_;

  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  
  // Convert euler angles to rotation matrix
  enum class EulerOrder { XYZ, XZY, YXZ, YZX, ZXY, ZYX };
  Eigen::Matrix3d eulerToMatrix(const Eigen::Vector3d &euler, EulerOrder order = EulerOrder::XYZ) const {
    auto r = Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX());
    auto p = Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY());
    auto y = Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ());
    switch (order) {
      case EulerOrder::XYZ:
        return (y * p * r).matrix();
      case EulerOrder::XZY:
        return (p * y * r).matrix();
      case EulerOrder::YXZ:
        return (y * r * p).matrix();
      case EulerOrder::YZX:
        return (r * y * p).matrix();
      case EulerOrder::ZXY:
        return (p * r * y).matrix();
      case EulerOrder::ZYX:
        return (r * p * y).matrix();
      default:
        return Eigen::Matrix3d::Identity();
    }
  }
};

}  // namespace rm_auto_aim
#endif // RUNE_SOLVER_SOLVER_HPP_
