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

#include "rune_solver/rune_solver.hpp"

// ros2
#include <cv_bridge/cv_bridge.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/logging.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// std
#include <memory>
// third party
#include <angles/angles.h>

#include <Eigen/Geometry>
// project
#include "rune_solver/types.hpp"

namespace rm_auto_aim {

// 构造函数，初始化 RuneSolver
RuneSolver::RuneSolver(const RuneSolverParams &rsp, std::shared_ptr<tf2_ros::Buffer> buffer)
: rune_solver_params(rsp), tf2_buffer_(buffer) {
  // 初始化状态
  tracker_state = LOST;
  curve_fitter = std::make_unique<CurveFitter>(MotionType::UNKNOWN);
  curve_fitter->setAutoTypeDetermined(rsp.auto_type_determined);
  ekf_state_ = Eigen::Vector4d::Zero();
}

// 初始化 RuneSolver
double RuneSolver::init(const auto_aim_interfaces::msg::Rune::SharedPtr received_target) {
  if (received_target->is_lost) {
    return 0;
  }

  RCLCPP_INFO(rclcpp::get_logger("rune_solver"), "Init!");

  // 初始化 EKF
  try {
    Eigen::Matrix4d t_odom_2_rune = solvePose(*received_target); //pnp解算

    // 过滤掉异常值
    Eigen::Vector3d t = t_odom_2_rune.block(0, 3, 3, 1);
    if (t.norm() < MIN_RUNE_DISTANCE || t.norm() > MAX_RUNE_DISTANCE) {
      RCLCPP_ERROR(rclcpp::get_logger("rune_solver"), "Rune position is out of range");
      return 0;
    }

    ekf_state_ = getStateFromTransform(t_odom_2_rune);
    ekf->setState(ekf_state_);
  } catch (...) {
    RCLCPP_ERROR(rclcpp::get_logger("rune_solver"), "EKF init failed");
    return 0;
  }

  // 初始化观测变量
  tracker_state = DETECTING;
  double observed_angle = getNormalAngle(received_target);
  double observed_time = 0;
  curve_fitter->update(observed_time, observed_angle);

  last_observed_angle_ = observed_angle;
  last_angle_ = last_observed_angle_;
  start_time_ = rclcpp::Time(received_target->header.stamp).seconds();
  last_time_ = start_time_;

  return observed_angle;
}

// 更新 RuneSolver
double RuneSolver::update(const auto_aim_interfaces::msg::Rune::SharedPtr received_target) {
  double now_time = rclcpp::Time(received_target->header.stamp).seconds();
  double delta_time = now_time - last_time_;

  if (received_target->is_big_rune) {
    curve_fitter->setType(MotionType::BIG);
  } else {
    curve_fitter->setType(MotionType::SMALL);
  }

  if (!received_target->is_lost) {
    // 更新 EKF
    try {
      Eigen::Matrix4d t_odom_2_rune = solvePose(*received_target);

      // 过滤掉异常值
      Eigen::Vector3d t = t_odom_2_rune.block(0, 3, 3, 1);
      if (t.norm() < MIN_RUNE_DISTANCE || t.norm() > MAX_RUNE_DISTANCE) {
        RCLCPP_ERROR(rclcpp::get_logger("rune_solver"), "Rune position is out of range");
        return 0;
      }

      Eigen::Vector4d measurement = getStateFromTransform(t_odom_2_rune);
      ekf->predict();
      ekf_state_ = ekf->update(measurement);
    } catch (...) {
      RCLCPP_ERROR(rclcpp::get_logger("rune_solver"), "EKF update failed");
      return 0;
    }

    // 获取拟合数据
    double observed_time = now_time - start_time_;
    double normal_angle = getNormalAngle(received_target);
    double observed_angle = getObservedAngle(normal_angle);

    // 更新拟合器
    curve_fitter->update(observed_time, observed_angle);

    last_time_ = now_time;
    last_angle_ = normal_angle;
    last_observed_angle_ = observed_angle;
  }

  // 更新跟踪器状态
  switch (tracker_state) {
    case DETECTING: {
      if (received_target->is_lost && delta_time > rune_solver_params.lost_time_thres) {
        tracker_state = LOST;
        curve_fitter->reset();
      } else if (curve_fitter->statusVerified()) {
        tracker_state = TRACKING;
      }
      break;
    }
    case TRACKING: {
      if (received_target->is_lost && delta_time > rune_solver_params.lost_time_thres) {
        tracker_state = LOST;
        curve_fitter->reset();
      }
      break;
    }
    case LOST: {
      if (!received_target->is_lost) {
        tracker_state = DETECTING;
      }
      break;
    }
  }
  return last_observed_angle_;
}

// pnp计算姿态
Eigen::Matrix4d RuneSolver::solvePose(const auto_aim_interfaces::msg::Rune &predicted_target) {
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  std::vector<cv::Point2f> image_points(predicted_target.pts.size());
  std::transform(predicted_target.pts.begin(),
                 predicted_target.pts.end(),
                 image_points.begin(),
                 [](const auto &pt) { return cv::Point2f(pt.x, pt.y); });

  cv::Mat rvec(3, 1, CV_64F), tvec(3, 1, CV_64F);
  if (pnp_solver && pnp_solver->solvePnP(image_points, rvec, tvec, "rune")) {
    // 获取从 rune 到 odom 的变换矩阵
    try {
      // 从 rvec 获取旋转矩阵
      cv::Mat rmat;
      cv::Rodrigues(rvec, rmat);
      Eigen::Matrix3d rot;
      rot << rmat.at<double>(0, 0), rmat.at<double>(0, 1), rmat.at<double>(0, 2),
             rmat.at<double>(1, 0), rmat.at<double>(1, 1), rmat.at<double>(1, 2), 
             rmat.at<double>(2, 0), rmat.at<double>(2, 1), rmat.at<double>(2, 2);
      Eigen::Quaterniond quat(rot);

      // 初始化姿态消息
      geometry_msgs::msg::PoseStamped ps;
      ps.header.frame_id = "camera_optical_frame";
      ps.header.stamp = predicted_target.header.stamp;

      // 填充姿态消息
      ps.pose.orientation.x = quat.x();
      ps.pose.orientation.y = quat.y();
      ps.pose.orientation.z = quat.z();
      ps.pose.orientation.w = quat.w();
      ps.pose.position.x = tvec.at<double>(0);
      ps.pose.position.y = tvec.at<double>(1);
      ps.pose.position.z = tvec.at<double>(2);

      // 转换到 odom 坐标系
      ps = tf2_buffer_->transform(ps, "odom");
      double roll, pitch, yaw;
      tf2::Quaternion quat_tf(ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w);
      tf2::Matrix3x3(quat_tf).getRPY(roll, pitch, yaw);

      // 填充姿态
      pose(0, 3) = ps.pose.position.x;
      pose(1, 3) = ps.pose.position.y;
      pose(2, 3) = ps.pose.position.z;

      Eigen::Quaterniond quat_odom;
      quat_odom.x() = ps.pose.orientation.x;
      quat_odom.y() = ps.pose.orientation.y;
      quat_odom.z() = ps.pose.orientation.z;
      quat_odom.w() = ps.pose.orientation.w;

      Eigen::Matrix3d rot_odom = quat_odom.toRotationMatrix();
      pose.block(0, 0, 3, 3) = rot_odom;

    } catch (tf2::TransformException &ex) {
      RCLCPP_ERROR(rclcpp::get_logger("rune_solver"), "%s", ex.what());
      throw ex;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("rune_solver"), "PnP failed");
    throw std::runtime_error("PnP failed");
  }
  return pose;
}

// 获取roll角
double RuneSolver::getNormalAngle(const auto_aim_interfaces::msg::Rune::SharedPtr received_target) {
  auto center_point = cv::Point2f(received_target->pts[0].x, received_target->pts[0].y);
  std::array<cv::Point2f, ARMOR_KEYPOINTS_NUM> armor_points;
  std::transform(received_target->pts.begin() + 3,
                 received_target->pts.end(),
                 armor_points.begin(),
                 [](const auto &pt) { return cv::Point2f(pt.x, pt.y); });

  cv::Point2f armor_center = getCenterPoint(armor_points);
  double x_diff = armor_center.x - center_point.x;
  double y_diff = -(armor_center.y - center_point.y);
  double normal_angle = std::atan2(y_diff, x_diff);
  // 归一化角度
  normal_angle = angles::normalize_angle_positive(normal_angle);

  return normal_angle;
}

// 处理跳变
double RuneSolver::getObservedAngle(double normal_angle) {
  double angle_diff = angles::shortest_angular_distance(last_angle_, normal_angle);
  // 处理符文目标切换
  if (std::abs(angle_diff) > rune_solver_params.angle_offset_thres) {
    angle_diff = normal_angle - last_angle_;
    int offset = std::round(double(angle_diff / DEG_72));
    angle_diff -= offset * DEG_72;
  }

  double observed_angle = last_observed_angle_ + angle_diff;

  return observed_angle;
}

// 获取中心位置
Eigen::Vector3d RuneSolver::getCenterPosition() const { return ekf_state_.head(3); }

// 发布目标位置
void RuneSolver::pubTargetPosition(auto_aim_interfaces::msg::RuneTarget &rune_target) const {
  rune_target.center.x = ekf_state_(0);
  rune_target.center.y = ekf_state_(1);
  rune_target.center.z = ekf_state_(2);
  rune_target.yaw = ekf_state_(3); 
  rune_target.roll = -last_angle_; 
  rune_target.tracking = tracker_state == TRACKING;
  rune_target.is_big = curve_fitter->getType() == MotionType::BIG; 
  rune_target.fitting_curve[0] = curve_fitter->getFittingParam()[0]; 
  rune_target.fitting_curve[1] = curve_fitter->getFittingParam()[1];
  rune_target.fitting_curve[2] = curve_fitter->getFittingParam()[2];
  rune_target.fitting_curve[3] = curve_fitter->getFittingParam()[3];
  rune_target.fitting_curve[4] = curve_fitter->getFittingParam()[4];
  rune_target.direction = curve_fitter->getDirection();
  rune_target.start_time = start_time_; 

  return; 
}

// 获取目标位置
Eigen::Vector3d RuneSolver::getTargetPosition(double angle_diff) const {
  Eigen::Vector3d t_odom_2_rune = ekf_state_.head(3);

  // 考虑到从 PnP 获取的方向存在较大的误差和抖动，
  // 并且符的位置在 odom 坐标系中是静态的，
  // 建议使用几何信息重新构建旋转矩阵
  double yaw = ekf_state_(3);
  double pitch = 0;
  double roll = -last_angle_;
  Eigen::Matrix3d R_odom_2_rune =
    eulerToMatrix(Eigen::Vector3d{roll, pitch, yaw}, EulerOrder::XYZ);

  // 计算符文坐标系中的装甲位置
  Eigen::Vector3d p_rune = Eigen::AngleAxisd(-angle_diff, Eigen::Vector3d::UnitX()).matrix() *
                           Eigen::Vector3d(0, -ARM_LENGTH, 0);

  // 转换到 odom 坐标系
  Eigen::Vector3d p_odom = R_odom_2_rune * p_rune + t_odom_2_rune;

  return p_odom;
}

// 从变换矩阵获取状态
Eigen::Vector4d RuneSolver::getStateFromTransform(const Eigen::Matrix4d &transform) const {
  // 获取 yaw 角
  Eigen::Matrix3d r_odom_2_rune = transform.block(0, 0, 3, 3);
  Eigen::Quaterniond q_eigen = Eigen::Quaterniond(r_odom_2_rune);
  tf2::Quaternion q_tf = tf2::Quaternion(q_eigen.x(), q_eigen.y(), q_eigen.z(), q_eigen.w());
  double roll, pitch, yaw;
  tf2::Matrix3x3(q_tf).getRPY(roll, pitch, yaw);
  yaw = angles::normalize_angle(yaw);

  // 使 yaw 连续
  yaw = ekf_state_(3) + angles::shortest_angular_distance(ekf_state_(3), yaw);

  Eigen::Vector4d state;
  state << transform(0, 3), transform(1, 3), transform(2, 3), yaw;
  return state;
}

// 获取当前角度
double RuneSolver::getCurAngle() const { return last_angle_; }

// 获取中心点
cv::Point2f RuneSolver::getCenterPoint(
  const std::array<cv::Point2f, ARMOR_KEYPOINTS_NUM> &armor_points) {
  return std::accumulate(armor_points.begin(), armor_points.end(), cv::Point2f(0, 0)) /
         ARMOR_KEYPOINTS_NUM;
}

}  // namespace rm_auto_aim