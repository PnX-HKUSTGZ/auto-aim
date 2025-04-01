// Created by Labor 2023.8.25
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

#include "armor_detector/graph_optimizer.hpp"
// std
#include <algorithm>
// third party
#include <Eigen/Core>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <rclcpp/clock.hpp>
#include <sophus/so3.hpp>
// project
#include "armor_detector/armor.hpp"

namespace rm_auto_aim {

void VertexYaw::oplusImpl(const double *update) {
  _estimate += update[0];
}

EdgeProjection::EdgeProjection(const Sophus::SO3d &R_camera_imu,
                               const Eigen::Vector3d &t,
                               const Eigen::Matrix3d &K)
    : R_camera_imu_(R_camera_imu), t_(t), K_(K) {
  M_ = K_ * (R_camera_imu_).matrix();
  vt_ = K_ * t_;
}

void EdgeProjection::computeError() {
  // 获取 yaw 角
  double yaw = static_cast<VertexYaw *>(_vertices[0])->estimate();

  // 计算绕 Z 轴旋转的旋转矩阵（闭式计算）
  double cy = std::cos(yaw), sy = std::sin(yaw);
  Eigen::Matrix3d Rz;
  Rz << cy, -sy, 0,
        sy,  cy, 0,
         0,   0, 1;

  // 获取 3D 点
  const Eigen::Vector3d &p_3d = static_cast<g2o::VertexPointXYZ *>(_vertices[1])->estimate();
  const Eigen::Vector2d &obs = _measurement;

  // 利用预计算结果减少乘法次数
  Eigen::Vector3d p = M_ * (Rz * p_3d) + vt_;
  p /= p.z();

  // 计算重投影误差
  _error = obs - p.head<2>();
}

} // namespace rm_auto_aim