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
#include <g2o/types/slam3d/vertex_pointxyz.h>

#include <Eigen/Core>
#include <rclcpp/clock.hpp>
#include <sophus/so3.hpp>
// project
#include "armor_detector/types.hpp"

namespace rm_auto_aim
{

void VertexYaw::oplusImpl(const double * update) { _estimate += update[0]; }
EdgeProjection::EdgeProjection() {}
void EdgeProjection::setCameraPose(
    const Sophus::SO3d & R_odom_to_camera, const Eigen::Vector3d & t_camera_armor)
{
    // 计算相机坐标系到装甲板坐标系的变换矩阵
    K_ = R_odom_to_camera.matrix();
    t_ = t_camera_armor;
}
void EdgeProjection::computeError()
{
    // 获取 yaw 角
    double yaw = static_cast<VertexYaw *>(_vertices[0])->estimate();

    // 计算绕 Z 轴旋转的旋转矩阵（闭式计算）
    double cy = std::cos(yaw), sy = std::sin(yaw);
    Eigen::Matrix3d Rz;
    Rz << cy, -sy, 0, sy, cy, 0, 0, 0, 1;

    // 获取 3D 点
    const Eigen::Vector3d & p_3d = static_cast<g2o::VertexPointXYZ *>(_vertices[1])->estimate();
    const Eigen::Vector2d & obs = _measurement;

    // 利用预计算结果减少乘法次数
    Eigen::Vector3d p = K_ * (Rz * p_3d) + t_;
    p /= p.z();

    // 计算重投影误差
    _error = obs - p.head<2>();
}

void VertexR::oplusImpl(const double * update) { _estimate += update[0]; }

void VertexXY::oplusImpl(const double * update)
{
    _estimate += Eigen::Vector2d(update[0], update[1]);
}

EdgeTwoArmors::EdgeTwoArmors()
{
    // 设置边的维度
    resize(5);
}
void EdgeTwoArmors::setCameraPose(
    const Sophus::SO3d & R_odom_to_camera, const Eigen::Vector3d & t_odom_to_camera)
{
    // 计算相机坐标系到装甲板坐标系的变换矩阵
    K_ = R_odom_to_camera.matrix();
    t_ = t_odom_to_camera;
}
void EdgeTwoArmors::computeError()
{
    // 获取车辆中心点
    Eigen::Vector2d xy = static_cast<VertexXY *>(_vertices[0])->estimate();
    // 获取两个装甲板半径
    double r = static_cast<VertexR *>(_vertices[1])->estimate();
    // 获取yaw角
    double yaw = static_cast<FixedScalarVertex *>(_vertices[2])->estimate();
    // 获取z
    double z = static_cast<FixedScalarVertex *>(_vertices[3])->estimate();
    // 获取角点相对于装甲板中心的位置
    Eigen::Vector3d p = static_cast<g2o::VertexPointXYZ *>(_vertices[4])->estimate();
    //获取2D点
    Eigen::Vector2d obs = _measurement;
    // 计算重投影误差
    Eigen::Vector3d p_camera =
        K_ * (p + Eigen::Vector3d(xy.x() - r * cos(yaw), xy.y() - r * sin(yaw), z)) + t_;
    p_camera /= p_camera.z();
    _error = obs - p_camera.head<2>();
}

}  // namespace rm_auto_aim