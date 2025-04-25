// Created by Labor 2023.8.25
// Maintained by Labor, Chengfu Zou
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

#ifndef ARMOR_DETECTOR_GRAPH_OPTIMIZER_HPP_
#define ARMOR_DETECTOR_GRAPH_OPTIMIZER_HPP_

// std
#include <array>
// g2o
#include <g2o/core/auto_differentiation.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/base_multi_edge.h>
// 3rd party
#include <Eigen/Dense>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace rm_auto_aim {
// Vertex of graph optimization algorithm for the yaw angle
class VertexYaw : public g2o::BaseVertex<1, double> { 
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexYaw() = default;
   void setToOriginImpl() override { _estimate = 0; } // 重置
   void oplusImpl(const double *update) override; // 更新
   //不需要读写
   bool read(std::istream &in) override { return true; }
   bool write(std::ostream &out) const override { return true; }
};

class VertexXY : public g2o::BaseVertex<2, Eigen::Vector2d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  VertexXY() = default;
  void setToOriginImpl() override { _estimate = Eigen::Vector2d::Zero(); }
  void oplusImpl(const double *update) override;
  bool read(std::istream &in) override { return true; }
  bool write(std::ostream &out) const override { return true; }
};

class FixedScalarVertex : public g2o::BaseVertex<1, double> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  FixedScalarVertex() = default;
  void setToOriginImpl() override { _estimate = 0; }
  void oplusImpl(const double *update) override { _estimate += update[0]; }
  bool read(std::istream &in) override { return true; }
  bool write(std::ostream &out) const override { return true; }
};

class VertexR : public g2o::BaseVertex<1, double> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  VertexR() = default;
  void setToOriginImpl() override { _estimate = 0; }
  void oplusImpl(const double *update) override;
  bool read(std::istream &in) override { return true; }
  bool write(std::ostream &out) const override { return true; }
};

// Edge of graph optimization algorithm for reporjection error calculation using
// yaw angle and observation
// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class EdgeProjection : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexYaw,
                                                  g2o::VertexPointXYZ> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  using InfoMatrixType = Eigen::Matrix<double, 2, 2>;

  EdgeProjection(); 
  void setCameraPose(const Sophus::SO3d &R_odom_to_camera,
                 const Eigen::Vector3d &t_camera_armor); 
  void computeError() override; // 误差

  bool read(std::istream &in) override { return true; }
  bool write(std::ostream &out) const override { return true; }

private:
  Eigen::Vector3d t_;
  Eigen::Matrix3d K_;
};
// 误差模型 模板参数：观测值维度，类型
class EdgeTwoArmors : public g2o::BaseMultiEdge<2, Eigen::Vector2d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  void computeError() override;

  EdgeTwoArmors();
  void setCameraPose(const Sophus::SO3d &R_odom_to_camera); 

  bool read(std::istream &in) override { return true; }
  bool write(std::ostream &out) const override { return true; }

private:
  Eigen::Matrix3d K_;
};

} // namespace rm_auto_aim
#endif // ARMOR_DETECTOR_GRAPH_OPTIMIZER_HPP_