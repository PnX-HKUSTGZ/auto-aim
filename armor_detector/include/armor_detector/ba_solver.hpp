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

#ifndef ARMOR_DETECTOR_BA_SOLVER_HPP_
#define ARMOR_DETECTOR_BA_SOLVER_HPP_

// std
#include <array>
#include <cstddef>
#include <tuple>
#include <vector>
#include <thread>
#include <mutex>
// 3rd party
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <sophus/so3.hpp>
#include <std_msgs/msg/float32.hpp>
// g2o
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/sparse_optimizer.h>
// project
#include "armor_detector/graph_optimizer.hpp"
#include "armor_detector/armor.hpp"
#include "armor_detector/thread_pool.hpp"

namespace rm_auto_aim {

// BA algorithm based Optimizer for the armor pose estimation (Particularly for
// the Yaw angle)
class BaSolver {
public:
  BaSolver(const std::array<double, 9> &camera_matrix,
           const std::vector<double> &dist_coeffs);

  // Solve the armor pose using the BA algorithm, return the optimized rotation
  void solveBa(Armor &armor, 
               const Eigen::Matrix3d &R_odom_to_camera,
                const Eigen::Vector3d &t_odom_to_camera) noexcept;
  
  // 单线程版本的BA求解方法 (多线程实现内部使用)
  void solveBaSingleThread(Armor &armor, 
                         const Eigen::Matrix3d &R_odom_to_camera,
                         const Eigen::Vector3d &t_odom_to_camera) noexcept;
  
  void solveTwoArmorsBa(const double &yaw1, const double &yaw2, const double &z1, const double &z2, 
                        double &x, double &y, double &r1, double &r2,
                        const std::vector<cv::Point2f> &landmarks, 
                        const Eigen::Matrix3d &R_odom_to_camera, 
                        std::string number, ArmorType type);

  bool fixTwoArmors(Armor &armor1, Armor &armor2, const Eigen::Matrix3d &R_odom_to_camera);
  
  // 初始化线程池，用于多线程优化
  void startThreadPoolIfNeeded(int num_threads = 4);

private:
  Eigen::Matrix3d K_;
  g2o::SparseOptimizer optimizer_;
  g2o::OptimizationAlgorithmProperty solver_property_;
  g2o::OptimizationAlgorithmLevenberg *lm_algorithm_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  double shortest_angular_distance(double a1, double a2);
  
  // 线程池相关成员
  std::shared_ptr<ThreadPool> thread_pool_;
  std::mutex solver_mutex_; // 保护优化器访问的互斥锁
};

} // namespace rm_auto_aim
#endif // ARMOR_DETECTOR_BAS_SOLVER_HPP_