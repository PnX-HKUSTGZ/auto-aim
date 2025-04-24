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

#include "armor_detector/ba_solver.hpp"
// std
#include <memory>
#include <cmath>
// g2o
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/slam3d/types_slam3d.h>
// 3rd party
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <rclcpp/clock.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include "rclcpp/rclcpp.hpp"
#include <opencv2/core/eigen.hpp>
// project
#include "armor_detector/graph_optimizer.hpp"
#include "armor_detector/armor.hpp"

namespace rm_auto_aim {
G2O_USE_OPTIMIZATION_LIBRARY(dense)

BaSolver::BaSolver(const std::array<double, 9> &camera_matrix,
                   const std::vector<double> &dist_coeffs)
                 : camera_matrix_(cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone()),
                   dist_coeffs_(cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data())).clone()) {
  K_ = Eigen::Matrix3d::Identity();
  K_(0, 0) = camera_matrix[0];
  K_(1, 1) = camera_matrix[4];
  K_(0, 2) = camera_matrix[2];
  K_(1, 2) = camera_matrix[5];

  // Optimization information
  optimizer_.setVerbose(false);
  // Optimization method
  optimizer_.setAlgorithm(
      g2o::OptimizationAlgorithmFactory::instance()->construct(
          "lm_dense", solver_property_));
  // Initial step size
  lm_algorithm_ = dynamic_cast<g2o::OptimizationAlgorithmLevenberg *>(
      const_cast<g2o::OptimizationAlgorithm *>(optimizer_.algorithm()));
  lm_algorithm_->setUserLambdaInit(0.1);
}

void BaSolver::solveBa(Armor &armor, 
                  const Eigen::Matrix3d &R_odom_to_camera, 
                  const Eigen::Vector3d &t_odom_to_camera) noexcept {
  // 检查线程池是否已初始化，如果没有则使用单线程模式
  if (!thread_pool_) {
    // 如果线程池未初始化，使用原始的单线程实现
    solveBaSingleThread(armor, R_odom_to_camera, t_odom_to_camera);
    return;
  }

  // 使用线程池异步处理BA优化
  try {
    // 将BA优化任务提交到线程池
    auto future = thread_pool_->enqueue(
      [this, &armor, R_odom_to_camera, t_odom_to_camera]() {
        // 在线程中执行BA优化
        this->solveBaSingleThread(armor, R_odom_to_camera, t_odom_to_camera);
      }
    );
    
    // 等待任务完成
    future.get();
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("armor_detector"), "多线程BA优化失败: %s", e.what());
    // 如果多线程处理失败，回退到单线程处理
    solveBaSingleThread(armor, R_odom_to_camera, t_odom_to_camera);
  }
}

void BaSolver::solveBaSingleThread(Armor &armor, 
                  const Eigen::Matrix3d &R_odom_to_camera, 
                  const Eigen::Vector3d &t_odom_to_camera) noexcept {
  // Reset optimizer
  {
    std::lock_guard<std::mutex> lock(solver_mutex_);
    optimizer_.clear();
  }

  // Essential coordinate system transformation
  Eigen::Matrix3d R_camera_to_odom = R_odom_to_camera.inverse(); 
  Eigen::Vector3d t_camera_to_odom = -R_camera_to_odom * t_odom_to_camera;
  Eigen::Vector3d & t_camera_armor = armor.t_camera_armor;
  Eigen::Matrix3d & r_odom_armor = armor.r_odom_armor;
  Eigen::Vector3d & t_odom_armor = armor.t_odom_armor;

  // Compute the initial yaw from rotation matrix
  double initial_armor_yaw;
  auto theta_by_sin = std::asin(-r_odom_armor(0, 1));
  auto theta_by_cos = std::acos(r_odom_armor(1, 1));
  if (std::abs(theta_by_sin) > 1e-5) {
    initial_armor_yaw = theta_by_sin > 0 ? theta_by_cos : -theta_by_cos;
  } else {
    initial_armor_yaw = r_odom_armor(1, 1) > 0 ? 0 : M_PI;
  }

  // Get the pitch angle of the armor
  double armor_pitch =
      armor.number == "outpost" ? -0.2618 : 0.2618;

  // Get the 3D points of the armor
  const auto armor_size =
      armor.type == ArmorType::SMALL
          ? Eigen::Vector2d(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT)
          : Eigen::Vector2d(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT);
  const auto object_points =
      Armor::buildObjectPoints<Eigen::Vector3d>(armor_size(0), armor_size(1));

  // 使用本地优化器，避免多线程干扰
  g2o::SparseOptimizer local_optimizer;
  {
    std::lock_guard<std::mutex> lock(solver_mutex_);
    // 复制优化器设置
    local_optimizer.setVerbose(false);
    local_optimizer.setAlgorithm(
        g2o::OptimizationAlgorithmFactory::instance()->construct(
            "lm_dense", solver_property_));
    auto* local_lm_algorithm = dynamic_cast<g2o::OptimizationAlgorithmLevenberg *>(
        const_cast<g2o::OptimizationAlgorithm *>(local_optimizer.algorithm()));
    local_lm_algorithm->setUserLambdaInit(0.1);
  }

  // Fill the optimizer
  size_t id_counter = 0;
  Sophus::SO3d R_pitch = Sophus::SO3d::exp(Eigen::Vector3d(0, armor_pitch, 0)); 

  VertexYaw *v_yaw = new VertexYaw();
  v_yaw->setId(id_counter++);
  v_yaw->setEstimate(initial_armor_yaw);
  local_optimizer.addVertex(v_yaw);

  auto landmarks = armor.landmarks();
  auto normalized_landmarks = landmarks; 
  cv::undistortPoints(landmarks, normalized_landmarks, camera_matrix_, dist_coeffs_);
  for (size_t i = 0; i < 4; i++) {
    g2o::VertexPointXYZ *v_point = new g2o::VertexPointXYZ();
    v_point->setId(id_counter++);
    v_point->setEstimate(R_pitch * Eigen::Vector3d(
        object_points[i].x(), object_points[i].y(), object_points[i].z()));
    v_point->setFixed(true);
    local_optimizer.addVertex(v_point);

    EdgeProjection *edge =
        new EdgeProjection(R_odom_to_camera, t_camera_armor, K_);
    edge->setId(id_counter++);
    edge->setVertex(0, v_yaw);
    edge->setVertex(1, v_point);
    edge->setMeasurement(Eigen::Vector2d(normalized_landmarks[i].x, normalized_landmarks[i].y));
    edge->setInformation(EdgeProjection::InfoMatrixType::Identity());
    edge->setRobustKernel(new g2o::RobustKernelHuber);
    local_optimizer.addEdge(edge);
  }

  // 执行优化
  local_optimizer.initializeOptimization();
  local_optimizer.optimize(30);
  
  // Get yaw angle after optimization
  double yaw_optimized = v_yaw->estimate();

  if (std::isnan(yaw_optimized)) {
    RCLCPP_ERROR(rclcpp::get_logger("armor_detector"), "Yaw angle is nan after optimization"); 
    return;
  }
  
  Sophus::SO3d R_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw_optimized));
  r_odom_armor = (R_yaw * R_pitch).matrix(); 
  
  //通过面积比矫正距离
  double area_measure = 0, l = 0, w = 0;
  l = sqrt(pow(landmarks[0].x - landmarks[1].x, 2) + pow(landmarks[0].y - landmarks[1].y, 2)) + sqrt(pow(landmarks[2].x - landmarks[3].x, 2) + pow(landmarks[2].y - landmarks[3].y, 2));
  w = sqrt(pow(landmarks[1].x - landmarks[2].x, 2) + pow(landmarks[1].y - landmarks[2].y, 2)) + sqrt(pow(landmarks[3].x - landmarks[0].x, 2) + pow(landmarks[3].y - landmarks[0].y, 2));
  area_measure = l * w / 4;
  
  double area_expect = 0; 
  Eigen::Vector2d expect_points[4];
  for (size_t i = 0; i < 4; i++) {
    expect_points[i] = (K_ * (R_odom_to_camera * r_odom_armor * object_points[i] + t_camera_armor)).hnormalized();
  }
  
  l = sqrt(pow(expect_points[0].x() - expect_points[1].x(), 2) + pow(expect_points[0].y() - expect_points[1].y(), 2)) + sqrt(pow(expect_points[2].x() - expect_points[3].x(), 2) + pow(expect_points[2].y() - expect_points[3].y(), 2));
  w = sqrt(pow(expect_points[1].x() - expect_points[2].x(), 2) + pow(expect_points[1].y() - expect_points[2].y(), 2)) + sqrt(pow(expect_points[3].x() - expect_points[0].x(), 2) + pow(expect_points[3].y() - expect_points[0].y(), 2));
  area_expect = l * w / 4;
  t_camera_armor *= sqrt(area_expect / area_measure);
  t_odom_armor = R_camera_to_odom * t_camera_armor + t_camera_to_odom;
  
  return;
}

void BaSolver::solveTwoArmorsBa(const double &yaw1, const double &yaw2, const double &z1, const double &z2, 
                                double &x, double &y, double &r1, double &r2,
                                const std::vector<cv::Point2f> &landmarks, 
                                const Eigen::Matrix3d &R_odom_to_camera, 
                                std::string number, ArmorType type){
  // Reset optimizer
  optimizer_.clear();
  
  // 预计算三角函数值，避免在循环中重复计算
  const double cos_yaw1 = cos(yaw1);
  const double sin_yaw1 = sin(yaw1);
  const double cos_yaw2 = cos(yaw2);
  const double sin_yaw2 = sin(yaw2);
  
  // Get the pitch angle of the armor
  const double armor_pitch =
      number == "outpost" ? -0.2618 : 0.2618;
  Sophus::SO3d R_pitch = Sophus::SO3d::exp(Eigen::Vector3d(0, armor_pitch, 0)); 
  Sophus::SO3d R_yaw1 = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw1));
  Sophus::SO3d R_yaw2 = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw2));

  // Get the 3D points of the armor
  const auto armor_size =
      type == ArmorType::SMALL
          ? Eigen::Vector2d(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT)
          : Eigen::Vector2d(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT);
  const auto object_points_ =
      Armor::buildObjectPoints<Eigen::Vector3d>(armor_size(0), armor_size(1));
  
  // 预计算旋转后的点以避免在循环中计算
  Eigen::Vector3d object_points[8];
  for(size_t i = 0; i < 4; i++){
    object_points[i] = R_yaw1 * R_pitch * object_points_[i];
    object_points[i + 4] = R_yaw2 * R_pitch * object_points_[i];
  }
  
  // 需要优化的节点
  VertexXY *v_xy = new VertexXY();
  v_xy->setId(0);
  v_xy->setEstimate(Eigen::Vector2d(x, y));
  optimizer_.addVertex(v_xy);
  
  VertexR *v_r1 = new VertexR();
  v_r1->setId(1);
  v_r1->setEstimate(r1);
  optimizer_.addVertex(v_r1);
  
  VertexR *v_r2 = new VertexR();
  v_r2->setId(2);
  v_r2->setEstimate(r2);
  optimizer_.addVertex(v_r2);
  
  // 固定节点 - 使用预计算的值
  FixedScalarVertex *cos_v_yaw1 = new FixedScalarVertex();
  cos_v_yaw1->setId(3);
  cos_v_yaw1->setEstimate(cos_yaw1);
  cos_v_yaw1->setFixed(true);
  optimizer_.addVertex(cos_v_yaw1);
  
  FixedScalarVertex *sin_v_yaw1 = new FixedScalarVertex();
  sin_v_yaw1->setId(4);
  sin_v_yaw1->setEstimate(sin_yaw1);
  sin_v_yaw1->setFixed(true);
  optimizer_.addVertex(sin_v_yaw1);
  
  FixedScalarVertex *cos_v_yaw2 = new FixedScalarVertex();
  cos_v_yaw2->setId(5); 
  cos_v_yaw2->setEstimate(cos_yaw2);
  cos_v_yaw2->setFixed(true);
  optimizer_.addVertex(cos_v_yaw2);
  
  FixedScalarVertex *sin_v_yaw2 = new FixedScalarVertex();
  sin_v_yaw2->setId(6); 
  sin_v_yaw2->setEstimate(sin_yaw2);
  sin_v_yaw2->setFixed(true);
  optimizer_.addVertex(sin_v_yaw2);
  
  FixedScalarVertex *v_z1 = new FixedScalarVertex();
  v_z1->setId(7);
  v_z1->setEstimate(z1);
  v_z1->setFixed(true);
  optimizer_.addVertex(v_z1);
  
  FixedScalarVertex *v_z2 = new FixedScalarVertex();
  v_z2->setId(8);
  v_z2->setEstimate(z2);
  v_z2->setFixed(true);
  optimizer_.addVertex(v_z2);
  
  // 边 - 一次性计算非畸变点
  std::vector<cv::Point2f> normalized_landmarks;
  cv::undistortPoints(landmarks, normalized_landmarks, camera_matrix_, dist_coeffs_);
  
  // 创建边对象的通用信息
  EdgeProjection::InfoMatrixType info_matrix = EdgeProjection::InfoMatrixType::Identity();
  
  // 添加第一组边（前4个点）
  for(size_t i = 0; i < 4; i++){
    g2o::VertexPointXYZ *v_point = new g2o::VertexPointXYZ();
    v_point->setId(i + 9);
    v_point->setEstimate(object_points[i]);
    v_point->setFixed(true);
    optimizer_.addVertex(v_point);
    
    EdgeTwoArmors *edge = new EdgeTwoArmors(R_odom_to_camera, K_);
    edge->setId(i);
    edge->setVertex(0, v_xy);
    edge->setVertex(1, v_r1);
    edge->setVertex(2, cos_v_yaw1);
    edge->setVertex(3, sin_v_yaw1);
    edge->setVertex(4, v_z1);
    edge->setVertex(5, v_point);
    edge->setMeasurement(Eigen::Vector2d(normalized_landmarks[i].x, normalized_landmarks[i].y));
    edge->setInformation(info_matrix);
    edge->setRobustKernel(new g2o::RobustKernelHuber);
    optimizer_.addEdge(edge);
  }
  
  // 添加第二组边（后4个点）
  for(size_t i = 4; i < 8; i++){
    g2o::VertexPointXYZ *v_point = new g2o::VertexPointXYZ();
    v_point->setId(i + 9);
    v_point->setEstimate(object_points[i]);
    v_point->setFixed(true);
    optimizer_.addVertex(v_point);
    
    EdgeTwoArmors *edge = new EdgeTwoArmors(R_odom_to_camera, K_);
    edge->setId(i);
    edge->setVertex(0, v_xy);
    edge->setVertex(1, v_r2);
    edge->setVertex(2, cos_v_yaw2);
    edge->setVertex(3, sin_v_yaw2);
    edge->setVertex(4, v_z2);
    edge->setVertex(5, v_point);
    edge->setMeasurement(Eigen::Vector2d(normalized_landmarks[i].x, normalized_landmarks[i].y));
    edge->setInformation(info_matrix);
    edge->setRobustKernel(new g2o::RobustKernelHuber);
    optimizer_.addEdge(edge);
  }
  
  // 执行优化
  optimizer_.initializeOptimization();
  optimizer_.optimize(30);
  
  // Get results after optimization
  r1 = v_r1->estimate();
  r2 = v_r2->estimate();
  x = v_xy->estimate().x();
  y = v_xy->estimate().y();
}

bool BaSolver::fixTwoArmors(Armor &armor1, Armor &armor2, const Eigen::Matrix3d &R_odom_to_camera){
  // 取出各个数据
  Eigen::Matrix3d & r_odom_armor1 = armor1.r_odom_armor;
  Eigen::Matrix3d & r_odom_armor2 = armor2.r_odom_armor;
  Eigen::Vector3d & t_odom_armor1 = armor1.t_odom_armor;
  Eigen::Vector3d & t_odom_armor2 = armor2.t_odom_armor;
  
  // 矫正yaw角
  double yaw_armor1 = std::atan2(r_odom_armor1(1, 0), r_odom_armor1(0, 0));
  double yaw_armor2 = std::atan2(r_odom_armor2(1, 0), r_odom_armor2(0, 0));
  double yaw_diff = shortest_angular_distance(yaw_armor1, yaw_armor2);
  
  if (abs(abs(yaw_diff) - M_PI / 2) > M_PI / 6) {
    RCLCPP_ERROR(rclcpp::get_logger("armor_detector"), "Yaw angle is %f and %f", yaw_armor1, yaw_armor2);
    return false;
  }
  
  // 预计算一些常量以避免重复计算
  const double correction = yaw_diff > 0 ? yaw_diff - M_PI / 2 : yaw_diff + M_PI / 2;
  const double half_correction = correction / 2.0;
  
  yaw_armor1 += half_correction;
  yaw_armor2 -= half_correction;
  yaw_armor1 = std::fmod(yaw_armor1 + M_PI, 2.0 * M_PI) - M_PI;
  yaw_armor2 = std::fmod(yaw_armor2 + M_PI, 2.0 * M_PI) - M_PI;
  
  if(std::abs(std::abs(shortest_angular_distance(yaw_armor1, yaw_armor2)) - M_PI / 2) > 1e-6){
    RCLCPP_ERROR(rclcpp::get_logger("armor_detector"), "Yaw angle diff is %f degrees after correction", shortest_angular_distance(yaw_armor1, yaw_armor2)); 
    return false;
  }
  
  // 预计算三角函数值
  const double cos_yaw1 = cos(yaw_armor1);
  const double sin_yaw1 = sin(yaw_armor1);
  const double cos_yaw2 = cos(yaw_armor2);
  const double sin_yaw2 = sin(yaw_armor2);
  
  // 计算旋转矩阵
  Sophus::SO3d R_yaw1 = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw_armor1));
  Sophus::SO3d R_yaw2 = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw_armor2));
  const double armor_pitch = armor1.number == "outpost" ? -0.2618 : 0.2618;
  Sophus::SO3d R_pitch = Sophus::SO3d::exp(Eigen::Vector3d(0, armor_pitch, 0)); 
  
  // 计算各个传入优化的数据 (使用预计算的三角函数值)
  const double x_armor1 = t_odom_armor1(0), y_armor1 = t_odom_armor1(1);
  const double x_armor2 = t_odom_armor2(0), y_armor2 = t_odom_armor2(1);
  
  const double offset1 = 0.26 * cos_yaw1;
  const double offset2 = 0.26 * sin_yaw1;
  const double offset3 = 0.26 * cos_yaw2;
  const double offset4 = 0.26 * sin_yaw2;
  
  double x_center = (x_armor1 + offset1 + x_armor2 + offset3) / 2.0;
  double y_center = (y_armor1 + offset2 + y_armor2 + offset4) / 2.0;
  
  double z1 = t_odom_armor1(2), z2 = t_odom_armor2(2);
  double r1 = 0.26, r2 = 0.26; 
  
  // 收集装甲板关键点，使用预分配大小避免动态扩容
  const auto &landmarks1 = armor1.landmarks();
  const auto &landmarks2 = armor2.landmarks();
  std::vector<cv::Point2f> landmarks;
  landmarks.reserve(landmarks1.size() + landmarks2.size());
  landmarks.insert(landmarks.end(), landmarks1.begin(), landmarks1.end());
  landmarks.insert(landmarks.end(), landmarks2.begin(), landmarks2.end());
  
  // 执行优化
  solveTwoArmorsBa(yaw_armor1, yaw_armor2, z1, z2, x_center, y_center, r1, r2, landmarks, R_odom_to_camera, armor1.number, armor1.type);
  
  // 计算各个返回的数据 (使用预计算的三角函数值)
  t_odom_armor1 = Eigen::Vector3d(x_center - r1 * cos_yaw1, y_center - r1 * sin_yaw1, z1);
  t_odom_armor2 = Eigen::Vector3d(x_center - r2 * cos_yaw2, y_center - r2 * sin_yaw2, z2);
  r_odom_armor1 = (R_yaw1 * R_pitch).matrix();
  r_odom_armor2 = (R_yaw2 * R_pitch).matrix();
  
  return true; 
}
double BaSolver::shortest_angular_distance(double a1, double a2) {
    double diff = std::fmod(a2 - a1, 2.0 * M_PI);
    if (diff > M_PI) diff -= 2.0 * M_PI;
    if (diff < -M_PI) diff += 2.0 * M_PI;
    return diff;
}

void BaSolver::startThreadPoolIfNeeded(int num_threads) {
  if (!thread_pool_) {
    thread_pool_ = std::make_shared<ThreadPool>(num_threads);
    RCLCPP_INFO(rclcpp::get_logger("armor_detector"), "BA线程池已启动，线程数: %d", num_threads);
  }
}
} // namespace rm_auto_aim