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
// g2o
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
// 3rd party
#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <rclcpp/clock.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include "rclcpp/rclcpp.hpp"
// project
#include "armor_detector/graph_optimizer.hpp"
#include "armor_detector/types.hpp"

namespace rm_auto_aim
{
G2O_USE_OPTIMIZATION_LIBRARY(dense)

BaSolver::BaSolver(
    const std::array<double, 9> & camera_matrix, const std::vector<double> & dist_coeffs)
: camera_matrix_(cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone()),
  dist_coeffs_(cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data())).clone())
{
    K_ = Eigen::Matrix3d::Identity();
    K_(0, 0) = camera_matrix[0];
    K_(1, 1) = camera_matrix[4];
    K_(0, 2) = camera_matrix[2];
    K_(1, 2) = camera_matrix[5];

    // initialize
    initializeOneArmorsOptimization(optimizer_);
    initializeTwoArmorsOptimization(two_armor_optimizer_);
}

void BaSolver::solveBa(
    Armor & armor, const Eigen::Matrix3d & R_odom_to_camera,
    const Eigen::Vector3d & t_odom_to_camera) noexcept
{
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
    double armor_pitch = armor.number == "outpost" ? -0.2618 : 0.2618;

    // Get the 3D points of the armor
    const auto armor_size = armor.type == ArmorType::SMALL
                                ? Eigen::Vector2d(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT)
                                : Eigen::Vector2d(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT);
    const auto object_points =
        Armor::buildObjectPoints<Eigen::Vector3d>(armor_size(0), armor_size(1));

    // Fill the optimizer
    size_t id_counter = 0;
    Sophus::SO3d R_pitch = Sophus::SO3d::exp(Eigen::Vector3d(0, armor_pitch, 0));

    auto * v_yaw = dynamic_cast<VertexYaw *>(optimizer_.vertex(id_counter++));
    v_yaw->setEstimate(initial_armor_yaw);
    for (size_t i = 0; i < 4; i++) {
        auto * v_point = dynamic_cast<g2o::VertexPointXYZ *>(optimizer_.vertex(id_counter++));
        v_point->setEstimate(
            R_pitch *
            Eigen::Vector3d(object_points[i].x(), object_points[i].y(), object_points[i].z()));
    }

    auto landmarks = armor.landmarks();
    cv::undistortPoints(landmarks, landmarks, camera_matrix_, dist_coeffs_);
    auto edgeset = optimizer_.edges();
    for (auto edge : edgeset) {
        auto * edge_proj = dynamic_cast<EdgeProjection *>(edge);
        edge_proj->setMeasurement(
            Eigen::Vector2d(landmarks[edge->id()].x, landmarks[edge->id()].y));
        edge_proj->setCameraPose(R_odom_to_camera, t_camera_armor);
    }

    // 执行优化
    optimizer_.initializeOptimization();
    optimizer_.optimize(30);
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
    // 计算两点之间的欧几里得距离
    auto calcDistance = [](const auto & p1, const auto & p2) -> double {
        if constexpr (std::is_same_v<std::decay_t<decltype(p1)>, cv::Point2f>) {
            return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
        } else {
            return sqrt(pow(p1.x() - p2.x(), 2) + pow(p1.y() - p2.y(), 2));
        }
    };

    l = calcDistance(landmarks[0], landmarks[1]) + calcDistance(landmarks[2], landmarks[3]);
    w = calcDistance(landmarks[1], landmarks[2]) + calcDistance(landmarks[3], landmarks[0]);
    area_measure = l * w / 4;
    double area_expect = 0;
    Eigen::Vector2d expect_points[4];
    for (size_t i = 0; i < 4; i++) {
        expect_points[i] =
            (R_odom_to_camera * r_odom_armor * object_points[i] + t_camera_armor).hnormalized();
    }

    l = calcDistance(expect_points[0], expect_points[1]) +
        calcDistance(expect_points[2], expect_points[3]);
    w = calcDistance(expect_points[1], expect_points[2]) +
        calcDistance(expect_points[3], expect_points[0]);
    area_expect = l * w / 4;
    t_camera_armor *= sqrt(area_expect / area_measure);
    t_odom_armor = R_camera_to_odom * t_camera_armor + t_camera_to_odom;

    return;
}
void BaSolver::solveTwoArmorsBa(
    const double & yaw1, const double & yaw2, const double & z1, const double & z2, double & x,
    double & y, double & r1, double & r2, const std::vector<cv::Point2f> & landmarks,
    const Eigen::Matrix3d & R_odom_to_camera, const Eigen::Vector3d & t_odom_to_camera,
    std::string number, ArmorType type)
{
    // Get the pitch angle of the armor
    double armor_pitch = number == "outpost" ? -0.2618 : 0.2618;
    Sophus::SO3d R_pitch = Sophus::SO3d::exp(Eigen::Vector3d(0, armor_pitch, 0));
    Sophus::SO3d R_yaw1 = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw1));
    Sophus::SO3d R_yaw2 = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw2));

    // Get the 3D points of the armor
    const auto armor_size = type == ArmorType::SMALL
                                ? Eigen::Vector2d(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT)
                                : Eigen::Vector2d(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT);
    const auto object_points_ =
        Armor::buildObjectPoints<Eigen::Vector3d>(armor_size(0), armor_size(1));
    Eigen::Vector3d object_points[8];
    for (size_t i = 0; i < 4; i++) {
        object_points[i] =
            R_yaw1 * R_pitch *
            Eigen::Vector3d(object_points_[i].x(), object_points_[i].y(), object_points_[i].z());
    }
    for (size_t i = 0; i < 4; i++) {
        object_points[i + 4] =
            R_yaw2 * R_pitch *
            Eigen::Vector3d(object_points_[i].x(), object_points_[i].y(), object_points_[i].z());
    }
    //需要优化的节点
    int id_counter = 0;
    auto * v_xy = dynamic_cast<VertexXY *>(two_armor_optimizer_.vertex(id_counter++));
    v_xy->setEstimate(Eigen::Vector2d(x, y));
    VertexR * v_r1 = dynamic_cast<VertexR *>(two_armor_optimizer_.vertex(id_counter++));
    v_r1->setEstimate(r1);
    VertexR * v_r2 = dynamic_cast<VertexR *>(two_armor_optimizer_.vertex(id_counter++));
    v_r2->setEstimate(r2);
    //固定节点
    FixedScalarVertex * v_yaw1 =
        dynamic_cast<FixedScalarVertex *>(two_armor_optimizer_.vertex(id_counter++));
    v_yaw1->setEstimate(yaw1);
    FixedScalarVertex * v_yaw2 =
        dynamic_cast<FixedScalarVertex *>(two_armor_optimizer_.vertex(id_counter++));
    v_yaw2->setEstimate(yaw2);
    FixedScalarVertex * v_z1 =
        dynamic_cast<FixedScalarVertex *>(two_armor_optimizer_.vertex(id_counter++));
    v_z1->setEstimate(z1);
    FixedScalarVertex * v_z2 =
        dynamic_cast<FixedScalarVertex *>(two_armor_optimizer_.vertex(id_counter++));
    v_z2->setEstimate(z2);
    for (size_t i = 0; i < 8; i++) {
        g2o::VertexPointXYZ * v_point =
            dynamic_cast<g2o::VertexPointXYZ *>(two_armor_optimizer_.vertex(id_counter++));
        v_point->setEstimate(object_points[i]);
    }
    //边
    auto edgeset = two_armor_optimizer_.edges();
    for (auto edge : edgeset) {
        auto * edge_proj = dynamic_cast<EdgeTwoArmors *>(edge);
        edge_proj->setMeasurement(
            Eigen::Vector2d(landmarks[edge->id()].x, landmarks[edge->id()].y));
        edge_proj->setCameraPose(R_odom_to_camera, t_odom_to_camera);
    }
    //执行优化
    two_armor_optimizer_.initializeOptimization();
    two_armor_optimizer_.optimize(30);
    // Get yaw angle after optimization
    r1 = v_r1->estimate();
    r2 = v_r2->estimate();
    x = v_xy->estimate().x();
    y = v_xy->estimate().y();
}
bool BaSolver::fixTwoArmors(
    Armor & armor1, Armor & armor2, const Eigen::Matrix3d & R_odom_to_camera,
    const Eigen::Vector3d & t_odom_to_camera)
{
    // 取出各个数据
    Eigen::Matrix3d & r_odom_armor1 = armor1.r_odom_armor;
    Eigen::Matrix3d & r_odom_armor2 = armor2.r_odom_armor;
    Eigen::Vector3d & t_odom_armor1 = armor1.t_odom_armor;
    Eigen::Vector3d & t_odom_armor2 = armor2.t_odom_armor;
    // 矫正yaw角
    double yaw_armor1 = std::atan2(r_odom_armor1(1, 0), r_odom_armor1(0, 0));
    double yaw_armor2 = std::atan2(r_odom_armor2(1, 0), r_odom_armor2(0, 0));
    double yaw_diff = shortest_angular_distance(yaw_armor1, yaw_armor2);
    if (armor1.number != "outpost") {
        if (abs(abs(yaw_diff) - M_PI / 2) > M_PI / 6) {
            RCLCPP_ERROR(
                rclcpp::get_logger("armor_detector"), "Yaw angle is %f and %f", yaw_armor1,
                yaw_armor2);
            return false;
        }
        double correction = yaw_diff > 0 ? yaw_diff - M_PI / 2 : yaw_diff + M_PI / 2;
        yaw_armor1 += correction / 2.0;
        yaw_armor2 -= correction / 2.0;
        yaw_armor1 = std::fmod(yaw_armor1 + M_PI, 2.0 * M_PI) - M_PI;
        yaw_armor2 = std::fmod(yaw_armor2 + M_PI, 2.0 * M_PI) - M_PI;
        if (std::abs(std::abs(shortest_angular_distance(yaw_armor1, yaw_armor2)) - M_PI / 2) >
            1e-6) {
            RCLCPP_ERROR(
                rclcpp::get_logger("armor_detector"),
                "Yaw angle diff is %f degrees after correction",
                shortest_angular_distance(yaw_armor1, yaw_armor2));
            return false;
        }
    } else {
        if (abs(abs(yaw_diff) - M_PI * 2 / 3) > M_PI / 6) {
            RCLCPP_ERROR(
                rclcpp::get_logger("armor_detector"), "Yaw angle is %f and %f", yaw_armor1,
                yaw_armor2);
            return false;
        }
        double correction = yaw_diff > 0 ? yaw_diff - M_PI * 2 / 3 : yaw_diff + M_PI * 2 / 3;
        yaw_armor1 += correction / 2.0;
        yaw_armor2 -= correction / 2.0;
        yaw_armor1 = std::fmod(yaw_armor1 + M_PI, 2.0 * M_PI) - M_PI;
        yaw_armor2 = std::fmod(yaw_armor2 + M_PI, 2.0 * M_PI) - M_PI;
        if (std::abs(std::abs(shortest_angular_distance(yaw_armor1, yaw_armor2)) - M_PI * 2 / 3) >
            1e-6) {
            RCLCPP_ERROR(
                rclcpp::get_logger("armor_detector"),
                "Yaw angle diff is %f degrees after correction",
                shortest_angular_distance(yaw_armor1, yaw_armor2));
            return false;
        }
    }
    // 计算旋转矩阵
    Sophus::SO3d R_yaw1 = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw_armor1));
    Sophus::SO3d R_yaw2 = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw_armor2));
    double armor_pitch = armor1.number == "outpost" ? -0.2618 : 0.2618;
    Sophus::SO3d R_pitch = Sophus::SO3d::exp(Eigen::Vector3d(0, armor_pitch, 0));
    // 计算各个传入优化的数据
    double x_armor1 = t_odom_armor1(0), y_armor1 = t_odom_armor1(1);
    double x_armor2 = t_odom_armor2(0), y_armor2 = t_odom_armor2(1);
    double x_center = (x_armor1 + 0.26 * cos(yaw_armor1) + x_armor2 + 0.26 * cos(yaw_armor2)) / 2.0;
    double y_center = (y_armor1 + 0.26 * sin(yaw_armor1) + y_armor2 + 0.26 * sin(yaw_armor2)) / 2.0;
    double z1 = t_odom_armor1(2), z2 = t_odom_armor2(2);
    double r1 = 0.26, r2 = 0.26;
    const auto & landmarks1 = armor1.landmarks();
    const auto & landmarks2 = armor2.landmarks();
    std::vector<cv::Point2f> landmarks;
    landmarks.reserve(landmarks1.size() + landmarks2.size());
    landmarks.insert(landmarks.end(), landmarks1.begin(), landmarks1.end());
    landmarks.insert(landmarks.end(), landmarks2.begin(), landmarks2.end());
    cv::undistortPoints(landmarks, landmarks, camera_matrix_, dist_coeffs_);
    // 执行优化
    solveTwoArmorsBa(
        yaw_armor1, yaw_armor2, z1, z2, x_center, y_center, r1, r2, landmarks, R_odom_to_camera,
        t_odom_to_camera, armor1.number, armor1.type);
    // 计算各个返回的数据
    t_odom_armor1 =
        Eigen::Vector3d(x_center - r1 * cos(yaw_armor1), y_center - r1 * sin(yaw_armor1), z1);
    t_odom_armor2 =
        Eigen::Vector3d(x_center - r2 * cos(yaw_armor2), y_center - r2 * sin(yaw_armor2), z2);
    r_odom_armor1 = (R_yaw1 * R_pitch).matrix();
    r_odom_armor2 = (R_yaw2 * R_pitch).matrix();
    return true;
}
double BaSolver::shortest_angular_distance(double a1, double a2)
{
    double diff = std::fmod(a2 - a1, 2.0 * M_PI);
    if (diff > M_PI) diff -= 2.0 * M_PI;
    if (diff < -M_PI) diff += 2.0 * M_PI;
    return diff;
}
void BaSolver::initializeOneArmorsOptimization(g2o::SparseOptimizer & optimizer)
{
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(
        g2o::OptimizationAlgorithmFactory::instance()->construct("lm_dense", solver_property_));

    // Initial step size
    lm_algorithm_ = dynamic_cast<g2o::OptimizationAlgorithmLevenberg *>(
        const_cast<g2o::OptimizationAlgorithm *>(optimizer.algorithm()));
    lm_algorithm_->setUserLambdaInit(0.1);
    //填充优化器
    size_t id_counter = 0;
    VertexYaw * v_yaw = new VertexYaw();
    v_yaw->setId(id_counter++);
    optimizer.addVertex(v_yaw);

    // cv::undistortPoints(landmarks, landmarks, camera_matrix_, dist_coeffs_);
    for (size_t i = 0; i < 4; i++) {
        g2o::VertexPointXYZ * v_point = new g2o::VertexPointXYZ();
        v_point->setId(id_counter++);
        v_point->setFixed(true);
        optimizer.addVertex(v_point);

        EdgeProjection * edge = new EdgeProjection();
        edge->setId(i);
        edge->setVertex(0, v_yaw);
        edge->setVertex(1, v_point);
        edge->setInformation(EdgeProjection::InfoMatrixType::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer.addEdge(edge);
    }
}
void BaSolver::initializeTwoArmorsOptimization(g2o::SparseOptimizer & optimizer)
{
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(
        g2o::OptimizationAlgorithmFactory::instance()->construct("lm_dense", solver_property_));
    // Initial step size
    lm_algorithm_ = dynamic_cast<g2o::OptimizationAlgorithmLevenberg *>(
        const_cast<g2o::OptimizationAlgorithm *>(optimizer.algorithm()));
    lm_algorithm_->setUserLambdaInit(0.1);
    int id_counter = 0;
    //需要优化的节点
    VertexXY * v_xy = new VertexXY();
    v_xy->setId(id_counter++);
    optimizer.addVertex(v_xy);
    VertexR * v_r1 = new VertexR();
    v_r1->setId(id_counter++);
    optimizer.addVertex(v_r1);
    VertexR * v_r2 = new VertexR();
    v_r2->setId(id_counter++);
    optimizer.addVertex(v_r2);
    //固定节点
    FixedScalarVertex * v_yaw1 = new FixedScalarVertex();
    v_yaw1->setId(id_counter++);
    v_yaw1->setFixed(true);
    optimizer.addVertex(v_yaw1);
    FixedScalarVertex * v_yaw2 = new FixedScalarVertex();
    v_yaw2->setId(id_counter++);
    v_yaw2->setFixed(true);
    optimizer.addVertex(v_yaw2);
    FixedScalarVertex * v_z1 = new FixedScalarVertex();
    v_z1->setId(id_counter++);
    v_z1->setFixed(true);
    optimizer.addVertex(v_z1);
    FixedScalarVertex * v_z2 = new FixedScalarVertex();
    v_z2->setId(id_counter++);
    v_z2->setFixed(true);
    optimizer.addVertex(v_z2);
    //边
    for (size_t i = 0; i < 4; i++) {
        g2o::VertexPointXYZ * v_point = new g2o::VertexPointXYZ();
        v_point->setId(id_counter++);
        v_point->setFixed(true);
        optimizer.addVertex(v_point);
        EdgeTwoArmors * edge = new EdgeTwoArmors();
        edge->setId(i);
        edge->setVertex(0, v_xy);
        edge->setVertex(1, v_r1);
        edge->setVertex(2, v_yaw1);
        edge->setVertex(3, v_z1);
        edge->setVertex(4, v_point);
        edge->setInformation(EdgeProjection::InfoMatrixType::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer.addEdge(edge);
    }
    for (size_t i = 4; i < 8; i++) {
        g2o::VertexPointXYZ * v_point = new g2o::VertexPointXYZ();
        v_point->setId(id_counter++);
        v_point->setFixed(true);
        optimizer.addVertex(v_point);
        EdgeTwoArmors * edge = new EdgeTwoArmors();
        edge->setId(i);
        edge->setVertex(0, v_xy);
        edge->setVertex(1, v_r2);
        edge->setVertex(2, v_yaw2);
        edge->setVertex(3, v_z2);
        edge->setVertex(4, v_point);
        edge->setInformation(EdgeProjection::InfoMatrixType::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer.addEdge(edge);
    }
    //填充优化器
}
}  // namespace rm_auto_aim