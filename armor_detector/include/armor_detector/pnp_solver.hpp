// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__PNP_SOLVER_HPP_
#define ARMOR_DETECTOR__PNP_SOLVER_HPP_
// 防止头文件被重复包含
#include <geometry_msgs/msg/point.hpp>
#include <opencv2/core.hpp>

// STD
#include <array>
#include <vector>

#include "armor_detector/types.hpp"

namespace rm_auto_aim
{
/**
 * @brief 装甲板的 PnP 解算器，用于从 2D 图像坐标计算 3D 位姿
 * 
 * PnP (Perspective-n-Point) 解算器通过已知的特征点坐标和相机内参，
 * 来估计物体在相机坐标系下的姿态和位置。
 */
class PnPSolver
{
public:
    /**
     * @brief 构造一个 PnP 解算器
     * 
     * @param camera_matrix 相机内参矩阵，3x3 矩阵的数组表示
     * @param distortion_coefficients 相机畸变系数
     */
    PnPSolver(
        const std::array<double, 9> & camera_matrix,
        const std::vector<double> & distortion_coefficients);

    /**
     * @brief 对装甲板进行 PnP 解算，获取其 3D 位置
     * 
     * 通过将装甲板的特征点（两个灯条的顶点和底点）与预定义的 3D 模型点进行匹配，
     * 使用 OpenCV 的 solvePnP 函数来计算装甲板在相机坐标系下的位姿。
     * 
     * @param armor 待解算的装甲板对象
     * @param rvec 输出的旋转向量
     * @param tvec 输出的平移向量
     * @return true 如果 PnP 解算成功
     * @return false 如果 PnP 解算失败
     */
    bool solvePnP(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec);

    /**
     * @brief 计算图像点到图像中心的距离
     * 
     * 用于评估装甲板在图像中的位置，常用于目标选择时的权重计算。
     * 
     * @param image_point 图像中的点坐标
     * @return float 点到图像中心的欧氏距离
     */
    float calculateDistanceToCenter(const cv::Point2f & image_point);

private:
    cv::Mat camera_matrix_;  ///< 相机内参矩阵
    cv::Mat dist_coeffs_;    ///< 相机畸变系数

    // 装甲板在 3D 空间中的四个顶点
    std::vector<cv::Point3f> small_armor_points_;  ///< 小装甲板的 3D 模型点
    std::vector<cv::Point3f> large_armor_points_;  ///< 大装甲板的 3D 模型点
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__PNP_SOLVER_HPP_
