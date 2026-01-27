#ifndef ARMOR_DETECTOR_BA_SOLVER_HPP_
#define ARMOR_DETECTOR_BA_SOLVER_HPP_

// std
#include <array>
#include <cstddef>
#include <tuple>
#include <vector>
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
#include "armor_detector/types.hpp"

namespace rm_auto_aim
{

/**
 * @brief 基于最小二乘法 (Bundle Adjustment) 的求解器
 * 
 * 该类使用 BA 算法优化装甲板的位姿估计，尤其关注偏航角的精确计算。
 * BA 算法通过最小化重投影误差来同时优化相机位姿和 3D 点的位置。
 */
class BaSolver
{
public:
    /**
     * @brief 构造 BA 求解器
     * 
     * @param camera_matrix 相机内参矩阵，3x3 矩阵的数组表示
     * @param dist_coeffs 相机畸变系数
     */
    BaSolver(const std::array<double, 9> & camera_matrix, const std::vector<double> & dist_coeffs);

    /**
     * @brief 使用 BA 算法求解单个装甲板的位姿
     * 
     * 通过最小化重投影误差，优化单个装甲板在相机坐标系下的旋转和平移
     * 
     * @param armor 需要优化的装甲板对象，结果将更新至此对象
     * @param R_odom_to_camera 从里程计坐标系到相机坐标系的旋转矩阵
     * @param t_odom_to_camera 从里程计坐标系到相机坐标系的平移向量
     */
    void solveBa(
        Armor & armor, const Eigen::Matrix3d & R_odom_to_camera,
        const Eigen::Vector3d & t_odom_to_camera) noexcept;

    /**
     * @brief 使用 BA 算法同时求解两个装甲板的位姿
     * 
     * 通过考虑装甲板之间的几何约束，提高位姿估计精度
     * 
     * @param yaw1 第一个装甲板的初始偏航角
     * @param yaw2 第二个装甲板的初始偏航角
     * @param z1 第一个装甲板的 Z 坐标
     * @param z2 第二个装甲板的 Z 坐标
     * @param x [输出] 优化后的 X 坐标
     * @param y [输出] 优化后的 Y 坐标
     * @param r1 [输出] 优化后的第一个装甲板半径
     * @param r2 [输出] 优化后的第二个装甲板半径
     * @param landmarks 特征点坐标
     * @param R_odom_to_camera 从里程计坐标系到相机坐标系的旋转矩阵
     * @param t_odom_to_camera 从里程计坐标系到相机坐标系的平移向量
     * @param number 装甲板数字
     * @param type 装甲板类型
     */
    void solveTwoArmorsBa(
        const double & yaw1, const double & yaw2, const double & z1, const double & z2, double & x,
        double & y, double & r1, double & r2, const std::vector<cv::Point2f> & landmarks,
        const Eigen::Matrix3d & R_odom_to_camera, const Eigen::Vector3d & t_odom_to_camera,
        std::string number, ArmorType type);

    /**
     * @brief 修正两个装甲板的位姿估计
     * 
     * 使用两个装甲板的几何约束关系优化位姿估计
     * 
     * @param armor1 第一个装甲板
     * @param armor2 第二个装甲板
     * @param R_odom_to_camera 从里程计坐标系到相机坐标系的旋转矩阵
     * @param t_odom_to_camera 从里程计坐标系到相机坐标系的平移向量
     * @return true 如果修正成功
     * @return false 如果修正失败
     */
    bool fixTwoArmors(
        Armor & armor1, Armor & armor2, const Eigen::Matrix3d & R_odom_to_camera,
        const Eigen::Vector3d & t_odom_to_camera);

private:
    Eigen::Matrix3d K_;                                   ///< 相机内参矩阵（Eigen 格式）
    g2o::SparseOptimizer optimizer_;                      ///< 单装甲板优化器
    g2o::SparseOptimizer two_armor_optimizer_;            ///< 双装甲板优化器
    g2o::OptimizationAlgorithmProperty solver_property_;  ///< 求解器属性
    g2o::OptimizationAlgorithmLevenberg * lm_algorithm_;  ///< Levenberg-Marquardt 优化算法
    cv::Mat camera_matrix_;                               ///< 相机内参矩阵（OpenCV 格式）
    cv::Mat dist_coeffs_;                                 ///< 相机畸变系数

    /**
     * @brief 计算两个角度之间的最短距离
     * 
     * @param a1 第一个角度（弧度）
     * @param a2 第二个角度（弧度）
     * @return double 两个角度之间的最短距离（弧度）
     */
    double shortest_angular_distance(double a1, double a2);

    /**
     * @brief 初始化单装甲板优化
     * 
     * 配置用于单装甲板位姿估计的图优化器
     * 
     * @param optimizer 要初始化的优化器
     */
    void initializeOneArmorsOptimization(g2o::SparseOptimizer & optimizer);

    /**
     * @brief 初始化双装甲板优化
     * 
     * 配置用于双装甲板位姿估计的图优化器
     * 
     * @param optimizer 要初始化的优化器
     */
    void initializeTwoArmorsOptimization(g2o::SparseOptimizer & optimizer);
};

}  // namespace rm_auto_aim
#endif  // ARMOR_DETECTOR_BAS_SOLVER_HPP_