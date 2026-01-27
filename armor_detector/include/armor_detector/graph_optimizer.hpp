#ifndef ARMOR_DETECTOR_GRAPH_OPTIMIZER_HPP_
#define ARMOR_DETECTOR_GRAPH_OPTIMIZER_HPP_

// std
#include <array>
// g2o
#include <g2o/core/auto_differentiation.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/sparse_optimizer.h>
// 3rd party
#include <g2o/types/slam3d/vertex_pointxyz.h>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace rm_auto_aim
{
/**
 * @brief 图优化中的偏航角顶点
 * 
 * 用于表示并优化装甲板在车体坐标系中的偏航角
 */
class VertexYaw : public g2o::BaseVertex<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexYaw() = default;
    void setToOriginImpl() override { _estimate = 0; }  // 重置
    void oplusImpl(const double * update) override;     // 更新
    //不需要读写
    bool read(std::istream & in) override { return true; }
    bool write(std::ostream & out) const override { return true; }
};

/**
 * @brief 图优化中的二维坐标顶点
 * 
 * 用于表示并优化装甲板在平面上的 X-Y 坐标
 */
class VertexXY : public g2o::BaseVertex<2, Eigen::Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexXY() = default;
    void setToOriginImpl() override { _estimate = Eigen::Vector2d::Zero(); }  // 重置
    void oplusImpl(const double * update) override;                           // 更新
    bool read(std::istream & in) override { return true; }
    bool write(std::ostream & out) const override { return true; }
};

/**
 * @brief 固定标量顶点
 * 
 * 用于在优化过程中表示固定的标量参数
 */
class FixedScalarVertex : public g2o::BaseVertex<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    FixedScalarVertex() = default;
    void setToOriginImpl() override { _estimate = 0; }                          // 重置
    void oplusImpl(const double * update) override { _estimate += update[0]; }  // 更新
    bool read(std::istream & in) override { return true; }
    bool write(std::ostream & out) const override { return true; }
};

/**
 * @brief 半径顶点
 * 
 * 用于表示并优化装甲板相对中心的半径距离
 */
class VertexR : public g2o::BaseVertex<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexR() = default;
    void setToOriginImpl() override { _estimate = 0; }  // 重置
    void oplusImpl(const double * update) override;     // 更新
    bool read(std::istream & in) override { return true; }
    bool write(std::ostream & out) const override { return true; }
};

/**
 * @brief 投影误差边
 * 
 * 图优化中用于计算特定偏航角下装甲板的重投影误差
 * 该边连接 VertexYaw 和 VertexPointXYZ 两种顶点
 */
class EdgeProjection
: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexYaw, g2o::VertexPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using InfoMatrixType = Eigen::Matrix<double, 2, 2>;

    EdgeProjection();

    /**
     * @brief 设置相机姿态参数
     * 
     * @param R_odom_to_camera 从里程计坐标系到相机坐标系的旋转矩阵
     * @param t_camera_armor 相机坐标系下装甲板的平移向量
     */
    void setCameraPose(
        const Sophus::SO3d & R_odom_to_camera, const Eigen::Vector3d & t_camera_armor);

    /**
     * @brief 计算重投影误差
     * 
     * 根据当前顶点状态计算预测的投影点与观测点之间的误差
     */
    void computeError() override;

    bool read(std::istream & in) override { return true; }
    bool write(std::ostream & out) const override { return true; }

private:
    Eigen::Vector3d t_;  ///< 平移向量
    Eigen::Matrix3d K_;  ///< 相机内参矩阵
};

/**
 * @brief 两个装甲板之间的关系误差边
 * 
 * 用于约束两个装甲板之间的几何关系，提高整体估计精度
 * 这是一个多边，连接车辆位姿信息等共五个顶点
 */
class EdgeTwoArmors : public g2o::BaseMultiEdge<2, Eigen::Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * @brief 计算装甲板间关系误差
     * 
     * 根据当前顶点状态计算两个装甲板之间的位置关系与观测值之间的误差
     */
    void computeError() override;

    EdgeTwoArmors();

    /**
     * @brief 设置相机姿态参数
     * 
     * @param R_odom_to_camera 从里程计坐标系到相机坐标系的旋转矩阵
     * @param t_odom_to_camera 从里程计坐标系到相机坐标系的平移向量
     */
    void setCameraPose(
        const Sophus::SO3d & R_odom_to_camera, const Eigen::Vector3d & t_odom_to_camera);

    bool read(std::istream & in) override { return true; }
    bool write(std::ostream & out) const override { return true; }

private:
    Eigen::Matrix3d K_;  ///< 相机内参矩阵
    Eigen::Vector3d t_;  ///< 平移向量
};

}  // namespace rm_auto_aim
#endif  // ARMOR_DETECTOR_GRAPH_OPTIMIZER_HPP_