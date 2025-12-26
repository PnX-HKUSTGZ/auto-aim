#ifndef BALLISTIC_CALCULATION_MATH_UTIL_HPP_
#define BALLISTIC_CALCULATION_MATH_UTIL_HPP_

#include <angles/angles.h>
#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <auto_aim_interfaces/msg/detail/target__struct.hpp>
#include <auto_aim_interfaces/msg/rune_target.hpp>
#include <map>

namespace rm_auto_aim
{
/**
 * @brief 欧拉角旋转顺序枚举
 */
enum class EulerOrder { XYZ, XZY, YXZ, YZX, ZXY, ZYX };

inline double limit_rad(double rad)
{
    rad = fmod(rad, 2 * M_PI);
    if (rad > M_PI) {
        rad -= 2 * M_PI;
    } else if (rad < -M_PI) {
        rad += 2 * M_PI;
    }
    return rad;
}

/**
 * @brief 计算两个角度之间的最短角距离
 * 
 * 将角度差规范化到 [-π, π] 范围
 * 
 * @param a 角度1（弧度）
 * @param b 角度2（弧度）
 * @return double 最短角距离（弧度）
 */
inline double shortest_angular_distance(double a, double b)
{
    return limit_rad(a-b);
}

/**
 * @brief 将欧拉角转换为旋转矩阵
 * 
 * @param euler 欧拉角向量 [roll, pitch, yaw]
 * @param order 旋转顺序，默认为XYZ
 * @return Eigen::Matrix3d 3x3旋转矩阵
 */
Eigen::Matrix3d eulerToMatrix(const Eigen::Vector3d & euler, EulerOrder order = EulerOrder::XYZ);

/**
 * @brief 偏航角优化残差类
 * 
 * 用于Ceres优化器，计算最优放弃角度的残差函数。
 * 主要用于高速运动目标的装甲板切换策略。
 */
class YawResidual
{
public:
    /**
     * @brief 构造函数
     * 
     * @param v_yaw 目标偏航角速度
     * @param v_yaw_gimble 云台最大偏航角速度
     * @param distance 到目标中心的距离
     * @param radius 装甲板半径
     * @param armors_num 装甲板数量
     */
    YawResidual(double v_yaw, double v_yaw_gimble, double distance, double radius, int armors_num)
    : v_yaw_(v_yaw),
      v_yaw_gimble_(v_yaw_gimble),
      distance_(distance),
      radius_(radius),
      armors_num_(armors_num)
    {
    }

    /**
     * @brief 残差计算操作符
     * 
     * 计算放弃角度优化问题的残差值
     * 
     * @tparam T 数值类型（支持自动微分）
     * @param yaw 优化变量：放弃角度
     * @param residual 输出残差
     * @return true 计算成功
     */
    template <typename T>
    bool operator()(const T * const yaw, T * residual) const
    {
        // 计算云台偏航角速度
        T numerator = ((T(2.0) * T(M_PI) / T(armors_num_)) - T(2.0) * yaw[0]) * T(v_yaw_gimble_);
        T denominator = T(2.0) * T(v_yaw_);
        T yaw_gimble = numerator / denominator;

        // 计算几何约束方程：sin(yaw + yaw_gimble) / distance - sin(yaw_gimble) / radius = 0
        residual[0] =
            ceres::sin(yaw[0] + yaw_gimble) / T(distance_) - ceres::sin(yaw_gimble) / T(radius_);

        return true;
    }

private:
    const double v_yaw_;        // 目标偏航角速度
    const double v_yaw_gimble_; // 云台最大偏航角速度
    const double distance_;     // 到目标中心的距离
    const double radius_;       // 装甲板半径
    const int armors_num_;      // 装甲板数量
};

} // namespace rm_auto_aim
#endif  // BALLISTIC_CALCULATION_MATH_UTIL_HPP_