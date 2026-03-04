#include "ballistic_calculation/math_util.hpp"

namespace rm_auto_aim {
    Eigen::Matrix3d eulerToMatrix(const Eigen::Vector3d & euler, EulerOrder order) {
        // 构建各轴旋转
        auto r = Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX());  // 绕X轴旋转（roll）
        auto p = Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY());  // 绕Y轴旋转（pitch）
        auto y = Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ());  // 绕Z轴旋转（yaw）
        
        // 根据旋转顺序组合旋转矩阵
        switch (order) {
            case EulerOrder::XYZ:
                return (y * p * r).matrix();
            case EulerOrder::XZY:
                return (p * y * r).matrix();
            case EulerOrder::YXZ:
                return (y * r * p).matrix();
            case EulerOrder::YZX:
                return (r * y * p).matrix();
            case EulerOrder::ZXY:
                return (p * r * y).matrix();
            case EulerOrder::ZYX:
                return (r * p * y).matrix();
            default:
                return Eigen::Matrix3d::Identity();
        }
    }
}