#ifndef BALLISTIC_CALCULATION_AIM_INFO_HPP_
#define BALLISTIC_CALCULATION_AIM_INFO_HPP_

#include <angles/angles.h>
#include <bits/types/time_t.h>
#include <ceres/ceres.h>
#include "rclcpp/rclcpp.hpp"

#include <Eigen/Dense>
#include <auto_aim_interfaces/msg/detail/target__struct.hpp>
#include <auto_aim_interfaces/msg/rune_target.hpp>
#include "math_util.hpp"
#include <map>
namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::Target;
using rune_target = auto_aim_interfaces::msg::RuneTarget;

/**
 * @brief 瞄准信息基类
 * 
 * 抽象基类，定义了目标位置预测和坐标变换的接口。
 * 支持从odom坐标系到枪口坐标系的变换。
 */
class AimInfo 
{
public:
    virtual ~AimInfo() = default;
    
    /**
     * @brief 构造函数
     * 
     * @param odom2gunxyz 枪口在odom坐标系中的位置
     * @param odom2gunrpy 枪口在odom坐标系中的姿态（RPY角）
     */
    AimInfo(const Eigen::Vector3d & odom2gunxyz, const Eigen::Vector3d & odom2gunrpy)
    : odom2gunxyz(odom2gunxyz), odom2gunrpy(odom2gunrpy)
    {
    }
    
    /**
     * @brief 纯虚函数：获取odom坐标系中的目标位置
     * 
     * @param time 预测时间
     * @return Eigen::Vector3d odom坐标系中的目标位置
     */
    virtual Eigen::Vector3d getOdomTarget(double time) = 0;
    
    /**
     * @brief 获取枪口坐标系中的目标位置
     * 
     * 将odom坐标系中的目标位置变换到枪口坐标系
     * 
     * @param time 预测时间
     * @return Eigen::Vector3d 枪口坐标系中的目标位置
     */
    virtual Eigen::Vector3d getGunTarget(double time){
        // 获取odom坐标系中的预测位置
        Eigen::Vector3d target_position_odom = getOdomTarget(time);
        
        // 执行坐标变换：从odom坐标系到枪口坐标系
        // 枪口坐标系相对odom坐标系有旋转odom2gunrpy和平移odom2gunxyz
        Eigen::Matrix3d R_odom2gun = eulerToMatrix(odom2gunrpy);
        Eigen::Vector3d target_position_gun = R_odom2gun * target_position_odom + odom2gunxyz;

        return target_position_gun;
    }
    
    /**
     * @brief 获取目标的水平距离
     * 
     * @param time 预测时间
     * @return double 水平距离（X-Y平面内的距离）
     */
    virtual double getHorizontalDistance(double time){
        Eigen::Vector3d target_position_gun = getGunTarget(time);
        return sqrt(pow(target_position_gun.x(), 2) + pow(target_position_gun.y(), 2));
    }

protected:
    rclcpp::Time now() const {
        static rclcpp::Clock clock(RCL_ROS_TIME);
        return clock.now();
    }

private:
    Eigen::Vector3d odom2gunxyz;  // 枪口在odom坐标系中的位置
    Eigen::Vector3d odom2gunrpy;  // 枪口在odom坐标系中的姿态
};

/**
 * @brief 车辆信息类
 * 
 * 处理装甲车辆目标的位置预测，假设车辆做匀速直线运动
 */
class CarInfo : public AimInfo
{
private:
    target target_msg;  // 目标消息缓存

public:
    CarInfo(const Eigen::Vector3d & odom2gunxyz, const Eigen::Vector3d & odom2gunrpy)
    : AimInfo(odom2gunxyz, odom2gunrpy) {}
    
    /**
     * @brief 更新目标信息
     * 
     * @param new_target_msg 新的目标消息
     */
    void updateTarget(const target & new_target_msg) {
        auto msg_delay = (this->now() - rclcpp::Time(new_target_msg.header.stamp)).seconds();
        target_msg = new_target_msg;
        target_msg.yaw = new_target_msg.yaw + msg_delay * new_target_msg.v_yaw;
        target_msg.position.x = new_target_msg.position.x + msg_delay * new_target_msg.velocity.x;
        target_msg.position.y = new_target_msg.position.y + msg_delay * new_target_msg.velocity.y;
        target_msg.position.z = new_target_msg.position.z + msg_delay * new_target_msg.velocity.z;
    }
    
    Eigen::Vector3d getOdomTarget(double time) override
    {
        // 使用匀速直线运动模型：position = initial_position + velocity * time
        Eigen::Vector3d target_position_odom = Eigen::Vector3d(
            target_msg.position.x + target_msg.velocity.x * time,
            target_msg.position.y + target_msg.velocity.y * time,
            target_msg.position.z + target_msg.velocity.z * time);
        return target_position_odom;
    }
};

/**
 * @brief 装甲板信息类
 * 
 * 处理特定装甲板的位置预测，考虑车辆的旋转运动
 */
class ArmorInfo : public AimInfo
{
private:
    target target_msg;  // 目标消息缓存
    double yaw, z, r;   // 装甲板的偏航角、Z坐标和半径

public:
    ArmorInfo(const Eigen::Vector3d & odom2gunxyz, const Eigen::Vector3d & odom2gunrpy)
    : AimInfo(odom2gunxyz, odom2gunrpy) {}
    
    /**
     * @brief 更新目标和装甲板信息
     * 
     * @param new_target_msg 新的目标消息
     * @param new_yaw 装甲板偏航角
     * @param new_z 装甲板Z坐标
     * @param new_r 装甲板半径
     */
    void updateTarget(const target & new_target_msg, double new_yaw, double new_z, double new_r)
    {
        target_msg = new_target_msg;
        yaw = new_yaw;
        z = new_z;
        r = new_r;
    }
    
    Eigen::Vector3d getOdomTarget(double time) override
    {
        // 装甲板位置 = 车辆中心位置 + 旋转偏移
        // 旋转偏移考虑装甲板相对车辆中心的圆周运动
        Eigen::Vector3d target_position_odom = Eigen::Vector3d(
            target_msg.position.x + target_msg.velocity.x * time -
                r * cos(yaw + target_msg.v_yaw * time),  // X坐标包含旋转分量
            target_msg.position.y + target_msg.velocity.y * time -
                r * sin(yaw + target_msg.v_yaw * time),  // Y坐标包含旋转分量
            z + target_msg.velocity.z * time);           // Z坐标只考虑平移
        return target_position_odom;
    }
};

/**
 * @brief 能量机关信息类
 * 
 * 处理能量机关目标的位置预测，基于拟合曲线模型
 */
class RuneInfo : public AimInfo
{
private:
    rune_target target_msg;  // 能量机关目标消息缓存

    // 能量机关拟合曲线宏定义
    // 大能量机关：复杂正弦函数 + 线性函数
#define BIG_RUNE_CURVE(x, a, omega, b, c, d, sign)                                             \
    (((-((a) / (omega)*std::cos((omega) * ((x) + (d)))) + (b) * ((x) + (d)) + (c)) * (sign)) - \
     ((-((a) / (omega)*std::cos((omega) * ((d)))) + (b) * ((d)) + (c)) * (sign)))

    // 小能量机关：线性函数
#define SMALL_RUNE_CURVE(x, a, b, c, sign) \
    ((((a) * ((x) + (b)) + (c)) * (sign)) - (((a) * (b) + (c)) * (sign)))

    // 能量机关臂长，单位：米
    const double ARM_LENGTH = 0.700;

public:
    RuneInfo(const Eigen::Vector3d & odom2gunxyz, const Eigen::Vector3d & odom2gunrpy)
    : AimInfo(odom2gunxyz, odom2gunrpy) {}
    
    /**
     * @brief 更新能量机关目标信息
     * 
     * @param new_target_msg 新的能量机关目标消息
     */
    void updateTarget(const rune_target & new_target_msg) { target_msg = new_target_msg; }
    
    Eigen::Vector3d getOdomTarget(double time) override
    {
        int sign = target_msg.direction ? 1 : -1;
        // 根据能量机关类型选择相应的拟合曲线
        double angle_diff =
            target_msg.is_big ? BIG_RUNE_CURVE(
                                    time, target_msg.fitting_curve[0], target_msg.fitting_curve[1],
                                    target_msg.fitting_curve[2], target_msg.fitting_curve[3],
                                    target_msg.fitting_curve[4], sign)
                              : SMALL_RUNE_CURVE(
                                    time, target_msg.fitting_curve[0], target_msg.fitting_curve[1],
                                    target_msg.fitting_curve[2], sign);

        // 能量机关中心在odom坐标系中的位置
        Eigen::Vector3d t_odom_2_rune =
            Eigen::Vector3d(target_msg.center.x, target_msg.center.y, target_msg.center.z);

        // 计算装甲板在能量机关坐标系中的位置
        // 考虑角度变化和初始roll角
        Eigen::Vector3d p_rune =
            eulerToMatrix(Eigen::Vector3d{target_msg.roll - angle_diff, 0, target_msg.yaw}) *
            Eigen::Vector3d(0, -ARM_LENGTH, 0);  // 装甲板相对中心的位置

        // 变换到odom坐标系
        Eigen::Vector3d p_odom = p_rune + t_odom_2_rune;

        return p_odom;
    }
};

}  // namespace rm_auto_aim
#endif  // BALLISTIC_CALCULATION_AIM_INFO_HPP_