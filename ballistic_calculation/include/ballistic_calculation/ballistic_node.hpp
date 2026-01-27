#ifndef BLNODELIBRARY_HPP
#define BLNODELIBRARY_HPP

#include <auto_aim_interfaces/msg/detail/firecontrol__struct.hpp>
#include <auto_aim_interfaces/msg/firecontrol.hpp>
#include <auto_aim_interfaces/msg/target.hpp>
#include <auto_aim_interfaces/msg/rune_target.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
// STD
#include <cv_bridge/cv_bridge.h>

// visualization
#include <visualization_msgs/msg/marker.hpp>

#include <memory>
#include <opencv2/core.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <string>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float32.hpp>

#include "ballistic_calculation/ballistic_calculator.hpp"
#include "ballistic_calculation/aim_info.hpp"
#include "ballistic_calculation/armor_selector.hpp"
#include "ballistic_calculation/mpc_controller.hpp"
#include <geometry_msgs/msg/point_stamped.hpp>

namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::Target;

class BallisticCalculateNode : public rclcpp::Node
{
public:
    /**
     * @brief 构造函数
     * 
     * 初始化节点参数、订阅者、发布者和各种计算组件
     * 
     * @param options ROS2节点选项
     */
    explicit BallisticCalculateNode(const rclcpp::NodeOptions & options);

private:

    /**
     * @brief 装甲车辆目标回调函数
     * 
     * 处理装甲车辆目标信息，进行弹道计算和装甲板选择，
     * 最终发布火控指令
     * 
     * @param msg 目标消息
     */
    void carTargetCallback(const auto_aim_interfaces::msg::Target::SharedPtr msg);
    
    /**
     * @brief 能量机关目标回调函数
     * 
     * 处理能量机关目标信息，进行弹道计算，
     * 最终发布火控指令
     * 
     * @param msg 能量机关目标消息
     */
    void runeTargetCallback(const auto_aim_interfaces::msg::RuneTarget::SharedPtr msg);

    // 核心计算组件
    std::unique_ptr<Ballistic> calculator;           // 弹道计算器
    std::shared_ptr<ArmorSelector> armor_selector_;  // 装甲板选择器
    
    // ROS2通信组件
    rclcpp::Publisher<auto_aim_interfaces::msg::Firecontrol>::SharedPtr publisher_;  // 火控指令发布者
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr aim_point_pub_;  // 瞄准点可视化发布者

    // 装甲车辆目标相关
    rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr car_target_sub_;  // 目标订阅者
    auto_aim_interfaces::msg::Target::SharedPtr car_target_msg;  // 目标消息缓存
    std::shared_ptr<CarInfo> car_info_;     // 车辆信息处理器
    std::unique_ptr<ArmorInfo> armor_info_; // 装甲板信息处理器

    // 能量机关目标相关
    rclcpp::Subscription<auto_aim_interfaces::msg::RuneTarget>::SharedPtr rune_target_sub_;  // 能量机关订阅者
    auto_aim_interfaces::msg::RuneTarget::SharedPtr rune_target_msg;  // 能量机关消息缓存
    std::unique_ptr<RuneInfo> rune_info_;  // 能量机关信息处理器

    // tf2坐标变换
    std::shared_ptr<tf2_ros::Buffer> tfBuffer;         // tf2缓冲区
    std::shared_ptr<tf2_ros::TransformListener> tfListener;  // tf2监听器
    std::string target_frame_;  // 目标坐标系

    // 算法参数（通过ROS参数服务器配置）
    double K1;             // 第一次大迭代时的步长
    double K2;             // 第二次大迭代时的步长
    double K;              // 空气阻力系数
    double BULLET_V;       // 子弹出膛速度
    static const double THRES1;  // 第一次迭代的收敛阈值
    static const double THRES2; // 第二次迭代的收敛阈值
    double min_v;          // 一级策略切换二级策略速度临界值
    double max_v;          // 二级策略切换三级策略速度临界值
    double v_yaw_gimble;   // 云台最大yaw速度
    double stop_fire_time; // 停止开火时间
    std::vector<double> xyz_vec;  // 枪口的xyz坐标
    std::vector<double> rpy_vec;  // 枪口的rpy角度

    // 状态变量
    bool ifstart = false;           // 是否开始标志
    int rate = 1000;               // 节点运行频率
    rclcpp::Time last_fire_time;   // 上次开火时间
    float current_yaw_vel = 0.0;
    float current_pitch_vel = 0.0;

    bool use_mpc_default;

    // 相机信息和图像投影
    sensor_msgs::msg::CameraInfo cam_info_;  // 相机内参信息
    cv::Point2f cam_center_;                 // 相机中心点
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;  // 相机信息订阅者

    /**
     * @brief 将3D点投影到图像平面
     * 
     * 使用相机内参和坐标变换，将odom坐标系中的3D点
     * 投影到图像平面上
     * 
     * @param point_3d odom坐标系中的3D点
     * @return cv::Point2f 图像平面上的2D点（归一化坐标）
     */
    cv::Point2f projectPointToImage(const Eigen::Vector3d & point_3d);

    Eigen::Vector4d getCurrentGimbalState();

    std::unique_ptr<rm_auto_aim::MPCController> mpc_controller_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr gimbal_vel_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr mpc_yaw_tracker_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr iffire_pub_;
};
}  // namespace rm_auto_aim

#endif  // BLNODELIBRARY_HPP