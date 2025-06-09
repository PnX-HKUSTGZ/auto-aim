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

#include <memory>
#include <opencv2/core.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <string>

#include "ballistic_calculator.hpp"
#include "types.hpp"

namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::Target;

class BallisticCalculateNode : public rclcpp::Node
{
public:
    explicit BallisticCalculateNode(const rclcpp::NodeOptions & options);

private:
    //接收串口的云台位姿，当云台位姿和枪口预测位置相差小于某一个阈值时，返回true
    bool ifFire(double prepitch, double preyaw);

    void carTargetCallback(const auto_aim_interfaces::msg::Target::SharedPtr msg);
    void runeTargetCallback(const auto_aim_interfaces::msg::RuneTarget::SharedPtr msg);

    std::unique_ptr<Ballistic> calculator;
    rclcpp::Publisher<auto_aim_interfaces::msg::Firecontrol>::SharedPtr publisher_;

    rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr car_target_sub_;
    auto_aim_interfaces::msg::Target::SharedPtr car_target_msg;
    std::shared_ptr<CarInfo> car_info_;
    std::unique_ptr<ArmorInfo> armor_info_;

    rclcpp::Subscription<auto_aim_interfaces::msg::RuneTarget>::SharedPtr rune_target_sub_;
    auto_aim_interfaces::msg::RuneTarget::SharedPtr rune_target_msg;
    std::unique_ptr<RuneInfo> rune_info_;

    //tf2
    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    std::string target_frame_;

    double K1;             //第一次大迭代时的步长，需要parameter_declare来调整参数
    double K2;             //第一次大迭代时的步长，需要parameter_declare来调整参数
    double K;              //空气阻力系数，需要parameter_declare来调整参数
    double BULLET_V;       //子弹出膛速度，需要parameter_declare来调整参数
    double THRES1 = 0.01;  //第一次迭代的阈值，需要parameter_declare来调整参数
    double THRES2 = 0.005;  //第二次迭代的阈值，需要parameter_declare来调整参数
    double ifFireK_;        //判断是否开火的阈值，需要parameter_declare来调整参数
    double min_v;  //一级策略切换二级策略速度临界，需要parameter_declare来调整参数
    double max_v;  //二级策略切换三级策略速度临界，需要parameter_declare来调整参数
    double v_yaw_gimble;             //云台最大yaw速度，需要parameter_declare来调整参数
    double stop_fire_time;        //停止开火时间，需要parameter_declare来调整参数
    double fire_delay;  //开火延迟，需要parameter_declare来调整参数
    std::vector<double> xyz_vec;  //枪口的xyz坐标
    std::vector<double> rpy_vec;  //枪口的rpy角度

    bool ifstart = false;
    int rate = 1000;
    double ifFireK;
    rclcpp::Time last_fire_time;

    // 相机信息
    sensor_msgs::msg::CameraInfo cam_info_;
    cv::Point2f cam_center_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;

    // 重投影相关函数
    cv::Point2f projectPointToImage(const Eigen::Vector3d & point_3d);
};

}  // namespace rm_auto_aim

#endif  // BLNODELIBRARY_HPP