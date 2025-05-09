#ifndef RUNE_BALLISTIC_NODE_HPP
#define RUNE_BALLISTIC_NODE_HPP

#include <auto_aim_interfaces/msg/detail/firecontrol__struct.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/subscription.hpp>

#include <auto_aim_interfaces/msg/rune_target.hpp>
#include <auto_aim_interfaces/msg/firecontrol.hpp>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
// STD
#include <memory> 
#include <string>


#include "rune_ballistic_calculation.hpp"


namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::RuneTarget;


class RuneBallisticNode : public rclcpp::Node
{
public:
    explicit RuneBallisticNode(const rclcpp::NodeOptions & options);


private:
    //接收串口的云台位姿，当云台位姿和枪口预测位置相差小于某一个阈值时，返回true
    bool ifFire(double prepitch , double preyaw);

    void targetCallback(const auto_aim_interfaces::msg::RuneTarget::SharedPtr msg);

    void timerCallback();

    rclcpp::Subscription<auto_aim_interfaces::msg::RuneTarget>::SharedPtr subscription_;
    rclcpp::Publisher<auto_aim_interfaces::msg::Firecontrol>::SharedPtr publisher_;
    std::unique_ptr<RuneBallistic>calculator;
    auto_aim_interfaces::msg::RuneTarget::SharedPtr target_msg;
    rclcpp::TimerBase::SharedPtr timer_;

//tf2
    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    geometry_msgs::msg::TransformStamped t;
    std::string target_frame_;

    double K;//迭代时的步长，需要parameter_declare来调整参数
    double k; //空气阻力系数，需要parameter_declare来调整参数
    double BULLET_V;//子弹出膛速度，需要parameter_declare来调整参数
    double THRES = 0.005;//迭代的阈值，需要parameter_declare来调整参数
    double ifFireK;//判断是否开火的阈值，需要parameter_declare来调整参数
    Eigen::Vector3d odom2gun = Eigen::Vector3d(0.0, 0.0, 0.0);//枪口到odom的坐标变换

    bool ifstart = false;
    int rate = 1000;

};



}// namespace rm_auto_aim





#endif // RUNE_BALLISTIC_NODE_HPP