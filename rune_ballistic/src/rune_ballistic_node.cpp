// STD
#include <auto_aim_interfaces/msg/detail/firecontrol__struct.hpp>
#include <cmath>
#include <memory> 
#include <string>
#include <vector>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/subscription.hpp>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

#include <auto_aim_interfaces/msg/rune_target.hpp>
#include <auto_aim_interfaces/msg/firecontrol.hpp>

#include "rune_ballistic/rune_ballistic_node.hpp"
#include "rune_ballistic/rune_ballistic_calculation.hpp"






namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::RuneTarget;
using firemsg = auto_aim_interfaces::msg::Firecontrol;


RuneBallisticNode::RuneBallisticNode(const rclcpp::NodeOptions & options)
: Node("rune_ballistic", options)
{
    RCLCPP_INFO(this->get_logger(), "start rune ballistic calculation!");
    K  = this->declare_parameter("iteration_coeffcient",0.1);
    k   = this->declare_parameter("air_resistence",0.1);
    BULLET_V = this->declare_parameter("bullet_speed",24.8);
    ifFireK = this->declare_parameter("ifFireK",0.05);
    std::vector<double> odom2gun_vec = this->declare_parameter("odom2gun", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    Eigen::Vector3d odom2gunxyz(odom2gun_vec[0], odom2gun_vec[1], odom2gun_vec[2]);
    Eigen::Vector3d odom2gunrpy(odom2gun_vec[3], odom2gun_vec[4], odom2gun_vec[5]);
    calculator = std::make_unique<rm_auto_aim::RuneBallistic>(k , K , BULLET_V, odom2gunxyz, odom2gunrpy);
    
    //创建监听器，监听云台位姿    
    tfBuffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());   
    tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer,this);

    //创建订阅者
    subscription_ = this->create_subscription<auto_aim_interfaces::msg::RuneTarget>(
        "rune_solver/rune_target",rclcpp::SensorDataQoS() ,
        std::bind(&RuneBallisticNode::targetCallback, this, std::placeholders::_1));
    //创建发布者
    publisher_ = this->create_publisher<auto_aim_interfaces::msg::Firecontrol>("/firecontrol", 10);
    //设置时间callback
    timer_ = this->create_wall_timer(std::chrono::milliseconds(4), std::bind(&RuneBallisticNode::timerCallback, this));
    
    
  
}

void RuneBallisticNode::targetCallback( auto_aim_interfaces::msg::RuneTarget::SharedPtr _target_msg)
{
    this->target_msg = std::move(_target_msg);
    ifstart = this->target_msg->tracking;
}


bool RuneBallisticNode::ifFire(double targetpitch, double targetyaw)
{
    //获取当前云台位姿
    try{
        t = tfBuffer->lookupTransform("gimbal_link", "odom", tf2::TimePointZero);
    }
    catch (tf2::TransformException &ex) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "%s", ex.what());
        return false;
    }
    
    tf2::Quaternion q(
    t.transform.rotation.x,
    t.transform.rotation.y,
    t.transform.rotation.z,
    t.transform.rotation.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    
    //计算云台位姿和预测位置的差值,当差值小于某一个阈值时，返回true
    return std::abs(-yaw - targetyaw) < ifFireK;

}


void RuneBallisticNode::timerCallback()
{
    if(!ifstart) return;
    //add duration to update targetmsg.stamp
    calculator->target_msg = *target_msg;
    Eigen::Vector3d target = calculator->getTarget(target_msg->header.stamp.sec); 
    
    //进入迭代
    double init_pitch = std::atan(target[2] / std::sqrt(target[0] * target[0] + target[1] * target[1]));
    double init_t = std::sqrt(target[0] * target[0] + target[1] * target[1]) / (cos(init_pitch) * BULLET_V);
    
    
    std::pair<double,double> iteration_result = this->calculator->iteration(THRES , init_pitch , init_t);
    //发布消息
    firemsg fire_msg;
    fire_msg.header = target_msg->header;
    fire_msg.pitch = iteration_result.first;
    fire_msg.yaw = iteration_result.second;
    fire_msg.tracking = target_msg->tracking;
    fire_msg.id = "rune"; 
    fire_msg.iffire = ifFire(iteration_result.first,iteration_result.second);
    publisher_->publish(fire_msg);
    
    
    //新图像数据未到来，进行预测

  
}

    

}// namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::RuneBallisticNode)

