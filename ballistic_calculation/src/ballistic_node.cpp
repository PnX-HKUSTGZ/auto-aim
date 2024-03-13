// STD
#include <auto_aim_interfaces/msg/detail/firecontrol__struct.hpp>
#include <cmath>
#include <memory> 
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/subscription.hpp>
#include <auto_aim_interfaces/msg/target.hpp>
#include <auto_aim_interfaces/msg/firecontrol.hpp>

#include"ballistic_calculation/ballistic_node.hpp"



namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::Target;
using firemsg = auto_aim_interfaces::msg::Firecontrol;


BallisticCalculateNode::BallisticCalculateNode(const rclcpp::NodeOptions & options)
: Node("ballistic_calculate", options)
{
    RCLCPP_INFO(this->get_logger(), "start ballistic calculation!");
    subscription_ = this->create_subscription<auto_aim_interfaces::msg::Target>(
      "/tracker/target", 10, std::bind(&BallisticCalculateNode::targetCallback, this, std::placeholders::_1));
    
    publisher_ = this->create_publisher<auto_aim_interfaces::msg::Firecontrol>("/firecontrol", 10);

    K1  = this->declare_parameter("iteration_coeffcient_first",0.1);
    K2  = this->declare_parameter("iteration_coeffcient_second",0.05);
    K   = this->declare_parameter("air_resistence",0.1);
    BULLET_V = this->declare_parameter("bullet_speed",50.0);
  
}

void BallisticCalculateNode::targetCallback(const auto_aim_interfaces::msg::Target::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(),"Receive target ID");

    calculator = std::make_unique<Ballistic>(BULLET_V , *msg , K , K1 , K2);

    //进入第一次大迭代
    double init_t = std::sqrt(msg->position.z * 2 / 9.8);
    double init_pitch = 0.0;

    std::pair<double,double> first_iteration_result = calculator->iteration1(THRES1 , init_pitch , init_t);

    //预测并选择合适击打的装甲板
    double temp_theta = first_iteration_result.first;
    double temp_t = first_iteration_result.second;
    std::vector<double>hit_aim = calculator->predictBestArmor(temp_t);
    double x = hit_aim[0];
    double y = hit_aim[1];

    //进入第二次大迭代
    std::pair<double,double> final_result = calculator->iteration2(THRES2 , temp_theta , temp_t , x , y);
    
    //发布消息
    firemsg fire_msg;
    fire_msg.pitch = final_result.first;
    fire_msg.yaw = std::atan2(y , x);
    publisher_->publish(fire_msg);
    
}


    

}// namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::BallisticCalculateNode)

