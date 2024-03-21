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
    //创建订阅者
    subscription_ = this->create_subscription<auto_aim_interfaces::msg::Target>(
      "/tracker/target", 10, std::bind(&BallisticCalculateNode::targetCallback, this, std::placeholders::_1));
    //创建发布者
    publisher_ = this->create_publisher<auto_aim_interfaces::msg::Firecontrol>("/firecontrol", 10);
    //设置时间callback
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1), std::bind(&BallisticCalculateNode::timerCallback, this));
    
    K1  = this->declare_parameter("iteration_coeffcient_first",0.1);
    K2  = this->declare_parameter("iteration_coeffcient_second",0.05);
    K   = this->declare_parameter("air_resistence",0.1);
    BULLET_V = this->declare_parameter("bullet_speed",50.0);
  
}




void BallisticCalculateNode::targetCallback( auto_aim_interfaces::msg::Target::SharedPtr target_msg)
{

  RCLCPP_INFO(this->get_logger(),"Receive target ID");
  
  this->target_msg = std::move(target_msg);
  ifstart = this->target_msg->tracking;

}





void BallisticCalculateNode::timerCallback()
{
    

    if(!ifstart){
      
      return;
    }
  
    RCLCPP_INFO(this->get_logger(),"hey");

    rclcpp::Time now = this->now();
    rclcpp::Time msg_time = target_msg->header.stamp;
    rclcpp::Duration duration = now - msg_time;

    target_msg->position.x = target_msg->position.x + target_msg->velocity.x * duration.seconds();
    target_msg->position.y = target_msg->position.y + target_msg->velocity.y * duration.seconds();
    target_msg->position.z = target_msg->position.z + target_msg->velocity.z * duration.seconds();
    target_msg->yaw = target_msg->yaw + target_msg->v_yaw*  duration.seconds();

    calculator = std::make_unique<rm_auto_aim::Ballistic>(*target_msg);
    
    //进入第一次大迭代
    double init_pitch = std::atan(target_msg->position.z / std::sqrt(target_msg->position.x * target_msg->position.x + target_msg->position.y * target_msg->position.y));
    double init_t = std::sqrt(target_msg->position.x * target_msg->position.x + target_msg->position.y * target_msg->position.y) / (cos(init_pitch) * this->calculator->bulletV);
    
    
    std::pair<double,double> first_iteration_result = this->calculator->iteration1(THRES1 , init_pitch , init_t);
    
    //预测并选择合适击打的装甲板
    double temp_theta = first_iteration_result.first;
    double temp_t = first_iteration_result.second;

    //预测平衡步兵的最佳装甲板
    double chosen_yaw;
    double z;
    double r;

    if(target_msg->armors_num == 2){
      std::vector<double>hit_aim = calculator->predictBalanceBestArmor(temp_t);
        
      chosen_yaw = hit_aim[0];
      z = hit_aim[1];
      r = hit_aim[2];
    }
    else if (target_msg->armors_num == 4){
      std::vector<double>hit_aim = calculator->predictInfantryBestArmor(temp_t);
        
      chosen_yaw = hit_aim[0];
      z = hit_aim[1];
      r = hit_aim[2];
    }
    else{
      RCLCPP_ERROR(this->get_logger(),"The number of armors is not 2 or 4");
    
    }
    

    //进入第二次大迭代
    std::pair<double,double> final_result = calculator->iteration2(THRES2 , temp_theta , temp_t , chosen_yaw , z , r);
    
    
    //发布消息
    firemsg fire_msg;
    fire_msg.pitch = final_result.first;
    fire_msg.yaw = final_result.second;
    fire_msg.tracking = target_msg->tracking;
    fire_msg.id = target_msg->id;
    publisher_->publish(fire_msg);
    
    
    //新图像数据未到来，进行预测



    
    
  
}

    

}// namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::BallisticCalculateNode)

