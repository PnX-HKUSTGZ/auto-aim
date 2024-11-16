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
#include "ballistic_calculation/ballistic_calculation.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

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
    K1  = this->declare_parameter("iteration_coeffcient_first",0.1);
    K2  = this->declare_parameter("iteration_coeffcient_second",0.05);
    K   = this->declare_parameter("air_resistence",0.1);
    BULLET_V = this->declare_parameter("bullet_speed",24.8);
    ifFireK = this->declare_parameter("ifFireK",0.05);
    min_v = this->declare_parameter("swich_stategy_1",30) * M_PI / 30;
    max_v = this->declare_parameter("swich_stategy_2",120) * M_PI / 30;
    v_yaw_PTZ = this->declare_parameter("max_v_yaw_PTZ", 0.8); 

    calculator = std::make_unique<rm_auto_aim::Ballistic>(K , K1 , K2 , BULLET_V);
    
    //创建监听器，监听云台位姿    
    tfBuffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());   
    tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer,this);

    //创建订阅者
    subscription_ = this->create_subscription<auto_aim_interfaces::msg::Target>(
      "/tracker/target",rclcpp::SensorDataQoS() , std::bind(&BallisticCalculateNode::targetCallback, this, std::placeholders::_1));
    //创建发布者
    publisher_ = this->create_publisher<auto_aim_interfaces::msg::Firecontrol>("/firecontrol", 10);
    //设置时间callback
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1), std::bind(&BallisticCalculateNode::timerCallback, this));
    
    
  
}

void BallisticCalculateNode::targetCallback( auto_aim_interfaces::msg::Target::SharedPtr _target_msg)
{
  
  
  
  this->target_msg = std::move(_target_msg);

  ifstart = this->target_msg->tracking;
  
}


bool BallisticCalculateNode::ifFire(double targetpitch, double targetyaw)
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
        return std::abs(pitch - targetpitch) < ifFireK && std::abs(yaw - targetyaw) < ifFireK;
    
    }


void BallisticCalculateNode::timerCallback()
{
    
    if(!ifstart){
      
      return;
    }

  
    //use linear interpolation to update the target_msg when the image not arriving
    rclcpp::Time now = this->now();
    rclcpp::Time msg_time = target_msg->header.stamp;
    rclcpp::Duration duration = now - msg_time;
    
    target_msg->position.x = target_msg->position.x + target_msg->velocity.x * duration.seconds();
    target_msg->position.y = target_msg->position.y + target_msg->velocity.y * duration.seconds();
    target_msg->position.z = target_msg->position.z + target_msg->velocity.z * duration.seconds();
    target_msg->yaw = target_msg->yaw + target_msg->v_yaw*  duration.seconds();
    //add duration to update targetmsg.stamp
    target_msg->header.stamp = now;
    
    calculator->target_msg.header = target_msg->header;
    calculator->target_msg.tracking = target_msg->tracking;
    calculator->target_msg.id = target_msg->id;
    calculator->target_msg.armors_num = target_msg->armors_num;
    calculator->target_msg.position = target_msg->position;
    calculator->target_msg.velocity = target_msg->velocity;
    calculator->target_msg.yaw = target_msg->yaw;
    calculator->target_msg.v_yaw = target_msg->v_yaw;
    calculator->target_msg.radius_1 = target_msg->radius_1;
    calculator->target_msg.radius_2 = target_msg->radius_2;
    calculator->target_msg.dz = target_msg->dz;
    calculator->robotcenter = target_msg->position;
    calculator->velocity = target_msg->velocity;
    
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

    if (target_msg->armors_num == 4){
      std::vector<double>hit_aim = calculator->predictInfantryBestArmor(temp_t, min_v, max_v, v_yaw_PTZ);
      chosen_yaw = hit_aim[0];
      z = hit_aim[1];  
      r = hit_aim[2];
    }
    else{
      RCLCPP_ERROR(this->get_logger(),"The number of armors is not 4");
    
    }
    

    //进入第二次大迭代
    std::pair<double,double> final_result = calculator->iteration2(THRES2 , temp_theta , temp_t , chosen_yaw , z , r);
    
    
    
    

    //发布消息
    firemsg fire_msg;
    fire_msg.header = target_msg->header;
    fire_msg.pitch = final_result.first;
    fire_msg.yaw = final_result.second;
    fire_msg.tracking = target_msg->tracking;
    fire_msg.id = target_msg->id;
    fire_msg.iffire = ifFire(final_result.first,final_result.second);
    publisher_->publish(fire_msg);
    
    
    //新图像数据未到来，进行预测

  
}

    

}// namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::BallisticCalculateNode)

