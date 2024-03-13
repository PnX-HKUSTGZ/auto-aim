#ifndef BLNODELIBRARY_HPP
#define BLNODELIBRARY_HPP


#include <auto_aim_interfaces/msg/detail/firecontrol__struct.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/subscription.hpp>
#include <auto_aim_interfaces/msg/target.hpp>
#include <auto_aim_interfaces/msg/firecontrol.hpp>
// STD
#include <memory> 
#include <string>
#include <vector>

#include "ballistic_calculation.hpp"


namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::Target;


class BallisticCalculateNode : public rclcpp::Node
{
public:
  explicit BallisticCalculateNode(const rclcpp::NodeOptions & options);


private:
void targetCallback(const auto_aim_interfaces::msg::Target::SharedPtr msg);

    rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr subscription_;
    rclcpp::Publisher<auto_aim_interfaces::msg::Firecontrol>::SharedPtr publisher_;
    std::unique_ptr<Ballistic>calculator;



    double K1;//第一次大迭代时的步长，需要parameter_declare来调整参数
    double K2;//第一次大迭代时的步长，需要parameter_declare来调整参数
    double K; //空气阻力系数，需要parameter_declare来调整参数
    double BULLET_V;//子弹出膛速度，需要parameter_declare来调整参数
    double THRES1 = 0.05;//第一次迭代的阈值，需要parameter_declare来调整参数
    double THRES2 = 0.01;//第二次迭代的阈值，需要parameter_declare来调整参数

};



}// namespace rm_auto_aim





#endif // BLNODELIBRARY_HPP