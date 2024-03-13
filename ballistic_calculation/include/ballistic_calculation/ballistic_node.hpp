#ifndef BLNODELIBRARY_HPP
#define BLNODELIBRARY_HPP


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
void targetCallback(const auto_aim_interfaces::msg::Target msg){};

    rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr subscription_;
    std::unique_ptr<Ballistic>calculator;
};

} // namespace rm_auto_aim


#endif // BLNODELIBRARY_HPP