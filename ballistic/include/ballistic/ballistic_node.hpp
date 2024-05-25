#ifndef __BALLISTIC_NODE_H__
#define __BALLISTIC_NODE_H__

#include <auto_aim_interfaces/msg/firecontrol.hpp>
#include <auto_aim_interfaces/msg/target.hpp>
#include <memory>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/timer.hpp>

#include "ballistic.hpp"

namespace rm_auto_aim
{
class BallisticNode : public rclcpp::Node
{
public:
  explicit BallisticNode(const rclcpp::NodeOptions & options);

private:
  rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr subscription_;
  rclcpp::Publisher<auto_aim_interfaces::msg::Firecontrol>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  std::unique_ptr<Ballistic> ballistic;

  void targetCallback(const auto_aim_interfaces::msg::Target::SharedPtr msg);

  void timerCallback();

  Ballistic::BallisticParam loadParams();

  rcl_interfaces::msg::SetParametersResult parameterCallback(
    const std::vector<rclcpp::Parameter> & parameters);

  bool first_recv_target_ = false;
  rclcpp::Time last_target_time_;
  Ballistic::TargetInfo info_;
  auto_aim_interfaces::msg::Target::SharedPtr last_target_msg_;
};
}  // namespace rm_auto_aim

#endif  // !__BALLISTIC_NODE_H__