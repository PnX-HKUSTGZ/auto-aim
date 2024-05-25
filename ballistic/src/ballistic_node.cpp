// node definition
#include "ballistic/ballistic_node.hpp"
// msg & ros2
#include <auto_aim_interfaces/msg/detail/firecontrol__struct.hpp>
#include <auto_aim_interfaces/msg/firecontrol.hpp>
#include <auto_aim_interfaces/msg/target.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
// non-ros2 implementation
#include "ballistic/ballistic.hpp"
// STD
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace rm_auto_aim
{
BallisticNode::BallisticNode(const rclcpp::NodeOptions & options) : Node("ballistic", options)
{
  RCLCPP_INFO(this->get_logger(), "start ballistic calculation node");
  subscription_ = this->create_subscription<auto_aim_interfaces::msg::Target>(
    "/tracker/target", rclcpp::SensorDataQoS(),
    std::bind(&BallisticNode::targetCallback, this, std::placeholders::_1));
  publisher_ = this->create_publisher<auto_aim_interfaces::msg::Firecontrol>("/firecontrol", 10);
  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(1), std::bind(&BallisticNode::timerCallback, this));
  // register param callback
  param_cb_handle_ = this->add_on_set_parameters_callback(
    std::bind(&BallisticNode::parameterCallback, this, std::placeholders::_1));

  this->ballistic = std::make_unique<Ballistic>(loadParams());
}

void BallisticNode::targetCallback(const auto_aim_interfaces::msg::Target::SharedPtr target_msg)
{
  first_recv_target_ = true;
  // save target_msg and info
  last_target_msg_ = target_msg;
  last_target_time_ = target_msg->header.stamp;
  info_.vx = target_msg->velocity.x;
  info_.vy = target_msg->velocity.y;
  info_.vz = target_msg->velocity.z;
  info_.x = target_msg->position.x;
  info_.y = target_msg->position.y;
  info_.z = target_msg->position.z;
  info_.yaw = target_msg->yaw;
  info_.r1 = target_msg->radius_1;
  info_.r2 = target_msg->radius_2;
  info_.w = target_msg->v_yaw;
  info_.id = target_msg->id;
  info_.type = static_cast<TargetType>(target_msg->armors_num);

  // triger timerCallback for now
  timerCallback();
}

void BallisticNode::timerCallback()
{
  if (!first_recv_target_) return;

  rclcpp::Time now = this->now();  // @todo: + solve latency
  rclcpp::Duration duration = now - last_target_time_;
  double t = duration.seconds();
  // do prediction, update target info
  info_.x += info_.vx * t;
  info_.y += info_.vy * t;
  info_.z += info_.vz * t;
  info_.yaw += info_.w * t;

  this->ballistic->solveBallistic(info_);

  //发布消息
  auto_aim_interfaces::msg::Firecontrol fire_msg;
  fire_msg.header.stamp = now;  // @todo: align with now?
  // fill msg...
  // fire_msg.xxx = xxx;

  // debug msg
  // ...

  publisher_->publish(fire_msg);
}

Ballistic::BallisticParam BallisticNode::loadParams()
{
  Ballistic::BallisticParam param;
  param.bullet_speed = declare_parameter("bullet_speed", 27.0);
  param.gravity = declare_parameter("gravity", 9.805);
  param.air_resistance_k = declare_parameter("air_resistance_k", 0.01);
  param.step_cv = declare_parameter("step_cv", 1);
  param.thre_cv = declare_parameter("thre_cv", 1);
  param.mx_iter_cv = declare_parameter("mx_iter_cv", 5);
  param.step_holonomic = declare_parameter("step_holonomic", 1);
  param.thre_holonomic = declare_parameter("thre_holonomic", 1);
  param.mx_iter_holonomic = declare_parameter("mx_iter_holonomic", 5);
  param.step_calc_pitch = declare_parameter("step_calc_pitch", 1);
  param.thre_calc_pitch = declare_parameter("thre_calc_pitch", 1);
  param.mx_iter_calc_pitch = declare_parameter("mx_iter_calc_pitch", 20);
  param.extra_compensation = declare_parameter("extra_compensation", 0.1);
  return param;
}

rcl_interfaces::msg::SetParametersResult BallisticNode::parameterCallback(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto & param : parameters) {
    if (param.get_name() == "bullet_speed") {
      this->ballistic->param_.bullet_speed = param.as_double();
    } else if (param.get_name() == "gravity") {
      this->ballistic->param_.gravity = param.as_double();
    } else if (param.get_name() == "air_resistance_k") {
      this->ballistic->param_.air_resistance_k = param.as_double();
    } else if (param.get_name() == "step_cv") {
      this->ballistic->param_.step_cv = param.as_double();
    } else if (param.get_name() == "thre_cv") {
      this->ballistic->param_.thre_cv = param.as_double();
    } else if (param.get_name() == "mx_iter_cv") {
      this->ballistic->param_.mx_iter_cv = param.as_double();
    } else if (param.get_name() == "step_holonomic") {
      this->ballistic->param_.step_holonomic = param.as_double();
    } else if (param.get_name() == "thre_holonomic") {
      this->ballistic->param_.thre_holonomic = param.as_double();
    } else if (param.get_name() == "mx_iter_holonomic") {
      this->ballistic->param_.mx_iter_holonomic = param.as_double();
    } else if (param.get_name() == "step_calc_pitch") {
      this->ballistic->param_.step_calc_pitch = param.as_double();
    } else if (param.get_name() == "thre_calc_pitch") {
      this->ballistic->param_.thre_calc_pitch = param.as_double();
    } else if (param.get_name() == "mx_iter_calc_pitch") {
      this->ballistic->param_.mx_iter_calc_pitch = param.as_double();
    } else if (param.get_name() == "extra_compensation") {
      this->ballistic->param_.extra_compensation = param.as_double();
    } else {
      result.successful = false;
      result.reason = "unknown parameter";
    }
  }
  return result;
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::BallisticNode)
