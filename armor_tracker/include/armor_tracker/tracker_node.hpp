// Copyright 2022 Chen Jun

#ifndef ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_
#define ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_

// ROS
#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include <opencv2/core/types.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <image_transport/image_transport.hpp>

// STD
#include <memory> 
#include <string>
#include <vector>

#include "armor_tracker/types.hpp"
#include "armor_tracker/tracker.hpp"
#include "armor_tracker/tracker_manager.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/tracker_info.hpp"
#include "auto_aim_interfaces/srv/set_mode.hpp"

// OpenCV
#include <opencv2/core.hpp>
#include <cv_bridge/cv_bridge.h>

namespace rm_auto_aim
{
using tf2_filter = tf2_ros::MessageFilter<auto_aim_interfaces::msg::Armors>;
class ArmorTrackerNode : public rclcpp::Node
{
public:
  explicit ArmorTrackerNode(const rclcpp::NodeOptions & options);

private:
  void initializeEKF(); 
  void armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_ptr);

  void publishMarkers(const auto_aim_interfaces::msg::Target & target_msg);

  void setModeCallback(const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
    std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response);

  void publishImgAll(
    const auto_aim_interfaces::msg::Target & target_msg,
    cv::Mat & image,
    bool is_primary_target);

  // Maximum allowable armor distance in the XOY plane
  double max_armor_distance_;

  // The time when the last message was received
  rclcpp::Time last_time_ = rclcpp::Time(0);
  double dt_;
  bool debug_;

  // Armor tracker
  double s2qxy_, s2qz_, s2qyaw_, s2qr_;
  double r_xyz_factor, r_yaw, r_radius;
  double lost_time_thres_;
  std::unique_ptr<TrackerManager> tracker_manager_;
  

  // Reset tracker service
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_tracker_srv_;

  // set_mode service
  rclcpp::Service<auto_aim_interfaces::srv::SetMode>::SharedPtr set_mode_srv_;

  // Subscriber with tf2 message_filter
  std::string target_frame_;
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
  message_filters::Subscriber<auto_aim_interfaces::msg::Armors> armors_sub_;
  std::shared_ptr<tf2_filter> tf2_filter_;

  // Tracker info publisher
  rclcpp::Publisher<auto_aim_interfaces::msg::TrackerInfo>::SharedPtr info_pub_;

  // Publisher
  rclcpp::Publisher<auto_aim_interfaces::msg::Target>::SharedPtr target_pub_;

  // Visualization marker publisher
  visualization_msgs::msg::Marker position_marker_;
  visualization_msgs::msg::Marker linear_v_marker_;
  visualization_msgs::msg::Marker angular_v_marker_;
  visualization_msgs::msg::Marker armor_marker_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // 相机参数
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  sensor_msgs::msg::CameraInfo cam_info_;
  cv::Point2f cam_center_;

  // 发布图像
  image_transport::Publisher tracker_img_pub_;
  std_msgs::msg::Header_<std::allocator<void>>::_stamp_type last_img_time_; 

  VisionMode mode_ = VisionMode::AUTO; // 默认模式为AUTO
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_
