// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_NODE_HPP_
#define ARMOR_DETECTOR__DETECTOR_NODE_HPP_

// ROS
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

// STD
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>

#include "armor_detector/detector.hpp"
#include "armor_detector/number_classifier.hpp"
#include "armor_detector/light_corner_corrector.hpp"
#include "armor_detector/ba_solver.hpp"
#include "armor_detector/pnp_solver.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/srv/set_mode.hpp"


namespace rm_auto_aim
{

class ArmorDetectorNode : public rclcpp::Node
{
public:
  ArmorDetectorNode(const rclcpp::NodeOptions & options);

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);

  void setModeCallback(const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
                      std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response);

  std::unique_ptr<Detector> initDetector();
  std::vector<Armor> detectArmors(const sensor_msgs::msg::Image::ConstSharedPtr & img_msg, cv::Mat & img); 
  void drawResults(const sensor_msgs::msg::Image::ConstSharedPtr & img_msg, cv::Mat & img, const std::vector<Armor> & armors); 

  void createDebugPublishers();
  void destroyDebugPublishers();

  void publishMarkers();
  void chooseBestPose(Armor & armor, const std::vector<cv::Mat> & rvecs, const std::vector<cv::Mat> & tvecs);
  void fix_two_armors(Armor & armor1, Armor & armor2);
  // Light corner corrector
  LightCornerCorrector lcc;

  //dynamic parameter
  OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
  rcl_interfaces::msg::SetParametersResult onParameterChanged(const std::vector<rclcpp::Parameter> &parameters);
  
  // Armor Detector
  std::unique_ptr<Detector> detector_;

  // set_mode service
  rclcpp::Service<auto_aim_interfaces::srv::SetMode>::SharedPtr set_mode_srv_;

  // Detected armors publisher
  auto_aim_interfaces::msg::Armors armors_msg_;
  rclcpp::Publisher<auto_aim_interfaces::msg::Armors>::SharedPtr armors_pub_;

  // Visualization marker publisher
  visualization_msgs::msg::Marker armor_marker_;
  visualization_msgs::msg::Marker text_marker_;
  visualization_msgs::msg::MarkerArray marker_array_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // Camera info part
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  cv::Point2f cam_center_;
  std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
  std::unique_ptr<PnPSolver> pnp_solver_;
  std::unique_ptr<BaSolver> ba_solver_;

  // Image subscrpition
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

  // tf2
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
  Eigen::Matrix3d r_odom_to_camera;
  Eigen::Vector3d t_odom_to_camera;

  // Debug information
  bool debug_;
  std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;
  rclcpp::Publisher<auto_aim_interfaces::msg::DebugLights>::SharedPtr lights_data_pub_;
  rclcpp::Publisher<auto_aim_interfaces::msg::DebugArmors>::SharedPtr armors_data_pub_;
  image_transport::Publisher binary_img_pub_;
  image_transport::Publisher number_img_pub_;
  image_transport::Publisher result_img_pub_;

  // types
  enum VisionMode {
    OUTPOST = 0,
    HERO = 1, 
    ENGINEER = 2,
    INFANTRY_1 = 3,
    INFANTRY_2 = 4,
    INFANTRY_3 = 5,
    GUARD = 6,
    BASE = 7,
    RUNE = 8,
    AUTO = 9
  };
  inline std::string visionModeToString(VisionMode mode) {
    switch (mode) {
      case VisionMode::OUTPOST:
        return "OUTPOST";
      case VisionMode::HERO:
        return "HERO";
      case VisionMode::ENGINEER:
        return "ENGINEER";
      case VisionMode::INFANTRY_1:
        return "INFANTRY_1";
      case VisionMode::INFANTRY_2:
        return "INFANTRY_2";
      case VisionMode::INFANTRY_3:  
        return "INFANTRY_3";
      case VisionMode::GUARD:
        return "GUARD";
      case VisionMode::BASE:
        return "BASE";
      case VisionMode::RUNE:  
        return "RUNE";
      case VisionMode::AUTO:
        return "AUTO";
      default:
        return "UNKNOWN";
    }
  }
  bool enable_ = true; 
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_NODE_HPP_
