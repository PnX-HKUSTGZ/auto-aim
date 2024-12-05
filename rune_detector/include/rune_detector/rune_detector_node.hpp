// Copyright (C) FYT Vision Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DETECTOR_NODE_HPP_
#define DETECTOR_NODE_HPP_

// std
#include <algorithm>
#include <array>
#include <string>
#include <vector>
// ros2
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
// 3rd party
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
// project
#include "rune_detector/types.hpp"
#include "rune_detector/rune_detector.hpp"
#include "auto_aim_interfaces/srv/set_mode.hpp"
#include "auto_aim_interfaces/msg/rune.hpp"

namespace rm_auto_aim {
class RuneDetectorNode : public rclcpp::Node {
public:
  RuneDetectorNode(const rclcpp::NodeOptions &options);

private:
  std::unique_ptr<RuneDetector> initDetector();

  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);
  void inferResultCallback(std::vector<RuneObject> &rune_objects,
                           int64_t timestamp_nanosec,
                           const cv::Mat &img);

  void createDebugPublishers();
  void destroyDebugPublishers();

  void setModeCallback(const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
                       std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response);
  
  std::string resolveURL(const std::string &url); // Resolve URL
  // Dynamic Parameter
  rcl_interfaces::msg::SetParametersResult onSetParameters(
    std::vector<rclcpp::Parameter> parameters);
  rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr on_set_parameters_callback_handle_;

  // Image subscription
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

  //Target publisher
  std::string frame_id_;
  rclcpp::Publisher<auto_aim_interfaces::msg::Rune>::SharedPtr rune_pub_;

  // Enable/Disable Rune Detector
  rclcpp::Service<auto_aim_interfaces::srv::SetMode>::SharedPtr set_rune_mode_srv_;

  // Rune detector
  int requests_limit_;
  std::queue<std::future<bool>> detect_requests_;
  std::unique_ptr<RuneDetector> rune_detector_;

  // Rune params
  EnemyColor detect_color_;
  bool is_rune_;
  bool is_big_rune_;

  // For R tag detection
  bool detect_r_tag_;
  int binary_thresh_;

  // Debug infomation
  bool debug_;
  image_transport::Publisher result_img_pub_;

  rclcpp::Time timestamp; 
};
}  // namespace rm_auto_aim
#endif  // DETECTOR_NODE_HPP_
