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

#include "rune_detector/rune_detector_node.hpp"
// ros2
#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <opencv2/highgui.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
// std
#include <algorithm>
#include <array>
#include <filesystem>
#include <numeric>
#include <vector>
// third party
#include <opencv2/imgproc.hpp>
#include <fmt/core.h>
// project
#include "rune_detector/types.hpp"
#include "rune_detector/rune_detector.hpp"
#include "auto_aim_interfaces/srv/set_mode.hpp"
#include "auto_aim_interfaces/msg/rune.hpp"

namespace rm_auto_aim {

RuneDetectorNode::RuneDetectorNode(const rclcpp::NodeOptions &options)
: Node("rune_detector", options), is_rune_(false) {
  RCLCPP_INFO(this->get_logger(), "Starting RuneDetectorNode!");

  // 声明参数
  frame_id_ = declare_parameter("frame_id", "camera_optical_frame");
  detect_r_tag_ = declare_parameter("detect_r_tag", true);
  binary_thresh_ = declare_parameter("min_lightness", 100);
  detect_color_ = static_cast<EnemyColor>(declare_parameter("detect_color", 1)); 

  // 初始化检测器
  rune_detector_ = initDetector();
  // 创建 Rune 目标发布者
  rune_pub_ = this->create_publisher<auto_aim_interfaces::msg::Rune>("rune_detector/rune",
                                                                     rclcpp::SensorDataQoS());

  // 创建调试发布者
  this->debug_ = declare_parameter("debug", true);
  if (this->debug_) {
    createDebugPublishers();
  }
  auto qos = rclcpp::SensorDataQoS();
  qos.keep_last(1);
  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "image_raw", qos, std::bind(&RuneDetectorNode::imageCallback, this, std::placeholders::_1));
  set_rune_mode_srv_ = this->create_service<auto_aim_interfaces::srv::SetMode>(
    "rune_detector/set_mode",
    std::bind(
      &RuneDetectorNode::setModeCallback, this, std::placeholders::_1, std::placeholders::_2));
}

// 初始化检测器
std::unique_ptr<RuneDetector> RuneDetectorNode::initDetector() {
  // 设置动态参数回调
  rcl_interfaces::msg::SetParametersResult onSetParameters(
    std::vector<rclcpp::Parameter> parameters);
  on_set_parameters_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&RuneDetectorNode::onSetParameters, this, std::placeholders::_1));
  max_iterations_ = declare_parameter("max_iterations", 99);
  distance_threshold_ = declare_parameter("distance_threshold", 2.0);
  prob_threshold_ = declare_parameter("prob_threshold", 0.6);

  // 创建检测器
  auto rune_detector = std::make_unique<RuneDetector>(max_iterations_, distance_threshold_, prob_threshold_, detect_color_);

  return rune_detector;
}

// 图像回调函数
void RuneDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  if (!is_rune_) {
    return;
  }

  timestamp = rclcpp::Time(msg->header.stamp);
  frame_id_ = msg->header.frame_id;
  auto src_img = cv_bridge::toCvCopy(msg, "rgb8")->image;
  cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);

  // 将图像推送到检测器
  std::vector<RuneObject> objs = rune_detector_->detectRune(src_img);

  // 用于绘制调试信息
  cv::Mat debug_img;
  if (debug_) {
    debug_img = src_img.clone();
  }

  auto_aim_interfaces::msg::Rune rune_msg;
  rune_msg.header.frame_id = frame_id_;
  rune_msg.header.stamp = timestamp;

  if (!objs.empty()) {

    cv::Point2f r_tag;
    cv::Mat binary_roi = cv::Mat::zeros(1, 1, CV_8UC3);
    if (detect_r_tag_) {
      // 使用传统方法检测 R 标签
      cv::Point2f prior = std::accumulate(objs.begin(),
                              objs.end(),
                              cv::Point2f(0, 0),
                              [n = static_cast<float>(objs.size())](cv::Point2f p, auto &o) {
                                return p + o.pts.getRCenter() / n;
                              });
      std::tie(r_tag, binary_roi) =
        rune_detector_->detectRTag(src_img, prior);
    } else {
      // 使用所有对象的平均中心作为 R 标签的中心
      r_tag = std::accumulate(objs.begin(),
                              objs.end(),
                              cv::Point2f(0, 0),
                              [n = static_cast<float>(objs.size())](cv::Point2f p, auto &o) {
                                return p + o.pts.getRCenter() / n;
                              });
    }
    // 将 R 标签的中心分配给所有对象
    std::for_each(objs.begin(), objs.end(), [r = r_tag](RuneObject &obj) { obj.pts.r_center = r; });

    // 绘制二值化 ROI
    if (debug_ && !debug_img.empty()) {
      cv::Rect roi =
        cv::Rect(debug_img.cols - binary_roi.cols, 0, binary_roi.cols, binary_roi.rows);
      binary_roi.copyTo(debug_img(roi));
      cv::rectangle(debug_img, roi, cv::Scalar(150, 150, 150), 2);
    }

    // 最终目标是未激活的符文
    auto result_it =
      std::find_if(objs.begin(), objs.end(), [](const auto &obj) -> bool {
        return obj.type == RuneType::ACTIVATED;
      });

    if (result_it != objs.end()) {
      RCLCPP_DEBUG(this->get_logger(), "Detected!");
      rune_msg.is_lost = false;
      rune_msg.pts[0].x = result_it->pts.r_center.x;
      rune_msg.pts[0].y = result_it->pts.r_center.y;
      rune_msg.pts[1].x = result_it->pts.arm_bottom.x; 
      rune_msg.pts[1].y = result_it->pts.arm_bottom.y;
      rune_msg.pts[2].x = result_it->pts.arm_top.x;
      rune_msg.pts[2].y = result_it->pts.arm_top.y;
      rune_msg.pts[3].x = result_it->pts.hit_bottom.x;
      rune_msg.pts[3].y = result_it->pts.hit_bottom.y;
      rune_msg.pts[4].x = result_it->pts.hit_left.x; 
      rune_msg.pts[4].y = result_it->pts.hit_left.y; 
      rune_msg.pts[5].x = result_it->pts.hit_top.x; 
      rune_msg.pts[5].y = result_it->pts.hit_top.y; 
      rune_msg.pts[6].x = result_it->pts.hit_right.x; 
      rune_msg.pts[6].y = result_it->pts.hit_right.y; 
    } else {
      // 所有符文都已激活
      rune_msg.is_lost = true;
    }
  } else {
    // 所有符文都不是目标颜色
    rune_msg.is_lost = true;
  }

  rune_pub_->publish(rune_msg);

  if (debug_) {
    if (debug_img.empty()) {
      // 避免在处理过程中更改 debug_mode
      return;
    }

    // 绘制检测结果
    for (auto &obj : objs) {
      auto pts = obj.pts.toVector2f();
      cv::Point2f aim_point = std::accumulate(pts.begin() + 3, pts.end(), cv::Point2f(0, 0)) / 4;

      cv::Scalar line_color =
        obj.type == RuneType::ACTIVATED ? cv::Scalar(50, 255, 50) : cv::Scalar(255, 50, 255);
      cv::polylines(debug_img, obj.pts.toVector2i(), true, line_color, 2);
      cv::circle(debug_img, aim_point, 5, line_color, -1);

      std::string rune_type = obj.type == RuneType::ACTIVATED ? "_HIT" : "_OK";
      std::string rune_color = enemyColorToString(detect_color_);
      cv::putText(debug_img,
                  rune_color + rune_type,
                  cv::Point2i(pts[2]),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.8,
                  line_color,
                  2);
    }

    auto end = this->get_clock()->now();
    auto duration = end.seconds() - timestamp.seconds();
    std::string letency = fmt::format("Latency: {:.3f}ms", duration * 1000);
    cv::putText(debug_img,
                letency,
                cv::Point2i(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 255),
                2);
    cv::cvtColor(debug_img, debug_img, cv::COLOR_BGR2RGB);
    result_img_pub_.publish(cv_bridge::CvImage(rune_msg.header, "rgb8", debug_img).toImageMsg());
  }
}

// 动态参数设置回调函数
rcl_interfaces::msg::SetParametersResult RuneDetectorNode::onSetParameters(
  std::vector<rclcpp::Parameter> parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  for (const auto &param : parameters) {
    if (param.get_name() == "binary_thresh") {
      binary_thresh_ = param.as_int();
    }
  }
  result.successful = true;
  return result;
}

// 设置模式回调函数
void RuneDetectorNode::setModeCallback(
  const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
  std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response) {
  response->success = true;

  VisionMode mode = static_cast<VisionMode>(request->mode);
  std::string mode_name = visionModeToString(mode);
  if (mode_name == "UNKNOWN") {
    RCLCPP_ERROR(this->get_logger(), "Invalid mode: %d", request->mode);
    return;
  }

  auto createImageSub = [this]() {
    if (img_sub_ == nullptr) {
      img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "image_raw",
        rclcpp::SensorDataQoS(),
        std::bind(&RuneDetectorNode::imageCallback, this, std::placeholders::_1));
    }
  };

  switch (mode) {
    case VisionMode::RUNE: {
      is_rune_ = true;
      createImageSub();
      break;
    }
    default: {
      is_rune_ = false;
      img_sub_.reset();
      break;
    }
  }

  RCLCPP_WARN(this->get_logger(), "Set Rune Mode: %s", visionModeToString(mode).c_str());
}

// 创建调试发布者
void RuneDetectorNode::createDebugPublishers() {
  result_img_pub_ = image_transport::create_publisher(this, "rune_detector/result_img");
}
std::string RuneDetectorNode::resolveURL(const std::string &url) {
  // 检查 URL 前缀
  const std::string package_prefix = "package://";
  if (url.substr(0, package_prefix.size()) != package_prefix) {
    throw std::runtime_error("Invalid URL: " + url);
  }

  // 提取包名和相对路径
  std::string package_name = url.substr(package_prefix.size(), url.find('/', package_prefix.size()) - package_prefix.size());
  std::string relative_path = url.substr(url.find('/', package_prefix.size()));

  // 获取包路径
  std::string package_path = ament_index_cpp::get_package_share_directory(package_name);

  // 组合成完整路径
  std::string resolved_path = package_path + "/" +  relative_path;
  return resolved_path;
}

// 销毁调试发布者
void RuneDetectorNode::destroyDebugPublishers() { result_img_pub_.shutdown(); }

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// 注册组件到 class_loader
// 这相当于一个入口点，允许在加载库时发现组件
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::RuneDetectorNode)