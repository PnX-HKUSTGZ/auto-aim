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

#include "rune_detector_node.hpp"
// ros2
#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>

#include <rclcpp/qos.hpp>
// std
#include <algorithm>
#include <array>
#include <filesystem>
#include <numeric>
#include <vector>
// third party
#include <opencv2/imgproc.hpp>
// project
#include "rune_detector/types.hpp"

namespace rm_auto_aim {

RuneDetectorNode::RuneDetectorNode(const rclcpp::NodeOptions &options)
: Node("rune_detector", options), is_rune_(false) {
  RCLCPP_INFO(this->get_logger(), "Starting RuneDetectorNode!");

  // 声明参数
  frame_id_ = declare_parameter("frame_id", "camera_optical_frame");
  detect_r_tag_ = declare_parameter("detect_r_tag", true);
  binary_thresh_ = declare_parameter("min_lightness", 100);
  requests_limit_ = declare_parameter("requests_limit", 5);

  // 初始化检测器
  rune_detector_ = initDetector();
  // 创建 Rune 目标发布者
  rune_pub_ = this->create_publisher<rm_interfaces::msg::RuneTarget>("rune_detector/rune_target",
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
  set_rune_mode_srv_ = this->create_service<rm_interfaces::srv::SetMode>(
    "rune_detector/set_mode",
    std::bind(
      &RuneDetectorNode::setModeCallback, this, std::placeholders::_1, std::placeholders::_2));

  // 心跳发布者
  heartbeat_ = HeartBeatPublisher::create(this);
}

// 初始化检测器
std::unique_ptr<RuneDetector> RuneDetectorNode::initDetector() {
  std::string model_path =
    this->declare_parameter("detector.model", "package://rune_detector/model/yolox_rune_3.6m.onnx");
  std::string device_type = this->declare_parameter("detector.device_type", "AUTO");
  RCLCPP_ASSERT(this->get_logger(), !model_path.empty());
  RCLCPP_INFO(this->get_logger(), "Model: %s, Device: %s", model_path.c_str(), device_type.c_str());

  float conf_threshold = this->declare_parameter("detector.confidence_threshold", 0.50);
  int top_k = this->declare_parameter("detector.top_k", 128);
  float nms_threshold = this->declare_parameter("detector.nms_threshold", 0.3);

  namespace fs = std::filesystem;
  fs::path resolved_path = utils::URLResolver::getResolvedPath(model_path);
  RCLCPP_ASSERT(this->get_logger(), fs::exists(resolved_path), "%s Not Found", resolved_path.string().c_str());

  // 设置动态参数回调
  rcl_interfaces::msg::SetParametersResult onSetParameters(
    std::vector<rclcpp::Parameter> parameters);
  on_set_parameters_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&RuneDetectorNode::onSetParameters, this, std::placeholders::_1));

  // 创建检测器
  auto rune_detector = std::make_unique<RuneDetector>(
    resolved_path, device_type, conf_threshold, top_k, nms_threshold);
  // 设置检测回调
  rune_detector->setCallback(std::bind(&RuneDetectorNode::inferResultCallback,
                                       this,
                                       std::placeholders::_1,
                                       std::placeholders::_2,
                                       std::placeholders::_3));
  // 初始化检测器
  rune_detector->init();
  return rune_detector;
}

// 图像回调函数
void RuneDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  if (is_rune_ == false) {
    return;
  }

  // 限制请求大小
  while (detect_requests_.size() > static_cast<size_t>(requests_limit_)) {
    detect_requests_.front().get();
    detect_requests_.pop();
  }

  auto timestamp = rclcpp::Time(msg->header.stamp);
  frame_id_ = msg->header.frame_id;
  auto img = cv_bridge::toCvCopy(msg, "rgb8")->image;

  // 将图像推送到检测器
  detect_requests_.push(rune_detector_->pushInput(img, timestamp.nanoseconds()));
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

// 推理结果回调函数
void RuneDetectorNode::inferResultCallback(std::vector<RuneObject> &objs,
                                           int64_t timestamp_nanosec,
                                           const cv::Mat &src_img) {
  auto timestamp = rclcpp::Time(timestamp_nanosec);
  // 用于绘制调试信息
  cv::Mat debug_img;
  if (debug_) {
    debug_img = src_img.clone();
  }

  rm_interfaces::msg::RuneTarget rune_msg;
  rune_msg.header.frame_id = frame_id_;
  rune_msg.header.stamp = timestamp;
  rune_msg.is_big_rune = is_big_rune_;

  // 删除所有不匹配颜色的对象
  objs.erase(
    std::remove_if(objs.begin(),
                   objs.end(),
                   [c = detect_color_](const auto &obj) -> bool { return obj.color != c; }),
    objs.end());

  if (!objs.empty()) {
    // 按概率排序
    std::sort(objs.begin(), objs.end(), [](const RuneObject &a, const RuneObject &b) {
      return a.prob > b.prob;
    });

    cv::Point2f r_tag;
    cv::Mat binary_roi = cv::Mat::zeros(1, 1, CV_8UC3);
    if (detect_r_tag_) {
      // 使用传统方法检测 R 标签
      std::tie(r_tag, binary_roi) =
        rune_detector_->detectRTag(src_img, binary_thresh_, objs.at(0).pts.r_center);
    } else {
      // 使用所有对象的平均中心作为 R 标签的中心
      r_tag = std::accumulate(objs.begin(),
                              objs.end(),
                              cv::Point2f(0, 0),
                              [n = static_cast<float>(objs.size())](cv::Point2f p, auto &o) {
                                return p + o.pts.r_center / n;
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

    // 最终目标是未激活的概率最高的符文
    auto result_it =
      std::find_if(objs.begin(), objs.end(), [c = detect_color_](const auto &obj) -> bool {
        return obj.type == RuneType::INACTIVATED && obj.color == c;
      });

    if (result_it != objs.end()) {
      // RCLCPP_DEBUG(this->get_logger(), "Detected!");
      rune_msg.is_lost = false;
      rune_msg.pts[0].x = result_it->pts.r_center.x;
      rune_msg.pts[0].y = result_it->pts.r_center.y;
      rune_msg.pts[1].x = result_it->pts.bottom_left.x;
      rune_msg.pts[1].y = result_it->pts.bottom_left.y;
      rune_msg.pts[2].x = result_it->pts.top_left.x;
      rune_msg.pts[2].y = result_it->pts.top_left.y;
      rune_msg.pts[3].x = result_it->pts.top_right.x;
      rune_msg.pts[3].y = result_it->pts.top_right.y;
      rune_msg.pts[4].x = result_it->pts.bottom_right.x;
      rune_msg.pts[4].y = result_it->pts.bottom_right.y;
    } else {
      // 所有符文都已激活
      rune_msg.is_lost = true;
    }
  } else {
    // 所有符文都不是目标颜色
    rune_msg.is_lost = true;
  }

  rune_pub_->publish(std::move(rune_msg));

  if (debug_) {
    if (debug_img.empty()) {
      // 避免在处理过程中更改 debug_mode
      return;
    }

    // 绘制检测结果
    for (auto &obj : objs) {
      auto pts = obj.pts.toVector2f();
      cv::Point2f aim_point = std::accumulate(pts.begin() + 1, pts.end(), cv::Point2f(0, 0)) / 4;

      cv::Scalar line_color =
        obj.type == RuneType::INACTIVATED ? cv::Scalar(50, 255, 50) : cv::Scalar(255, 50, 255);
      cv::putText(debug_img,
                  fmt::format("{:.2f}", obj.prob),
                  cv::Point2i(pts[1]),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.8,
                  line_color,
                  2);
      cv::polylines(debug_img, obj.pts.toVector2i(), true, line_color, 2);
      cv::circle(debug_img, aim_point, 5, line_color, -1);

      std::string rune_type = obj.type == RuneType::INACTIVATED ? "_HIT" : "_OK";
      std::string rune_color = enemyColorToString(obj.color);
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
    result_img_pub_.publish(cv_bridge::CvImage(rune_msg.header, "rgb8", debug_img).toImageMsg());
  }
}

// 设置模式回调函数
void RuneDetectorNode::setModeCallback(
  const std::shared_ptr<rm_interfaces::srv::SetMode::Request> request,
  std::shared_ptr<rm_interfaces::srv::SetMode::Response> response) {
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
    case VisionMode::SMALL_RUNE_RED: {
      is_rune_ = true;
      is_big_rune_ = false;
      detect_color_ = EnemyColor::RED;
      createImageSub();
      break;
    }
    case VisionMode::SMALL_RUNE_BLUE: {
      is_rune_ = true;
      is_big_rune_ = false;
      detect_color_ = EnemyColor::BLUE;
      createImageSub();
      break;
    }
    case VisionMode::BIG_RUNE_RED: {
      is_rune_ = true;
      is_big_rune_ = true;
      detect_color_ = EnemyColor::RED;
      createImageSub();
      break;
    }
    case VisionMode::BIG_RUNE_BLUE: {
      is_rune_ = true;
      is_big_rune_ = true;
      detect_color_ = EnemyColor::BLUE;
      createImageSub();
      break;
    }
    default: {
      is_rune_ = false;
      is_big_rune_ = false;
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

// 销毁调试发布者
void RuneDetectorNode::destroyDebugPublishers() { result_img_pub_.shutdown(); }

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// 注册组件到 class_loader
// 这相当于一个入口点，允许在加载库时发现组件
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::RuneDetectorNode)