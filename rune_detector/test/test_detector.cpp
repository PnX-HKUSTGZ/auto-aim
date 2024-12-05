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

// gest
#include <gtest/gtest.h>
// ros2
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <rclcpp/executors.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/utilities.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
// std
#include <memory>
// opencv
#include <opencv2/opencv.hpp>
// project
#include "rune_detector/rune_detector.hpp"
#include "rune_detector/types.hpp"

using namespace rm_auto_aim;
std::string resolveURL(const std::string &url) {
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
  std::filesystem::path resolved_path = std::filesystem::path(package_path) / relative_path;
  return resolved_path.string();
}
TEST(RuneDetectorNodeTest, NodeStartupTest) {
  // Init Params
  std::string model_path = "package://rune_detector/model/yolox_rune_3.6m.onnx";
  std::string device_type = "AUTO";

  float conf_threshold = 0.5;
  int top_k = 128;
  float nms_threshold = 0.3;

  namespace fs = std::filesystem;
  std::string resolved_path_str;
  try {
    resolved_path_str = resolveURL(model_path);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl; 
    rclcpp::shutdown();
    return ;
  }

  fs::path resolved_path = resolved_path_str;
  if (!fs::exists(resolved_path)) {
    std::cerr << "Model file not found: " << resolved_path.string() << std::endl;
    rclcpp::shutdown();
    return ;
  }

  // Create detector
  auto rune_detector = std::make_unique<RuneDetector>(
    resolved_path, device_type, conf_threshold, top_k, nms_threshold);

  // Set detect callback
  std::vector<RuneObject> runes;
  rune_detector->setCallback([&runes](std::vector<RuneObject> &objs,
                                      int64_t timestamp_nanosec,
                                      const cv::Mat &src_img) { runes = objs; });

  // init detector
  rune_detector->init();

  // Load test image
  std::string test_image_path_str;
  try {
    resolved_path_str = resolveURL("package://rune_detector/docs/test.png");
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl; 
    rclcpp::shutdown();
    return ;
  }

  fs::path test_image_path = resolved_path_str;
  if (!fs::exists(resolved_path)) {
    std::cerr << "image file not found: " << resolved_path.string() << std::endl;
    rclcpp::shutdown();
    return ;
  }
  cv::Mat test_image = cv::imread(test_image_path.string(), cv::IMREAD_COLOR);
  cv::cvtColor(test_image, test_image, cv::COLOR_BGR2RGB);

  auto future = rune_detector->pushInput(test_image, 0);
  future.get();

  EXPECT_EQ(runes.size(), static_cast<size_t>(3));
  std::sort(runes.begin(), runes.end(), [](const RuneObject &a, const RuneObject &b) {
    return a.type < b.type;
  });
  EXPECT_EQ(runes[0].type, RuneType::INACTIVATED);
  EXPECT_EQ(runes[1].type, RuneType::ACTIVATED);
  EXPECT_EQ(runes[2].type, RuneType::ACTIVATED);
}

