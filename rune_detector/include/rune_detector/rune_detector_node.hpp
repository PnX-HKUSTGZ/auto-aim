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
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
// 3rd party
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
// Tiger Core
#include "vc/camera/camera_param.h"
#include "vc/core/type_expansion.hpp"
#include "vc/detector/detector.h"
#include "vc/detector/rune_detector.h"
#include "vc/feature/feature_node_child_feature_type.h"
#include "vc/feature/rune_center.h"
#include "vc/feature/rune_combo.h"
#include "vc/feature/rune_fan.h"
#include "vc/feature/rune_group.h"
#include "vc/feature/rune_target.h"
#include "vc/feature/tracking_feature_node.h"
// project
#include "auto_aim_interfaces/msg/rune.hpp"
#include "auto_aim_interfaces/srv/set_mode.hpp"
#include "rune_detector/types.hpp"

namespace rm_auto_aim
{
class RuneDetectorNode : public rclcpp::Node
{
public:
    RuneDetectorNode(const rclcpp::NodeOptions & options);

private:
    std::unique_ptr<::RuneDetector> initDetector();

    static void rectLongEndpoints(const cv::RotatedRect & r, cv::Point2f & a, cv::Point2f & b);

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

    void createDebugPublishers();
    void destroyDebugPublishers();

    void setModeCallback(
        const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
        std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response);

    std::string resolveURL(const std::string & url);  // Resolve URL
    // Dynamic Parameter
    rcl_interfaces::msg::SetParametersResult onSetParameters(
        std::vector<rclcpp::Parameter> parameters);
    rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr on_set_parameters_callback_handle_;

    // Image subscription
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;

    //Target publisher
    std::string frame_id_;
    rclcpp::Publisher<auto_aim_interfaces::msg::Rune>::SharedPtr rune_pub_;

    // Enable/Disable Rune Detector
    rclcpp::Service<auto_aim_interfaces::srv::SetMode>::SharedPtr set_rune_mode_srv_;

    // Tiger Core detector与特征缓存
    std::unique_ptr<::RuneDetector> tiger_detector_;
    std::vector<FeatureNode_ptr> rune_groups_{};

    // Params
    PixChannel detect_color_;    // 检测颜色
    bool is_rune_{};
    bool has_camera_info_{};
    int binary_thresh_{};

    // Debug infomation
    bool debug_{};
    image_transport::Publisher result_img_pub_;

    rclcpp::Time timestamp;
};
}  // namespace rm_auto_aim
#endif  // DETECTOR_NODE_HPP_
