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
// ament
#include <ament_index_cpp/get_package_share_directory.hpp>
// ros2
#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>

#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
// std
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include <opencv2/imgproc.hpp>
// project
#include "auto_aim_interfaces/msg/rune.hpp"
#include "auto_aim_interfaces/srv/set_mode.hpp"

namespace rm_auto_aim
{

RuneDetectorNode::RuneDetectorNode(const rclcpp::NodeOptions & options)
: Node("rune_detector", options), is_rune_(false)
{
    RCLCPP_INFO(this->get_logger(), "Starting RuneDetectorNode with Tiger Core!");

    frame_id_ = declare_parameter("frame_id", "camera_optical_frame");
    binary_thresh_ = declare_parameter("binary_thresh", 100);
    detect_color_ = declare_parameter("detect_color", 1) == 0 ? PixChannel::RED : PixChannel::BLUE;

    tiger_detector_ = initDetector();

    rune_pub_ = this->create_publisher<auto_aim_interfaces::msg::Rune>(
        "rune_detector/rune", rclcpp::SensorDataQoS());

    debug_ = declare_parameter("debug", true);
    if (debug_) {
        createDebugPublishers();
    }

    auto qos = rclcpp::SensorDataQoS();
    qos.keep_last(1);
    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "image_raw", qos, std::bind(&RuneDetectorNode::imageCallback, this, std::placeholders::_1));
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", qos, std::bind(&RuneDetectorNode::cameraInfoCallback, this, std::placeholders::_1));

    set_rune_mode_srv_ = this->create_service<auto_aim_interfaces::srv::SetMode>(
        "rune_detector/set_mode", std::bind(
                                      &RuneDetectorNode::setModeCallback, this,
                                      std::placeholders::_1, std::placeholders::_2));
}

// 初始化检测器
std::unique_ptr<::RuneDetector> RuneDetectorNode::initDetector()
{
    rcl_interfaces::msg::SetParametersResult onSetParameters(
        std::vector<rclcpp::Parameter> parameters);
    on_set_parameters_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&RuneDetectorNode::onSetParameters, this, std::placeholders::_1));

    return ::RuneDetector::make_detector();
}

void RuneDetectorNode::rectLongEndpoints(const cv::RotatedRect & r, cv::Point2f & a, cv::Point2f & b)
{
    cv::Point2f pts[4];
    r.points(pts);
    double d01 = cv::norm(pts[0] - pts[1]);
    double d12 = cv::norm(pts[1] - pts[2]);
    if (d01 > d12) {
        a = (pts[0] + pts[3]) / 2;
        b = (pts[1] + pts[2]) / 2;
    } else {
        a = (pts[0] + pts[1]) / 2;
        b = (pts[2] + pts[3]) / 2;
    }
}

void RuneDetectorNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
    cv::Matx33f camera_matrix(
        static_cast<float>(msg->k[0]), static_cast<float>(msg->k[1]), static_cast<float>(msg->k[2]),
        static_cast<float>(msg->k[3]), static_cast<float>(msg->k[4]), static_cast<float>(msg->k[5]),
        static_cast<float>(msg->k[6]), static_cast<float>(msg->k[7]), static_cast<float>(msg->k[8]));

    cv::Matx<float, 5, 1> dist_coeffs(
        static_cast<float>(msg->d[0]), static_cast<float>(msg->d[1]), static_cast<float>(msg->d[2]),
        static_cast<float>(msg->d[3]), static_cast<float>(msg->d[4]));

    camera_param.cameraMatrix = camera_matrix;
    camera_param.distCoeff = dist_coeffs;
    camera_param.image_width = static_cast<int>(msg->width);
    camera_param.image_height = static_cast<int>(msg->height);

    has_camera_info_ = true;
    cam_info_sub_.reset();
}

// 图像回调函数
void RuneDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    if (!is_rune_ || !has_camera_info_) {
        return;
    }

    timestamp = rclcpp::Time(msg->header.stamp);
    frame_id_ = msg->header.frame_id;

    cv::Mat src_img;
    try {
        src_img = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception & e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    DetectorInput input;
    input.setImage(src_img);
    input.setTick(rclcpp::Time(msg->header.stamp).nanoseconds());
    input.setGyroData(GyroData());
    input.setColor(detect_color_);
    input.setColorThresh(static_cast<uint8_t>(binary_thresh_));
    input.setFeatureNodes(rune_groups_);

    DetectorOutput output;
    try {
        tiger_detector_->detect(input, output);
    } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Tiger Core detect error: %s", e.what());
        return;
    }

    rune_groups_ = output.getFeatureNodes();

    auto_aim_interfaces::msg::Rune rune_msg;
    rune_msg.header.frame_id = frame_id_;
    rune_msg.header.stamp = timestamp;
    rune_msg.is_lost = true;

    if (!rune_groups_.empty())
    {
        auto rune_group = RuneGroup::cast(rune_groups_.front());
        FeatureNode_ptr target_tracker = nullptr;
        for (auto & tr : rune_group->getTrackers())
        {
            auto tracker = TrackingFeatureNode::cast(tr);
            if (!tracker || tracker->getHistoryNodes().empty())
                continue;
            auto combo = RuneCombo::cast(tracker->getHistoryNodes().front());
            if (!combo)
                continue;
            auto type = combo->getRuneType();
            if (type == ::RuneType::PENDING_STRUCK)
            {
                target_tracker = tr;
                break;
            }
        }

        if (target_tracker)
        {
            auto tracker = TrackingFeatureNode::cast(target_tracker);
            auto combo = RuneCombo::cast(tracker->getHistoryNodes().front());
            auto target = RuneTarget::cast(combo->getChildFeatures().at(FeatureNode::ChildFeatureType::RUNE_TARGET));
            auto center = RuneCenter::cast(combo->getChildFeatures().at(FeatureNode::ChildFeatureType::RUNE_CENTER));
            auto fan = RuneFan::cast(combo->getChildFeatures().at(FeatureNode::ChildFeatureType::RUNE_FAN));

            if (target && center && fan && target->getImageCache().isSetCorners())
            {
                const auto & corners = target->getImageCache().getCorners();
                if (corners.size() >= 8)
                {
                    rune_msg.is_lost = false;
                    rune_msg.pts[0].x = center->getImageCache().getCenter().x;
                    rune_msg.pts[0].y = center->getImageCache().getCenter().y;

                    cv::Point2f a, b;
                    rectLongEndpoints(fan->getRotatedRect(), a, b);
                    const auto r_center = center->getImageCache().getCenter();
                    if (cv::norm(a - r_center) < cv::norm(b - r_center))
                    {
                        rune_msg.pts[1].x = a.x; rune_msg.pts[1].y = a.y;
                        rune_msg.pts[2].x = b.x; rune_msg.pts[2].y = b.y;
                    }
                    else
                    {
                        rune_msg.pts[1].x = b.x; rune_msg.pts[1].y = b.y;
                        rune_msg.pts[2].x = a.x; rune_msg.pts[2].y = a.y;
                    }

                    // 1/3/5/7 -> top/right/bottom/left
                    rune_msg.pts[3].x = corners[5].x; rune_msg.pts[3].y = corners[5].y; // bottom
                    rune_msg.pts[4].x = corners[7].x; rune_msg.pts[4].y = corners[7].y; // left
                    rune_msg.pts[5].x = corners[1].x; rune_msg.pts[5].y = corners[1].y; // top
                    rune_msg.pts[6].x = corners[3].x; rune_msg.pts[6].y = corners[3].y; // right
                }
            }
        }
    }

    rune_pub_->publish(rune_msg);

    if (debug_ && !rune_groups_.empty())
    {
        cv::Mat debug_img = src_img.clone();
        auto rune_group = RuneGroup::cast(rune_groups_.front());
        //rune_group->drawFeature(debug_img);

        // visualize polygon through r_center (0) and inactive corners (4,5,6)
        if (!rune_msg.is_lost)
        {
            std::vector<cv::Point> poly;
            const int indices[4] = {0, 4, 5, 6};
            for (int idx : indices)
            {
                poly.emplace_back(static_cast<int>(rune_msg.pts[idx].x), static_cast<int>(rune_msg.pts[idx].y));
            }
            cv::polylines(debug_img, poly, true, cv::Scalar(0, 255, 255), 2);
            cv::line(
                debug_img,
                cv::Point(static_cast<int>(rune_msg.pts[1].x), static_cast<int>(rune_msg.pts[1].y)),
                cv::Point(static_cast<int>(rune_msg.pts[2].x), static_cast<int>(rune_msg.pts[2].y)),
                cv::Scalar(255, 0, 255), 2);
        }

        result_img_pub_.publish(cv_bridge::CvImage(rune_msg.header, "bgr8", debug_img).toImageMsg());
    }
}

// 动态参数设置回调函数
rcl_interfaces::msg::SetParametersResult RuneDetectorNode::onSetParameters(
    std::vector<rclcpp::Parameter> parameters)
{
    rcl_interfaces::msg::SetParametersResult result;
    for (const auto & param : parameters) {
        if (param.get_name() == "binary_thresh") {
            binary_thresh_ = param.as_int();
        }
        if (param.get_name() == "detect_color") {
            detect_color_ = param.as_int() == 0 ? PixChannel::RED : PixChannel::BLUE;
        }
    }
    result.successful = true;
    return result;
}

// 设置模式回调函数
void RuneDetectorNode::setModeCallback(
    const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
    std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response)
{
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
                "image_raw", rclcpp::SensorDataQoS(),
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
void RuneDetectorNode::createDebugPublishers()
{
    result_img_pub_ = image_transport::create_publisher(this, "rune_detector/result_img");
}
std::string RuneDetectorNode::resolveURL(const std::string & url)
{
    // 检查 URL 前缀
    const std::string package_prefix = "package://";
    if (url.substr(0, package_prefix.size()) != package_prefix) {
        throw std::runtime_error("Invalid URL: " + url);
    }

    // 提取包名和相对路径
    std::string package_name = url.substr(
        package_prefix.size(), url.find('/', package_prefix.size()) - package_prefix.size());
    std::string relative_path = url.substr(url.find('/', package_prefix.size()));

    // 获取包路径
    std::string package_path = ament_index_cpp::get_package_share_directory(package_name);

    // 组合成完整路径
    std::string resolved_path = package_path + "/" + relative_path;
    return resolved_path;
}

// 销毁调试发布者
void RuneDetectorNode::destroyDebugPublishers() { result_img_pub_.shutdown(); }

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// 注册组件到 class_loader
// 这相当于一个入口点，允许在加载库时发现组件
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::RuneDetectorNode)