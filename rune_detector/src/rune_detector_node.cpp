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
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
// std
#include <algorithm>
#include <array>
#include <cmath>
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

    // TF2 for gyro extraction from odom -> gimbal_link
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

    frame_id_ = declare_parameter("frame_id", "camera_main_optical_frame");
    binary_thresh_ = declare_parameter("binary_thresh", 100);
    detect_color_ = declare_parameter("detect_color", 0) == 0 ? PixChannel::RED : PixChannel::BLUE;

    tiger_detector_ = initDetector();

    rune_pub_ = this->create_publisher<auto_aim_interfaces::msg::Rune>(
        "rune_detector/rune", rclcpp::SensorDataQoS());

    debug_ = declare_parameter("debug", true);
    debug_marker_ = declare_parameter("debug_marker", true);
    if (debug_) {
        createDebugPublishers();
    }
    if (debug_marker_) {
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "rune_detector/marker", rclcpp::QoS(10));
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

GyroData RuneDetectorNode::getGyroData(const rclcpp::Time & stamp)
{
    GyroData gyro;

    try {
        auto latest_tf = tf2_buffer_->lookupTransform("odom", "gimbal_link", tf2::TimePointZero);
        const rclcpp::Time latest_time = latest_tf.header.stamp;

        geometry_msgs::msg::TransformStamped tf_msg;
        if (stamp > latest_time) {
            tf_msg = latest_tf;
        } else {
            try {
                tf_msg = tf2_buffer_->lookupTransform(
                    "odom", "gimbal_link", stamp,
                    rclcpp::Duration::from_nanoseconds(1000000));
            } catch (const tf2::TransformException & ex) {
                tf_msg = latest_tf;
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "rune_detector gyro TF exact-time lookup failed, fallback to latest: %s",
                    ex.what());
            }
        }

        tf2::Quaternion q(
            tf_msg.transform.rotation.x, tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z, tf_msg.transform.rotation.w);
        double roll = 0.0, pitch = 0.0, yaw = 0.0;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        // Tiger Core expects degree units:
        // yaw: left negative, right positive
        // pitch: up negative, down positive
        constexpr double RAD2DEG = 180.0 / 3.14159265358979323846;
        gyro.rotation.yaw = static_cast<float>(-yaw * RAD2DEG);
        gyro.rotation.pitch = static_cast<float>(-pitch * RAD2DEG);
        gyro.rotation.roll = static_cast<float>(roll * RAD2DEG);
    } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "rune_detector gyro TF lookup failed, use zero gyro: %s", ex.what());
    }

    return gyro;
}

void RuneDetectorNode::clearRune3DMarkers(const rclcpp::Time & stamp)
{
    if (!marker_pub_) {
        return;
    }
    visualization_msgs::msg::MarkerArray marker_array;
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = "odom";
    clear_marker.header.stamp = stamp;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);
    marker_pub_->publish(marker_array);
}

void RuneDetectorNode::publishRune3DMarkers(
    const rclcpp::Time & stamp, const std::string & camera_frame,
    const PoseNode & rune_to_camera,
    const PoseNode * pending_target_to_camera)
{
    if (!marker_pub_) {
        return;
    }

    try {
        auto latest_tf = tf2_buffer_->lookupTransform("odom", camera_frame, tf2::TimePointZero);
        const rclcpp::Time latest_time = latest_tf.header.stamp;

        geometry_msgs::msg::TransformStamped odom_to_cam_tf;
        if (stamp > latest_time) {
            odom_to_cam_tf = latest_tf;
        } else {
            try {
                odom_to_cam_tf = tf2_buffer_->lookupTransform(
                    "odom", camera_frame, stamp,
                    rclcpp::Duration::from_nanoseconds(1000000));
            } catch (const tf2::TransformException & ex) {
                odom_to_cam_tf = latest_tf;
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "rune_detector marker TF exact-time lookup failed, fallback to latest: %s",
                    ex.what());
            }
        }

        tf2::Quaternion q_odom_cam(
            odom_to_cam_tf.transform.rotation.x, odom_to_cam_tf.transform.rotation.y,
            odom_to_cam_tf.transform.rotation.z, odom_to_cam_tf.transform.rotation.w);
        tf2::Vector3 t_odom_cam(
            odom_to_cam_tf.transform.translation.x, odom_to_cam_tf.transform.translation.y,
            odom_to_cam_tf.transform.translation.z);
        tf2::Transform odom_T_cam(tf2::Matrix3x3(q_odom_cam), t_odom_cam);

        const auto r = rune_to_camera.rmat();
        const auto t = rune_to_camera.tvec();
        constexpr double kMillimeterToMeter = 1e-3;
        tf2::Matrix3x3 cam_R_rune(
            r(0, 0), r(0, 1), r(0, 2),
            r(1, 0), r(1, 1), r(1, 2),
            r(2, 0), r(2, 1), r(2, 2));
        tf2::Vector3 cam_t_rune(
            t(0) * kMillimeterToMeter,
            t(1) * kMillimeterToMeter,
            t(2) * kMillimeterToMeter);
        tf2::Transform cam_T_rune(cam_R_rune, cam_t_rune);

        tf2::Transform odom_T_rune = odom_T_cam * cam_T_rune;

        visualization_msgs::msg::MarkerArray marker_array;

        if (pending_target_to_camera != nullptr) {
            const auto r_target = pending_target_to_camera->rmat();
            const auto t_target = pending_target_to_camera->tvec();
            tf2::Matrix3x3 cam_R_target(
                r_target(0, 0), r_target(0, 1), r_target(0, 2),
                r_target(1, 0), r_target(1, 1), r_target(1, 2),
                r_target(2, 0), r_target(2, 1), r_target(2, 2));
            tf2::Vector3 cam_t_target(
                t_target(0) * kMillimeterToMeter,
                t_target(1) * kMillimeterToMeter,
                t_target(2) * kMillimeterToMeter);
            tf2::Transform cam_T_target(cam_R_target, cam_t_target);
            tf2::Transform odom_T_target = odom_T_cam * cam_T_target;

            visualization_msgs::msg::Marker target_marker;
            target_marker.header.frame_id = "odom";
            target_marker.header.stamp = stamp;
            target_marker.ns = "pending_target_center";
            target_marker.id = 10;
            target_marker.type = visualization_msgs::msg::Marker::SPHERE;
            target_marker.action = visualization_msgs::msg::Marker::ADD;
            target_marker.scale.x = target_marker.scale.y = target_marker.scale.z = 0.09;
            target_marker.color.a = 1.0;
            target_marker.color.r = 1.0;
            target_marker.color.g = 0.0;
            target_marker.color.b = 1.0;
            target_marker.pose.position.x = odom_T_target.getOrigin().x();
            target_marker.pose.position.y = odom_T_target.getOrigin().y();
            target_marker.pose.position.z = odom_T_target.getOrigin().z();
            marker_array.markers.push_back(target_marker);
        }

        visualization_msgs::msg::Marker center_marker;
        center_marker.header.frame_id = "odom";
        center_marker.header.stamp = stamp;
        center_marker.ns = "rune_center";
        center_marker.id = 0;
        center_marker.type = visualization_msgs::msg::Marker::SPHERE;
        center_marker.action = visualization_msgs::msg::Marker::ADD;
        center_marker.scale.x = center_marker.scale.y = center_marker.scale.z = 0.12;
        center_marker.color.a = 1.0;
        center_marker.color.r = 1.0;
        center_marker.color.g = 0.6;
        center_marker.color.b = 0.0;
        center_marker.pose.position.x = odom_T_rune.getOrigin().x();
        center_marker.pose.position.y = odom_T_rune.getOrigin().y();
        center_marker.pose.position.z = odom_T_rune.getOrigin().z();
        marker_array.markers.push_back(center_marker);

        auto make_axis_arrow = [&](int id, const std::string & ns, const tf2::Vector3 & axis,
                                   float r_c, float g_c, float b_c) {
            visualization_msgs::msg::Marker axis_marker;
            axis_marker.header.frame_id = "odom";
            axis_marker.header.stamp = stamp;
            axis_marker.ns = ns;
            axis_marker.id = id;
            axis_marker.type = visualization_msgs::msg::Marker::ARROW;
            axis_marker.action = visualization_msgs::msg::Marker::ADD;
            axis_marker.scale.x = 0.02;
            axis_marker.scale.y = 0.04;
            axis_marker.color.a = 1.0;
            axis_marker.color.r = r_c;
            axis_marker.color.g = g_c;
            axis_marker.color.b = b_c;

            geometry_msgs::msg::Point p0, p1;
            const tf2::Vector3 origin = odom_T_rune.getOrigin();
            const tf2::Vector3 axis_end = odom_T_rune * (axis * 0.2);
            p0.x = origin.x();
            p0.y = origin.y();
            p0.z = origin.z();
            p1.x = axis_end.x();
            p1.y = axis_end.y();
            p1.z = axis_end.z();
            axis_marker.points.push_back(p0);
            axis_marker.points.push_back(p1);
            marker_array.markers.push_back(axis_marker);
        };

        make_axis_arrow(1, "rune_axis_x", tf2::Vector3(1.0, 0.0, 0.0), 1.0, 0.0, 0.0);
        make_axis_arrow(2, "rune_axis_y", tf2::Vector3(0.0, 1.0, 0.0), 0.0, 1.0, 0.0);
        make_axis_arrow(3, "rune_axis_z", tf2::Vector3(0.0, 0.0, 1.0), 0.0, 0.0, 1.0);

        marker_pub_->publish(marker_array);
    } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "rune_detector marker TF lookup failed, clear markers: %s", ex.what());
        clearRune3DMarkers(stamp);
    }
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
    input.setGyroData(getGyroData(timestamp));
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
    std::shared_ptr<RuneGroup> current_rune_group = nullptr;
    bool has_pending_target_pose = false;
    PoseNode pending_target_to_camera;

    if (!rune_groups_.empty())
    {
        auto rune_group = RuneGroup::cast(rune_groups_.front());
        current_rune_group = rune_group;
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

            if (target &&
                target->getPoseCache().getPoseNodes().count(CoordFrame::CAMERA) > 0) {
                pending_target_to_camera =
                    target->getPoseCache().getPoseNodes().at(CoordFrame::CAMERA);
                has_pending_target_pose = true;
            }

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

    if (debug_marker_) {
        bool published_marker = false;
        if (current_rune_group) {
            PoseNode rune_to_camera;
            if (current_rune_group->getCamPnpDataFromFilter(rune_to_camera)) {
                publishRune3DMarkers(
                    timestamp, frame_id_, rune_to_camera,
                    has_pending_target_pose ? &pending_target_to_camera : nullptr);
                published_marker = true;
            }
        }
        if (!published_marker) {
            clearRune3DMarkers(timestamp);
        }
    }

    if (debug_ && !rune_groups_.empty())
    {
        cv::Mat debug_img = src_img.clone();
        auto rune_group = RuneGroup::cast(rune_groups_.front());
        rune_group->drawFeature(debug_img);

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