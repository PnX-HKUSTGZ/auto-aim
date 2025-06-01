// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__ARMOR_HPP_
#define ARMOR_DETECTOR__ARMOR_HPP_

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <rclcpp/duration.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// STL
#include <algorithm>
#include <string>

namespace rm_auto_aim
{

// Unit: mm
static constexpr float SMALL_ARMOR_WIDTH = 138.0 / 1000.0;
static constexpr float SMALL_ARMOR_HEIGHT = 48.0 / 1000.0;
static constexpr float LARGE_ARMOR_WIDTH = 228.0 / 1000.0;
static constexpr float LARGE_ARMOR_HEIGHT = 48.0 / 1000.0;

//灯条和装甲板的结构体定义
const int RED = 0;
const int BLUE = 1;

enum class ArmorType { SMALL, LARGE, INVALID };
const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};

struct Light : public cv::RotatedRect
{
    Light() = default;
    explicit Light(cv::RotatedRect box) : cv::RotatedRect(box)
    {
        cv::Point2f p[4];
        box.points(p);
        std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        tilt_angle = std::atan2(bottom.x - top.x, bottom.y - top.y);
        tilt_angle = tilt_angle / CV_PI * 180;
    }

    int color;
    cv::Point2f top, bottom, axis;
    double length;
    double width;
    float tilt_angle;
};

struct Armor
{
    Armor() = default;
    Armor(const Light & l1, const Light & l2)
    {
        if (l1.center.x < l2.center.x) {
            left_light = l1, right_light = l2;
        } else {
            left_light = l2, right_light = l1;
        }
        center = (left_light.center + right_light.center) / 2;
    }

    template <typename PointType>
    static inline std::vector<PointType> buildObjectPoints(const double & w, const double & h)
    {
        return {
            PointType(0, w / 2, -h / 2), PointType(0, w / 2, h / 2), PointType(0, -w / 2, h / 2),
            PointType(0, -w / 2, -h / 2)};
    }
    std::vector<cv::Point2f> landmarks() const
    {
        return {left_light.bottom, left_light.top, right_light.top, right_light.bottom};
    }

    void setCameraArmor(
        const Eigen::Matrix3d & r_odom_to_camera, const Eigen::Vector3d & t_odom_to_camera)
    {
        t_camera_armor = r_odom_to_camera * t_odom_armor + t_odom_to_camera;
        r_camera_armor = r_odom_to_camera * r_odom_armor;
    }

    // Light pairs part
    Light left_light, right_light;
    cv::Point2f center;
    bool sign = false;  // 灯条和y轴夹角，0指向右下，1指向左下
    ArmorType type;

    // Number part
    cv::Mat number_img;
    std::string number;
    float confidence;
    std::string classfication_result;
    Eigen::Matrix3d r_odom_armor;
    Eigen::Vector3d t_odom_armor;
    Eigen::Matrix3d r_camera_armor;
    Eigen::Vector3d t_camera_armor;
};

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
inline std::string visionModeToString(VisionMode mode)
{
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

// Visualization marker publisher
// See http://wiki.ros.org/rviz/DisplayTypes/Marker
static visualization_msgs::msg::Marker armor_marker_ = [] {
    visualization_msgs::msg::Marker marker;
    marker.ns = "armors";
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.scale.x = 0.05;
    marker.scale.z = 0.125;
    marker.color.a = 1.0;
    marker.color.g = 0.5;
    marker.color.b = 1.0;
    marker.lifetime = rclcpp::Duration::from_seconds(0.1);
    return marker;
}();

static visualization_msgs::msg::Marker text_marker_ = [] {
    visualization_msgs::msg::Marker marker;
    marker.ns = "classification";
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.scale.z = 0.1;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.lifetime = rclcpp::Duration::from_seconds(0.1);
    return marker;
}();

static visualization_msgs::msg::MarkerArray marker_array_;
}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_HPP_
