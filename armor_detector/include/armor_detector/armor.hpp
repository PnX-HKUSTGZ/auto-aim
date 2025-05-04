// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__ARMOR_HPP_
#define ARMOR_DETECTOR__ARMOR_HPP_

#include <opencv2/core.hpp>
#include <Eigen/Core>

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
  static inline std::vector<PointType> buildObjectPoints(const double &w,
                                                         const double &h) {
  return {PointType(0, w / 2, -h / 2),
          PointType(0, w / 2, h / 2),
          PointType(0, -w / 2, h / 2),
          PointType(0, -w / 2, -h / 2)};
  }
  std::vector<cv::Point2f> landmarks() const {
    return {left_light.bottom, left_light.top, right_light.top, right_light.bottom};
  }

  void setCameraArmor(const Eigen::Matrix3d & r_odom_to_camera, 
                      const Eigen::Vector3d & t_odom_to_camera)
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

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_HPP_
