// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__ARMOR_HPP_
#define ARMOR_DETECTOR__ARMOR_HPP_

#include <opencv2/core.hpp>

// STL
#include <algorithm>
#include <string>

namespace rm_auto_aim
{

// Unit: mm
static constexpr float SMALL_ARMOR_WIDTH = 130.0 / 1000.0;
static constexpr float SMALL_ARMOR_HEIGHT = 47.0 / 1000.0;
static constexpr float LARGE_ARMOR_WIDTH = 220.0 / 1000.0;
static constexpr float LARGE_ARMOR_HEIGHT = 47.0 / 1000.0;

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
  double yaw, pitch, roll; 
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_HPP_
