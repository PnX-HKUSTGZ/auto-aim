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

#ifndef RUNE_DETECTOR_TYPES_HPP_
#define RUNE_DETECTOR_TYPES_HPP_

// 3rd party
#include <opencv2/opencv.hpp>
// project

namespace rm_auto_aim {

enum class EnemyColor {
  RED = 0,
  BLUE = 1,
  WHITE = 2,
};
inline std::string enemyColorToString(EnemyColor color) {
  switch (color) {
    case EnemyColor::RED:
      return "RED";
    case EnemyColor::BLUE:
      return "BLUE";
    case EnemyColor::WHITE:
      return "WHITE";
    default:
      return "UNKNOWN";
  }
}

enum VisionMode {
  AUTO_AIM_SLOPE = 0,
  AUTO_AIM_FLAT = 1,
  SMALL_RUNE = 2,
  BIG_RUNE = 3,
};
inline std::string visionModeToString(VisionMode mode) {
  switch (mode) {
    case VisionMode::AUTO_AIM_SLOPE:
      return "AUTO_AIM_SLOPE";
    case VisionMode::AUTO_AIM_FLAT:
      return "AUTO_AIM_FLAT";
    case VisionMode::SMALL_RUNE:
      return "SMALL_RUNE";
    case VisionMode::BIG_RUNE:
      return "BIG_RUNE";
    default:
      return "UNKNOWN";
  }
}

enum class RuneType { INACTIVATED = 0, ACTIVATED };

struct FeaturePoints {
  FeaturePoints() {
    r_center = cv::Point2f(-1, -1);
    bottom_right = cv::Point2f(-1, -1);
    top_right = cv::Point2f(-1, -1);
    top_left = cv::Point2f(-1, -1);
    bottom_left = cv::Point2f(-1, -1);
  }

  void reset() {
    r_center = cv::Point2f(-1, -1);
    bottom_right = cv::Point2f(-1, -1);
    top_right = cv::Point2f(-1, -1);
    top_left = cv::Point2f(-1, -1);
    bottom_left = cv::Point2f(-1, -1);
  }

  FeaturePoints operator+(const FeaturePoints &other) {
    FeaturePoints res;
    res.r_center = r_center + other.r_center;
    res.bottom_right = bottom_right + other.bottom_right;
    res.top_right = top_right + other.top_right;
    res.top_left = top_left + other.top_left;
    res.bottom_left = bottom_left + other.bottom_left;
    return res;
  }

  FeaturePoints operator/(const float &other) {
    FeaturePoints res;
    res.r_center = r_center / other;
    res.bottom_right = bottom_right / other;
    res.top_right = top_right / other;
    res.top_left = top_left / other;
    res.bottom_left = bottom_left / other;
    return res;
  }

  std::vector<cv::Point2f> toVector2f() const {
    return {r_center, bottom_left, top_left, top_right, bottom_right};
  }
  std::vector<cv::Point> toVector2i() const {
    return {r_center, bottom_left, top_left, top_right, bottom_right};
  }

  cv::Point2f r_center;
  cv::Point2f bottom_right;
  cv::Point2f top_right;
  cv::Point2f top_left;
  cv::Point2f bottom_left;

  std::vector<FeaturePoints> children;
};

struct RuneObject {
  EnemyColor color;
  RuneType type;
  float prob;
  FeaturePoints pts;
  cv::Rect box;
};

}  // namespace rm_auto_aim
#endif  // RUNE_DETECTOR_TYPES_HPP_
