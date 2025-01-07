// Created by Labor 2024.1.27
// Maintained by Chengfu Zou, Labor
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

#ifndef RUNE_SOLVER_TYPES_HPP_
#define RUNE_SOLVER_TYPES_HPP_

// STD
#include <array>

// 3rd party
#include <ceres/ceres.h>

#include <opencv2/opencv.hpp>

namespace rm_auto_aim {

constexpr double DEG_72 = 0.4 * CV_PI;
constexpr int ARMOR_KEYPOINTS_NUM = 4;
constexpr int KEYPOINTS_NUM = 5;

// Motion type
enum class MotionType { SMALL, BIG, UNKNOWN };

// Moving direction
enum Direction { CLOCKWISE = -1, ANTI_CLOCKWISE = 1, UNKNOWN = 0 };

// Rune arm length, Unit: m
constexpr double ARM_LENGTH = 0.700;

// Acceptable distance between robot and rune, Unit: m
// True value = 6.436 m
constexpr double MIN_RUNE_DISTANCE = 4.0;
constexpr double MAX_RUNE_DISTANCE = 9.0;

// Rune object points
// r_tag, bottom_left, top_left, top_right, bottom_right
const std::vector<cv::Point3f> RUNE_OBJECT_POINTS = {cv::Point3f(0, 0, 0) / 1000,
                                                     cv::Point3f(0, -184, 186) / 1000,
                                                     cv::Point3f(0, -514, 0) / 1000,
                                                     cv::Point3f(0, -550, 0) / 1000,
                                                     cv::Point3f(0, -700, 150) / 1000,
                                                     cv::Point3f(0, -700, -150) / 1000,
                                                     cv::Point3f(0, -850, 0) / 1000};

#define BIG_RUNE_CURVE(x, a, omega, b, c, d, sign) \
  ((-((a) / (omega) * ceres::cos((omega) * ((x) + (d)))) + (b) * ((x) + (d)) + (c)) * (sign))

#define SMALL_RUNE_CURVE(x, a, b, c, sign) (((a) * ((x) + (b)) + (c)) * (sign))

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

}  // namespace rm_auto_aim
#endif  // RUNE_SOLVER_TYPES_HPP_
