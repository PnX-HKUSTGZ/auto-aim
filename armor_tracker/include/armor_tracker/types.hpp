// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_TRACKER__TYPES_HPP_
#define ARMOR_TRACKER__TYPES_HPP_

#include <opencv2/core.hpp>

// STL
#include <algorithm>
#include <string>

namespace rm_auto_aim
{
// 状态
enum CarState
{
    XC = 0,
    VXC,
    YC,
    VYC,
    ZC1,
    ZC2, 
    VZC,
    VYAW,
    R1,
    R2, 
    YAW1,
    YAW2 // 11
};
// 跟踪目标
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
inline std::string visionModeToString(VisionMode mode) {
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
}  // namespace rm_auto_aim

#endif  // ARMOR_TRACKER__TYPES_HPP_
