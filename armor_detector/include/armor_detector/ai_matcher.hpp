// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__AI_MATCHER_HPP_
#define ARMOR_DETECTOR__AI_MATCHER_HPP_

#include <vector>

#include "armor_detector/types.hpp"

namespace rm_auto_aim
{

bool isNegativeArmor(const Armor & armor);

void setArmorNegative(Armor & armor);

bool isAIClassificationCompatibleWithCandidate(const Armor & ai_armor, const Armor & candidate);

double calculateArmorIoU(const Armor & lhs, const Armor & rhs);

double calculateCandidateCenterRatio(const Armor & candidate, const Armor & ai_armor);

int applyAIClassificationToCandidates(
    std::vector<Armor> & candidates, const std::vector<Armor> & ai_armors, double min_iou,
    double max_center_ratio);

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__AI_MATCHER_HPP_
