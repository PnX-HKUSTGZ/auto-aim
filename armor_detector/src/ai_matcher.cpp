// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#include "armor_detector/ai_matcher.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>

namespace rm_auto_aim
{
namespace
{
cv::Rect2f armorRect(const Armor & armor)
{
    auto points = armor.landmarks();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto & point : points) {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
    }

    return cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
}

cv::Point2f rectCenter(const cv::Rect2f & rect)
{
    return cv::Point2f(rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f);
}
}  // namespace

bool isNegativeArmor(const Armor & armor) { return armor.number == "negative"; }

void setArmorNegative(Armor & armor)
{
    armor.number = "negative";
    armor.confidence = 0.0f;
    armor.classfication_result = "negative: 0.0%";
}

bool isAIClassificationCompatibleWithCandidate(const Armor & ai_armor, const Armor & candidate)
{
    if (candidate.type == ArmorType::INVALID || ai_armor.type == ArmorType::INVALID) {
        return false;
    }

    return ai_armor.type == candidate.type;
}

double calculateArmorIoU(const Armor & lhs, const Armor & rhs)
{
    const auto lhs_rect = armorRect(lhs);
    const auto rhs_rect = armorRect(rhs);
    const auto intersection = lhs_rect & rhs_rect;
    const double intersection_area = intersection.area();
    const double union_area = lhs_rect.area() + rhs_rect.area() - intersection_area;

    if (union_area <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }

    return intersection_area / union_area;
}

double calculateCandidateCenterRatio(const Armor & candidate, const Armor & ai_armor)
{
    const auto candidate_rect = armorRect(candidate);
    const auto ai_rect = armorRect(ai_armor);
    const double diagonal = std::sqrt(
        candidate_rect.width * candidate_rect.width +
        candidate_rect.height * candidate_rect.height);

    if (diagonal <= std::numeric_limits<double>::epsilon()) {
        return std::numeric_limits<double>::infinity();
    }

    return cv::norm(rectCenter(candidate_rect) - rectCenter(ai_rect)) / diagonal;
}

int applyAIClassificationToCandidates(
    std::vector<Armor> & candidates, const std::vector<Armor> & ai_armors, double min_iou,
    double max_center_ratio)
{
    for (auto & candidate : candidates) {
        setArmorNegative(candidate);
    }

    std::vector<size_t> ai_indices(ai_armors.size());
    std::iota(ai_indices.begin(), ai_indices.end(), 0);
    std::sort(ai_indices.begin(), ai_indices.end(), [&ai_armors](size_t lhs, size_t rhs) {
        return ai_armors[lhs].confidence > ai_armors[rhs].confidence;
    });

    std::vector<bool> used_candidates(candidates.size(), false);
    int matched_count = 0;

    for (const auto ai_index : ai_indices) {
        const auto & ai_armor = ai_armors[ai_index];
        int best_candidate = -1;
        double best_iou = -1.0;
        double best_center_ratio = std::numeric_limits<double>::infinity();

        for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
            if (used_candidates[candidate_index]) {
                continue;
            }

            const auto & candidate = candidates[candidate_index];
            if (!isAIClassificationCompatibleWithCandidate(ai_armor, candidate)) {
                continue;
            }

            const double iou = calculateArmorIoU(candidate, ai_armor);
            const double center_ratio = calculateCandidateCenterRatio(candidate, ai_armor);
            if (iou < min_iou || center_ratio > max_center_ratio) {
                continue;
            }

            constexpr double SAME_IOU_EPS = 1e-6;
            if (iou > best_iou + SAME_IOU_EPS ||
                (std::abs(iou - best_iou) <= SAME_IOU_EPS && center_ratio < best_center_ratio)) {
                best_candidate = static_cast<int>(candidate_index);
                best_iou = iou;
                best_center_ratio = center_ratio;
            }
        }

        if (best_candidate < 0) {
            continue;
        }

        auto & candidate = candidates[static_cast<size_t>(best_candidate)];
        candidate.number = ai_armor.number;
        candidate.confidence = ai_armor.confidence;
        candidate.classfication_result = ai_armor.classfication_result;
        used_candidates[static_cast<size_t>(best_candidate)] = true;
        ++matched_count;
    }

    return matched_count;
}

}  // namespace rm_auto_aim
