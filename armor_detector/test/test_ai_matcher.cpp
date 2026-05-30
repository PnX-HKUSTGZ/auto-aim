// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "armor_detector/ai_matcher.hpp"

namespace
{
rm_auto_aim::Armor makeArmor(
    float x, float y, float width, float height, rm_auto_aim::ArmorType type,
    const std::string & number, float confidence)
{
    rm_auto_aim::Light left(rm_auto_aim::RED, cv::Point2f(x, y), cv::Point2f(x, y + height));
    rm_auto_aim::Light right(
        rm_auto_aim::RED, cv::Point2f(x + width, y), cv::Point2f(x + width, y + height));
    rm_auto_aim::Armor armor(left, right);
    armor.type = type;
    armor.number = number;
    armor.confidence = confidence;
    armor.classfication_result = number + ": test";
    return armor;
}
}  // namespace

TEST(AIMatcherTest, AIResultMatchesBestTraditionalCandidate)
{
    std::vector<rm_auto_aim::Armor> candidates{
        makeArmor(4.0f, 0.0f, 20.0f, 10.0f, rm_auto_aim::ArmorType::SMALL, "", 0.0f),
        makeArmor(0.0f, 0.0f, 20.0f, 10.0f, rm_auto_aim::ArmorType::SMALL, "", 0.0f)};
    const std::vector<rm_auto_aim::Armor> ai_armors{
        makeArmor(0.0f, 0.0f, 20.0f, 10.0f, rm_auto_aim::ArmorType::SMALL, "3", 0.9f)};

    const int matched =
        rm_auto_aim::applyAIClassificationToCandidates(candidates, ai_armors, 0.1, 0.5);

    EXPECT_EQ(matched, 1);
    EXPECT_TRUE(rm_auto_aim::isNegativeArmor(candidates[0]));
    EXPECT_EQ(candidates[1].number, "3");
    EXPECT_FLOAT_EQ(candidates[1].confidence, 0.9f);
}

TEST(AIMatcherTest, CandidateCanOnlyBeOccupiedOnce)
{
    std::vector<rm_auto_aim::Armor> candidates{
        makeArmor(0.0f, 0.0f, 20.0f, 10.0f, rm_auto_aim::ArmorType::SMALL, "", 0.0f)};
    const std::vector<rm_auto_aim::Armor> ai_armors{
        makeArmor(0.0f, 0.0f, 20.0f, 10.0f, rm_auto_aim::ArmorType::SMALL, "3", 0.9f),
        makeArmor(0.0f, 0.0f, 20.0f, 10.0f, rm_auto_aim::ArmorType::SMALL, "4", 0.8f)};

    const int matched =
        rm_auto_aim::applyAIClassificationToCandidates(candidates, ai_armors, 0.1, 0.5);

    EXPECT_EQ(matched, 1);
    EXPECT_EQ(candidates[0].number, "3");
}

TEST(AIMatcherTest, UnmatchedCandidatesBecomeNegative)
{
    std::vector<rm_auto_aim::Armor> candidates{
        makeArmor(0.0f, 0.0f, 20.0f, 10.0f, rm_auto_aim::ArmorType::SMALL, "", 0.0f),
        makeArmor(100.0f, 0.0f, 20.0f, 10.0f, rm_auto_aim::ArmorType::SMALL, "", 0.0f)};
    const std::vector<rm_auto_aim::Armor> ai_armors;

    const int matched =
        rm_auto_aim::applyAIClassificationToCandidates(candidates, ai_armors, 0.1, 0.5);

    EXPECT_EQ(matched, 0);
    EXPECT_TRUE(rm_auto_aim::isNegativeArmor(candidates[0]));
    EXPECT_TRUE(rm_auto_aim::isNegativeArmor(candidates[1]));
}

TEST(AIMatcherTest, TypeMismatchInvalidatesMatch)
{
    std::vector<rm_auto_aim::Armor> candidates{
        makeArmor(0.0f, 0.0f, 20.0f, 10.0f, rm_auto_aim::ArmorType::SMALL, "", 0.0f)};
    const std::vector<rm_auto_aim::Armor> ai_armors{
        makeArmor(0.0f, 0.0f, 20.0f, 10.0f, rm_auto_aim::ArmorType::LARGE, "base", 0.9f)};

    const int matched =
        rm_auto_aim::applyAIClassificationToCandidates(candidates, ai_armors, 0.1, 0.5);

    EXPECT_EQ(matched, 0);
    EXPECT_TRUE(rm_auto_aim::isNegativeArmor(candidates[0]));
}
