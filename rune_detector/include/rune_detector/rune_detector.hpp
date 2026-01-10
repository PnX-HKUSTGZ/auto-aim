// Copyright 2023 Yunlong Feng
//
// Additional modifications and features by Chengfu Zou, 2024.
//
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

#ifndef DETECTOR_HPP_
#define DETECTOR_HPP_

// std
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <opencv2/core/types.hpp>
#include <random>
#include <string>
#include <tuple>
#include <vector>
// third party
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
// project
#include "rune_detector/types.hpp"

namespace rm_auto_aim
{

class RuneDetector
{
public:
    struct Params
    {
        int gray_threshold_red = 80;
        int gray_threshold_blue = 80;
        double min_contour_area = 20.0;
        double max_contour_area = 20000.0;
        double big_active_fan_area = 1000.0;
        double active_fan_concentricity_ratio = 0.08;
        double min_force_construct_angle_deg = 45.0;
        double max_distance_ratio = 1.5;
        double max_match_deviation_ratio = 0.2;
        double center_force_construct_window_ratio = 0.2;
        bool enable_center_force_construct_window = true;

        // fan active
        double fan_active_min_area = 100.0;
        double fan_active_max_area = 4000.0;
        double fan_active_min_area_ratio = 0.05;
        double fan_active_max_area_ratio = 0.60;
        double fan_active_min_area_peri_ratio = 0.0002;
        double fan_active_max_area_peri_ratio = 0.030;
        double fan_active_max_side_ratio = 2.0;
        double fan_active_min_area_incomplete = 30.0;
        double fan_active_max_area_peri_ratio_incomplete = 0.07;
        double fan_active_max_direction_delta_incomplete = 45.0;

        // fan inactive
        double fan_inactive_min_area = 100.0;
        double fan_inactive_max_area = 4000.0;
        double fan_inactive_min_side_ratio = 2.0;
        double fan_inactive_max_side_ratio = 6.0;
        double fan_inactive_max_distance_ratio = 6.0;

        // target active
        double target_active_min_area = 60.0;
        double target_active_max_area = 6000.0;
        double target_active_min_side_ratio = 0.99;
        double target_active_max_side_ratio = 1.55;
        double target_active_min_area_ratio = 0.8;
        double target_active_max_area_ratio = 1.20;

        // target inactive
        double target_inactive_min_area = 60.0;
        double target_inactive_max_area = 6000.0;
        double target_inactive_min_side_ratio = 0.99;
        double target_inactive_max_side_ratio = 1.55;
        double target_inactive_min_area_ratio = 0.8;
        double target_inactive_max_area_ratio = 1.20;

        // center
        double center_min_area = 10.0;
        double center_max_area = 1000.0;
        double center_min_side_ratio = 0.3;
        double center_max_side_ratio = 2.5;
        double center_min_convex_area_ratio = 0.9;
        double center_min_roundness = 0.2;
        double center_max_roundness = 0.9;
    };

    RuneDetector() = default;
    explicit RuneDetector(const Params & params);

    // 检测能量机关（输出未激活靶心对应的 7 点，如无则空）
    std::vector<RuneObject> detectRune(const cv::Mat & img, int min_lightness);

    // 检测 R 标（若未找到中心则回退）
    std::tuple<cv::Point2f, cv::Mat> detectRTag(const cv::Mat & img, const cv::Point2f & prior);

    // 更新参数
    void setParams(const Params & params) { params_ = params; }

    EnemyColor detect_color;

private:
    struct Candidate
    {
        cv::RotatedRect rect;
        double area = 0.0;
    };

    // helpers
    cv::Mat makeBinary(const cv::Mat & rgb, int min_lightness) const;
    std::vector<std::vector<cv::Point>> findValidContours(const cv::Mat & bin) const;
    std::vector<Candidate> classifyCenters(const std::vector<std::vector<cv::Point>> & contours) const;
    std::vector<Candidate> classifyTargets(const std::vector<std::vector<cv::Point>> & contours, bool active) const;
    std::vector<Candidate> classifyFans(const std::vector<std::vector<cv::Point>> & contours, bool active) const;
    cv::Point2f chooseCenter(const std::vector<Candidate> & centers, const std::vector<Candidate> & targets, const std::vector<Candidate> & fans) const;
    std::vector<std::tuple<Candidate, Candidate, Candidate>> buildCombos(const std::vector<Candidate> & targets_inactive, const std::vector<Candidate> & targets_active, const std::vector<Candidate> & fans_inactive, const std::vector<Candidate> & fans_active, const cv::Point2f & center) const;
    static double rectLongSide(const cv::RotatedRect & r);
    static double aspect(const cv::RotatedRect & r);
    static double roundness(const std::vector<cv::Point> & c);
    static cv::Point2f rectDir(const cv::RotatedRect & r);
    static void rectLongEndpoints(const cv::RotatedRect & r, cv::Point2f & a, cv::Point2f & b);
    static FeaturePoints toFeaturePoints(const Candidate & center, const Candidate & target, const Candidate & fan);

    Params params_;
};
}  // namespace rm_auto_aim
#endif  // DETECTOR_HPP_
