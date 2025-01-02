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
#include <string>
#include <vector>
#include <tuple>
#include <random>
// third party
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
// project
#include "rune_detector/types.hpp"

namespace rm_auto_aim {

class RuneDetector {
public:
    // Construct a new OpenVINO Detector object
    RuneDetector(int max_iterations, double distance_threshold, double prob_threshold, EnemyColor detect_color);
    std::vector<cv::Point2f> processHittingLights(); // 处理击打灯条
    std::vector<std::vector<cv::Point2f>> processhitLights(); // 处理 已击中灯条
    std::tuple<cv::Point2f, cv::Mat> detectRTag(const cv::Mat &img, const cv::Point2f &prior); //检查R标
    std::vector<RuneObject> detectRune(const cv::Mat &img); // 检测能量机关

private:
    void preprocess(); // 预处理函数
    double calculateAngleDifference(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse); // 计算角度差
    double calculateAxisLength(const cv::RotatedRect& ellipse, const cv::Point2f& direction); // 计算椭圆沿特定方向的轴的长度
    double calculateRatioDifferenceHitting(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse); // 计算比例差
    double calculateMatchScoreHitting(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse); // 计算匹配程度
    double calculateRatioDifferencehit(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse); // 计算比例差
    double calculateMatchScorehit(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse); // 计算匹配程度
    std::vector<cv::Point2f> ellipseIntersections(const cv::RotatedRect& ellipse, const cv::Point2f& dir); // 计算指定方向与椭圆的两个交点
    cv::Rect calculateROI(const cv::RotatedRect& rect); // 计算疑似打击目标的ROI
    double pointToEllipseDistance(const cv::Point2f& point, const cv::RotatedRect& ellipse); // 计算点到椭圆的距离
    cv::RotatedRect fitEllipseRANSAC(const std::vector<cv::Point>& points); // 修改后的RANSAC拟合椭圆函数
    std::vector<cv::RotatedRect> detectEllipses(const cv::Mat& src); // 使用随机霍夫变换检测椭圆
    cv::RotatedRect detectBestEllipse(const cv::Mat& src); // 检测最佳椭圆
    std::vector<cv::Point2f> getSignalPoints(const cv::RotatedRect& ellipse, const cv::RotatedRect& rect); // 提取 6 个 signal points
    bool isRightColor(const cv::RotatedRect& rect); // 判断颜色是否正确

    //image
    cv::Mat frame;
    cv::Mat flow_img, arm_img, hit_img, aim_img; 
    cv::Mat hitting_light_mask;
    cv::Point2f center;
    //parameters
    int max_iterations; // 最大迭代次数(RANSAC)
    double distance_threshold; // 距离阈值(RANSAC)
    double prob_threshold; // 可信度阈值(RANSAC)
    EnemyColor detect_color; // 检测颜色



};
}  // namespace rm_auto_aim
#endif  // DETECTOR_HPP_
