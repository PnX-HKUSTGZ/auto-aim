// Copyright 2022 Chen Jun

#include "armor_detector/pnp_solver.hpp"

#include <iostream>
#include <opencv2/calib3d.hpp>
#include <vector>

namespace rm_auto_aim
{
PnPSolver::PnPSolver(
    const std::array<double, 9> & camera_matrix, const std::vector<double> & dist_coeffs)
: camera_matrix_(cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone()),
  dist_coeffs_(cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data())).clone())
{
    // Unit: m,是真实长度
    constexpr double small_half_y = SMALL_ARMOR_WIDTH / 2.0;
    constexpr double small_half_z = SMALL_ARMOR_HEIGHT / 2.0;
    constexpr double large_half_y = LARGE_ARMOR_WIDTH / 2.0;
    constexpr double large_half_z = LARGE_ARMOR_HEIGHT / 2.0;

    // Start from bottom left in clockwise order
    // Model coordinate: x forward, y left, z up
    small_armor_points_.emplace_back(cv::Point3f(0, 0, small_half_z));
    small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, 0));
    small_armor_points_.emplace_back(cv::Point3f(0, 0, -small_half_z));
    small_armor_points_.emplace_back(cv::Point3f(0, -small_half_y, 0));

    large_armor_points_.emplace_back(cv::Point3f(0, 0, large_half_z));
    large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, 0));
    large_armor_points_.emplace_back(cv::Point3f(0, 0, -large_half_z));
    large_armor_points_.emplace_back(cv::Point3f(0, -large_half_y, 0));
}

bool PnPSolver::solvePnP(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec)
{
    std::vector<cv::Point2f> image_armor_points;

    // Fill in image points
    cv::Point2f toppoint = (armor.left_light.top + armor.right_light.top) / 2.0;
    cv::Point2f bottompoint = (armor.left_light.bottom + armor.right_light.bottom) / 2.0;
    cv::Point2f leftpoint = (armor.left_light.top + armor.left_light.bottom) / 2.0;
    cv::Point2f rightpoint = (armor.right_light.top + armor.right_light.bottom) / 2.0;
    image_armor_points.emplace_back(toppoint);
    image_armor_points.emplace_back(leftpoint);
    image_armor_points.emplace_back(bottompoint);
    image_armor_points.emplace_back(rightpoint);

    // Solve pnp
    auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
    bool success = cv::solvePnP(
        object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false,
        cv::SOLVEPNP_IPPE);
    return success;
}

float PnPSolver::calculateDistanceToCenter(const cv::Point2f & image_point)
{
    float cx = camera_matrix_.at<double>(0, 2);
    float cy = camera_matrix_.at<double>(1, 2);
    return cv::norm(image_point - cv::Point2f(cx, cy));
}

}  // namespace rm_auto_aim
