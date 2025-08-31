// Copyright (c) 2022 ChenJun
// Licensed under the MIT License.

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "armor_detector/detector.hpp"

namespace rm_auto_aim
{
Detector::Detector(
    const int & bin_thres, const LightParams & l, const ArmorParams & a,
    const std::string & model_path, const std::string & label_path, const float & threshold,
    const std::vector<std::string> & ignore_classes)
: binary_thres(bin_thres), l(l), a(a)
{
    this->classifier =
        std::make_unique<NumberClassifier>(model_path, label_path, threshold, ignore_classes);
}

std::vector<Armor> Detector::detect(
    const cv::Mat & input, int detect_color)  //侦测，分类装甲板的主函数
{
    preprocessImage(input);  //生成二值化后的图片
    lights_ = findLights(input, binary_img);
    armors_ = matchLights(lights_, detect_color);

    if (!armors_.empty()) {
        classifier->extractNumbers(input, armors_);
        classifier->classify(armors_);
    }

    for (auto & armor : armors_) {
        lcc.correctCorners(armor, gray_img);
    }

    return armors_;
}

void Detector::preprocessImage(const cv::Mat & rgb_img)  //生成二值化后的图片
{
    cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);

    cv::threshold(gray_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);
}

std::vector<Light> Detector::findLights(
    const cv::Mat & rbg_img, const cv::Mat & binary_img)  //找到灯条
{
    using std::vector;
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    vector<Light> lights;

    for (const auto & contour : contours) {
        if (contour.size() < 5) continue;

        auto r_rect = cv::minAreaRect(contour);
        auto light = Light(r_rect);

        if (isLight(light)) {
            auto rect = light.boundingRect();
            if (  // Avoid assertion failed
                0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols &&
                0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= rbg_img.rows) {
                auto roi = rbg_img(rect);
                cv::Scalar mean_val = cv::mean(roi, binary_img(rect));
                double mean_r = mean_val[0];
                double mean_b = mean_val[2];

                // Sum of red pixels > sum of blue pixels ?
                light.color = mean_r > mean_b ? RED : BLUE;
                lights.emplace_back(light);
            }
        }
    }

    return lights;
}

bool Detector::isLight(const Light & light)  //findlights中用于判断是否为灯条的bool函数
{
    // The ratio of light (short side / long side)
    float ratio = light.width / light.length;
    bool ratio_ok = l.min_ratio < ratio && ratio < l.max_ratio;

    bool angle_ok = abs(light.tilt_angle) < l.max_angle;

    bool is_light = ratio_ok && angle_ok;

    return is_light;
}

std::vector<Armor> Detector::matchLights(
    const std::vector<Light> & lights, int detect_color)  //将灯条合成装甲板的魔法，返回合格的装甲板
{
    std::vector<Armor> armors;

    // Loop all the pairing of lights
    for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
        for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
            if (light_1->color != detect_color || light_2->color != detect_color) continue;

            if (containLight(*light_1, *light_2, lights)) {
                continue;
            }

            auto type = isArmor(*light_1, *light_2);
            if (type != ArmorType::INVALID) {
                auto armor = Armor(*light_1, *light_2);
                armor.type = type;
                armors.emplace_back(armor);
            }
        }
    }

    return armors;
}

// Check if there is another light in the boundingRect formed by the 2 lights
bool Detector::containLight(
    const Light & light_1, const Light & light_2, const std::vector<Light> & lights)
{
    auto points =
        std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
    auto bounding_rect = cv::boundingRect(points);

    for (const auto & test_light : lights) {
        if (test_light.center == light_1.center || test_light.center == light_2.center) continue;

        if (bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
            bounding_rect.contains(test_light.center)) {
            return true;
        }
    }

    return false;
}

ArmorType Detector::isArmor(
    const Light & light_1, const Light & light_2)  //应用在matchLights函数中，判断是否为一个装甲
{
    // Ratio of the length of 2 lights (short side / long side)
    float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                               : light_2.length / light_1.length;
    //判断两个灯条的长度差异是否在合理范围内
    bool light_ratio_ok = light_length_ratio > a.min_light_ratio;

    // Distance between the center of 2 lights (unit : light length)
    float avg_light_length = (light_1.length + light_2.length) / 2;
    //两灯条之间的距离和灯条长度的比值是否合理
    float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
    bool center_distance_ok = (a.min_small_center_distance <= center_distance &&
                               center_distance < a.max_small_center_distance) ||
                              (a.min_large_center_distance <= center_distance &&
                               center_distance < a.max_large_center_distance);

    // Angle of light center connection
    cv::Point2f diff = light_1.center - light_2.center;
    float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
    bool angle_ok = angle < a.max_angle;

    bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

    // Judge armor type
    ArmorType type;
    if (is_armor) {
        type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
    } else {
        type = ArmorType::INVALID;
    }

    return type;
}

cv::Mat Detector::getAllNumbersImage()
{
    if (armors_.empty()) {
        return cv::Mat(cv::Size(20, 28), CV_8UC1);
    } else {
        std::vector<cv::Mat> number_imgs;
        number_imgs.reserve(armors_.size());
        for (auto & armor : armors_) {
            number_imgs.emplace_back(armor.number_img);
        }
        cv::Mat all_num_img;
        cv::vconcat(number_imgs, all_num_img);
        return all_num_img;
    }
}

cv::Mat Detector::getBinaryImage() { return binary_img; }

void Detector::drawResults(cv::Mat & img)
{
    // Draw Lights
    for (const auto & light : lights_) {
        cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
        cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
        auto line_color = light.color == RED ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
        cv::line(img, light.top, light.bottom, line_color, 5);
    }

    // Draw armors
    for (const auto & armor : armors_) {
        cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
        cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
    }

    // Show numbers and confidence
    for (const auto & armor : armors_) {
        cv::putText(
            img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 255), 2);
    }
}

}  // namespace rm_auto_aim
