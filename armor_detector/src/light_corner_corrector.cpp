// Maintained by Shenglin Qin, Chengfu Zou
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

#include "armor_detector/light_corner_corrector.hpp"

#include <iostream>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>


namespace rm_auto_aim
{

// 辅助函数：获取Mat类型的字符串表示
std::string getMatType(const cv::Mat & mat)
{
    int type = mat.type();

    std::string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:
            r = "8U";
            break;
        case CV_8S:
            r = "8S";
            break;
        case CV_16U:
            r = "16U";
            break;
        case CV_16S:
            r = "16S";
            break;
        case CV_32S:
            r = "32S";
            break;
        case CV_32F:
            r = "32F";
            break;
        case CV_64F:
            r = "64F";
            break;
        default:
            r = "User";
            break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

// 修正装甲板的灯条角点
void LightCornerCorrector::correctCorners(Armor & armor, const cv::Mat & gray_img)
{
    // 如果灯条的宽度过小，则不进行修正
    constexpr int PASS_OPTIMIZE_WIDTH = 3;

    if (armor.left_light.width > PASS_OPTIMIZE_WIDTH) {
        // 寻找左灯条的对称轴
        SymmetryAxis left_axis = findSymmetryAxis(gray_img, armor.left_light);
        armor.left_light.center = left_axis.centroid;
        armor.left_light.axis = left_axis.direction;
        // 寻找左灯条的角点
        if (cv::Point2f t = findCorner(gray_img, armor.left_light, left_axis, "top"); t.x > 0) {
            armor.left_light.top = t;
        }
        if (cv::Point2f b = findCorner(gray_img, armor.left_light, left_axis, "bottom"); b.x > 0) {
            armor.left_light.bottom = b;
        }
        armor.left_light.tilt_angle = std::atan2(
                                          armor.left_light.bottom.x - armor.left_light.top.x,
                                          armor.left_light.bottom.y - armor.left_light.top.y) *
                                      180 / CV_PI;
    }

    if (armor.right_light.width > PASS_OPTIMIZE_WIDTH) {
        // 寻找右灯条的对称轴
        SymmetryAxis right_axis = findSymmetryAxis(gray_img, armor.right_light);
        armor.right_light.center = right_axis.centroid;
        armor.right_light.axis = right_axis.direction;
        // 寻找右灯条的角点
        if (cv::Point2f t = findCorner(gray_img, armor.right_light, right_axis, "top"); t.x > 0) {
            armor.right_light.top = t;
        }
        if (cv::Point2f b = findCorner(gray_img, armor.right_light, right_axis, "bottom");
            b.x > 0) {
            armor.right_light.bottom = b;
        }
        armor.right_light.tilt_angle = std::atan2(
                                           armor.right_light.bottom.x - armor.right_light.top.x,
                                           armor.right_light.bottom.y - armor.right_light.top.y) *
                                       180 / CV_PI;
    }
    double theta_1 = armor.left_light.tilt_angle, theta_2 = armor.right_light.tilt_angle;
    armor.sign = (theta_1 + theta_2) / 2 <= 0;
}

// 寻找灯条的对称轴
// C++
SymmetryAxis LightCornerCorrector::findSymmetryAxis(const cv::Mat & gray_img, const Light & light)
{
    constexpr float MAX_BRIGHTNESS = 25.0f;  // 最大亮度值
    constexpr float scale = 0.07f;           // 缩放比例

    // 创建扩展的旋转矩形并限制在图像范围内
    cv::RotatedRect scaled_rect(
        light.center, 
        cv::Size2f(light.width * (1 + scale * 2), light.length * (1 + scale * 2)),
        light.angle);
    cv::Rect light_box = scaled_rect.boundingRect() & cv::Rect(0, 0, gray_img.cols, gray_img.rows);

    // 提取ROI并创建掩码
    cv::Mat roi = gray_img(light_box).clone();
    cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);

    // 调整旋转矩形到ROI坐标系
    scaled_rect.center -= cv::Point2f(light_box.x, light_box.y);

    // 创建多边形掩码
    cv::Point2f vertices[4];
    scaled_rect.points(vertices);
    std::vector<cv::Point> polygon;
    for (const auto& vertex : vertices) {
        polygon.emplace_back(static_cast<int>(vertex.x), static_cast<int>(vertex.y));
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{polygon}, 255);

    // 应用掩码并计算均值
    roi.setTo(0, ~mask);
    float mean_val = cv::mean(roi)[0];  // 计算平均亮度值
    roi.convertTo(roi, CV_32F);
    cv::normalize(roi, roi, 0, MAX_BRIGHTNESS, cv::NORM_MINMAX);

    // 获取非零像素点坐标
    std::vector<cv::Point2f> points;
    cv::findNonZero(roi, points);

    // 计算质心（重心）
    cv::Moments moments = cv::moments(roi, true);
    cv::Point2f centroid =
        cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00) +
        cv::Point2f(static_cast<float>(light_box.x), static_cast<float>(light_box.y));

    // 进行主成分分析 (PCA)
    cv::Mat data_pts(static_cast<int>(points.size()), 2, CV_32F);
    for (size_t i = 0; i < points.size(); i++) {
        data_pts.at<float>(static_cast<int>(i), 0) = points[i].x;
        data_pts.at<float>(static_cast<int>(i), 1) = points[i].y;
    }
    cv::PCA pca(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);

    // 获取第一主成分作为对称轴方向
    cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));

    // 归一化方向向量
    axis /= cv::norm(axis);

    // 调整方向，确保 y 分量为负方向
    if (axis.y > 0) {
        axis = -axis;
    }

    return SymmetryAxis{centroid, axis, mean_val};
}

// 寻找灯条的角点
cv::Point2f LightCornerCorrector::findCorner(
    const cv::Mat & gray_img, const Light & light, const SymmetryAxis & axis, std::string order)
{
    constexpr float START = 0.8f / 2;  // 搜索起始位置占灯条长度的比例
    constexpr float END = 1.2f / 2;    // 搜索结束位置占灯条长度的比例

    // 检查点是否在图像内
    auto inImage = [&gray_img](const cv::Point & point) -> bool {
        return point.x >= 0 && point.x < gray_img.cols && point.y >= 0 && point.y < gray_img.rows;
    };

    // 计算两点之间的距离
    auto distance = [](float x0, float y0, float x1, float y1) -> float {
        return std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
    };

    int oper = order == "top" ? 1 : -1;  // 决定搜索方向，顶端或底端
    float L = light.length;              // 灯条长度
    float dx = axis.direction.x * oper;  // 方向向量的 x 分量
    float dy = axis.direction.y * oper;  // 方向向量的 y 分量

    std::vector<cv::Point2f> candidates;  // 候选角点集合

    // 选择多个角点候选，取平均值作为最终角点
    int n = light.width - 2;
    int half_n = std::round(n / 2);
    for (int i = -half_n; i <= half_n; i++) {
        float x0 = axis.centroid.x + L * START * dx + i;
        float y0 = axis.centroid.y + L * START * dy;

        cv::Point2f prev = cv::Point2f(x0, y0);
        cv::Point2f corner = cv::Point2f(x0, y0);
        float max_brightness_diff = 0;
        bool has_corner = false;
        // 沿对称轴方向搜索，寻找亮度差最大的点作为角点
        for (float x = x0 + dx, y = y0 + dy; distance(x, y, x0, y0) < L * (END - START);
             x += dx, y += dy) {
            cv::Point2f cur = cv::Point2f(x, y);
            if (!inImage(cv::Point(cur))) {
                break;
            }

            float brightness_diff = gray_img.at<uchar>(prev) - gray_img.at<uchar>(cur);
            if (brightness_diff > max_brightness_diff && gray_img.at<uchar>(prev) > axis.mean_val) {
                max_brightness_diff = brightness_diff;
                corner = prev;
                has_corner = true;
            }

            prev = cur;
        }

        if (has_corner) {
            candidates.emplace_back(corner);
        }
    }
    if (!candidates.empty()) {
        // 计算所有候选角点的平均值
        cv::Point2f result =
            std::accumulate(candidates.begin(), candidates.end(), cv::Point2f(0, 0));
        return result / static_cast<float>(candidates.size());
    }

    return cv::Point2f(-1, -1);  // 未找到有效角点
}

}  // namespace rm_auto_aim