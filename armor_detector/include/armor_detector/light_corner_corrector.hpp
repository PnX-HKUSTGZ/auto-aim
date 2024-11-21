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

#ifndef ARMOR_DETECTOR_LIGHT_CORNER_CORRECTOR_HPP_
#define ARMOR_DETECTOR_LIGHT_CORNER_CORRECTOR_HPP_

// OpenCV 头文件
#include <opencv2/opencv.hpp>
// 项目头文件
#include "armor_detector/armor.hpp"

namespace rm_auto_aim {

// 定义对称轴结构体，包含质心、方向和平均亮度值
struct SymmetryAxis {
  cv::Point2f centroid;    // 对称轴的质心
  cv::Point2f direction;   // 对称轴的方向向量
  float mean_val;          // 平均亮度值
};

// 该类用于提高灯条角点的精度。
// 首先使用 PCA 算法找到灯条的对称轴，
// 然后沿着对称轴根据亮度梯度寻找灯条的角点。
class LightCornerCorrector {
public:
  explicit LightCornerCorrector() noexcept {}

  // 修正装甲板的灯条角点
  void correctCorners(Armor &armor, const cv::Mat &gray_img);

private:
  // 寻找灯条的对称轴
  SymmetryAxis findSymmetryAxis(const cv::Mat &gray_img, const Light &light);

  // 寻找灯条的角点
  cv::Point2f findCorner(const cv::Mat &gray_img,
                         const Light &light,
                         const SymmetryAxis &axis,
                         std::string order);
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_LIGHT_CORNER_CORRECTOR_HPP_