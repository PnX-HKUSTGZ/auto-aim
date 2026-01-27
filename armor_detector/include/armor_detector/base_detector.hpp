// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__BASE_DETECTOR_HPP_
#define ARMOR_DETECTOR__BASE_DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>

// STD
#include <string>
#include <vector>

#include "armor_detector/light_corner_corrector.hpp"
#include "armor_detector/types.hpp"

namespace rm_auto_aim
{

/**
 * @brief 装甲板检测器基类
 * 
 * 定义了所有装甲板检测器的统一接口
 */
class BaseDetector
{
public:
    /**
     * @brief 构造函数
     */
    BaseDetector() = default;

    /**
     * @brief 虚析构函数
     */
    virtual ~BaseDetector() = default;

    /**
     * @brief 检测装甲板（纯虚函数）
     * 
     * @param input 输入图像
     * @param detect_color 检测颜色 (0: red, 1: blue)
     * @return std::vector<Armor> 检测到的装甲板数组
     */
    virtual std::vector<Armor> detect(const cv::Mat & input, int detect_color) = 0;

    /**
     * @brief 在图像上绘制检测结果（纯虚函数）
     * 
     * @param img 要绘制结果的图像
     */
    virtual void drawResults(cv::Mat & img) = 0;

    /**
     * @brief 获取检测器类型名称
     * 
     * @return std::string 检测器类型名称
     */
    virtual std::string getDetectorType() const = 0;

protected:
    // 装甲板
    std::vector<Armor> armors_;  ///< 装甲板信息

    // 图像信息
    cv::Mat gray_img;  ///< 灰度图像

    // Light corner corrector
    LightCornerCorrector lcc;  ///< 用于校正灯条角点的工具
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__BASE_DETECTOR_HPP_
