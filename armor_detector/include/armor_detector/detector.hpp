// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

// STD
#include <cmath>
#include <string>
#include <vector>

#include "armor_detector/number_classifier.hpp"
#include "armor_detector/types.hpp"
#include "auto_aim_interfaces/msg/debug_armors.hpp"
#include "auto_aim_interfaces/msg/debug_lights.hpp"

namespace rm_auto_aim
{
/**
 * @brief 装甲板检测器类
 * 
 * 负责从图像中检测装甲板，包括灯条的提取和装甲板的识别与匹配
 */
class Detector
{
public:
    /**
     * @brief 灯条参数结构体
     * 
     * 定义了灯条选择的限制参数
     */
    struct LightParams
    {
        double min_ratio;  ///< 灯条宽高比的最小值
        double max_ratio;  ///< 灯条宽高比的最大值
        double max_angle;  ///< 灯条允许的最大垂直倾斜角度
    };

    /**
     * @brief 装甲板参数结构体
     * 
     * 定义了装甲板选择的限制参数
     */
    struct ArmorParams
    {
        double min_light_ratio;  ///< 两灯条的最小比例值
        // 灯条对中心距离
        double min_small_center_distance;  ///< 小装甲板灯条中心的最小距离
        double max_small_center_distance;  ///< 小装甲板灯条中心的最大距离
        double min_large_center_distance;  ///< 大装甲板灯条中心的最小距离
        double max_large_center_distance;  ///< 大装甲板灯条中心的最大距离
        double max_angle;                  ///< 装甲板允许的最大水平倾斜角度
    };

    /**
     * @brief 构造一个装甲板检测器
     * 
     * @param bin_thres 二值化阈值
     * @param color 检测的颜色（如蓝色或红色）
     * @param l 灯条参数
     * @param a 装甲板参数
     */
    Detector(
        const int & bin_thres, const int & color, const LightParams & l, const ArmorParams & a);

    /**
     * @brief 处理输入图像并检测装甲板
     * 
     * 整个检测的主流程，包括预处理、灯条检测和装甲板匹配
     * 
     * @param input 输入的图像
     * @return std::vector<Armor> 检测到的装甲板数组
     */
    std::vector<Armor> detect(const cv::Mat & input);

    /**
     * @brief 对输入图像进行预处理
     * 
     * 包括灰度转换、二值化等操作以便于后续的灯条检测
     * 
     * @param input 输入的原始图像
     */
    void preprocessImage(const cv::Mat & input);

    /**
     * @brief 从图像中找出所有可能的灯条
     * 
     * @param rbg_img 原始的RGB图像
     * @param binary_img 二值化后的图像
     * @return std::vector<Light> 检测到的灯条数组
     */
    std::vector<Light> findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img);

    /**
     * @brief 从灯条中匹配装甲板
     * 
     * 根据灯条之间的几何关系，匹配可能的装甲板
     * 
     * @param lights 检测到的灯条数组
     * @return std::vector<Armor> 匹配得到的装甲板数组
     */
    std::vector<Armor> matchLights(const std::vector<Light> & lights);

    /**
     * @brief 获取所有数字图像用于调试
     * 
     * @return cv::Mat 包含所有数字的图像
     */
    cv::Mat getAllNumbersImage();

    /**
     * @brief 在图像上绘制检测结果
     * 
     * @param img 要绘制结果的图像
     */
    void drawResults(cv::Mat & img);

    int binary_thres;  ///< 二值化阈值
    int detect_color;  ///< 检测的颜色
    LightParams l;     ///< 灯条参数
    ArmorParams a;     ///< 装甲板参数

    std::unique_ptr<NumberClassifier> classifier;  ///< 数字分类器

    // 调试信息
    cv::Mat binary_img, gray_img;                        ///< 处理过程中的图像用于调试
    auto_aim_interfaces::msg::DebugLights debug_lights;  ///< 灯条调试信息
    auto_aim_interfaces::msg::DebugArmors debug_armors;  ///< 装甲板调试信息

private:
    /**
     * @brief 判断一个轮廓是否符合灯条特征
     * 
     * @param possible_light 可能的灯条
     * @return true 如果符合灯条特征
     * @return false 如果不符合灯条特征
     */
    bool isLight(const Light & possible_light);

    /**
     * @brief 检查两个灯条之间是否有其他灯条
     * 
     * @param light_1 第一个灯条
     * @param light_2 第二个灯条
     * @param lights 所有检测到的灯条
     * @return true 如果有其他灯条在这两个灯条之间
     * @return false 如果没有其他灯条在这两个灯条之间
     */
    bool containLight(
        const Light & light_1, const Light & light_2, const std::vector<Light> & lights);

    /**
     * @brief 判断两个灯条是否能组成装甲板，并确定装甲板类型
     * 
     * @param light_1 第一个灯条
     * @param light_2 第二个灯条
     * @return ArmorType 装甲板类型（大装甲板、小装甲板或无效）
     */
    ArmorType isArmor(const Light & light_1, const Light & light_2);

    std::vector<Light> lights_;  ///< 当前检测到的所有灯条
    std::vector<Armor> armors_;  ///< 当前检测到的所有装甲板
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_HPP_
