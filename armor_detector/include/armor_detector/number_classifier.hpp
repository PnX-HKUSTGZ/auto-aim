// Copyright 2022 Chen Jun

#ifndef ARMOR_DETECTOR__NUMBER_CLASSIFIER_HPP_
#define ARMOR_DETECTOR__NUMBER_CLASSIFIER_HPP_

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "armor_detector/types.hpp"

namespace rm_auto_aim
{
/**
 * @brief 装甲板数字分类器
 * 
 * 使用深度学习模型对装甲板上的数字进行识别和分类
 */
class NumberClassifier
{
public:
    /**
     * @brief 构造一个数字分类器
     * 
     * @param model_path 深度学习模型路径
     * @param label_path 标签文件路径，包含可识别的数字类别
     * @param threshold 分类置信度阈值，低于此阈值的分类结果将被忽略
     * @param ignore_classes 需要忽略的数字类别列表
     */
    NumberClassifier(
        const std::string & model_path, const std::string & label_path, const double threshold,
        const std::vector<std::string> & ignore_classes = {});

    /**
     * @brief 从原始图像中提取装甲板上的数字图像
     * 
     * 根据装甲板的位置和姿态，从原始图像中提取出装甲板数字部分，
     * 并对提取的图像进行预处理，为后续的分类做准备。
     * 
     * @param src 原始图像
     * @param armors 待提取数字的装甲板数组，函数会更新其中的number_img属性
     */
    void extractNumbers(const cv::Mat & src, std::vector<Armor> & armors);

    /**
     * @brief 对装甲板上的数字进行分类
     * 
     * 使用加载的深度学习模型，对提取的装甲板数字图像进行分类，
     * 识别出对应的数字类别。
     * 
     * @param armors 需要分类的装甲板数组，函数会更新其中的number和confidence属性
     */
    void classify(std::vector<Armor> & armors);

    double threshold;  ///< 分类置信度阈值

private:
    cv::dnn::Net net_;                         ///< 深度学习网络模型
    std::vector<std::string> class_names_;     ///< 可识别的数字类别名称
    std::vector<std::string> ignore_classes_;  ///< 需要忽略的数字类别
};
}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__NUMBER_CLASSIFIER_HPP_
