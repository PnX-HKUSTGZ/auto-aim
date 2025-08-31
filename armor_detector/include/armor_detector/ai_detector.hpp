// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__AI_DETECTOR_HPP_
#define ARMOR_DETECTOR__AI_DETECTOR_HPP_

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

// OpenVINO
#include <openvino/openvino.hpp>

// STD
#include <cmath>
#include <string>
#include <vector>
#include <memory>

#include "armor_detector/types.hpp"
#include "auto_aim_interfaces/msg/debug_armors.hpp"

namespace rm_auto_aim
{

// 图像尺寸常量
static constexpr int IMAGE_WIDTH = 640;
static constexpr int IMAGE_HEIGHT = 640;

/**
 * @brief AI检测对象结构体
 */
struct Object 
{
    cv::Rect rect;              ///< 检测框
    int label;                  ///< 类别标签 (0-8 对应数字)
    int color;                  ///< 颜色 (0: red, 1: blue)
    float prob;                 ///< 置信度
    float landmarks[8];         ///< 装甲板四个角点坐标 (x1,y1,x2,y2,x3,y3,x4,y4)
    float length;               ///< 装甲板长度
    float width;                ///< 装甲板宽度
    float ratio;                ///< 长宽比
};

/**
 * @brief 基于 OpenVINO 的装甲板 AI 检测器
 * 
 * 使用深度学习模型进行装甲板检测和数字识别
 */
class AIDetector
{
public:
    /**
     * @brief 构造 AI 检测器
     * 
     * @param model_path ONNX 模型文件路径
     * @param device 推理设备 ("CPU", "GPU", etc.)
     * @param conf_threshold 置信度阈值
     * @param nms_threshold NMS 阈值
     */
    AIDetector(
        const std::string & model_path, 
        const std::string & device = "CPU",
        float conf_threshold = 0.65f,
        float nms_threshold = 0.45f);

    /**
     * @brief 析构函数
     */
    ~AIDetector();

    /**
     * @brief 检测装甲板
     * 
     * @param input 输入图像
     * @param detect_color 检测颜色 (0: red, 1: blue)
     * @return std::vector<Armor> 检测到的装甲板数组
     */
    std::vector<Armor> detect(const cv::Mat & input, int detect_color);

    /**
     * @brief 在图像上绘制检测结果
     * 
     * @param img 要绘制结果的图像
     */
    void drawResults(cv::Mat & img);

    // 调试信息
    auto_aim_interfaces::msg::DebugArmors debug_armors;  ///< 装甲板调试信息

private:
    /**
     * @brief 执行模型推理
     * 
     * @param img 输入图像
     * @param detect_color 检测颜色
     */
    void infer(const cv::Mat & img, int detect_color);

    /**
     * @brief Sigmoid 激活函数
     * 
     * @param x 输入值
     * @return float Sigmoid 输出
     */
    inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    /**
     * @brief 将检测对象转换为装甲板
     * 
     * @param obj 检测对象
     * @return Armor 装甲板
     */
    Armor objectToArmor(const Object & obj);

    // OpenVINO 相关
    ov::Core core;                                          ///< OpenVINO 核心
    std::shared_ptr<ov::Model> model;                      ///< 模型
    ov::CompiledModel compiled_model;                       ///< 编译后的模型
    std::unique_ptr<ov::preprocess::PrePostProcessor> ppp; ///< 预处理器
    
    // 参数
    float conf_threshold_;  ///< 置信度阈值
    float nms_threshold_;   ///< NMS 阈值
    std::vector<size_t> input_shape;  ///< 输入形状
    
    // 原始图像尺寸 (用于坐标缩放)
    int original_width_;    ///< 原始图像宽度
    int original_height_;   ///< 原始图像高度

    // 检测结果
    std::vector<Object> objects_;      ///< 原始检测对象
    std::vector<Object> tmp_objects_;  ///< NMS 后的检测对象
    std::vector<float> ious_;          ///< IoU 数组
    std::vector<Armor> armors_;         ///< 检测到的装甲板
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__AI_DETECTOR_HPP_