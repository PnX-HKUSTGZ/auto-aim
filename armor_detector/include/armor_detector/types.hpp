// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__ARMOR_HPP_
#define ARMOR_DETECTOR__ARMOR_HPP_

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <rclcpp/duration.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// STL
#include <algorithm>
#include <string>

namespace rm_auto_aim
{

// ============================================================================
// 常量定义 Constants
// ============================================================================

// 颜色常量
const int RED = 0;
const int BLUE = 1;

// 装甲板尺寸常量 (单位: 米)
static constexpr float SMALL_ARMOR_WIDTH = 138.0 / 1000.0;
static constexpr float SMALL_ARMOR_HEIGHT = 48.0 / 1000.0;
static constexpr float LARGE_ARMOR_WIDTH = 228.0 / 1000.0;
static constexpr float LARGE_ARMOR_HEIGHT = 48.0 / 1000.0;

// 图像尺寸常量
static constexpr int IMAGE_WIDTH = 640;
static constexpr int IMAGE_HEIGHT = 640;

// ============================================================================
// 枚举类型 Enumerations
// ============================================================================

/**
 * @brief 装甲板类型枚举
 */
enum class ArmorType { SMALL, LARGE, INVALID };
const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};

/**
 * @brief 视觉模式枚举
 * 定义不同机器人角色对应的视觉识别模式
 */
enum VisionMode {
    OUTPOST = 0,    ///< 前哨站
    HERO = 1,       ///< 英雄机器人
    ENGINEER = 2,   ///< 工程机器人
    INFANTRY_1 = 3, ///< 步兵机器人1
    INFANTRY_2 = 4, ///< 步兵机器人2
    INFANTRY_3 = 5, ///< 步兵机器人3
    GUARD = 6,      ///< 哨兵机器人
    BASE = 7,       ///< 基地
    RUNE = 8,       ///< 能量机关
    AUTO = 9        ///< 自动模式
};
/**
 * @brief 视觉模式转换为字符串的工具函数
 * @param mode 视觉模式枚举值
 * @return 对应的字符串表示
 */
inline std::string visionModeToString(VisionMode mode)
{
    switch (mode) {
        case VisionMode::OUTPOST:
            return "OUTPOST";
        case VisionMode::HERO:
            return "HERO";
        case VisionMode::ENGINEER:
            return "ENGINEER";
        case VisionMode::INFANTRY_1:
            return "INFANTRY_1";
        case VisionMode::INFANTRY_2:
            return "INFANTRY_2";
        case VisionMode::INFANTRY_3:
            return "INFANTRY_3";
        case VisionMode::GUARD:
            return "GUARD";
        case VisionMode::BASE:
            return "BASE";
        case VisionMode::RUNE:
            return "RUNE";
        case VisionMode::AUTO:
            return "AUTO";
        default:
            return "UNKNOWN";
    }
}

// ============================================================================
// 参数结构体 Parameter Structures
// ============================================================================

/**
 * @brief 灯条检测参数结构体
 * 定义了灯条选择的限制参数
 */
struct LightParams
{
    double min_ratio;  ///< 灯条宽高比的最小值
    double max_ratio;  ///< 灯条宽高比的最大值
    double max_angle;  ///< 灯条允许的最大垂直倾斜角度
};

/**
 * @brief 装甲板检测参数结构体
 * 定义了装甲板选择的限制参数
 */
struct ArmorParams
{
    double min_light_ratio;            ///< 两灯条的最小比例值
    // 灯条对中心距离参数
    double min_small_center_distance;  ///< 小装甲板灯条中心的最小距离
    double max_small_center_distance;  ///< 小装甲板灯条中心的最大距离
    double min_large_center_distance;  ///< 大装甲板灯条中心的最小距离
    double max_large_center_distance;  ///< 大装甲板灯条中心的最大距离
    double max_angle;                  ///< 装甲板允许的最大水平倾斜角度
};

// ============================================================================
// AI检测相关结构体 AI Detection Structures
// ============================================================================

/**
 * @brief AI检测目标对象结构体
 * 存储AI模型检测到的装甲板信息
 */
struct Object
{
    cv::Rect rect;       ///< 检测框
    int label;           ///< 类别标签 (0-8 对应数字)
    int color;           ///< 颜色 (0: red, 1: blue)
    float prob;          ///< 置信度
    float landmarks[8];  ///< 装甲板四个角点坐标 (x1,y1,x2,y2,x3,y3,x4,y4)
    float length;        ///< 装甲板长度
    float width;         ///< 装甲板宽度
    float ratio;         ///< 长宽比
};

// ============================================================================
// 视觉检测核心结构体 Vision Detection Core Structures
// ============================================================================

/**
 * @brief 灯条结构体
 * 继承自cv::RotatedRect，包含灯条的几何信息和颜色信息
 */
struct Light : public cv::RotatedRect
{
    Light() = default;
    
    /**
     * @brief 从旋转矩形构造灯条
     * @param box 旋转矩形
     */
    explicit Light(cv::RotatedRect box) : cv::RotatedRect(box)
    {
        cv::Point2f p[4];
        box.points(p);
        std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        tilt_angle = std::atan2(bottom.x - top.x, bottom.y - top.y);
        tilt_angle = tilt_angle / CV_PI * 180;
    }

    /**
     * @brief 从颜色和端点构造灯条
     * @param color 灯条颜色
     * @param top 顶部端点
     * @param bottom 底部端点
     */
    explicit Light(int color, cv::Point2f top, cv::Point2f bottom)
    : cv::RotatedRect(
          (top + bottom) / 2, cv::Size2f(cv::norm(top - bottom), cv::norm(top - bottom) / 4),
          -std::atan2(bottom.x - top.x, bottom.y - top.y) * 180 / CV_PI),
      color(color),
      top(top),
      bottom(bottom)
    {
        length = cv::norm(top - bottom);
        width = length / 4;
        tilt_angle = std::atan2(bottom.x - top.x, bottom.y - top.y);
        tilt_angle = tilt_angle / CV_PI * 180;
    }

    int color;                    ///< 灯条颜色
    cv::Point2f top, bottom, axis; ///< 顶部、底部和轴心点
    double length;                ///< 灯条长度
    double width;                 ///< 灯条宽度
    float tilt_angle;            ///< 倾斜角度
};

/**
 * @brief 装甲板结构体
 * 包含装甲板的所有相关信息，包括灯条对、位置、类型、数字识别结果等
 */
struct Armor
{
    Armor() = default;
    
    /**
     * @brief 从两个灯条构造装甲板
     * @param l1 第一个灯条
     * @param l2 第二个灯条
     */
    Armor(const Light & l1, const Light & l2)
    {
        if (l1.center.x < l2.center.x) {
            left_light = l1, right_light = l2;
        } else {
            left_light = l2, right_light = l1;
        }
        center = (left_light.center + right_light.center) / 2;
    }

    /**
     * @brief 构建3D物体点的模板函数
     * @param w 宽度
     * @param h 高度
     * @return 3D点的向量
     */
    template <typename PointType>
    static inline std::vector<PointType> buildObjectPoints(const double & w, const double & h)
    {
        return {
            PointType(0, w / 2, -h / 2), PointType(0, w / 2, h / 2), PointType(0, -w / 2, h / 2),
            PointType(0, -w / 2, -h / 2)};
    }
    
    /**
     * @brief 获取装甲板角点坐标
     * @return 装甲板四个角点的2D坐标
     */
    std::vector<cv::Point2f> landmarks() const
    {
        return {left_light.bottom, left_light.top, right_light.top, right_light.bottom};
    }

    /**
     * @brief 设置相机坐标系下的装甲板位姿
     * @param r_odom_to_camera 里程计到相机的旋转矩阵
     * @param t_odom_to_camera 里程计到相机的平移向量
     */
    void setCameraArmor(
        const Eigen::Matrix3d & r_odom_to_camera, const Eigen::Vector3d & t_odom_to_camera)
    {
        t_camera_armor = r_odom_to_camera * t_odom_armor + t_odom_to_camera;
        r_camera_armor = r_odom_to_camera * r_odom_armor;
    }

    // 灯条对相关
    Light left_light, right_light;  ///< 左右灯条
    cv::Point2f center;             ///< 装甲板中心点
    bool sign = false;              ///< 灯条和y轴夹角，0指向右下，1指向左下
    ArmorType type;                 ///< 装甲板类型

    // 数字识别相关
    cv::Mat number_img;                    ///< 数字区域图像
    std::string number;                    ///< 识别出的数字
    float confidence;                      ///< 识别置信度
    std::string classfication_result;      ///< 分类结果

    // 位姿信息
    Eigen::Matrix3d r_odom_armor;    ///< 里程计坐标系下装甲板旋转矩阵
    Eigen::Vector3d t_odom_armor;    ///< 里程计坐标系下装甲板位置向量
    Eigen::Matrix3d r_camera_armor;  ///< 相机坐标系下装甲板旋转矩阵
    Eigen::Vector3d t_camera_armor;  ///< 相机坐标系下装甲板位置向量
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_HPP_