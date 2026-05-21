// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_TRACKER__TYPES_HPP_
#define ARMOR_TRACKER__TYPES_HPP_

#include <opencv2/core.hpp>

// STL
#include <algorithm>
#include <string>

namespace rm_auto_aim
{
/**
 * @brief 机器人状态向量枚举
 * 
 * 定义EKF状态向量中各个状态分量的索引位置。
 * 状态向量包含机器人中心位置、速度、两个装甲板的Z坐标、偏航角速度、
 * 两个装甲板的半径和偏航角等状态变量。
 * 前哨站修改：当前为16维向量
 */
enum CarState {
    XC = 0,    /**< 机器人中心X坐标 */
    VXC,       /**< 机器人中心X方向速度 */
    YC,        /**< 机器人中心Y坐标 */
    VYC,       /**< 机器人中心Y方向速度 */
    ZC1,       /**< 装甲板1的Z坐标 */
    ZC2,       /**< 装甲板2的Z坐标 */
    ZC3,       /**< 新增：前哨站装甲板3的Z坐标 */
    VZC,       /**< Z方向速度 */
    VYAW,      /**< 偏航角速度 */
    R1,        /**< 装甲板1到中心半径 */
    R2,        /**< 装甲板2到中心半径 */
    R3,        /**< 新增：前哨站装甲板3到中心半径 */
    YAW1,      /**< 装甲板1的偏航角 */
    YAW2,      /**< 装甲板2的偏航角 */
    YAW3,      /**< 新增：前哨站装甲板3的偏航角 */
    R_OUTPOST  /**< 新增：前哨站基准半径（固定值） */
};

/**
 * @brief 视觉模式枚举
 * 
 * 定义不同的视觉识别模式，用于指定当前需要优先追踪的目标类型。
 * 每种模式对应不同的机器人类型或建筑物，影响目标选择的评分权重。
 */
enum VisionMode {
    OUTPOST = 0,    /**< 前哨站模式 */
    HERO = 1,       /**< 英雄机器人模式 */
    ENGINEER = 2,   /**< 工程机器人模式 */
    INFANTRY_1 = 3, /**< 步兵机器人1模式 */
    INFANTRY_2 = 4, /**< 步兵机器人2模式 */
    INFANTRY_3 = 5, /**< 步兵机器人3模式 */
    GUARD = 6,      /**< 哨兵机器人模式 */
    BASE = 7,       /**< 基地模式 */
    RUNE = 8,       /**< 神符模式 */
    AUTO = 9        /**< 自动模式，自动选择最佳目标 */
};

/**
 * @brief 将视觉模式枚举转换为字符串
 * 
 * 用于日志输出和调试，将枚举值转换为可读的字符串表示。
 * 
 * @param mode 视觉模式枚举值
 * @return 对应的字符串表示，未知模式返回"UNKNOWN"
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

/**
 * @brief 机器人和建筑物ID列表
 * 
 * 定义了系统中可识别的所有目标类型ID，包括不同编号的机器人
 * 和各种建筑物目标。用于初始化和管理追踪器。
 */
inline std::vector<std::string> ID_LIST{
    "1",       /**< 1号机器人（通常为英雄） */
    "2",       /**< 2号机器人（通常为工程） */
    "3",       /**< 3号机器人（通常为步兵1） */
    "4",       /**< 4号机器人（通常为步兵2） */
    "5",       /**< 5号机器人（通常为步兵3） */
    "outpost", /**< 前哨站建筑 */
    "guard",   /**< 哨兵机器人 */
    "base"     /**< 基地建筑 */
};

/**
 * @brief 装甲板数量枚举，定义机器人装甲板的数量类型
 */
enum class ArmorsNum {
    NORMAL_4 = 4,  /**< 普通机器人，4块装甲板 */
    BALANCE_2 = 2, /**< 平衡步兵，2块装甲板 */
    OUTPOST_3 = 3  /**< 前哨站，3块装甲板 */
};
}  // namespace rm_auto_aim

#endif  // ARMOR_TRACKER__TYPES_HPP_
