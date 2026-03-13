// Copyright 2022 Chen Jun

#ifndef ARMOR_PROCESSOR__TRACKER_HPP_
#define ARMOR_PROCESSOR__TRACKER_HPP_

// Eigen
#include <Eigen/Eigen>

// ROS
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/rclcpp.hpp>

// STD
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "armor_tracker/extended_kalman_filter.hpp"
#include "armor_tracker/types.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"

namespace rm_auto_aim
{

/**
 * @brief 单个装甲板目标追踪器类
 * 
 * 该类负责追踪单个装甲板目标，使用扩展卡尔曼滤波器进行状态估计和预测。
 * 支持不同数量装甲板的机器人追踪，包括状态管理和匹配算法。
 */
class Tracker
{
public:
    /**
     * @brief 构造追踪器对象
     * 
     * @param max_match_distance 装甲板匹配的最大位置距离阈值
     * @param max_match_yaw_diff 装甲板匹配的最大偏航角差阈值
     */
    Tracker(double max_match_distance, double max_match_yaw_diff);

    using Armors = auto_aim_interfaces::msg::Armors;
    using Armor = auto_aim_interfaces::msg::Armor;

    /**
     * @brief 初始化追踪器
     * 
     * 根据观测到的装甲板信息初始化追踪器状态和EKF滤波器。
     * 支持单装甲板和双装甲板的初始化。
     * 
     * @param armors_msg 包含装甲板信息的消息
     */
    void init(const Armors::SharedPtr & armors_msg);

    /**
     * @brief 更新追踪器状态
     *
     * 执行EKF预测和更新，并返回当前帧是否成功匹配到装甲板。
     * 匹配成功时会刷新last_update_time_，否则仅输出预测结果。
     *
     * @param armors_msg 包含最新装甲板观测数据的消息
     * @return true 当前帧找到匹配装甲板
     * @return false 当前帧未找到匹配装甲板
     */
    bool update(const Armors::SharedPtr & armors_msg, bool is_main_camera);

    /**
     * @brief 获取/维护检测计数
     *
     * 由TrackerManager外部状态机调用，避免重复在Tracker内部维护。
     */
    int getDetectCount() const { return detect_count_; }
    void resetDetectCount() { detect_count_ = 0; }
    void increaseDetectCount() { ++detect_count_; }

    /**
     * @brief 基于时间阈值更新状态机
     *
     * 使用msg_time与last_update_time_的时间差，对应匹配结果切换
     * DETECTING/TRACKING/TEMP_LOST/LOST状态。
     *
     * @param matched 当前帧是否匹配到装甲板
     * @param msg_time 当前消息时间戳
     * @param temp_lost_time 进入TEMP_LOST的时间阈值
     * @param lost_time_thres 进入LOST的时间阈值
     * @param tracking_thres 进入TRACKING所需的连续检测帧数
     */
    void updateState(
        bool matched, const rclcpp::Time & msg_time, double temp_lost_time,
        double lost_time_thres, int tracking_thres, bool is_main_camera);

    ExtendedKalmanFilter ekf;

    int tracking_thres;

    enum State {
        LOST,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } tracker_state;

    std::string tracked_id;
    Armor tracked_armor;
    Armor tracked_armor_2;
    ArmorsNum tracked_armors_num = ArmorsNum::NORMAL_4;

    double info_position_diff;
    double info_yaw_diff;
    double twoD_distance;

    Eigen::VectorXd measurement;

    Eigen::VectorXd target_state;

    rclcpp::Time last_update_time_;  // 上次有匹配的更新时间，也作为EKF时间基准
    rclcpp::Time last_main_update_time_; // 上次主相机更新时间

private:
    /**
     * @brief 使用单个装甲板初始化EKF
     * 
     * 根据单个装甲板的位置和方向信息初始化扩展卡尔曼滤波器的状态向量，
     * 包括机器人中心位置、速度、角度等。
     * 
     * @param a 用于初始化的装甲板信息
     */
    void initEKF(const Armor & a);

    /**
     * @brief 使用两个装甲板初始化EKF
     * 
     * 根据两个装甲板的位置和方向信息计算机器人中心位置和方向，
     * 初始化扩展卡尔曼滤波器的状态向量。
     * 
     * @param a 第一个装甲板信息
     * @param b 第二个装甲板信息
     */
    void initEKFTwo(const Armor & a, const Armor & b);

    /**
     * @brief 更新追踪目标的装甲板数量信息
     * 
     * 根据追踪的目标ID确定该目标的装甲板数量类型（2、3或4块）。
     */
    void updateArmorsNum();

    /**
     * @brief 将四元数转换为偏航角，考虑目标偏航角
     * 
     * @param q 四元数姿态
     * @param position 装甲板位置（未使用）
     * @param yaw_target 目标偏航角，用于选择合适的角度范围
     * @return 转换后的偏航角（弧度）
     */
    double orientationToYaw(
        const geometry_msgs::msg::Quaternion & q, geometry_msgs::msg::Point & position,
        const double & yaw_target);

    /**
     * @brief 将四元数转换为偏航角（重载版本）
     * 
     * @param q 四元数姿态
     * @return 转换后的偏航角（弧度）
     */
    double orientationToYaw(const geometry_msgs::msg::Quaternion & q);

    /**
     * @brief 计算两个偏航角之间的最小角度差
     * 
     * 考虑角度的周期性，计算两个偏航角之间的最小角度差异。
     * 对于多装甲板目标，还会考虑装甲板之间的角度关系。
     * 
     * @param yaw1 第一个偏航角（弧度）
     * @param yaw2 第二个偏航角（弧度）
     * @return 最小角度差（弧度）
     */
    double calYawDiff(double yaw1, double yaw2);

    /**
     * @brief 匹配观测装甲板与预测装甲板
     * 
     * 根据位置距离和偏航角差异，判断观测到的装甲板与EKF预测的装甲板是否匹配。
     * 
     * @param armor 观测到的装甲板
     * @param ekf_prediction EKF预测的状态向量
     * @return 匹配结果：0-不匹配，1-匹配装甲板1，2-匹配装甲板2
     */
    int matchArmor(const Armor & armor, const Eigen::VectorXd & ekf_prediction);

    /**
     * @brief 对 outpost 单装甲观测场景下的未观测装甲板高度进行硬约束
     *
     * 第一个参数为当前观测到的装甲板高度，后两个参数为未观测装甲板高度。
     * 该函数用于后续在 outpost 且 armors.size()==1 的更新流程中施加高度差约束。
     *
     * @param observed_armor_z 当前观测到的装甲板高度
     * @param unobserved_armor_z_1 未观测装甲板1的高度（可被函数内修正）
     * @param unobserved_armor_z_2 未观测装甲板2的高度（可被函数内修正）
     */
    void constrainOutpostHeights(
        double observed_armor_z, double & unobserved_armor_z_1, double & unobserved_armor_z_2,
        double height_diff = 0.102, double height_diff_threshold = 0.02);

    /**
     * @brief 从状态向量计算装甲板位置
     * 
     * 根据EKF的状态向量（包含机器人中心位置、半径、偏航角等），
     * 计算各个装甲板的3D位置。
     * 
     * @param x EKF状态向量
     * @return 装甲板位置向量列表
     */
    std::vector<Eigen::Vector3d> getArmorPositionFromState(const Eigen::VectorXd & x);

    double max_match_distance_;
    double max_match_yaw_diff_;

    // // 前哨站初始化轮转板序号：1 -> 2 -> 3 -> 1
    // int r_ = 1;
    // // z高度缓存槽位：0号槽对应1号板，1号槽对应2号板，2号槽对应3号板
    // std::array<double, 3> z_slots_{{0.0, 0.0, 0.0}};

    int detect_count_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__TRACKER_HPP_
