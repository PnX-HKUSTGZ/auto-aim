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
     * 根据新的装甲板观测数据更新追踪器状态，包括EKF预测和更新、
     * 装甲板匹配、状态机转换等。
     * 
     * @param armors_msg 包含最新装甲板观测数据的消息
     */
    void update(const Armors::SharedPtr & armors_msg);

    ExtendedKalmanFilter ekf;

    int tracking_thres;
    int lost_thres;

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

    rclcpp::Time last_update_time_;  // 上次更新时间

    // 前哨站(3板)：仅缓存每块装甲板最近一次观测到的z（不参与EKF更新）
    bool getOutpostArmorZ(size_t armor_index, const rclcpp::Time & now, double & z) const;

    void setOutpostZCacheWindowSec(double window_sec) { outpost_z_cache_window_sec_ = window_sec; }
    void setOutpostZMergeTolerance(double merge_tol) { outpost_z_merge_tol_ = merge_tol; }

    // 前哨站(3板)编号推断参数：用VYAW方向 + 相对中板的高度差来推断新观测装甲板编号
    void setOutpostHeightTolerance(double tol_m) { outpost_height_tol_ = tol_m; }
    void setOutpostVyawDeadband(double deadband) { outpost_vyaw_deadband_ = deadband; }
    void setOutpostVyawPositiveIsCCW(bool positive_is_ccw)
    {
        outpost_vyaw_positive_is_ccw_ = positive_is_ccw;
    }
    void setOutpostHeightMismatchPenalty(double penalty)
    {
        outpost_height_mismatch_penalty_ = penalty;
    }

private:
    struct OutpostZCache
    {
        bool valid = false;
        double z = 0.0;
        rclcpp::Time stamp;
    };

    std::array<OutpostZCache, 3> outpost_z_cache_;
    double outpost_z_cache_window_sec_ = 2;  // z缓存时间窗(秒)
    double outpost_z_merge_tol_ = 0.05;         // z合并阈值(米)：<=该阈值视为同一块板

    // 前哨站z缓存状态日志节流（每0.2s输出一次）
    rclcpp::Time last_outpost_cache_log_time_{0, 0, RCL_ROS_TIME};

    // outpost高度推断：以“单板默认中板(1号)”为锚点，利用高度差区分 2/3 号
    // - outpost_height_tol_: |z - z_middle| <= tol 认为仍是中板
    // - outpost_vyaw_deadband_: |vyaw| 小于该值认为旋转方向不可靠
    // - outpost_vyaw_positive_is_ccw_: true表示VYAW>0对应逆时针
    // - outpost_height_mismatch_penalty_: 在matchArmor评分中对不符合高度推断的编号加惩罚
    double outpost_height_tol_ = 0.04;
    double outpost_vyaw_deadband_ = 0.2;
    bool outpost_vyaw_positive_is_ccw_ = true;
    double outpost_height_mismatch_penalty_ = 10.0;

    void mergeOutpostZCache(const rclcpp::Time & now);

    int inferOutpostArmorIdByHeight(double meas_z, const Eigen::VectorXd & ekf_prediction) const;

    rclcpp::Time current_msg_time_{0, 0, RCL_ROS_TIME};

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

    int detect_count_;
    int lost_count_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__TRACKER_HPP_
