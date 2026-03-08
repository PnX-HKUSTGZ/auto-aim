// Copyright 2022 Chen Jun

#ifndef ARMOR_PROCESSOR__TRACKER_MANAGER_HPP_
#define ARMOR_PROCESSOR__TRACKER_MANAGER_HPP_

// Eigen
#include <Eigen/Eigen>

// ROS
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/rclcpp.hpp>

// STD
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "armor_tracker/extended_kalman_filter.hpp"
#include "armor_tracker/tracker.hpp"
#include "armor_tracker/types.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"

namespace rm_auto_aim
{

/**
 * @brief 追踪器管理类
 * 
 * 该类负责管理多个装甲板追踪器，包括创建、更新、选择最佳目标和清理非活跃追踪器。
 * 支持多目标同时追踪，并提供基于评分的目标选择机制。
 */
class TrackerManager
{
private:
    // 按ID管理多个追踪器
    std::map<std::string, std::shared_ptr<Tracker>> trackers_;

    // 当前选中的追踪目标
    std::string current_tracked_id_;

    // 评分历史和目标切换冷却
    rclcpp::Time last_switch_time_;
    double switch_cooldown_;

    // 计算评分的参数
    double max_match_distance_;
    double max_match_yaw_diff_;
    int tracking_thres_;
    double lost_time_thres_;

    // 前哨站(3板)z缓存参数（不参与EKF）
    double outpost_z_cache_window_sec_ = 7;
    double outpost_z_merge_tol_ = 0.01;

    // 前哨站(3板)编号推断参数：高度差 + VYAW方向
    double outpost_height_tol_ = 0.04;
    double outpost_vyaw_deadband_ = 0.2;
    bool outpost_vyaw_positive_is_ccw_ = true;
    double outpost_height_mismatch_penalty_ = 20.0;

    // 权重参数
    double w_distance_;       // 距离图像中心的权重_3D
    double w_twoD_distance_;  // 距离图像中心的权重_2D
    // EKF 模板，用于初始化新的 Tracker
    ExtendedKalmanFilter ekf_template_;

    VisionMode mode_ = VisionMode::AUTO;

public:
    void setOutpostZCacheParams(double window_sec, double merge_tol)
    {
        outpost_z_cache_window_sec_ = window_sec;
        outpost_z_merge_tol_ = merge_tol;
        for (auto & kv : trackers_) {
            if (!kv.second) {
                continue;
            }
            kv.second->setOutpostZCacheWindowSec(outpost_z_cache_window_sec_);
            kv.second->setOutpostZMergeTolerance(outpost_z_merge_tol_);
        }
    }

    void setOutpostHeightInferParams(
        double height_tol_m, double vyaw_deadband, bool vyaw_positive_is_ccw,
        double height_mismatch_penalty)
    {
        outpost_height_tol_ = height_tol_m;
        outpost_vyaw_deadband_ = vyaw_deadband;
        outpost_vyaw_positive_is_ccw_ = vyaw_positive_is_ccw;
        outpost_height_mismatch_penalty_ = height_mismatch_penalty;
        for (auto & kv : trackers_) {
            if (!kv.second) {
                continue;
            }
            kv.second->setOutpostHeightTolerance(outpost_height_tol_);
            kv.second->setOutpostVyawDeadband(outpost_vyaw_deadband_);
            kv.second->setOutpostVyawPositiveIsCCW(outpost_vyaw_positive_is_ccw_);
            kv.second->setOutpostHeightMismatchPenalty(outpost_height_mismatch_penalty_);
        }
    }
    /**
     * @brief 设置评分权重参数
     * 
     * @param w_distance 3D距离权重，用于距离评分计算
     * @param w_twoD_distance 2D图像距离权重，用于图像中心距离评分
     */
    void setWeights(double w_distance, double w_twoD_distance_);

    /**
     * @brief 设置视觉模式
     * 
     * 根据不同的视觉模式调整目标选择策略，为特定目标类型提供额外的评分加成。
     * 
     * @param mode 视觉模式枚举值
     * @return 设置是否成功
     */
    bool setMode(VisionMode mode)
    {
        try {
            mode_ = mode;
            return true;
        } catch (const std::exception & e) {
            RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Failed to set mode: %s", e.what());
            return false;
        }
    }

    /**
     * @brief 设置EKF模板
     * 
     * 为所有新创建的追踪器提供EKF滤波器模板
     * 
     * @param ekf_template EKF滤波器模板
     */
    void setEKFTemplate(const ExtendedKalmanFilter & ekf_template) { ekf_template_ = ekf_template; }

    /**
     * @brief 更新EKF模板的时间间隔
     * 
     * 根据当前帧率更新EKF模板中的时间间隔参数，保证状态预测的准确性。
     * 
     * @param dt 时间间隔（秒）
     */
    void updateEKFTemplate(double dt);

    /**
     * @brief 构造追踪器管理器
     * 
     * @param max_match_distance 装甲板匹配的最大位置距离阈值
     * @param max_match_yaw_diff 装甲板匹配的最大偏航角差阈值  
     * @param tracking_thres 追踪状态转换阈值
     * @param lost_time_thres 目标丢失时间阈值
     * @param switch_cooldown 目标切换冷却时间，默认1.0秒
     */
    TrackerManager(
        double max_match_distance, double max_match_yaw_diff, int tracking_thres,
        double lost_time_thres, double switch_cooldown = 1.0);

    /**
     * @brief 更新所有追踪器
     * 
     * 根据新的装甲板观测数据更新所有活跃的追踪器状态，
     * 包括创建新追踪器和更新现有追踪器。
     * 
     * @param armors_msg 包含所有观测到装甲板的消息
     * @param dt 时间间隔
     */
    void update(const auto_aim_interfaces::msg::Armors::SharedPtr & armors_msg, double dt);

    /**
     * @brief 选择最佳追踪目标
     * 
     * 基于多种评分因子（距离、追踪状态、模式偏好等）选择当前最佳的追踪目标，
     * 具有切换冷却机制以避免频繁切换。
     */
    void selectBestTarget();

    /**
     * @brief 根据ID获取指定的追踪器
     * 
     * @param id 追踪器ID
     * @return 追踪器指针，如果不存在则返回nullptr
     */
    std::shared_ptr<Tracker> getTracker(const std::string & id) const
    {
        if (trackers_.find(id) != trackers_.end()) {
            return trackers_.at(id);
        }
        return nullptr;
    }

    /**
     * @brief 获取当前选中的目标ID
     * 
     * @return 当前追踪目标的ID字符串
     */
    std::string getCurrentTargetID() const;

    /**
     * @brief 根据ID获取目标信息
     * 
     * 将追踪器的状态信息转换为ROS消息格式，用于发布给其他节点。
     * 
     * @param input_tracked_id_ 要获取的目标ID
     * @return 目标信息消息
     */
    bool getIDTarget(std::string input_tracked_id_, auto_aim_interfaces::msg::Target & target_msg) const;

    /**
     * @brief 获取所有活跃追踪器的ID列表
     * 
     * @return 活跃追踪器ID的字符串向量
     */
    std::vector<std::string> getActiveTrackerIDs() const;

    /**
     * @brief 清理非活跃的追踪器
     * 
     * 移除超过丢失时间阈值或状态为LOST的追踪器，释放内存资源。
     * 
     * @param now 当前时间
     */
    void cleanInactiveTrackers(rclcpp::Time now);

    /**
     * @brief 重置所有追踪器
     * 
     * 清除所有追踪器和目标选择状态，通常在系统重启或模式切换时使用。
     */
    void reset();

private:
    rclcpp::Clock clock_;

    /**
     * @brief 计算追踪器的综合评分
     * 
     * 基于追踪状态、距离、2D图像距离和模式偏好等因素计算综合评分，
     * 用于选择最佳追踪目标。
     * 
     * @param id 追踪器ID
     * @param tracker 追踪器指针
     * @return 计算得到的评分
     */
    double calculateScore(const std::string & id, const std::shared_ptr<Tracker> & tracker);

    /**
     * @brief 初始化新的追踪器
     * 
     * 为指定ID创建新的追踪器实例，使用EKF模板进行初始化，
     * 并设置相关参数。
     * 
     * @param id 追踪器ID
     * @param armors 用于初始化的装甲板数据
     * @param msg_time 消息时间戳
     */
    void initNewTracker(
        const std::string & id, const std::vector<auto_aim_interfaces::msg::Armor> & armors,
        rclcpp::Time msg_time);
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__TRACKER_MANAGER_HPP_