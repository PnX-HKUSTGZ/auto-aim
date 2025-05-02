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
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "armor_tracker/types.hpp"
#include "armor_tracker/extended_kalman_filter.hpp"
#include "armor_tracker/tracker.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"

namespace rm_auto_aim
{

class TrackerManager {
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

    
    // 权重参数
    double w_distance_;    // 距离图像中心的权重_3D
    double w_twoD_distance_; // 距离图像中心的权重_2D
    // EKF 模板，用于初始化新的 Tracker
    ExtendedKalmanFilter ekf_template_;

    VisionMode mode_ = VisionMode::AUTO; 
public:
    void setWeights(
        double w_distance,
        double w_twoD_distance_
    );

    bool setMode(VisionMode mode) {
        try{
            mode_ = mode;
            return true;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Failed to set mode: %s", e.what());
            return false;
        }
    }
    // 添加一个方法来设置 EKF 模板
    void setEKFTemplate(const ExtendedKalmanFilter& ekf_template) {
        ekf_template_ = ekf_template;
    }
    TrackerManager(
        double max_match_distance,
        double max_match_yaw_diff,
        int tracking_thres,
        double lost_time_thres,
        double switch_cooldown = 1.0);
    
    // 更新所有追踪器
    void update(const auto_aim_interfaces::msg::Armors::SharedPtr& armors_msg);
    //
    std::shared_ptr<Tracker> getTracker(const std::string& id) const {
        if (trackers_.find(id) != trackers_.end()) {
            return trackers_.at(id);
        }
        return nullptr;
    }
    // 获取当前目标
    auto_aim_interfaces::msg::Target getCurrentTarget() const;
    auto_aim_interfaces::msg::Target getIDTarget(std::string input_tracked_id_) const ;
    std::vector<std::string> getActiveTrackerIDs() const;
    
    // 清理不活跃的追踪器
    void cleanInactiveTrackers(rclcpp::Time now);
    
    // 重置所有追踪器
    void reset();
      
private:
    rclcpp::Clock clock_;
    // 评分函数
    double calculateScore(const std::string& id, const std::shared_ptr<Tracker>& tracker);
    
    // 选择最佳目标
    void selectBestTarget();
    
    // 初始化新追踪器
    void initNewTracker(const std::string& id
                      , const std::vector<auto_aim_interfaces::msg::Armor>& armors
                      , rclcpp::Time msg_time);
  };

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__TRACKER_MANAGER_HPP_