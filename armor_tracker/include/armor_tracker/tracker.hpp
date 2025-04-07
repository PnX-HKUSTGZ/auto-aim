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
#include <memory>
#include <string>
#include <vector>

#include "armor_tracker/extended_kalman_filter.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"

namespace rm_auto_aim
{

enum class ArmorsNum { NORMAL_4 = 4, BALANCE_2 = 2, OUTPOST_3 = 3 };

class Tracker
{
public:
  Tracker(double max_match_distance, double max_match_yaw_diff);

  using Armors = auto_aim_interfaces::msg::Armors;
  using Armor = auto_aim_interfaces::msg::Armor;

  void init(const Armors::SharedPtr & armors_msg);

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
  ArmorsNum tracked_armors_num;

  double info_position_diff;
  double info_yaw_diff;

  Eigen::VectorXd measurement;

  Eigen::VectorXd target_state;

  // 新增成员用于评分
  rclcpp::Time last_update_time_;      // 上次更新时间
  double tracking_duration_;           // 追踪持续时间
  std::vector<double> velocity_history_;  // 速度历史记录
  double avg_innovation_;              // 平均创新值(预测误差)
  
  // 获取状态和指标
  double getVelocityStability() const;
  enum CarState
  {
    XC = 0,
    VXC,
    YC,
    VYC,
    ZC1,
    ZC2, 
    VZC,
    VYAW,
    R1,
    R2, 
    YAW1,
    YAW2 // 11
  };
  

private:
  void initEKF(const Armor & a); 
  void initEKFTwo(const Armor & a, const Armor & b);

  void updateArmorsNum(const Armor & a);

  double orientationToYaw(const geometry_msgs::msg::Quaternion & q, geometry_msgs::msg::Point & position, const double & yaw_target);
  double orientationToYaw(const geometry_msgs::msg::Quaternion & q); //overload
  double calYawDiff(double yaw1, double yaw2); 

  std::vector<Eigen::Vector3d> getArmorPositionFromState(const Eigen::VectorXd & x);

  double max_match_distance_;
  double max_match_yaw_diff_;

  int detect_count_;
  int lost_count_;
  
};

class TrackerManager {
  private:
      // 按ID管理多个追踪器
      std::map<std::string, std::shared_ptr<Tracker>> trackers_;
      
      // 当前选中的追踪目标
      std::string current_tracked_id_;
      
      // 评分历史和目标切换冷却
      std::map<std::string, double> scores_history_;
      rclcpp::Time last_switch_time_;
      double switch_cooldown_;
      
      // 计算评分的参数
      double max_match_distance_;
      double max_match_yaw_diff_;
      int tracking_thres_;
      double lost_time_thres_;

      
      // 权重参数
      double w_distance_;    // 距离图像中心的权重
      double w_velocity_;    // 速度稳定性的权重
      double w_tracking_;    // 追踪状态权重
      double w_size_;        // 装甲板大小权重
      double w_confidence_;  // 置信度权重
      double w_history_;     // 历史评分权重
      // EKF 模板，用于初始化新的 Tracker
      ExtendedKalmanFilter ekf_template_;
      
      
  public:
      void setWeights(
        double w_distance,
        double w_velocity,
        double w_tracking, 
        double w_size,
        double w_confidence,
        double w_history);
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

#endif  // ARMOR_PROCESSOR__TRACKER_HPP_
