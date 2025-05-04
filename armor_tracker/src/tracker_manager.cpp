// Copyright 2022 Chen Jun

#include "armor_tracker/tracker_manager.hpp"

#include <iostream>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include "armor_tracker/types.hpp"

namespace rm_auto_aim
{

// TrackerManager类的实现
TrackerManager::TrackerManager(
    double max_match_distance,
    double max_match_yaw_diff,
    int tracking_thres,
    double lost_time_thres,
    double switch_cooldown)
    : trackers_(),
    current_tracked_id_(""),
    last_switch_time_(rclcpp::Clock().now()),
    switch_cooldown_(switch_cooldown),
    max_match_distance_(max_match_distance),
    max_match_yaw_diff_(max_match_yaw_diff),
    tracking_thres_(tracking_thres),
    lost_time_thres_(lost_time_thres),
    w_distance_(0.5),
    w_twoD_distance_(0.5)
    {}
    void TrackerManager::setWeights(
        double w_distance,
        double w_twoD_distance)
    {
    w_distance_ = w_distance;
    w_twoD_distance_ = w_twoD_distance;
    }

  //从这一部分开始是状态更新相关函数

  void TrackerManager::update(const auto_aim_interfaces::msg::Armors::SharedPtr& armors_msg) {
    // 计算时间差
    double dt = 0.0;
    rclcpp::Time msg_time = armors_msg->header.stamp;
    if (!trackers_.empty()) {
      for (auto& [_, tracker] : trackers_) {
        dt = (msg_time - tracker->last_update_time_).seconds();
        break;
      }
    }
    if (dt <= 0) {
        dt = 0.01;
    }
    if (trackers_.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "No active trackers available. Initializing new trackers if possible.");
        
    }
    
    //std::cerr << "Tracker size at update: " << trackers_.size() << std::endl;
    
    
    // 根据时间差计算lost_thres
    int lost_thres = static_cast<int>(lost_time_thres_ / dt);
    // 1. 按ID对装甲板分组
    std::map<std::string, std::vector<auto_aim_interfaces::msg::Armor>> armors_by_id;
  
    for (const auto& armor : armors_msg->armors) {
        armors_by_id[armor.number].push_back(armor);
    }
    
    // 2. 更新现有追踪器
    for (auto& [id, tracker] : trackers_) {
        if (armors_by_id.find(id) != armors_by_id.end() && !armors_by_id[id].empty()) {
            // 设置lost_thres（只要dt有效）
            if (dt > 0) {
            tracker->lost_thres = lost_thres;
            }
            // 创建仅包含特定ID装甲板的消息
            auto id_armors_msg = std::make_shared<auto_aim_interfaces::msg::Armors>();
            id_armors_msg->header = armors_msg->header;
            
            // Add bounds checking when adding armors
            for (const auto& armor : armors_by_id[id]) {
              id_armors_msg->armors.push_back(armor);
            }
            
            // Only update if message is valid
            if (!id_armors_msg->armors.empty()) {
              tracker->update(id_armors_msg);
            }
            
            
            tracker->last_update_time_ = msg_time;
            
            // 记录已处理的ID
            armors_by_id.erase(id);
        } else {
            // 此ID在当前帧中未检测到，使用空消息更新
            auto empty_msg = std::make_shared<auto_aim_interfaces::msg::Armors>();
            empty_msg->header = armors_msg->header;
            tracker->update(empty_msg);
        }
    }
    
    // 3. 为新的ID创建追踪器
    for (const auto& [id, armors] : armors_by_id) {
        initNewTracker(id, armors, msg_time);
        //std::cerr << "New tracker initialized for ID: " << id << std::endl;
        
        
    }
    
    // 4. 选择最佳追踪目标
    selectBestTarget();
  }
  
  void TrackerManager::initNewTracker(const std::string& id
                                      , const std::vector<auto_aim_interfaces::msg::Armor>& armors
                                      , rclcpp::Time msg_time) {
    auto tracker = std::make_shared<Tracker>(max_match_distance_, max_match_yaw_diff_);
    tracker->tracking_thres = tracking_thres_;
  
    // 复制 EKF 模板
    tracker->ekf = ekf_template_;
    
    // 创建仅含特定ID装甲板的消息
    auto id_armors_msg = std::make_shared<auto_aim_interfaces::msg::Armors>();
    id_armors_msg->header.stamp = msg_time;  // 当前时间
    id_armors_msg->header.frame_id = "odom";  // 假设使用odom坐标系
    for (const auto& armor : armors) {
        id_armors_msg->armors.push_back(armor);
    }
    
    // 初始化追踪器
    tracker->init(id_armors_msg);
    tracker->tracker_state = Tracker::DETECTING;  // 设置初始状态为检测中
    // 初始化评分指标
    tracker->last_update_time_ = id_armors_msg->header.stamp;
    
    trackers_[id] = tracker;
  }
  
  void TrackerManager::cleanInactiveTrackers(rclcpp::Time now) {
    std::vector<std::string> ids_to_remove;
    
    for (const auto& [id, tracker] : trackers_) {
        // 如果追踪器长时间没有更新，或者处于LOST状态，则标记为移除
        if ((now - tracker->last_update_time_).seconds() > lost_time_thres_ ||
            tracker->tracker_state == Tracker::LOST) {
            ids_to_remove.push_back(id);
        }
    }
    
    // 移除不活跃的追踪器
    for (const auto& id : ids_to_remove) {
        trackers_.erase(id);
    
        
        // 如果当前选中的追踪器被移除，需要重新选择
        if (id == current_tracked_id_) {
            current_tracked_id_ = "";
            selectBestTarget();
        }
    }
  }
  
  void TrackerManager::reset() 
  {
    // 清除所有追踪器
    trackers_.clear();
    
    // 重置当前选中的目标ID
    current_tracked_id_ = "";
    
    
    // 重置切换冷却时间
    last_switch_time_ = rclcpp::Clock().now();
    
    RCLCPP_INFO(rclcpp::get_logger("armor_tracker"), "TrackerManager has been reset");
  }

  //从这一部分开始是评分相关函数

  
  
  double TrackerManager::calculateScore(const std::string& id, const std::shared_ptr<Tracker>& tracker) {
    // 追踪状态分数
    double state_score = 0.0;
    switch (tracker->tracker_state) {
        case Tracker::TRACKING: state_score = 1.0; break;
        case Tracker::TEMP_LOST: state_score = 0.8; break;
        case Tracker::DETECTING: state_score = 0.3; break;
        case Tracker::LOST: state_score = 0.0; break;
    }
    
    // 装甲板距图像中心的距离分数
    double center_distance = 0.0;
    if (tracker->tracker_state != Tracker::LOST) {
        double px = tracker->target_state(XC);
        double py = tracker->target_state(YC);
    
        // 将3D位置投影到图像平面（简化计算）
        // 实际实现中应使用相机内参进行正确的投影
        double normalized_distance = (std::sqrt(px*px + py*py) -3.0)/ 2.0;  // 假设最大距离为5m
        center_distance = std::min(normalized_distance, 1.0);
        center_distance = std::max(center_distance, 0.0);
    }
    double distance_score = 1.0 - center_distance;
    double twoD_distance_ = static_cast<double>(tracker->twoD_distance);
    double twoD_center_score = 1.0 - std::min(twoD_distance_ / 1000, 1.0);
    // 综合评分
    double score = 
                  (w_distance_ * distance_score + w_twoD_distance_ * twoD_center_score)*state_score 
                ;
    if(id == "1" && mode_ == VisionMode::HERO) score += 2.0; 
    if(id == "2" && mode_ == VisionMode::ENGINEER) score += 2.0;
    if(id == "3" && mode_ == VisionMode::INFANTRY_1) score += 2.0;
    if(id == "4" && mode_ == VisionMode::INFANTRY_2) score += 2.0;
    if(id == "5" && mode_ == VisionMode::INFANTRY_3) score += 2.0;
    if(id == "outpost" && mode_ == VisionMode::OUTPOST) score += 2.0;
    if(id == "guard" && mode_ == VisionMode::GUARD) score += 2.0;
    if(id == "base" && mode_ == VisionMode::BASE) score += 2.0;

    // std::cerr << "Tracker ID: " << id << std::endl;
    // std::cerr << " Score: " << score << std::endl;
    // std::cerr << "Distance: " << distance_score  << std::endl;
    // std::cerr << "2DD:"<< twoD_center_score<<std::endl;
    // std::cerr << "State: " << state_score << std::endl;
    return score;
  }
  
  void TrackerManager::selectBestTarget() {
    if (trackers_.empty()) {
        current_tracked_id_ = "";
        return;
    }
    
    std::string best_id = "";
    double best_score = -1.0;
    double switch_threshold = 0.2;  // 切换阈值
    
    // 计算每个追踪器的评分
    for (const auto& [id, tracker] : trackers_) {
        if (tracker->tracker_state == Tracker::LOST) {
            continue;  // 忽略已丢失的目标
        }
        
        double score = calculateScore(id, tracker);
        
        
        // 维持当前目标的稳定性：如果当前目标的评分接近最高分，则保持不变
        if (id == current_tracked_id_ && score > best_score - switch_threshold) {
            best_id = id;
            best_score = score;
        } 
        // 如果有明显更好的目标，则切换
        else if (score > best_score + switch_threshold) {
            best_id = id;
            best_score = score;
        }
    }
    
    // 如果选定了新的目标，并且已经过了冷却时间，则更新
    auto now = rclcpp::Clock().now();
    if (!best_id.empty() && best_id != current_tracked_id_) {
        if ((now - last_switch_time_).seconds() > switch_cooldown_) {
            current_tracked_id_ = best_id;
            last_switch_time_ = now;
        }
    }
  }

  //从这一部分开始是目标数据发布与可视化相关函数

  auto_aim_interfaces::msg::Target TrackerManager::getCurrentTarget() const 
  {
      // 初始化target消息
      auto_aim_interfaces::msg::Target target_msg;
      
      // 设置默认帧ID
      target_msg.header.frame_id = "odom";
      target_msg.tracking = false;
      
      // 如果没有正在追踪的目标，返回空消息
      if (current_tracked_id_.empty() || trackers_.find(current_tracked_id_) == trackers_.end()) {
          return target_msg;
      }
      
      // 获取当前追踪的目标
      const auto& tracker = trackers_.at(current_tracked_id_);
      
      // 设置消息的时间戳
      target_msg.header.stamp = tracker->last_update_time_;
      
      // 根据追踪状态填充消息
      if (tracker->tracker_state == Tracker::DETECTING) {
          target_msg.tracking = false;
      } else if (
          tracker->tracker_state == Tracker::TRACKING ||
          tracker->tracker_state == Tracker::TEMP_LOST) {
          target_msg.tracking = true;
          
          // 填充目标消息
          const auto& state = tracker->target_state;
          target_msg.id = tracker->tracked_id;
          target_msg.armors_num = static_cast<int>(tracker->tracked_armors_num);
          
          // 位置和速度信息
          target_msg.position.x = state(XC);
          target_msg.velocity.x = state(VXC);
          target_msg.position.y = state(YC);
          target_msg.velocity.y = state(VYC);
          target_msg.position.z = state(ZC1);
          target_msg.velocity.z = state(VZC);
          
          // 角度和旋转信息
          target_msg.yaw = state(YAW1);
          target_msg.v_yaw = state(VYAW);
          
          // 半径信息
          target_msg.radius_1 = state(R1);
          target_msg.radius_2 = state(R2);
          
          // 装甲板高度差
          target_msg.dz = state(ZC2) - state(ZC1);
      }
      
      return target_msg;
  }
  
  auto_aim_interfaces::msg::Target TrackerManager::getIDTarget(std::string input_tracked_id_) const 
  {
      // 初始化target消息
      auto_aim_interfaces::msg::Target target_msg;
      
      // 设置默认帧ID
      target_msg.header.frame_id = "odom";
      target_msg.tracking = false;
      
      
      // 获取当前追踪的目标
      const auto& tracker = trackers_.at(input_tracked_id_);
      
      // 设置消息的时间戳
      target_msg.header.stamp = tracker->last_update_time_;
      
      // 根据追踪状态填充消息
      if (tracker->tracker_state == Tracker::DETECTING) {
          target_msg.tracking = false;
      } else if (
          tracker->tracker_state == Tracker::TRACKING ||
          tracker->tracker_state == Tracker::TEMP_LOST) {
          target_msg.tracking = true;
          
          // 填充目标消息
          const auto& state = tracker->target_state;
          target_msg.id = tracker->tracked_id;
          target_msg.armors_num = static_cast<int>(tracker->tracked_armors_num);
          
          // 位置和速度信息
          target_msg.position.x = state(XC);
          target_msg.velocity.x = state(VXC);
          target_msg.position.y = state(YC);
          target_msg.velocity.y = state(VYC);
          target_msg.position.z = state(ZC1);
          target_msg.velocity.z = state(VZC);
          
          // 角度和旋转信息
          target_msg.yaw = state(YAW1);
          target_msg.v_yaw = state(VYAW);
          
          // 半径信息
          target_msg.radius_1 = state(R1);
          target_msg.radius_2 = state(R2);
          
          // 装甲板高度差
          target_msg.dz = state(ZC2) - state(ZC1);
      }
      
      return target_msg;
  }
  
  std::vector<std::string> TrackerManager::getActiveTrackerIDs() const {
    std::vector<std::string> active_ids;
    for (const auto& [id, tracker] : trackers_) {
        if (tracker->tracker_state != Tracker::LOST) {
            active_ids.push_back(id);
        }
    }
    return active_ids;
  }

}  // namespace rm_auto_aim