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
    double max_match_distance, double max_match_yaw_diff, int tracking_thres,
    double lost_time_thres,double miss_match_time_thres, double switch_cooldown)
: trackers_(),
  current_tracked_id_(""),
  last_switch_time_(rclcpp::Clock().now()),
  switch_cooldown_(switch_cooldown),
  max_match_distance_(max_match_distance),
  max_match_yaw_diff_(max_match_yaw_diff),
  tracking_thres_(tracking_thres),
  lost_time_thres_(lost_time_thres),
  miss_match_time_thres_(miss_match_time_thres),
  w_distance_(0.5),
  w_twoD_distance_(0.5)
{
}
void TrackerManager::setWeights(double w_distance, double w_twoD_distance)
{
    w_distance_ = w_distance;
    w_twoD_distance_ = w_twoD_distance;
}

void TrackerManager::updateEKFTemplate(double dt)
{
    // 更新 EKF 模板的时间间隔
    ekf_template_.setTimeInterval(dt);
}
//从这一部分开始是状态更新相关函数

void TrackerManager::update(
    const auto_aim_interfaces::msg::Armors::SharedPtr & armors_msg, bool is_main_camera)
{
    rclcpp::Time msg_time = armors_msg->header.stamp;
    if (trackers_.empty()) {
        static rclcpp::Clock warn_clock(RCL_SYSTEM_TIME);
        RCLCPP_WARN_THROTTLE(
            rclcpp::get_logger("armor_tracker"), warn_clock, 2000,
            "No active trackers available. Initializing new trackers if possible.");
    }

    //std::cerr << "Tracker size at update: " << trackers_.size() << std::endl;

    const double temp_lost_time = lost_time_thres_ / 5.0;
    // 1. 按ID对装甲板分组
    std::map<std::string, std::vector<auto_aim_interfaces::msg::Armor>> armors_by_id;

    for (const auto & armor : armors_msg->armors) {
        armors_by_id[armor.number].push_back(armor);
    }

    // 更新和初始化tracker
    for (const auto & id : ID_LIST) {
        bool has_tracker = trackers_.find(id) != trackers_.end();
        bool has_armors = armors_by_id.find(id) != armors_by_id.end() && !armors_by_id[id].empty();
        if (has_tracker && has_armors) {
            // 如果追踪器存在且当前帧中有装甲板，更新追踪器
            // 创建仅包含特定ID装甲板的消息
            auto id_armors_msg = std::make_shared<auto_aim_interfaces::msg::Armors>();
            id_armors_msg->header = armors_msg->header;
            id_armors_msg->armors = armors_by_id[id];
            bool matched = trackers_[id]->update(id_armors_msg, is_main_camera);
            if (!matched && is_main_camera) {
                static rclcpp::Clock warn_clock(RCL_SYSTEM_TIME);
                RCLCPP_WARN_THROTTLE(
                    rclcpp::get_logger("armor_tracker"), warn_clock, 1000,
                    "Tracker %s did not match any armors with %s data.",
                    id.c_str(), is_main_camera ? "main camera" : "wide camera");
            }
            else{
                RCLCPP_DEBUG(
                    rclcpp::get_logger("armor_tracker"),
                    "Tracker %s successfully matched armors with %s data.", id.c_str(), is_main_camera ? "main camera" : "wide camera");
            }
            trackers_[id]->updateState(matched, msg_time, temp_lost_time, lost_time_thres_, tracking_thres_, miss_match_time_thres_,is_main_camera);
        } else if (has_tracker && !has_armors) {
            // 如果追踪器存在但当前帧中没有装甲板，使用空消息更新
            auto empty_msg = std::make_shared<auto_aim_interfaces::msg::Armors>();
            empty_msg->header = armors_msg->header;
            RCLCPP_DEBUG(
                rclcpp::get_logger("armor_tracker"),
                "No armors for tracker %s with %s data.", id.c_str(), is_main_camera ? "main camera" : "wide camera");
            bool matched = trackers_[id]->update(empty_msg, is_main_camera);
            trackers_[id]->updateState(matched, msg_time, temp_lost_time, lost_time_thres_, tracking_thres_, miss_match_time_thres_,is_main_camera);
        } else if (!has_tracker && has_armors) {
            // 如果追踪器不存在但当前帧中有装甲板，初始化新的追踪器
            if(!is_main_camera){
                RCLCPP_WARN(
                    rclcpp::get_logger("armor_tracker"),
                    "Initializing new tracker %s with wide camera data.", id.c_str());
            }
            initNewTracker(id, armors_by_id[id], msg_time);
            if(!is_main_camera) {
                RCLCPP_WARN(
                    rclcpp::get_logger("armor_tracker"),
                    "New tracker %s initialized with wide camera data.", id.c_str());
            }
        }
    }
}

void TrackerManager::initNewTracker(
    const std::string & id, const std::vector<auto_aim_interfaces::msg::Armor> & armors,
    rclcpp::Time msg_time)
{
    auto tracker = std::make_shared<Tracker>(max_match_distance_, max_match_yaw_diff_);
    tracker->tracking_thres = tracking_thres_;

    // 复制 EKF 模板
    tracker->ekf = ekf_template_;

    // 创建仅含特定ID装甲板的消息
    auto id_armors_msg = std::make_shared<auto_aim_interfaces::msg::Armors>();
    id_armors_msg->header.stamp = msg_time;   // 当前时间
    id_armors_msg->header.frame_id = "odom";  // 假设使用odom坐标系
    id_armors_msg->armors = armors;

    // 初始化追踪器
    tracker->init(id_armors_msg);

    trackers_[id] = tracker;
}

void TrackerManager::cleanInactiveTrackers()
{
    auto it = trackers_.begin();
    while (it != trackers_.end()) {
        // 移除不活跃的追踪器
        if (it->second->tracker_state == Tracker::LOST) {
            it = trackers_.erase(it);
        } else {
            ++it;
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

double TrackerManager::calculateScore(
    const std::string & id, const std::shared_ptr<Tracker> & tracker)
{
    // 追踪状态分数
    double state_score = 0.0;
    switch (tracker->tracker_state) {
        case Tracker::TRACKING:
            state_score = 1.0;
            break;
        case Tracker::TEMP_LOST:
            state_score = 0.8;
            break;
        case Tracker::DETECTING:
            state_score = 0.3;
            break;
        case Tracker::MISS_MATCH:
            state_score = 0.9;
            break;
        case Tracker::LOST:
            state_score = 0.0;
            break;
    }

    // 装甲板距图像中心的距离分数
    double center_distance = 0.0;
    if (tracker->tracker_state != Tracker::LOST) {
        double px = tracker->target_state(XC);
        double py = tracker->target_state(YC);

        double normalized_distance =
            (std::sqrt(px * px + py * py) - 3.0) / 2.0;  // 假设最大距离为5m
        center_distance = std::min(normalized_distance, 1.0);
        center_distance = std::max(center_distance, 0.0);
    }
    double distance_score = 1.0 - center_distance;
    double two_d_distance = static_cast<double>(tracker->twoD_distance);
    double two_d_center_score = 1.0 - std::min(two_d_distance / 1000, 1.0);
    // 综合评分
    double score =
        (w_distance_ * distance_score + w_twoD_distance_ * two_d_center_score) * state_score;
    if (tracker->last_update_time_.seconds() - tracker->last_main_update_time_.seconds() < 0.3) {
        score  += 1.0;
    }
    if (id == "1" && mode_ == VisionMode::HERO) score += 2.0;
    if (id == "2" && mode_ == VisionMode::ENGINEER) score += 2.0;
    if (id == "3" && mode_ == VisionMode::INFANTRY_1) score += 2.0;
    if (id == "4" && mode_ == VisionMode::INFANTRY_2) score += 2.0;
    if (id == "5" && mode_ == VisionMode::INFANTRY_3) score += 2.0;
    if (id == "outpost" && mode_ == VisionMode::OUTPOST) score += 2.0;
    if (id == "guard" && mode_ == VisionMode::GUARD) score += 2.0;
    if (id == "base" && mode_ == VisionMode::BASE) score += 2.0;

    // std::cerr << "Tracker ID: " << id << std::endl;
    // std::cerr << " Score: " << score << std::endl;
    // std::cerr << "Distance: " << distance_score  << std::endl;
    // std::cerr << "2DD:"<< twoD_center_score<<std::endl;
    // std::cerr << "State: " << state_score << std::endl;
    return score;
}

void TrackerManager::selectBestTarget()
{
    if (trackers_.empty()) {
        current_tracked_id_ = "";
        return;
    }

    std::string best_id = "";
    double best_score = -1.0;
    double switch_threshold = 0.2;  // 切换阈值

    // 计算每个追踪器的评分
    for (const auto & [id, tracker] : trackers_) {
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


std::string TrackerManager::getCurrentTargetID() const { return current_tracked_id_; }

bool TrackerManager::getIDTarget(
    std::string input_tracked_id_, auto_aim_interfaces::msg::Target & target_msg) const
{
    target_msg.tracking = false;

    // 如果没有正在追踪的目标，返回空消息
    if (input_tracked_id_.empty() || trackers_.find(input_tracked_id_) == trackers_.end()) {
        return false;
    }
    // 获取当前追踪的目标
    const auto & tracker = trackers_.at(input_tracked_id_);

    // 设置消息的时间戳
    target_msg.header.stamp = tracker->last_update_time_;

    // 根据追踪状态填充消息
    if (tracker->tracker_state == Tracker::DETECTING) {
        target_msg.tracking = false;
    } else if (
        tracker->tracker_state == Tracker::TRACKING ||
        tracker->tracker_state == Tracker::TEMP_LOST) {
        target_msg.tracking = true;
    }
    // 填充目标消息
    const auto & state = tracker->target_state;
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

    return true;
}

std::vector<std::string> TrackerManager::getActiveTrackerIDs() const
{
    std::vector<std::string> active_ids;
    for (const auto & [id, tracker] : trackers_) {
        if (tracker->tracker_state != Tracker::LOST) {
            active_ids.push_back(id);
        }
    }
    return active_ids;
}

}  // namespace rm_auto_aim