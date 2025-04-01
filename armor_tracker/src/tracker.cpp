// Copyright 2022 Chen Jun

#include "armor_tracker/tracker.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <angles/angles.h>
#include <kdl/utilities/utility.h>
#include <math.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>

#include <cmath>
#include <iostream>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <cfloat>
#include <memory>
#include <string>

namespace rm_auto_aim


// 利用扩展卡尔曼滤波器（EKF）来推算出当前目标装甲板所在的机器人的速度、角速度（偏航速度）等状态信息
{
// 构造追踪器为空的状态
Tracker::Tracker(double max_match_distance, double max_match_yaw_diff)
  :tracker_state(LOST),
  tracked_id(std::string("")),
  measurement(Eigen::VectorXd::Zero(4)),
  target_state(Eigen::VectorXd::Zero(9)),
  last_update_time_(rclcpp::Clock().now()),
  tracking_duration_(0.0),
  avg_innovation_(0.0),
  max_match_distance_(max_match_distance),
  max_match_yaw_diff_(max_match_yaw_diff)
{
  velocity_history_.clear();
}
//初始化追踪器
void Tracker::init(const Armors::SharedPtr & armors_msg)
{
  if (armors_msg->armors.empty()) {
    return;
  }

  // Simply choose the armor that is closest to image center
  double min_distance = DBL_MAX;
  tracked_armor = armors_msg->armors[0];//基于输入的装甲板消息选择最接近图像中心的装甲板作为追踪目标
  for (const auto & armor : armors_msg->armors) {
    if (armor.distance_to_image_center < min_distance) {
      min_distance = armor.distance_to_image_center;
      tracked_armor = armor;
    }
  }
  bool found = false;
  for (const auto & armor : armors_msg->armors) {
    if(armor.number == tracked_armor.number && armor != tracked_armor){
      if(!found){
        tracked_armor_2 = armor;
        found = true;
      }
      else{
        RCLCPP_ERROR(rclcpp::get_logger("tracker"), "More than two armor with same id found!"); 
      }
    }
  }
  //中心的装甲板作为追踪目标，并初始化EKF
  if(found) initEKFTwo(tracked_armor, tracked_armor_2);
  else initEKF(tracked_armor);
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init EKF!");

  tracked_id = tracked_armor.number;
  tracker_state = DETECTING;//将追踪状态设为detecting

  updateArmorsNum(tracked_armor);//对追踪的装甲板进行分类？
}

void Tracker::update(const Armors::SharedPtr & armors_msg)
//根据经过EKF加权后的观测和预测来更新装甲板的追踪状态
{
  // KF predict
  Eigen::VectorXd ekf_prediction = ekf.predict();//
  RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF predict");

  bool matched = false;
  // Use KF prediction as default target state if no matched armor is found
  target_state = ekf_prediction;
  if (!armors_msg->armors.empty()) {
    // Find the closest armor with the same id
    Armor same_id_armor;
    int same_id_armors_count = 0, found_armors = 0;
    auto predicted_position = getArmorPositionFromState(ekf_prediction);//预测
    double min_position_diff_1 = DBL_MAX, min_position_diff_2 = DBL_MAX;//导入差值上限
    double min_yaw_diff_1 = DBL_MAX, min_yaw_diff_2 = DBL_MAX;//导入差值上限
    for (const auto & armor : armors_msg->armors) {//遍历所有观测到的装甲板
      // Only consider armors with the same id
      if (armor.number == tracked_id) {
        same_id_armor = armor;
        same_id_armors_count++;
        // Calculate the difference between the predicted position and the current armor position
        auto p = armor.pose.position;
        double yaw = orientationToYaw(armor.pose.orientation); 
        Eigen::Vector3d position_vec(p.x, p.y, p.z);

        double position_diff_1 = fmin((predicted_position[0] - position_vec).norm(), (predicted_position[2] - position_vec).norm()); 
        double position_diff_2 = fmin((predicted_position[1] - position_vec).norm(), (predicted_position[3] - position_vec).norm());
        double yaw_diff_1 = calYawDiff(yaw, ekf_prediction(YAW1));
        double yaw_diff_2 = calYawDiff(yaw, ekf_prediction(YAW2));

        if (yaw_diff_1 <= yaw_diff_2) {
          // Find the closest armor
          if(yaw_diff_1 < min_yaw_diff_1){
            min_position_diff_1 = position_diff_1; 
            min_yaw_diff_1 = yaw_diff_1;
            tracked_armor = armor;
            found_armors = found_armors | 1;
          }
        }
        else{
          // Find the closest armor
          if(yaw_diff_2 < min_yaw_diff_2){
            min_position_diff_2 = position_diff_2;
            min_yaw_diff_2 = yaw_diff_2;
            tracked_armor_2 = armor;
            found_armors = found_armors | 2;
          }
        }
      }
    }
    // Store tracker info
    info_position_diff = fmin(min_position_diff_1, min_position_diff_2);
    info_yaw_diff = fmin(min_yaw_diff_1, min_yaw_diff_2);
    if(same_id_armors_count > 2){
      RCLCPP_ERROR(rclcpp::get_logger("tracker"), "More than two armor with same id found!");
    }
    else if(same_id_armors_count == 2){
      if(found_armors != 3){
        RCLCPP_ERROR(rclcpp::get_logger("tracker"), "2 armors are too close!");
        if(found_armors == 2) initEKF(tracked_armor_2);
        else initEKF(tracked_armor);
      } 
      else{
        // Check if the distance and yaw difference of closest armor are within the threshold
        if (min_position_diff_1 < max_match_distance_ && min_yaw_diff_1 < max_match_yaw_diff_ && min_position_diff_2 < max_match_distance_ && min_yaw_diff_2 < max_match_yaw_diff_) {
          // Matched armor found
          matched = true;
          auto p1 = tracked_armor.pose.position;  
          auto p2 = tracked_armor_2.pose.position;
          // Update EKF
          double yaw_a = orientationToYaw(tracked_armor.pose.orientation, p1, target_state(YAW1)); //四元数方向转换为偏航角
          double yaw_b = orientationToYaw(tracked_armor_2.pose.orientation, p2, target_state(YAW2)); //四元数方向转换为偏航角

          measurement = Eigen::VectorXd(10); 
          double xa = p1.x, ya = p1.y, xb = p2.x, yb = p2.y;
          // double yaw_avg = (yaw_a + yaw_b) / 2;
          // yaw_a = yaw_a > yaw_avg ? yaw_avg + M_PI / 4 : yaw_avg - M_PI / 4;
          // yaw_b = yaw_b > yaw_avg ? yaw_avg + M_PI / 4 : yaw_avg - M_PI / 4;
          double A = sin(yaw_b - yaw_a);
          double r1 = (sin(yaw_b) * (xb - xa) - cos(yaw_b) * (yb - ya))/A; 
          double r2 = (sin(yaw_a) * (xb - xa) - cos(yaw_a) * (yb - ya))/A;
          if(0){
            measurement << p1.x, p1.y, p1.z, yaw_a, p2.x, p2.y, p2.z, yaw_b, target_state(R1), target_state(R2);
          }
          else{
            measurement << p1.x, p1.y, p1.z, yaw_a, p2.x, p2.y, p2.z, yaw_b, r1, r2;
          }
          
          target_state = ekf.updateTwo(measurement);
          RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
        }
        else if (min_position_diff_1 < max_match_distance_ && min_yaw_diff_1 < max_match_yaw_diff_) {
          // Matched armor1 found
          matched = true;
          auto p = tracked_armor.pose.position;
          // Update EKF
          double measured_yaw = orientationToYaw(tracked_armor.pose.orientation, p, target_state(YAW1)); //四元数方向转换为偏航角
          measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
          target_state = ekf.update1(measurement);
          RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
        } 
        else if (min_position_diff_2 < max_match_distance_ && min_yaw_diff_2 < max_match_yaw_diff_) {
          // Matched armor2 found
          matched = true;
          auto p = tracked_armor.pose.position;
          // Update EKF
          double measured_yaw = orientationToYaw(tracked_armor.pose.orientation, p, target_state(YAW2)); //四元数方向转换为偏航角
          measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
          target_state = ekf.update2(measurement);
          RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
        } 
        else{
          initEKFTwo(tracked_armor, tracked_armor_2);
          RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Reset State by Two Armors!"); 
        }
      }
    }
    else if(same_id_armors_count == 1){
      if(found_armors == 1){
        // Check if the distance and yaw difference of closest armor are within the threshold
        if (min_position_diff_1 < max_match_distance_ && min_yaw_diff_1 < max_match_yaw_diff_) {
          // Matched armor1 found
          matched = true;
          auto p = tracked_armor.pose.position;
          // Update EKF
          double measured_yaw = orientationToYaw(tracked_armor.pose.orientation, p, target_state(YAW1)); //四元数方向转换为偏航角
          measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
          target_state = ekf.update1(measurement);
          RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
        } 
        else{
          initEKF(tracked_armor);
          RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Reset State by Armor1"); 
        }
      }
      else if(found_armors == 2){
        // Check if the distance and yaw difference of closest armor are within the threshold
        if (min_position_diff_2 < max_match_distance_ && min_yaw_diff_2 < max_match_yaw_diff_) {
          // Matched armor2 found
          matched = true;
          auto p = tracked_armor_2.pose.position;
          // Update EKF
          double measured_yaw = orientationToYaw(tracked_armor_2.pose.orientation, p, target_state(YAW2)); //四元数方向转换为偏航角
          measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
          target_state = ekf.update2(measurement);
          RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
        } 
        else{
          // std::cerr << "min_position_diff_2: " << min_position_diff_2 << std::endl;
          // std::cerr << "min_yaw_diff_2: " << yaw_diff_2 << std::endl;
          // std::cerr << "min_position_diff: " << min_position_diff_1 << std::endl;
          // std::cerr << "min_yaw_diff: " << yaw_diff_1 << std::endl;
          initEKF(tracked_armor_2);
          RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Reset State by Armor2"); 
        }
      }
      else{
        RCLCPP_ERROR(rclcpp::get_logger("tracker"), "No matched armor found!");
      }
    }
    else {
      // No matched armor found
      RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "No matched armor found!");
    }
  }

  // Prevent radius from spreading
  if (target_state(R1) < 0.12) {
    target_state(R1) = 0.12;
    ekf.setState(target_state);
  } else if (target_state(R1) > 0.4) {
    target_state(R1) = 0.4;
    ekf.setState(target_state);
  }
  if (target_state(R2) < 0.12) {
    target_state(R2) = 0.12;
    ekf.setState(target_state);
  } else if (target_state(R2) > 0.4) {
    target_state(R2) = 0.4;
    ekf.setState(target_state);
  }
  // Prevent angle of two armors from spreading
  if(target_state(YAW1) < -M_PI){ //暂时不确定
    target_state(YAW1) += M_PI;
    target_state(YAW2) += M_PI;
    ekf.setState(target_state);
  }
  if(target_state(YAW2) > M_PI){
    target_state(YAW1) -= M_PI;
    target_state(YAW2) -= M_PI;
    ekf.setState(target_state);
  }
  double yaw_average = (target_state(YAW1) + target_state(YAW2)) / 2;
  target_state(YAW1) = yaw_average - M_PI / 4;
  target_state(YAW2) = yaw_average + M_PI / 4;
  ekf.setState(target_state);

  // Tracking state machine
  if (tracker_state == DETECTING) {
    if (matched) {
      detect_count_++;
      if (detect_count_ > tracking_thres) {
        detect_count_ = 0;
        tracker_state = TRACKING;
      }
    } else {
      detect_count_ = 0;
      tracker_state = LOST;
    }
  } else if (tracker_state == TRACKING) {
    if (!matched) {
      tracker_state = TEMP_LOST;
      lost_count_++;
    }
  } else if (tracker_state == TEMP_LOST) {
    if (!matched) {
      lost_count_++;
      if (lost_count_ > lost_thres) {
        lost_count_ = 0;
        tracker_state = LOST;
      }
    } else {
      tracker_state = TRACKING;
      lost_count_ = 0;
    }
  }
}

void Tracker::initEKF(const Armor & a)
{
  auto p = a.pose.position;
  double yaw = orientationToYaw(a.pose.orientation);
  if(yaw < 0){ //暂时不确定
    // Set initial position at 0.2m behind the target
    target_state = Eigen::VectorXd::Zero(12);
    double r = 0.26;
    double xc = p.x + r * cos(yaw);
    double yc = p.y + r * sin(yaw);
    target_state(XC) = xc, target_state(YC) = yc, target_state(ZC1) = target_state(ZC2) = p.z;
    target_state(YAW1) = yaw, target_state(R1) = r;
    target_state(YAW2) = yaw + M_PI / 2, target_state(R2) = r;

    ekf.setState(target_state);
  }
  else {
    // Set initial position at 0.2m behind the target
    target_state = Eigen::VectorXd::Zero(12);
    double r = 0.26;
    double xc = p.x + r * cos(yaw);
    double yc = p.y + r * sin(yaw);
    target_state(XC) = xc, target_state(YC) = yc, target_state(ZC1) = target_state(ZC2) = p.z;
    target_state(VXC) = 0, target_state(VYC) = 0, target_state(VZC) = 0, target_state(VYAW) = 0;
    target_state(YAW2) = yaw, target_state(R2) = r;
    target_state(YAW1) = yaw - M_PI / 2, target_state(R1) = r;
    ekf.setState(target_state);
  }
}
void Tracker::initEKFTwo(const Armor & a, const Armor & b)
{
  double xa = a.pose.position.x;
  double ya = a.pose.position.y;
  double za = a.pose.position.z;
  double xb = b.pose.position.x;
  double yb = b.pose.position.y;
  double zb = b.pose.position.z;
  double yaw_a = orientationToYaw(a.pose.orientation);
  double yaw_b = orientationToYaw(b.pose.orientation);
  if(yaw_a > yaw_b){
    std::swap(yaw_a, yaw_b);
    std::swap(xa, xb);
    std::swap(ya, yb);
    std::swap(za, zb);
  }
  if(yaw_b - yaw_a < M_PI / 3){
    RCLCPP_ERROR(rclcpp::get_logger("tracker"), "Init failed");
    return; 
  }
  // double yaw_avg = (yaw_a + yaw_b) / 2;
  // yaw_a = yaw_avg - M_PI / 4; 
  // yaw_b = yaw_avg + M_PI / 4; 

  target_state = Eigen::VectorXd::Zero(12);
  double r1 = (sin(yaw_b) * (xb - xa) - cos(yaw_b) * (yb - ya)); 
  double r2 = (sin(yaw_a) * (xb - xa) - cos(yaw_a) * (yb - ya));
  double xc = xa + r1 * cos(yaw_a);
  double yc = ya + r1 * sin(yaw_a);
  if(yaw_a < -M_PI / 2 || yaw_a > M_PI / 2) {
    yaw_a = yaw_a < 0 ? yaw_a + M_PI : yaw_a - M_PI;
  }
  if(yaw_b < -M_PI / 2 || yaw_b > M_PI / 2) {
    yaw_b = yaw_b < 0 ? yaw_b + M_PI : yaw_b - M_PI;
  }
  if(yaw_a > yaw_b){
    std::swap(yaw_a, yaw_b);
    std::swap(r1, r2);
    std::swap(za, zb);
  }
  target_state(XC) = xc, target_state(YC) = yc, target_state(ZC1) = za, target_state(ZC2) = zb;
  target_state(VXC) = 0, target_state(VYC) = 0, target_state(VZC) = 0, target_state(VYAW) = 0;
  target_state(YAW1) = yaw_a, target_state(YAW2) = yaw_b;
  target_state(R1) = 0.26, target_state(R2) = 0.26;
  ekf.setState(target_state);
}

void Tracker::updateArmorsNum(const Armor & armor)
{
  if (armor.type == "large" && (tracked_id == "3" || tracked_id == "4" || tracked_id == "5")) {
    tracked_armors_num = ArmorsNum::BALANCE_2;
  } else if (tracked_id == "outpost") {
    tracked_armors_num = ArmorsNum::OUTPOST_3;
  } else {
    tracked_armors_num = ArmorsNum::NORMAL_4;
  }
}
double Tracker::orientationToYaw(const geometry_msgs::msg::Quaternion & q, geometry_msgs::msg::Point & position, const double & yaw_target)//将四元数转换为偏航角
{
  // Get armor yaw
  tf2::Quaternion tf_q;
  tf2::fromMsg(q, tf_q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw); 
  // Make yaw rang right (-pi~pi to -pi/2~pi/2)
  if (abs(yaw - yaw_target) > M_PI / 2) {
    Eigen::Vector3d center(target_state(XC), target_state(YC), target_state(ZC1));
    double r; 
    if(yaw_target == target_state(YAW1)) r = target_state(R1);
    else r = target_state(R2);
    if(yaw > yaw_target){
      position.x += 2 * r * cos(yaw);
      position.y += 2 * r * sin(yaw);
      yaw -= M_PI;
    }
    else if(yaw < yaw_target){
      position.x += 2 * r * cos(yaw);
      position.y += 2 * r * sin(yaw);
      yaw += M_PI;
    }
  }
  return yaw;
}
double Tracker::orientationToYaw(const geometry_msgs::msg::Quaternion & q)
{
  // Get armor yaw
  tf2::Quaternion tf_q;
  tf2::fromMsg(q, tf_q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw); 
  return yaw;
}

std::vector<Eigen::Vector3d> Tracker::getArmorPositionFromState(const Eigen::VectorXd & x)//从EKF的状态向量计算装甲板1的预测位置。
{
  // Calculate predicted position of the current armor
  double xc = x(XC), yc = x(YC), za1 = x(ZC1), za2 = x(ZC2);
  double yaw1 = x(YAW1), r1 = x(R1);
  double yaw2 = x(YAW2), r2 = x(R2);
  std::vector<Eigen::Vector3d> armor_position;
  double xa = xc - r1 * cos(yaw1);
  double ya = yc - r1 * sin(yaw1);
  armor_position.push_back(Eigen::Vector3d(xa, ya, za1));
  xa = xc - r2 * cos(yaw2);
  ya = yc - r2 * sin(yaw2);
  armor_position.push_back(Eigen::Vector3d(xa, ya, za2));
  xa = xc + r1 * cos(yaw1);
  ya = yc + r1 * sin(yaw1);
  armor_position.push_back(Eigen::Vector3d(xa, ya, za1));
  xa = xc + r2 * cos(yaw2);
  ya = yc + r2 * sin(yaw2);
  armor_position.push_back(Eigen::Vector3d(xa, ya, za2));
  return armor_position;
}
double Tracker::calYawDiff(double yaw1, double yaw2)
{
  double diff = abs(angles::shortest_angular_distance(yaw1, yaw2)); 
  if(diff > M_PI / 2){
    diff = M_PI - diff;
  }
  return diff;
}
// TrackerManager类的实现
TrackerManager::TrackerManager(
  double max_match_distance,
  double max_match_yaw_diff,
  int tracking_thres,
  double lost_time_thres,
  double switch_cooldown)
  : trackers_(),
  current_tracked_id_(""),
  scores_history_(),
  last_switch_time_(rclcpp::Clock().now()),
  switch_cooldown_(switch_cooldown),
  max_match_distance_(max_match_distance),
  max_match_yaw_diff_(max_match_yaw_diff),
  tracking_thres_(tracking_thres),
  lost_time_thres_(lost_time_thres),
  lost_thres_(0),
  w_distance_(0.3),
  w_velocity_(0.2),
  w_tracking_(0.3),
  w_size_(0.1),
  w_confidence_(0.05),
  w_history_(0.05),
  clock_(RCL_SYSTEM_TIME)
{}
void TrackerManager::setWeights(
  double w_distance,
  double w_velocity,
  double w_tracking, 
  double w_size,
  double w_confidence,
  double w_history)
{
  w_distance_ = w_distance;
  w_velocity_ = w_velocity;
  w_tracking_ = w_tracking;
  w_size_ = w_size;
  w_confidence_ = w_confidence;
  w_history_ = w_history;
}
void TrackerManager::update(const auto_aim_interfaces::msg::Armors::SharedPtr& armors_msg) {
  // 计算时间差
  double dt = 0.0;
  if (!trackers_.empty()) {
    for (auto& [_, tracker] : trackers_) {
        // 使用 fromMsg 转换时间格式
        rclcpp::Time msg_time = rclcpp::Time(armors_msg->header.stamp);
        dt = (msg_time - tracker->last_update_time_).seconds();
        break;
    }
  }
  if (dt <= 0) {
    dt = 0.01;
  }
  
  // 根据时间差计算lost_thres
  int lost_thres = static_cast<int>(lost_time_thres_ / dt);
  // 1. 按ID对装甲板分组
  std::map<std::string, std::vector<auto_aim_interfaces::msg::Armor>> armors_by_id;
  for (const auto& armor : armors_msg->armors) {
      armors_by_id[armor.number].push_back(armor);
  }
  
  // 2. 更新现有追踪器
  for (auto& [id, tracker] : trackers_) {
      if (armors_by_id.find(id) != armors_by_id.end()) {
          // 设置lost_thres（只要dt有效）
          if (dt > 0) {
          tracker->lost_thres = lost_thres;
          }
          // 创建仅包含特定ID装甲板的消息
          auto id_armors_msg = std::make_shared<auto_aim_interfaces::msg::Armors>();
          id_armors_msg->header = armors_msg->header;
          for (const auto& armor : armors_by_id[id]) {
              id_armors_msg->armors.push_back(armor);
          }
          
          // 更新对应ID的追踪器
          tracker->update(id_armors_msg);
          tracker->tracking_duration_ += 
          (rclcpp::Time(armors_msg->header.stamp) - tracker->last_update_time_).seconds();
          tracker->last_update_time_ = rclcpp::Time(armors_msg->header.stamp);
          
    
          
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
      initNewTracker(id, armors);
  }
  
  // 4. 选择最佳追踪目标
  selectBestTarget();
}

void TrackerManager::initNewTracker(const std::string& id, const std::vector<auto_aim_interfaces::msg::Armor>& armors) {
  auto tracker = std::make_shared<Tracker>(max_match_distance_, max_match_yaw_diff_);
  tracker->tracking_thres = tracking_thres_;

  // 复制 EKF 模板
  tracker->ekf = ekf_template_;
  
  // 创建仅含特定ID装甲板的消息
  auto id_armors_msg = std::make_shared<auto_aim_interfaces::msg::Armors>();
  id_armors_msg->header.stamp = rclcpp::Clock().now();  // 当前时间
  id_armors_msg->header.frame_id = "odom";  // 假设使用odom坐标系
  for (const auto& armor : armors) {
      id_armors_msg->armors.push_back(armor);
  }
  
  // 初始化追踪器
  tracker->init(id_armors_msg);
  
  // 初始化评分指标
  tracker->tracking_duration_ = 0.0;
  tracker->last_update_time_ = id_armors_msg->header.stamp;
  tracker->avg_innovation_ = 0.0;
  
  trackers_[id] = tracker;
}

void TrackerManager::cleanInactiveTrackers(double inactive_threshold) {
  auto now = clock_.now();
  std::vector<std::string> ids_to_remove;
  
  for (const auto& [id, tracker] : trackers_) {
      // 如果追踪器长时间没有更新，或者处于LOST状态，则标记为移除
      if ((now - tracker->last_update_time_).seconds() > inactive_threshold ||
          tracker->tracker_state == Tracker::LOST) {
          ids_to_remove.push_back(id);
      }
  }
  
  // 移除不活跃的追踪器
  for (const auto& id : ids_to_remove) {
      trackers_.erase(id);
      scores_history_.erase(id);
      
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
  
  // 清除评分历史
  scores_history_.clear();
  
  // 重置切换冷却时间
  last_switch_time_ = rclcpp::Clock().now();
  
  RCLCPP_INFO(rclcpp::get_logger("armor_tracker"), "TrackerManager has been reset");
}

double Tracker::getVelocityStability() const
{
    // 没有足够的历史数据时，返回默认值
    if (velocity_history_.size() < 3) {
        return 0.5;  // 默认中等稳定性
    }
    
    // 计算速度变化的标准差
    double sum = 0.0;
    double mean = 0.0;
    
    // 首先计算平均值
    for (const auto& vel : velocity_history_) {
        mean += vel;
    }
    mean /= velocity_history_.size();
    
    // 然后计算方差
    for (const auto& vel : velocity_history_) {
        sum += std::pow(vel - mean, 2);
    }
    double variance = sum / velocity_history_.size();
    
    // 标准差越小，稳定性越高
    // 将标准差转换为0~1的稳定性分数，值越大越稳定
    double stability = 1.0 / (1.0 + std::sqrt(variance));
    
    return stability;
}

double TrackerManager::calculateScore(const std::string& id, const std::shared_ptr<Tracker>& tracker) {
  // 追踪状态分数
  double state_score = 0.0;
  switch (tracker->tracker_state) {
      case Tracker::TRACKING: state_score = 1.0; break;
      case Tracker::TEMP_LOST: state_score = 0.6; break;
      case Tracker::DETECTING: state_score = 0.3; break;
      case Tracker::LOST: state_score = 0.0; break;
  }
  
  // 装甲板距图像中心的距离分数
  double center_distance = 0.0;
  if (tracker->tracker_state != Tracker::LOST) {
      double px = tracker->target_state(Tracker::XC);
      double py = tracker->target_state(Tracker::YC);
  
      // 将3D位置投影到图像平面（简化计算）
      // 实际实现中应使用相机内参进行正确的投影
      double normalized_distance = std::sqrt(px*px + py*py) / 10.0;  // 假设最大距离为10m
      center_distance = std::min(normalized_distance, 1.0);
  }
  double distance_score = 1.0 - center_distance;
  
  // 速度稳定性分数
  double velocity_score = tracker->getVelocityStability();
  
  // 追踪时长分数 (最多贡献1分，追踪超过5秒获得满分)
  double duration_score = std::min(tracker->tracking_duration_ / 3.0, 1.0);
  
  // 装甲板类型和置信度分数
  double type_score = tracker->tracked_armor.type == "large" ? 1.0 : 0.8;
  
  double confidence_score = 0.5;
  
  // 历史评分
  double history_score = scores_history_.count(id) ? scores_history_[id] : 0.0;
  
  // 综合评分
  double score = w_tracking_ * state_score +
                w_distance_ * distance_score +
                w_velocity_ * velocity_score +
                w_size_ * type_score +
                w_confidence_ * confidence_score +
                w_history_ * history_score+
                0.1*duration_score;
  
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
      scores_history_[id] = score;
      
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
  auto now = clock_.now();
  if (!best_id.empty() && best_id != current_tracked_id_) {
      if ((now - last_switch_time_).seconds() > switch_cooldown_) {
          current_tracked_id_ = best_id;
          last_switch_time_ = now;
      }
  }
}

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
        target_msg.position.x = state(Tracker::XC);
        target_msg.velocity.x = state(Tracker::VXC);
        target_msg.position.y = state(Tracker::YC);
        target_msg.velocity.y = state(Tracker::VYC);
        target_msg.position.z = state(Tracker::ZC1);
        target_msg.velocity.z = state(Tracker::VZC);
        
        // 角度和旋转信息
        target_msg.yaw = state(Tracker::YAW1);
        target_msg.v_yaw = state(Tracker::VYAW);
        
        // 半径信息
        target_msg.radius_1 = state(Tracker::R1);
        target_msg.radius_2 = state(Tracker::R2);
        
        // 装甲板高度差
        target_msg.dz = state(Tracker::ZC2) - state(Tracker::ZC1);
    }
    
    return target_msg;
}

}  // namespace rm_auto_aim
