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
: tracker_state(LOST),
  tracked_id(std::string("")),
  measurement(Eigen::VectorXd::Zero(4)),
  target_state(Eigen::VectorXd::Zero(9)),
  max_match_distance_(max_match_distance),
  max_match_yaw_diff_(max_match_yaw_diff)
{
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
          double yaw_avg = (yaw_a + yaw_b) / 2;
          yaw_a = yaw_a > yaw_avg ? yaw_avg + M_PI / 4 : yaw_avg - M_PI / 4;
          yaw_b = yaw_b > yaw_avg ? yaw_avg + M_PI / 4 : yaw_avg - M_PI / 4;
          double A = sin(yaw_b - yaw_a);
          double r1 = (sin(yaw_b) * (xb - xa) - cos(yaw_b) * (yb - ya))/A; 
          double r2 = (sin(yaw_a) * (xb - xa) - cos(yaw_a) * (yb - ya))/A;
          if(r1 < 0.15 || r1 > 0.4 || r2 < 0.15 || r2 > 0.4){
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
  double yaw_average = (target_state(YAW1) + target_state(YAW2)) / 2;
  target_state(YAW1) = yaw_average - M_PI / 4;
  target_state(YAW2) = yaw_average + M_PI / 4;
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
  double yaw_avg = (yaw_a + yaw_b) / 2;
  yaw_a = yaw_avg - M_PI / 4; 
  yaw_b = yaw_avg + M_PI / 4; 

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
  target_state(R1) = r1, target_state(R2) = r2;
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
    double r1 = target_state(R1), r2 = target_state(R2); 
    if(yaw > yaw_target){
      position.x += 2 * r1 * cos(yaw);
      position.y += 2 * r1 * sin(yaw);
      yaw -= M_PI;
    }
    if(yaw < yaw_target){
      position.x += 2 * r2 * cos(yaw);
      position.y += 2 * r2 * sin(yaw);
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

}  // namespace rm_auto_aim
