// Copyright 2022 Chen Jun

#ifndef ARMOR_PROCESSOR__TRACKER_HPP_
#define ARMOR_PROCESSOR__TRACKER_HPP_

// Eigen
#include <Eigen/Eigen>

// ROS
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>

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
private:
  void initEKF(const Armor & a); 
  void initEKFTwo(const Armor & a, const Armor & b);

  void updateArmorsNum(const Armor & a);

  double orientationToYaw(const geometry_msgs::msg::Quaternion & q, geometry_msgs::msg::Point & position);
  double orientationToYaw(const geometry_msgs::msg::Quaternion & q); //overload
  double calYawDiff(double yaw1, double yaw2); 

  Eigen::Vector3d getArmorPositionFromState1(const Eigen::VectorXd & x);
  Eigen::Vector3d getArmorPositionFromState2(const Eigen::VectorXd & x);

  double max_match_distance_;
  double max_match_yaw_diff_;

  int detect_count_;
  int lost_count_;
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
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__TRACKER_HPP_
