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
  last_update_time_(0.0),
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
    if (armors_msg->armors.size() == 1) {
        tracked_armor = armors_msg->armors[0];
        twoD_distance = tracked_armor.distance_to_image_center;
        initEKF(tracked_armor);
        RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init EKF with single armor!");
    } else if (armors_msg->armors.size() == 2) {
        tracked_armor = armors_msg->armors[0];
        tracked_armor_2 = armors_msg->armors[1];
        twoD_distance =
            fmin(tracked_armor.distance_to_image_center, tracked_armor_2.distance_to_image_center);
        initEKFTwo(tracked_armor, tracked_armor_2);
        RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init EKF with two armors!");
    }
    updateArmorsNum();
    tracked_id = tracked_armor.number;
    tracker_state = DETECTING;  //将追踪状态设为detecting
    return;
}

void Tracker::update(const Armors::SharedPtr & armors_msg)
//根据经过EKF加权后的观测和预测来更新装甲板的追踪状态
{
    // KF predict
    Eigen::VectorXd ekf_prediction = ekf.predict();
    RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF predict");

    bool matched = false;
    // Use KF prediction as default target state if no matched armor is found
    target_state = ekf_prediction;
    if (armors_msg->armors.empty()) {
        RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "No armors found, using EKF prediction");
        return;
    }
    // init tracker info
    twoD_distance = DBL_MAX;
    info_position_diff = DBL_MAX;
    info_yaw_diff = DBL_MAX;
    if (armors_msg->armors.size() == 1) {
        int matched_id = matchArmor(armors_msg->armors[0], ekf_prediction);
        if (matched_id == 1) {
            // Matched armor1 found
            tracked_armor = armors_msg->armors[0];
            matched = true;
            auto p = tracked_armor.pose.position;
            // Update EKF
            double measured_yaw = orientationToYaw(
                tracked_armor.pose.orientation, p,
                target_state(YAW1));  //四元数方向转换为偏航角
            measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
            target_state = ekf.update1(measurement);
            RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update1");
        } else if (matched_id == 2) {
            // Matched armor2 found
            tracked_armor_2 = armors_msg->armors[0];
            matched = true;
            auto p = tracked_armor_2.pose.position;
            // Update EKF
            double measured_yaw = orientationToYaw(
                tracked_armor_2.pose.orientation, p,
                target_state(YAW2));  //四元数方向转换为偏航角
            measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
            target_state = ekf.update2(measurement);
            RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update2");
        } else {
            RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Reset tracker by single armor!");
            init(armors_msg);
            return;
        }
    }
    if (armors_msg->armors.size() == 2) {
        int matched_armor1 = matchArmor(armors_msg->armors[0], ekf_prediction);
        int matched_armor2 = matchArmor(armors_msg->armors[1], ekf_prediction);
        if (matched_armor1 == 0 || matched_armor2 == 0) {
            RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Reset tracker by two armors!");
            init(armors_msg);
            return;
        }
        if (matched_armor1 == 2 && matched_armor2 == 1) {
            std::swap(armors_msg->armors[0], armors_msg->armors[1]);
            std::swap(matched_armor1, matched_armor2);
        }
        if (matched_armor1 == 1 && matched_armor2 == 2) {
            // Matched armor found
            tracked_armor = armors_msg->armors[0];
            tracked_armor_2 = armors_msg->armors[1];
            matched = true;
            auto p1 = tracked_armor.pose.position;
            auto p2 = tracked_armor_2.pose.position;
            // Update EKF
            double yaw_a = orientationToYaw(
                tracked_armor.pose.orientation, p1,
                target_state(YAW1));  //四元数方向转换为偏航角
            double yaw_b = orientationToYaw(
                tracked_armor_2.pose.orientation, p2,
                target_state(YAW2));  //四元数方向转换为偏航角

            measurement = Eigen::VectorXd(10);
            double xa = p1.x, ya = p1.y, xb = p2.x, yb = p2.y;
            double A = sin(yaw_b - yaw_a);
            double r1 = (sin(yaw_b) * (xb - xa) - cos(yaw_b) * (yb - ya)) / A;
            double r2 = (sin(yaw_a) * (xb - xa) - cos(yaw_a) * (yb - ya)) / A;
            if (abs(r1 - target_state(R1)) > 0.1 || abs(r2 - target_state(R2)) > 0.1) {
                measurement << p1.x, p1.y, p1.z, yaw_a, p2.x, p2.y, p2.z, yaw_b, target_state(R1),
                    target_state(R2);
            } else {
                measurement << p1.x, p1.y, p1.z, yaw_a, p2.x, p2.y, p2.z, yaw_b, r1, r2;
            }

            target_state = ekf.updateTwo(measurement);
            RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
        } else {
            RCLCPP_ERROR(rclcpp::get_logger("tracker"), "2 armors are too close!");
            return;
        }
    }
    if (tracked_armors_num == ArmorsNum::OUTPOST_3) {
        target_state(R1) = target_state(R2) = 0.2765;  // 固定半径
        // 约束速度
        if (std::abs(std::abs(target_state(VYAW)) - 0.8 * M_PI) < 0.2 * M_PI) {
            target_state(VYAW) = target_state(VYAW) > 0 ? 0.8 * M_PI : -0.8 * M_PI;
        }
        ekf.setState(target_state);
    }
    // 防止半径扩散
    for (int r_idx : {R1, R2}) {
        target_state(r_idx) = fmax(target_state(r_idx), 0.2);
        target_state(r_idx) = fmin(target_state(r_idx), 0.3); // hero: 0.4
    }
    // 防止yaw角度扩散
    if (target_state(YAW1) < -M_PI) {
        target_state(YAW1) += 4 * M_PI / double(tracked_armors_num);
        target_state(YAW2) += 4 * M_PI / double(tracked_armors_num);
        ekf.setState(target_state);
    }
    if (target_state(YAW2) > M_PI) {
        target_state(YAW1) -= 4 * M_PI / double(tracked_armors_num);
        target_state(YAW2) -= 4 * M_PI / double(tracked_armors_num);
        ekf.setState(target_state);
    }
    double yaw_average = (target_state(YAW1) + target_state(YAW2)) / 2;
    target_state(YAW1) = yaw_average - M_PI / double(tracked_armors_num);
    target_state(YAW2) = yaw_average + M_PI / double(tracked_armors_num);
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
int Tracker::matchArmor(const Armor & armor, const Eigen::VectorXd & ekf_prediction)
{
    // Match armor with EKF prediction
    // Calculate the difference between the predicted position and the current armor position
    auto p = armor.pose.position;
    double yaw = orientationToYaw(armor.pose.orientation);
    Eigen::Vector3d position_vec(p.x, p.y, p.z);
    auto predicted_position = getArmorPositionFromState(ekf_prediction);  //预测
    double position_diff_1 = fmin(
        (predicted_position[0] - position_vec).norm(),
        (predicted_position[2] - position_vec).norm());
    double position_diff_2 = fmin(
        (predicted_position[1] - position_vec).norm(),
        (predicted_position[3] - position_vec).norm());
    double yaw_diff_1 = calYawDiff(yaw, ekf_prediction(YAW1));
    double yaw_diff_2 = calYawDiff(yaw, ekf_prediction(YAW2));

    if (yaw_diff_1 <= yaw_diff_2) {
        if (yaw_diff_1 < max_match_yaw_diff_ && position_diff_1 < max_match_distance_) {
            twoD_distance = fmin(armor.distance_to_image_center, twoD_distance);
            info_position_diff = fmin(info_position_diff, position_diff_1);
            info_yaw_diff = fmin(info_yaw_diff, yaw_diff_1);
            return 1;  // Matched armor1 found
        }
    } else {
        if (yaw_diff_2 < max_match_yaw_diff_ && position_diff_2 < max_match_distance_) {
            twoD_distance = fmin(armor.distance_to_image_center, twoD_distance);
            info_position_diff = fmin(info_position_diff, position_diff_2);
            info_yaw_diff = fmin(info_yaw_diff, yaw_diff_2);
            return 2;  // Matched armor2 found
        }
    }
    return 0;  // No matched armor found
}
void Tracker::initEKF(const Armor & a)
{
    auto p = a.pose.position;
    double yaw = orientationToYaw(a.pose.orientation);
    if (yaw < 0) {
        target_state = Eigen::VectorXd::Zero(12);
        double r = 0.2765;
        double xc = p.x + r * cos(yaw);
        double yc = p.y + r * sin(yaw);
        target_state(XC) = xc, target_state(YC) = yc, target_state(ZC1) = target_state(ZC2) = p.z;
        target_state(YAW1) = yaw, target_state(R1) = r;
        target_state(YAW2) = yaw + 2 * M_PI / double(tracked_armors_num), target_state(R2) = r;

        ekf.setState(target_state);
    } else {
        target_state = Eigen::VectorXd::Zero(12);
        double r = 0.2765;
        double xc = p.x + r * cos(yaw);
        double yc = p.y + r * sin(yaw);
        target_state(XC) = xc, target_state(YC) = yc, target_state(ZC1) = target_state(ZC2) = p.z;
        target_state(VXC) = 0, target_state(VYC) = 0, target_state(VZC) = 0, target_state(VYAW) = 0;
        target_state(YAW2) = yaw, target_state(R2) = r;
        target_state(YAW1) = yaw - 2 * M_PI / double(tracked_armors_num), target_state(R1) = r;
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
    if (yaw_a > yaw_b) {
        std::swap(yaw_a, yaw_b);
        std::swap(xa, xb);
        std::swap(ya, yb);
        std::swap(za, zb);
    }
    if (yaw_b - yaw_a < M_PI / 3) {
        RCLCPP_ERROR(rclcpp::get_logger("tracker"), "Init failed");
        return;
    }

    target_state = Eigen::VectorXd::Zero(12);
    double r1 = (sin(yaw_b) * (xb - xa) - cos(yaw_b) * (yb - ya));
    double r2 = (sin(yaw_a) * (xb - xa) - cos(yaw_a) * (yb - ya));
    double xc = xa + r1 * cos(yaw_a);
    double yc = ya + r1 * sin(yaw_a);
    if (yaw_a < -2 * M_PI / double(tracked_armors_num) ||
        yaw_a > 2 * M_PI / double(tracked_armors_num)) {
        yaw_a = yaw_a < 0 ? yaw_a + M_PI : yaw_a - M_PI;
    }
    if (yaw_b < -2 * M_PI / double(tracked_armors_num) ||
        yaw_b > 2 * M_PI / double(tracked_armors_num)) {
        yaw_b = yaw_b < 0 ? yaw_b + M_PI : yaw_b - M_PI;
    }
    if (yaw_a > yaw_b) {
        std::swap(yaw_a, yaw_b);
        std::swap(r1, r2);
        std::swap(za, zb);
    }
    target_state(XC) = xc, target_state(YC) = yc, target_state(ZC1) = za, target_state(ZC2) = zb;
    target_state(VXC) = 0, target_state(VYC) = 0, target_state(VZC) = 0, target_state(VYAW) = 0;
    target_state(YAW1) = yaw_a, target_state(YAW2) = yaw_b;
    target_state(R1) = 0.2765, target_state(R2) = 0.2765;
    ekf.setState(target_state);
}

void Tracker::updateArmorsNum()
{
    if (tracked_id == "outpost") {
        tracked_armors_num = ArmorsNum::OUTPOST_3;
    } else {
        tracked_armors_num = ArmorsNum::NORMAL_4;
    }
}
double Tracker::orientationToYaw(
    const geometry_msgs::msg::Quaternion & q, geometry_msgs::msg::Point & position,
    const double & yaw_target)  //将四元数转换为偏航角
{
    // Get armor yaw
    tf2::Quaternion tf_q;
    tf2::fromMsg(q, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    // 保证输出欧拉角的唯一主值
    if (abs(pitch) > M_PI / 2) {
        // 略去对pitch和roll的处理
        yaw = std::atan2(std::sin(M_PI + yaw), std::cos(M_PI + yaw));  // 旋转yaw 180度
    }
    // Make yaw rang right (-pi~pi to -pi/2~pi/2)
    if (abs(yaw - yaw_target) > (int(tracked_armors_num) == 4)
            ? M_PI / 2
            : M_PI / double(tracked_armors_num)) {
        Eigen::Vector3d center(target_state(XC), target_state(YC), target_state(ZC1));
        double r;
        if (yaw_target == target_state(YAW1))
            r = target_state(R1);
        else
            r = target_state(R2);
        double center_x = position.x + r * cos(yaw);
        double center_y = position.y + r * sin(yaw);
        double gap = (int(tracked_armors_num) == 4) ? M_PI : 2 * M_PI / double(tracked_armors_num);
        double yaw_diff = yaw_target - yaw;
        yaw += std::round(yaw_diff / gap) * gap;
        position.x = center_x - r * cos(yaw);
        position.y = center_y - r * sin(yaw);
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
    // 保证输出欧拉角的唯一主值
    if (abs(pitch) > M_PI / 2) {
        // 略去对pitch和roll的处理
        yaw = std::atan2(std::sin(M_PI + yaw), std::cos(M_PI + yaw));  // 旋转yaw 180度
    }
    return yaw;
}

std::vector<Eigen::Vector3d> Tracker::getArmorPositionFromState(
    const Eigen::VectorXd & x)  //从EKF的状态向量计算装甲板的预测位置。
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
    double diff = std::min(
        abs(angles::shortest_angular_distance(yaw1, yaw2)),
        abs(angles::shortest_angular_distance(yaw1 + 4 * M_PI / double(tracked_armors_num), yaw2)));
    return diff;
}

}  // namespace rm_auto_aim
