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
#include <algorithm>
#include <opencv2/core.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <cfloat>
#include <memory>
#include <string>
#include "armor_tracker/types.hpp"

namespace rm_auto_aim

// 利用扩展卡尔曼滤波器（EKF）来推算出当前目标装甲板所在的机器人的速度、角速度（偏航速度）等状态信息
{
// 构造追踪器为空的状态
Tracker::Tracker(double max_match_distance, double max_match_yaw_diff, double max_translation_speed)
: tracker_state(LOST),
  tracked_id(std::string("")),
  measurement(Eigen::VectorXd::Zero(4)),
        target_state(Eigen::VectorXd::Zero(15)),
  last_update_time_(0.0),
  last_main_update_time_(0.0),
  max_match_distance_(max_match_distance),
  max_match_yaw_diff_(max_match_yaw_diff),
    max_translation_speed_(max_translation_speed),
    detect_count_(0)
{
}
//初始化追踪器
void Tracker::init(const Armors::SharedPtr & armors_msg)
{
    RCLCPP_INFO(
        rclcpp::get_logger("armor_tracker"), "Tracker init called at %.6f s",
        rclcpp::Time(armors_msg->header.stamp).seconds());
    if (armors_msg->armors.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Failed to init EKF with empty msg!");
        return;
    }
    if (armors_msg->armors[0].number == "outpost") {
        // 单块板初始化：填充实测坐标+默认高差/半径
        tracked_armor = armors_msg->armors[0];
        twoD_distance = tracked_armor.distance_to_image_center;
        auto p = tracked_armor.pose.position;
        double yaw = orientationToYaw(tracked_armor.pose.orientation);
        // 15维状态初始化
        target_state.setZero();
        double r_outpost = 0.275;  // 前哨站固定半径
        target_state(XC) = p.x + r_outpost * cos(yaw);
        target_state(YC) = p.y + r_outpost * sin(yaw);
        target_state(ZC1) = target_state(ZC2) = target_state(ZC3) = p.z;
        target_state(R1) = target_state(R2) = target_state(R3) = r_outpost;
        target_state(YAW1) = yaw;
        target_state(YAW2) = yaw + 2 * M_PI / 3;  // 3块板间隔120°
        target_state(YAW3) = yaw + 4 * M_PI / 3;
        ekf.setState(target_state);
        RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init EKF for outpost (3 armors)!");
    } else {
        // 原有4块板逻辑（不变）
        if (armors_msg->armors.size() == 1) {
            tracked_armor = armors_msg->armors[0];
            twoD_distance = tracked_armor.distance_to_image_center;
            initEKF(tracked_armor);
            RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init EKF with single armor!");
        } else if (armors_msg->armors.size() == 2) {
            tracked_armor = armors_msg->armors[0];
            tracked_armor_2 = armors_msg->armors[1];
            twoD_distance = fmin(
                tracked_armor.distance_to_image_center, tracked_armor_2.distance_to_image_center);
            initEKFTwo(tracked_armor, tracked_armor_2);
            RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Init EKF with two armors!");
        }
    }
    tracked_id = tracked_armor.number;
    updateArmorsNum();
    tracker_state = DETECTING;  //将追踪状态设为detecting
    last_update_time_ = armors_msg->header.stamp;
    return;
}

bool Tracker::update(const Armors::SharedPtr & armors_msg, bool is_main_camera)
//根据经过EKF加权后的观测和预测来更新装甲板的追踪状态
{
    rclcpp::Time msg_time = armors_msg->header.stamp;

    // 主/广角相机 决策逻辑
    if (!is_main_camera) {
        // // 如果是广角相机，且主相机在 100ms 内更新过，则忽略此帧广角数据
        // std::cerr << "time since last main update: "
        //           << (msg_time.seconds() - last_main_update_time_.seconds()) * 1000 << " ms"
        //           << std::endl;
        if ((msg_time.seconds() - last_main_update_time_.seconds()) < 0.1) {
            RCLCPP_DEBUG(
                rclcpp::get_logger("armor_tracker"),
                "Ignoring wide camera data due to recent main camera update.");
            return false;
        }
    }

    if (msg_time.nanoseconds() < last_update_time_.nanoseconds()) {
        static rclcpp::Clock main_warn_clock(RCL_SYSTEM_TIME);
        static rclcpp::Clock wide_warn_clock(RCL_SYSTEM_TIME);
        auto & warn_clock = is_main_camera ? main_warn_clock : wide_warn_clock;
        RCLCPP_WARN_THROTTLE(
            rclcpp::get_logger("armor_tracker"), warn_clock, 1000,
            "msg received is too old, skip processing. source=%s",
            is_main_camera ? "main" : "wide");
        return false;
    }
    // KF predict（先做快照，未匹配则回滚）
    auto ekf_backup = ekf;

    ekf.setTimeInterval((msg_time - last_update_time_).seconds());
    Eigen::VectorXd ekf_prediction = ekf.predict();
    limitTranslationVelocity(ekf_prediction);
    ekf.setState(ekf_prediction);
    RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF predict");

    bool matched = false;
    // Use KF prediction as default target state if no matched armor is found
    target_state = ekf_prediction;

    if (armors_msg->armors.empty()) {
        RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "No armors found, using EKF prediction");
        ekf = ekf_backup;
        return matched;
    }
    // init tracker info
    twoD_distance = DBL_MAX;
    info_position_diff = DBL_MAX;
    info_yaw_diff = DBL_MAX;
    auto update_outpost_with_single_armor = [this, &matched, &ekf_prediction](const Armor & armor) {//因为极少双板情况，直接归到单板，逻辑有待完善
        int matched_id = matchArmor(armor, ekf_prediction);
        if (matched_id == 1 || matched_id == 2 || matched_id == 3) {
            tracked_armor = armor;
            matched = true;
            auto p = tracked_armor.pose.position;

            int yaw_idx = (matched_id == 1) ? YAW1 : ((matched_id == 2) ? YAW2 : YAW3);
            double measured_yaw = orientationToYaw(
                tracked_armor.pose.orientation, p,
                target_state(yaw_idx));  // 四元数方向转换为偏航角

            measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
            if (matched_id == 1) {
                target_state = ekf.update1(measurement);
                // Propagate observed yaw to the other two outpost panels
                {
                    double base_yaw = angles::normalize_angle(measured_yaw);
                    const double gap = 2.0 * M_PI / 3.0;
                    target_state(YAW1) = base_yaw;
                    target_state(YAW2) = angles::normalize_angle(base_yaw + gap);
                    target_state(YAW3) = angles::normalize_angle(base_yaw + 2.0 * gap);
                }
                RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update1 (outpost)");
            } else if (matched_id == 2) {
                target_state = ekf.update2(measurement);
                // Propagate observed yaw to the other two outpost panels
                {
                    double base_yaw = angles::normalize_angle(measured_yaw);
                    const double gap = 2.0 * M_PI / 3.0;
                    target_state(YAW2) = base_yaw;
                    target_state(YAW3) = angles::normalize_angle(base_yaw + gap);
                    target_state(YAW1) = angles::normalize_angle(base_yaw + 2.0 * gap);
                }
                RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update2 (outpost)");
            } else {
                target_state = ekf.update3(measurement);
                // Propagate observed yaw to the other two outpost panels
                {
                    double base_yaw = angles::normalize_angle(measured_yaw);
                    const double gap = 2.0 * M_PI / 3.0;
                    target_state(YAW3) = base_yaw;
                    target_state(YAW1) = angles::normalize_angle(base_yaw + gap);
                    target_state(YAW2) = angles::normalize_angle(base_yaw + 2.0 * gap);
                }
                RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update3 (outpost)");
            }
        } else {
            RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Reset outpost tracker by single armor!");
            if (tracker_state == TRACKING || tracker_state == DETECTING) {
                tracker_state = MISS_MATCH;
            }
        }
    };

    if (tracked_armors_num == ArmorsNum::OUTPOST_3 && armors_msg->armors.size() > 1) {
        // Outpost keeps a single-armor update path to avoid unstable two-armor geometry.
        const auto best_it = std::min_element(
            armors_msg->armors.begin(), armors_msg->armors.end(),
            [](const Armor & a, const Armor & b) {
                return a.distance_to_image_center < b.distance_to_image_center;
            });
        if (best_it != armors_msg->armors.end()) {
            update_outpost_with_single_armor(*best_it);
        }
    } else if (armors_msg->armors.size() == 1) {
        int matched_id = matchArmor(armors_msg->armors[0], ekf_prediction);
        if (tracked_armors_num == ArmorsNum::OUTPOST_3) {
            // 前哨站单板更新：根据匹配到的1/2/3号板分别使用 update1/update2/update3
            if (matched_id == 1 || matched_id == 2 || matched_id == 3) {
                tracked_armor = armors_msg->armors[0];
                matched = true;
                auto p = tracked_armor.pose.position;

                int yaw_idx = (matched_id == 1) ? YAW1 : ((matched_id == 2) ? YAW2 : YAW3);
                double measured_yaw = orientationToYaw(
                    tracked_armor.pose.orientation, p,
                    target_state(yaw_idx));  // 四元数方向转换为偏航角

                measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
                if (matched_id == 1) {
                    target_state = ekf.update1(measurement);
                    // Propagate observed yaw to the other two outpost panels
                    {
                        double base_yaw = angles::normalize_angle(measured_yaw);
                        const double gap = 2.0 * M_PI / 3.0;
                        target_state(YAW1) = base_yaw;
                        target_state(YAW2) = angles::normalize_angle(base_yaw + gap);
                        target_state(YAW3) = angles::normalize_angle(base_yaw + 2.0 * gap);
                    }
                    RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update1 (outpost)");
                } else if (matched_id == 2) {
                    target_state = ekf.update2(measurement);
                    // Propagate observed yaw to the other two outpost panels
                    {
                        double base_yaw = angles::normalize_angle(measured_yaw);
                        const double gap = 2.0 * M_PI / 3.0;
                        target_state(YAW2) = base_yaw;
                        target_state(YAW3) = angles::normalize_angle(base_yaw + gap);
                        target_state(YAW1) = angles::normalize_angle(base_yaw + 2.0 * gap);
                    }
                    RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update2 (outpost)");
                } else {
                    target_state = ekf.update3(measurement);
                    // Propagate observed yaw to the other two outpost panels
                    {
                        double base_yaw = angles::normalize_angle(measured_yaw);
                        const double gap = 2.0 * M_PI / 3.0;
                        target_state(YAW3) = base_yaw;
                        target_state(YAW1) = angles::normalize_angle(base_yaw + gap);
                        target_state(YAW2) = angles::normalize_angle(base_yaw + 2.0 * gap);
                    }
                    RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update3 (outpost)");
                
                } 
            }else {
                        RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "Reset outpost tracker by single armor!");
                        if (tracker_state == TRACKING || tracker_state == DETECTING) {
                            tracker_state = MISS_MATCH;
                        }
                    return matched;
            }
        } else {
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
            limitTranslationVelocity(target_state);
            ekf.setState(target_state);
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
            limitTranslationVelocity(target_state);
            ekf.setState(target_state);
                RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update2");
            } else {
                RCLCPP_WARN(
                rclcpp::get_logger("armor_tracker"),
                "Single armor miss, keep prediction and enter MISS_MATCH instead of reset.");
                if (tracker_state == TRACKING || tracker_state == DETECTING) {
                    tracker_state = MISS_MATCH;
                }
            }
        }
    
    }if (armors_msg->armors.size() == 2 && tracked_armors_num != ArmorsNum::OUTPOST_3) {
        int matched_armor1 = matchArmor(armors_msg->armors[0], ekf_prediction);
        int matched_armor2 = matchArmor(armors_msg->armors[1], ekf_prediction);
        if (matched_armor1 == 0 || matched_armor2 == 0) {
            RCLCPP_WARN(
                rclcpp::get_logger("armor_tracker"),
                "Two-armors miss, keep prediction and enter MISS_MATCH instead of reset.");
            if (tracker_state == TRACKING || tracker_state == DETECTING) {
                tracker_state = MISS_MATCH;
            }
        }
        if (matched_armor1 != 0 && matched_armor2 != 0) {
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
                if (std::abs(A) < 1e-6) {
                    RCLCPP_WARN(
                        rclcpp::get_logger("tracker"),
                        "Skip two-armor update due to near-zero denominator (yaw_a=%f, yaw_b=%f)",
                        yaw_a, yaw_b);
                    return matched;
                }
                double r1 = (sin(yaw_b) * (xb - xa) - cos(yaw_b) * (yb - ya)) / A;
                double r2 = (sin(yaw_a) * (xb - xa) - cos(yaw_a) * (yb - ya)) / A;
                if (abs(r1 - target_state(R1)) > 0.1 || abs(r2 - target_state(R2)) > 0.1) {
                    measurement << p1.x, p1.y, p1.z, yaw_a, p2.x, p2.y, p2.z, yaw_b,
                        target_state(R1), target_state(R2);
                } else {
                    measurement << p1.x, p1.y, p1.z, yaw_a, p2.x, p2.y, p2.z, yaw_b, r1, r2;
                }

                target_state = ekf.updateTwo(measurement);
                limitTranslationVelocity(target_state);
                ekf.setState(target_state);
                RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
            } else {
                RCLCPP_ERROR(rclcpp::get_logger("tracker"), "2 armors are too close!");
                ekf = ekf_backup;
                    return matched;
            }
        }
    }
    if (tracked_armors_num == ArmorsNum::OUTPOST_3) {
        target_state(R1) = target_state(R2) = 0.275;  // 固定半径
        // 锁住前哨站的XYZ位置
        target_state(VXC) = 0.0;
        target_state(VYC) = 0.0;
        target_state(VZC) = 0.0;
        // 约束速度
        double outpost_yaw_speed_const = 4*M_PI/5;
        if (std::abs(std::abs(target_state(VYAW)) - outpost_yaw_speed_const) < 0.2 * M_PI) {
            target_state(VYAW) = target_state(VYAW) > 0 ? outpost_yaw_speed_const : -1*outpost_yaw_speed_const;
        }
        limitTranslationVelocity(target_state);
        ekf.setState(target_state);

    }
    // 防止半径扩散
    for (int r_idx : {R1, R2}) {
        target_state(r_idx) = fmax(target_state(r_idx), 0.2);
        target_state(r_idx) = fmin(target_state(r_idx), 0.3);  // hero: 0.4
    }
    // 防止yaw角度扩散
    if (tracked_armors_num == ArmorsNum::NORMAL_4) {
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
        if(tracker_state == DETECTING){
        target_state(VXC) = 0.0;
        target_state(VYC) = 0.0;
        target_state(VZC) = 0.0;}
        double yaw_average = (target_state(YAW1) + target_state(YAW2)) / 2;
        target_state(YAW1) = yaw_average - M_PI / double(tracked_armors_num);
        target_state(YAW2) = yaw_average + M_PI / double(tracked_armors_num);
        limitTranslationVelocity(target_state);
        ekf.setState(target_state);
    } else {
        // OUTPOST_3：不做两板“平均化/固定间隔”的逻辑（会破坏120°间隔）；只归一化到[-pi, pi]
        target_state(YAW1) = angles::normalize_angle(target_state(YAW1));
        target_state(YAW2) = angles::normalize_angle(target_state(YAW2));
        target_state(YAW3) = angles::normalize_angle(target_state(YAW3));
        ekf.setState(target_state);
    }

    if (!matched) {
        ekf = ekf_backup;
    }

    return matched;
}

void Tracker::updateState(
    bool matched, const rclcpp::Time & msg_time, double temp_lost_time, double lost_time_thres,
    int tracking_thres, double miss_match_time_thres, bool is_main_camera)
{
    if (matched) {
        last_update_time_ = msg_time;
        if (is_main_camera) {
            last_main_update_time_ = msg_time;
        }
    }

    double time_since_update = (msg_time - last_update_time_).seconds();
    if (time_since_update < 0) time_since_update = 0.0;

    switch (tracker_state) {
        case DETECTING:
            if (matched) {
                increaseDetectCount();
                if (getDetectCount() > tracking_thres) {
                    resetDetectCount();
                    tracker_state = TRACKING;
                }
            } else {
                resetDetectCount();
                if (time_since_update > lost_time_thres) {
                    tracker_state = LOST;
                } else if (time_since_update > temp_lost_time) {
                    tracker_state = TEMP_LOST;
                }
            }
            break;
        case TRACKING:
            if (matched) {
                resetDetectCount();
            } else if (time_since_update > lost_time_thres) {
                tracker_state = LOST;
            } else if (time_since_update > temp_lost_time) {
                tracker_state = TEMP_LOST;
            }
            break;
        case TEMP_LOST:
            if (matched) {
                tracker_state = TRACKING;
                resetDetectCount();
            } else if (time_since_update > lost_time_thres) {
                tracker_state = LOST;
            }
            break;
        case MISS_MATCH:
            if (matched) {
                tracker_state = TRACKING;
                resetDetectCount();
            } else if (time_since_update > miss_match_time_thres) {
                tracker_state = LOST;  // 较短时间未匹配上，直接转为LOST
            }
            break;
        case LOST:
            resetDetectCount();
            break;
    }
}
int Tracker::matchArmor(const Armor & armor, const Eigen::VectorXd & ekf_prediction)
{
    // Match armor with EKF prediction
    // Calculate the difference between the predicted position and the current armor position
    auto p = armor.pose.position;
    double yaw = orientationToYaw(armor.pose.orientation);
    Eigen::Vector3d position_vec(p.x, p.y, p.z);
    auto predicted_position = getArmorPositionFromState(ekf_prediction);

    // 前哨站3块板匹配
    if (tracked_armors_num == ArmorsNum::OUTPOST_3) {
        // 只用xy平面距离进行匹配
        double position_diff_1 = (predicted_position[0].head<2>() - position_vec.head<2>()).norm();
        double position_diff_2 = (predicted_position[1].head<2>() - position_vec.head<2>()).norm();
        double position_diff_3 = (predicted_position[2].head<2>() - position_vec.head<2>()).norm();  // 第三块板
        double yaw_diff_1 = calYawDiff(yaw, ekf_prediction(YAW1));
        double yaw_diff_2 = calYawDiff(yaw, ekf_prediction(YAW2));
        double yaw_diff_3 = calYawDiff(yaw, ekf_prediction(YAW3));  // 第三块板

        // 优先匹配最小差值
        double min_diff = std::min({yaw_diff_1, yaw_diff_2, yaw_diff_3}); //yaw_diff是我观测到的yaw和ekf预测的yaw的差值
        if (min_diff == yaw_diff_1 && yaw_diff_1 < max_match_yaw_diff_  //&& //yaw_diff推出他是1号板并且差值要小于差值的阈值
            //position_diff_1 < max_match_distance_
            ) {
            twoD_distance = fmin(armor.distance_to_image_center, twoD_distance);
            info_position_diff = fmin(info_position_diff, position_diff_1);
            info_yaw_diff = fmin(info_yaw_diff, yaw_diff_1);
            return 1;
        } else if (
            min_diff == yaw_diff_2 && yaw_diff_2 < max_match_yaw_diff_  //&&
            //position_diff_2 < max_match_distance_
            ) {
            twoD_distance = fmin(armor.distance_to_image_center, twoD_distance);
            info_position_diff = fmin(info_position_diff, position_diff_2);
            info_yaw_diff = fmin(info_yaw_diff, yaw_diff_2);
            return 2;
        } else if (
            min_diff == yaw_diff_3 && yaw_diff_3 < max_match_yaw_diff_  //&&
            //position_diff_3 < max_match_distance_
            ) {
            twoD_distance = fmin(armor.distance_to_image_center, twoD_distance);
            info_position_diff = fmin(info_position_diff, position_diff_3);
            info_yaw_diff = fmin(info_yaw_diff, yaw_diff_3);
            return 3;  // 第三块板匹配
        }else{
            return 0;
        }
    } else {
        // 原有4块板匹配逻辑（不变）
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
        RCLCPP_WARN(
            rclcpp::get_logger("tracker"),
            "Tracker %s match failed for armor 1: yaw_diff %f (thres %f), pos_diff %f (thres %f)",
            tracked_id.c_str(), yaw_diff_1, max_match_yaw_diff_, position_diff_1,
            max_match_distance_);
    } else {
        if (yaw_diff_2 < max_match_yaw_diff_ && position_diff_2 < max_match_distance_) {
            twoD_distance = fmin(armor.distance_to_image_center, twoD_distance);
            info_position_diff = fmin(info_position_diff, position_diff_2);
            info_yaw_diff = fmin(info_yaw_diff, yaw_diff_2);
            return 2;  // Matched armor2 found
        }
        RCLCPP_WARN(
            rclcpp::get_logger("tracker"),
            "Tracker %s match failed for armor 2: yaw_diff %f (thres %f), pos_diff %f (thres %f)",
            tracked_id.c_str(), yaw_diff_2, max_match_yaw_diff_, position_diff_2,
            max_match_distance_);
    }
    return 0;  // No matched armor found
    }
}


void Tracker::initEKF(const Armor & a)
{
    auto p = a.pose.position;
    double yaw = orientationToYaw(a.pose.orientation);
    if (yaw < 0) {
        target_state = Eigen::VectorXd::Zero(15);
        double r = 0.2765;
        double xc = p.x + r * cos(yaw);
        double yc = p.y + r * sin(yaw);
        target_state(XC) = xc, target_state(YC) = yc,
        target_state(ZC1) = target_state(ZC2) = target_state(ZC3) = p.z;
        target_state(YAW1) = yaw, target_state(R1) = r;
        target_state(YAW2) = yaw + 2 * M_PI / double(tracked_armors_num), target_state(R2) = r;
        target_state(VXC) = 0;
        target_state(VYC) = 0;
        target_state(VZC) = 0;
        target_state(VYAW) = 0;
        target_state(R3) = r;                                              // 第三块板半径
        target_state(YAW3) = yaw + 4 * M_PI / double(tracked_armors_num);  // 第三块板偏航角
        limitTranslationVelocity(target_state);
        ekf.setState(target_state);
    } else {
        target_state = Eigen::VectorXd::Zero(15);
        double r = 0.2765;
        double xc = p.x + r * cos(yaw);
        double yc = p.y + r * sin(yaw);
        target_state(XC) = xc, target_state(YC) = yc,
        target_state(ZC1) = target_state(ZC2) = target_state(ZC3) = p.z;
        target_state(VXC) = 0, target_state(VYC) = 0, target_state(VZC) = 0, target_state(VYAW) = 0;
        target_state(YAW2) = yaw, target_state(R2) = r;
        target_state(YAW1) = yaw - 2 * M_PI / double(tracked_armors_num), target_state(R1) = r;
        target_state(VXC) = 0;
        target_state(VYC) = 0;
        target_state(VZC) = 0;
        target_state(VYAW) = 0;
        target_state(R3) = r;                                              // 第三块板半径
        target_state(YAW3) = yaw - 4 * M_PI / double(tracked_armors_num);  // 第三块板偏航角
        limitTranslationVelocity(target_state);
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

    target_state = Eigen::VectorXd::Zero(15);
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
    target_state(ZC3) = (za + zb) / 2;  // 第三块板z坐标（取均值，有问题）
    target_state(R3) = 0.2765;          // 第三块板半径
    target_state(YAW3) = yaw_b + 2 * M_PI / double(tracked_armors_num);  // 第三块板偏航角
    limitTranslationVelocity(target_state);
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
    if (abs(yaw - yaw_target) >
        ((int(tracked_armors_num) == 4) ? M_PI / 2 : M_PI / double(tracked_armors_num))) {
        Eigen::Vector3d center(target_state(XC), target_state(YC), target_state(ZC1));
        double r;
        if (yaw_target == target_state(YAW1))
            r = target_state(R1);
        else if (yaw_target == target_state(YAW2))
            r = target_state(R2);
        else
            r = target_state(R3);  // 新增：第三块板半径
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
    std::vector<Eigen::Vector3d> armor_position;
    double xc = x(XC), yc = x(YC);
    // 前哨站3块板
    if (tracked_armors_num == ArmorsNum::OUTPOST_3) {
        double za1 = x(ZC1), za2 = x(ZC2), za3 = x(ZC3);
        double yaw1 = x(YAW1), yaw2 = x(YAW2), yaw3 = x(YAW3);
        double r1 = x(R1), r2 = x(R2), r3 = x(R3);
        // 计算3块板位置
        armor_position.push_back(Eigen::Vector3d(xc - r1 * cos(yaw1), yc - r1 * sin(yaw1), za1));
        armor_position.push_back(Eigen::Vector3d(xc - r2 * cos(yaw2), yc - r2 * sin(yaw2), za2));
        armor_position.push_back(Eigen::Vector3d(xc - r3 * cos(yaw3), yc - r3 * sin(yaw3), za3));
    } else {
        // 原有4块板逻辑（不变）
        double za1 = x(ZC1), za2 = x(ZC2);
        double yaw1 = x(YAW1), yaw2 = x(YAW2);
        double r1 = x(R1), r2 = x(R2);
        armor_position.push_back(Eigen::Vector3d(xc - r1 * cos(yaw1), yc - r1 * sin(yaw1), za1));
        armor_position.push_back(Eigen::Vector3d(xc - r2 * cos(yaw2), yc - r2 * sin(yaw2), za2));
        armor_position.push_back(Eigen::Vector3d(xc + r1 * cos(yaw1), yc + r1 * sin(yaw1), za1));
        armor_position.push_back(Eigen::Vector3d(xc + r2 * cos(yaw2), yc + r2 * sin(yaw2), za2));
    }
    return armor_position;
}

void Tracker::limitTranslationVelocity(Eigen::VectorXd & state) const
{
    if (state.size() <= VZC) {
        return;
    }
    if (!std::isfinite(max_translation_speed_) || max_translation_speed_ <= 0.0) {
        return;
    }

    Eigen::Vector3d v(state(VXC), state(VYC), state(VZC));
    double speed = v.norm();
    if (!std::isfinite(speed)) {
        state(VXC) = 0.0;
        state(VYC) = 0.0;
        state(VZC) = 0.0;
        return;
    }

    if (speed > max_translation_speed_) {
        double scale = max_translation_speed_ / speed;
        state(VXC) *= scale;
        state(VYC) *= scale;
        state(VZC) *= scale;
    }
}

double Tracker::calYawDiff(double yaw1, double yaw2)
{
    double diff = std::min(
        abs(angles::shortest_angular_distance(yaw1, yaw2)),
        abs(angles::shortest_angular_distance(yaw1 + M_PI, yaw2)));
    return diff;
}

}  // namespace rm_auto_aim
