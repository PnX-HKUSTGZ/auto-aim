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
#include <array>
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
  target_state(Eigen::VectorXd::Zero(19)),
  last_update_time_(0.0),
  max_match_distance_(max_match_distance),
  max_match_yaw_diff_(max_match_yaw_diff)
{
}

int Tracker::inferOutpostArmorIdByHeight(double meas_z, const Eigen::VectorXd & ekf_prediction) const
{
    // 以“中板(编号1)”为锚点，利用高度差区分 2/3 号，并结合VYAW正负判顺/逆时针
    // 约定：
    // - 顺时针：高度序列(按编号1/2/3)为 1-2-3 (即2低3高)
    // - 逆时针：高度序列(按编号1/2/3)为 1-3-2 (即2高3低)
    // 注意：顺/逆时针与VYAW正负的对应关系由 outpost_vyaw_positive_is_ccw_ 控制

    // 需要中板缓存新鲜
    const auto & mid = outpost_z_cache_[0];
    if (!mid.valid) {
        return 0;
    }
    if ((current_msg_time_ - mid.stamp).seconds() > outpost_z_cache_window_sec_) {
        return 0;
    }

    const double dz = meas_z - mid.z;
    if (std::abs(dz) <= outpost_height_tol_) {
        return 1;
    }

    const double vyaw = ekf_prediction(VYAW);
    if (std::abs(vyaw) < outpost_vyaw_deadband_) {
        // 方向不可靠时，不强行区分 2/3
        return 0;
    }

    bool is_ccw = vyaw > 0.0;
    if (!outpost_vyaw_positive_is_ccw_) {
        is_ccw = !is_ccw;
    }

    if (is_ccw) {
        // 逆时针：2高3低
        return dz > 0.0 ? 2 : 3;
    }
    // 顺时针：3高2低
    return dz > 0.0 ? 3 : 2;
}

bool Tracker::getOutpostArmorZ(size_t armor_index, const rclcpp::Time & now, double & z) const
{
    if (armor_index >= outpost_z_cache_.size()) {
        return false;
    }
    const auto & c = outpost_z_cache_[armor_index];
    if (!c.valid) {
        return false;
    }
    if ((now - c.stamp).seconds() > outpost_z_cache_window_sec_) {
        return false;
    }
    z = c.z;
    return true;
}

void Tracker::mergeOutpostZCache(const rclcpp::Time & now)
{
    // 对缓存里的z做“近似相等去重”：差值<=outpost_z_merge_tol_ 的槽位视为同一块装甲板
    // 策略：同一簇只保留一个槽位（优先保留0号槽位；否则保留时间戳最新的），其它槽位置为invalid
    // 目的：当匹配ID偶发跳变导致同一块板被写进不同槽位时，避免两个/三个槽位长期占据相同高度
    auto is_fresh = [&](const OutpostZCache & c) {
        return c.valid && (now - c.stamp).seconds() <= outpost_z_cache_window_sec_;
    };

    int parent[3] = {0, 1, 2};
    auto find_root = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](int a, int b) {
        int ra = find_root(a);
        int rb = find_root(b);
        if (ra != rb) {
            parent[rb] = ra;
        }
    };

    for (int i = 0; i < 3; ++i) {
        if (!is_fresh(outpost_z_cache_[i])) {
            continue;
        }
        for (int j = i + 1; j < 3; ++j) {
            if (!is_fresh(outpost_z_cache_[j])) {
                continue;
            }
            if (std::abs(outpost_z_cache_[i].z - outpost_z_cache_[j].z) <= outpost_z_merge_tol_) {
                unite(i, j);
            }
        }
    }

    // 收集每个簇的信息
    double sum[3] = {0.0, 0.0, 0.0};
    int cnt[3] = {0, 0, 0};
    rclcpp::Time newest_stamp[3] = {
        rclcpp::Time(0, 0, RCL_ROS_TIME),
        rclcpp::Time(0, 0, RCL_ROS_TIME),
        rclcpp::Time(0, 0, RCL_ROS_TIME),
    };
    bool has_slot0[3] = {false, false, false};

    for (int i = 0; i < 3; ++i) {
        if (!is_fresh(outpost_z_cache_[i])) {
            continue;
        }
        const int r = find_root(i);
        sum[r] += outpost_z_cache_[i].z;
        cnt[r] += 1;
        if (cnt[r] == 1 || outpost_z_cache_[i].stamp > newest_stamp[r]) {
            newest_stamp[r] = outpost_z_cache_[i].stamp;
        }
        if (i == 0) {
            has_slot0[r] = true;
        }
    }

    // 对每个簇执行去重：保留一个槽位，其它置为invalid
    for (int r = 0; r < 3; ++r) {
        if (cnt[r] < 2) {
            continue;
        }

        const double mean_z = sum[r] / static_cast<double>(cnt[r]);

        int keep_idx = -1;
        if (has_slot0[r]) {
            keep_idx = 0;
        } else {
            // 保留时间戳最新的槽位
            rclcpp::Time best_stamp(0, 0, RCL_ROS_TIME);
            for (int i = 0; i < 3; ++i) {
                if (!is_fresh(outpost_z_cache_[i])) {
                    continue;
                }
                if (find_root(i) != r) {
                    continue;
                }
                if (keep_idx < 0 || outpost_z_cache_[i].stamp > best_stamp) {
                    keep_idx = i;
                    best_stamp = outpost_z_cache_[i].stamp;
                }
            }
        }

        if (keep_idx < 0) {
            continue;
        }

        for (int i = 0; i < 3; ++i) {
            if (!is_fresh(outpost_z_cache_[i])) {
                continue;
            }
            if (find_root(i) != r) {
                continue;
            }
            if (i == keep_idx) {
                outpost_z_cache_[i].z = mean_z;
                outpost_z_cache_[i].stamp = newest_stamp[r];
                outpost_z_cache_[i].valid = true;
            } else {
                outpost_z_cache_[i].valid = false;
            }
        }
    }
}

//初始化追踪器
void Tracker::init(const Armors::SharedPtr & armors_msg)
{
    if (armors_msg->armors.empty()) {
        return;
    }
    if (armors_msg->armors[0].number == "outpost") {
        for (auto & c : outpost_z_cache_) {
            c.valid = false;
        }
        // 单块板初始化：填充实测坐标+默认高差/半径
        tracked_armor = armors_msg->armors[0];
        twoD_distance = tracked_armor.distance_to_image_center;
        auto p = tracked_armor.pose.position;
        double yaw = orientationToYaw(tracked_armor.pose.orientation);
        // 19维状态初始化
        target_state.setZero();
        double r_outpost = 0.2765;  // 前哨站固定半径
        target_state(XC) = p.x + r_outpost * cos(yaw);
        target_state(YC) = p.y + r_outpost * sin(yaw);
        target_state(ZC1) = p.z;
        target_state(ZC2) = p.z + 0.102;  // 默认高差Δz1=0.5
        target_state(ZC3) = p.z - 0.102;  // 默认高差Δz2=-0.5
        target_state(R1) = target_state(R2) = target_state(R3) = r_outpost;
        target_state(YAW1) = yaw;
        target_state(YAW2) = yaw + 2 * M_PI / 3;  // 3块板间隔120°
        target_state(YAW3) = yaw + 4 * M_PI / 3;
        target_state(DZ1) = 0.102;
        target_state(DZ2) = -0.204;
        target_state(DZ3) = 0.102;
        target_state(R_OUTPOST) = r_outpost;
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
    return;
}

void Tracker::update(const Armors::SharedPtr & armors_msg)
//根据经过EKF加权后的观测和预测来更新装甲板的追踪状态
{
    current_msg_time_ = rclcpp::Time(armors_msg->header.stamp);
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

    // 前哨站专属更新逻辑（3块装甲板）
    if (tracked_armors_num == ArmorsNum::OUTPOST_3) {
        const rclcpp::Time msg_time(armors_msg->header.stamp);

        auto make_outpost_three_z_from_anchor =
            [&](int anchor_id, double anchor_z, const Eigen::VectorXd & prediction) {
                // dz1 = z2 - z1, dz2 = z3 - z2
                double dz1 = prediction(DZ1);
                double dz2 = prediction(DZ2);
                // Fallback to init defaults if dz becomes invalid
                if (!std::isfinite(dz1) || std::abs(dz1) < 1e-6) {
                    dz1 = 0.102;
                }
                if (!std::isfinite(dz2) || std::abs(dz2) < 1e-6) {
                    dz2 = -0.204;
                }

                double z1 = prediction(ZC1);
                double z2 = prediction(ZC2);
                double z3 = prediction(ZC3);
                if (anchor_id == 1) {
                    z1 = anchor_z;
                    z2 = z1 + dz1;
                    z3 = z2 + dz2;
                } else if (anchor_id == 2) {
                    z2 = anchor_z;
                    z1 = z2 - dz1;
                    z3 = z2 + dz2;
                } else {  // 3
                    z3 = anchor_z;
                    z2 = z3 - dz2;
                    z1 = z2 - dz1;
                }
                return std::array<double, 3>{z1, z2, z3};
            };

        auto make_outpost_three_meas_from_anchor = [&](
                                                          int anchor_id,
                                                          const geometry_msgs::msg::Point & anchor_p,
                                                          double anchor_yaw,
                                                          const Eigen::VectorXd & prediction) {
            Eigen::VectorXd z(16);
            const auto pred_pos = getArmorPositionFromState(prediction);
            const auto z123 = make_outpost_three_z_from_anchor(anchor_id, anchor_p.z, prediction);

            // Start with prediction-consistent x/y/yaw to make residual=0 for non-observed parts.
            double x1 = pred_pos[0].x(), y1 = pred_pos[0].y(), yaw1 = prediction(YAW1);
            double x2 = pred_pos[1].x(), y2 = pred_pos[1].y(), yaw2 = prediction(YAW2);
            double x3 = pred_pos[2].x(), y3 = pred_pos[2].y(), yaw3 = prediction(YAW3);

            if (anchor_id == 1) {
                x1 = anchor_p.x;
                y1 = anchor_p.y;
                yaw1 = anchor_yaw;
            } else if (anchor_id == 2) {
                x2 = anchor_p.x;
                y2 = anchor_p.y;
                yaw2 = anchor_yaw;
            } else {
                x3 = anchor_p.x;
                y3 = anchor_p.y;
                yaw3 = anchor_yaw;
            }

            z << x1, y1, z123[0], yaw1, x2, y2, z123[1], yaw2, x3, y3, z123[2], yaw3,
                0.2765, 0.2765, 0.2765, 0.2765;
            return z;
        };

        // 0.2s节流输出：outpost_z_cache_ 三个槽位的z缓存
        if ((msg_time - last_outpost_cache_log_time_).seconds() >= 0.2) {
            last_outpost_cache_log_time_ = msg_time;

            const auto & c0 = outpost_z_cache_[0];
            const auto & c1 = outpost_z_cache_[1];
            const auto & c2 = outpost_z_cache_[2];

            const double z0 = c0.valid ? c0.z : NAN;
            const double z1 = c1.valid ? c1.z : NAN;
            const double z2 = c2.valid ? c2.z : NAN;

            const double age0 = c0.valid ? (msg_time - c0.stamp).seconds() : -1.0;
            const double age1 = c1.valid ? (msg_time - c1.stamp).seconds() : -1.0;
            const double age2 = c2.valid ? (msg_time - c2.stamp).seconds() : -1.0;

            RCLCPP_INFO(
                rclcpp::get_logger("armor_tracker"),
                "outpost_z_cache_: z=[%.3f, %.3f, %.3f] valid=[%d,%d,%d] age=[%.3f,%.3f,%.3f] window=%.3f",
                z0, z1, z2, static_cast<int>(c0.valid), static_cast<int>(c1.valid),
                static_cast<int>(c2.valid), age0, age1, age2, outpost_z_cache_window_sec_);
        }

        // 单块板：尝试匹配到 1/2/3 号板，并更新对应槽位的z缓存
        // 目的：当视觉大多数时间只看到一块板时，不同装甲板轮流出现也能逐步填满三个槽位缓存
        if (armors_msg->armors.size() == 1) {
            const auto & armor = armors_msg->armors[0];
            auto p = armor.pose.position;

            int matched_id = matchArmor(armor, ekf_prediction);
            if (matched_id <= 0 || matched_id > 3) {
                // 回退：无法可靠匹配时，沿用旧逻辑把它当作1号
                matched_id = 1;
            }

            // 更新对应槽位z缓存（不参与EKF更新）
            {
                auto & cache = outpost_z_cache_[matched_id - 1];
                const bool was_valid = cache.valid;
                cache.valid = true;
                if (was_valid && std::abs(cache.z - p.z) <= outpost_z_merge_tol_) {
                    cache.z = 0.5 * (cache.z + p.z);
                } else {
                    cache.z = p.z;
                }
                cache.stamp = msg_time;
            }

            // 用预测的对应YAW作为目标角，保证Yaw展开连续
            const double yaw_target =
                matched_id == 1 ? ekf_prediction(YAW1)
                                : (matched_id == 2 ? ekf_prediction(YAW2) : ekf_prediction(YAW3));
            double measured_yaw = orientationToYaw(armor.pose.orientation, p, yaw_target);
            measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);

            // Outpost: synchronize z-observations for the other two armors using dz.
            // Non-observed armors use prediction-consistent x/y/yaw, so only z introduces residual.
            {
                const Eigen::VectorXd z_three =
                    make_outpost_three_meas_from_anchor(matched_id, p, measured_yaw, ekf_prediction);
                target_state = ekf.updateThreeArmors(z_three);
            }
            matched = true;

            mergeOutpostZCache(msg_time);

            // 前哨站强制约束
            target_state(R1) = target_state(R2) = target_state(R3) = 0.2765;
            ekf.setState(target_state);
        } else {//基本用不到，因为前哨站大多数时间只看到一块板，有待完善
        bool has_match[3] = {false, false, false};
        geometry_msgs::msg::Point matched_pos[3];
        double matched_yaw[3] = {0.0, 0.0, 0.0};
        // 匹配3块装甲板中的任意一块
        for (const auto & armor : armors_msg->armors) {
            int matched_id = matchArmor(armor, ekf_prediction);
            if (matched_id > 0 && matched_id <= 3 && !has_match[matched_id - 1]) {//该板没被占用才接受观测
                auto p = armor.pose.position;
                double measured_yaw = orientationToYaw(
                    armor.pose.orientation, p,
                    matched_id == 1
                        ? ekf_prediction(YAW1)
                        : (matched_id == 2 ? ekf_prediction(YAW2) : ekf_prediction(YAW3)));
                has_match[matched_id - 1] = true;
                matched_pos[matched_id - 1] = p;
                matched_yaw[matched_id - 1] = measured_yaw;

                // 仅用于可视化/输出的z缓存（不参与EKF更新）
                auto & cache = outpost_z_cache_[matched_id - 1];
                const bool was_valid = cache.valid;
                cache.valid = true;
                if (was_valid && std::abs(cache.z - p.z) <= outpost_z_merge_tol_) {
                    cache.z = 0.5 * (cache.z + p.z);
                } else {
                    cache.z = p.z;
                }
                cache.stamp = msg_time;
            }
        }

        mergeOutpostZCache(msg_time);

        if (has_match[0] && has_match[1] && has_match[2]) {
            Eigen::VectorXd z(16);
            z << matched_pos[0].x, matched_pos[0].y, matched_pos[0].z, matched_yaw[0],
                matched_pos[1].x, matched_pos[1].y, matched_pos[1].z, matched_yaw[1],
                matched_pos[2].x, matched_pos[2].y, matched_pos[2].z, matched_yaw[2],
                0.2765, 0.2765, 0.2765, 0.2765;
            target_state = ekf.updateThreeArmors(z);
            matched = true;
        } else {
            int anchor_id = 0;
            for (int idx = 0; idx < 3; ++idx) {
                if (has_match[idx]) {
                    anchor_id = idx + 1;
                    break;
                }
            }

            if (anchor_id > 0) {
                const auto & ap = matched_pos[anchor_id - 1];
                const double ay = matched_yaw[anchor_id - 1];
                measurement = Eigen::Vector4d(ap.x, ap.y, ap.z, ay);

                const Eigen::VectorXd z_three =
                    make_outpost_three_meas_from_anchor(anchor_id, ap, ay, ekf_prediction);
                target_state = ekf.updateThreeArmors(z_three);
                matched = true;
            }
        }
        // 前哨站强制约束
        target_state(R1) = target_state(R2) = target_state(R3) = 0.2765;  // 固定半径 
        // 约束速度
        // if (std::abs(std::abs(target_state(VYAW)) - 0.8 * M_PI) < 0.2 * M_PI) {
        //     target_state(VYAW) = target_state(VYAW) > 0 ? 0.8 * M_PI : -0.8 * M_PI;
        // }
        ekf.setState(target_state);
        }
    } else {
        // 原有4块板更新逻辑（不变）
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
                RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Reset tracker by single armor!");
                init(armors_msg);
                return;
            }
        }
        if (armors_msg->armors.size() == 2) {
            int matched_armor1 = matchArmor(armors_msg->armors[0], ekf_prediction);
            int matched_armor2 = matchArmor(armors_msg->armors[1], ekf_prediction);
            if (matched_armor1 == 0 || matched_armor2 == 0) {
                RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "Reset tracker by two armors!");
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
                    measurement << p1.x, p1.y, p1.z, yaw_a, p2.x, p2.y, p2.z, yaw_b,
                        target_state(R1), target_state(R2);
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
        // 原有约束逻辑（不变）
        for (int r_idx : {R1, R2}) {
            target_state(r_idx) = fmax(target_state(r_idx), 0.2);
            target_state(r_idx) = fmin(target_state(r_idx), 0.3);
        }
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
    }

    // 状态机逻辑（不变）
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
    auto predicted_position = getArmorPositionFromState(ekf_prediction);

    // 前哨站3块板匹配
    if (tracked_armors_num == ArmorsNum::OUTPOST_3) {
        const int expected_id = inferOutpostArmorIdByHeight(p.z, ekf_prediction);

        const double yaw_diff[3] = {
            calYawDiff(yaw, ekf_prediction(YAW1)),
            calYawDiff(yaw, ekf_prediction(YAW2)),
            calYawDiff(yaw, ekf_prediction(YAW3)),
        };

        // 使用XY距离进行门限（对高度初始化偏差更鲁棒），并使用跨帧z缓存作为编号判定的一部分
        const Eigen::Vector2d meas_xy(p.x, p.y);
        double pos_diff_3d[3] = {0.0, 0.0, 0.0};
        double pos_diff_xy[3] = {0.0, 0.0, 0.0};

        for (int i = 0; i < 3; ++i) {
            pos_diff_3d[i] = (predicted_position[i] - position_vec).norm();
            const Eigen::Vector2d pred_xy(predicted_position[i].x(), predicted_position[i].y());
            pos_diff_xy[i] = (pred_xy - meas_xy).norm();
        }

        auto cache_fresh = [&](int idx) {
            const auto & c = outpost_z_cache_[static_cast<size_t>(idx)];
            return c.valid && (current_msg_time_ - c.stamp).seconds() <= outpost_z_cache_window_sec_;
        };

        auto cache_z_diff = [&](int idx) {
            const auto & c = outpost_z_cache_[static_cast<size_t>(idx)];
            return std::abs(c.z - p.z);
        };

        int best_idx = -1;
        double best_score = 1e9;
        for (int i = 0; i < 3; ++i) {
            if (yaw_diff[i] >= max_match_yaw_diff_) {
                continue;
            }
            if (pos_diff_xy[i] >= max_match_distance_) {
                continue;
            }

            // score单位对齐：yaw(弧度) + 5*xy(米) + 5*z_cache(米)
            double score = yaw_diff[i] + 2.5 * pos_diff_xy[i];
            if (cache_fresh(i)) {
                score += 5.0 * cache_z_diff(i);
            }

            // 高度推断强约束：当能从“中板缓存+VYAW方向”推断编号时，给不符合的槽位加惩罚
            if (expected_id > 0 && (i != expected_id - 1)) {
                score += outpost_height_mismatch_penalty_;
            }

            if (score < best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        if (best_idx >= 0) {
            twoD_distance = fmin(armor.distance_to_image_center, twoD_distance);
            info_position_diff = fmin(info_position_diff, pos_diff_3d[best_idx]);
            info_yaw_diff = fmin(info_yaw_diff, yaw_diff[best_idx]);
            return best_idx + 1;
        }

        return 0;
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
}

void Tracker::initEKF(const Armor & a)
{
    auto p = a.pose.position;
    double yaw = orientationToYaw(a.pose.orientation);
    if (yaw < 0) {
        target_state = Eigen::VectorXd::Zero(19);
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
        target_state(DZ1) = 0.0;                                           // 高差1
        target_state(DZ2) = 0.0;                                           // 高差2
        target_state(DZ3) = 0.0;                                           // 高差3
        target_state(R3) = r;                                              // 第三块板半径
        target_state(YAW3) = yaw + 4 * M_PI / double(tracked_armors_num);  // 第三块板偏航角
        target_state(R_OUTPOST) = 0.2765;                                  // 前哨站半径（默认值）
        ekf.setState(target_state);
    } else {
        target_state = Eigen::VectorXd::Zero(19);
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
        target_state(DZ1) = 0.0;                                           // 高差1
        target_state(DZ2) = 0.0;                                           // 高差2
        target_state(DZ3) = 0.0;                                           // 高差3
        target_state(R3) = r;                                              // 第三块板半径
        target_state(YAW3) = yaw - 4 * M_PI / double(tracked_armors_num);  // 第三块板偏航角
        target_state(R_OUTPOST) = 0.2765;
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

    target_state = Eigen::VectorXd::Zero(19);
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
    target_state(DZ1) = 0.05;           // 高差1（有问题）
    target_state(DZ2) = -0.05;          // 高差2
    target_state(DZ3) = 0.0;            // 高差3
    target_state(R3) = 0.2765;          // 第三块板半径
    target_state(YAW3) = yaw_b + 2 * M_PI / double(tracked_armors_num);  // 第三块板偏航角
    target_state(R_OUTPOST) = 0.2765;                                    // 前哨站半径
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
double Tracker::calYawDiff(double yaw1, double yaw2)
{
    double diff = std::min(
        abs(angles::shortest_angular_distance(yaw1, yaw2)),
        abs(angles::shortest_angular_distance(yaw1 + 4 * M_PI / double(tracked_armors_num), yaw2)));
    return diff;
}

}  // namespace rm_auto_aim
