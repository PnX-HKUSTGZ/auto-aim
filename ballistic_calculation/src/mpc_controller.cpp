#include "ballistic_calculation/mpc_controller.hpp"
#include <Eigen/src/Core/Matrix.h>

#include <auto_aim_interfaces/msg/detail/firecontrol__struct.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <opencv2/opencv.hpp>

#include "ballistic_calculation/math_util.hpp"
#include "yaml-cpp/yaml.h"
#include "rclcpp/rclcpp.hpp"

namespace rm_auto_aim
{
namespace
{
static rclcpp::Clock g_mpc_log_clock(RCL_ROS_TIME);
}

MPCController::MPCController(rclcpp::Node *node)
{
    
    fire_delay = node->declare_parameter("fire_delay", 0.0);
    iffire_ = node->declare_parameter("ifFireK", 0.05);
    max_switch_speed_ = node->declare_parameter("mpc_max_switch_speed", 30.0);
    node->get_parameter("air_resistence", air_resistence_);
    setupYawSolver(node);
    setupPitchSolver(node);
}

MPCController::~MPCController() {
    if (yaw_solver_) free(yaw_solver_);
    if (pitch_solver_) free(pitch_solver_);
}

double MPCController::nowSeconds() const
{
    // 与目标消息时间源保持一致，避免 time source mismatch
    static rclcpp::Clock clock(RCL_ROS_TIME);
    return clock.now().seconds();
}

void MPCController::pruneCache(double min_stamp)
{
    while (!history_.empty() && history_.front().stamp < min_stamp) {
        history_.pop_front();
    }
}

void MPCController::storeOptimizedState(const CachedState & state)
{
    if (!std::isfinite(state.stamp)) return;
    // 保持时间递增的队列
    if (!history_.empty() && state.stamp < history_.back().stamp - 1e-6) {
        // 如果时间回拨，直接忽略，避免乱序
        return;
    }
    history_.push_back(state);
    if (history_.size() > MAX_CACHE_SIZE) {
        history_.pop_front();
    }
}

std::optional<MPCController::CachedState> MPCController::queryLatest(double stamp) const
{
    if (history_.empty()) return std::nullopt;
    for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
        if (it->stamp <= stamp + 1e-6) {
            return *it;
        }
    }
    return std::nullopt;
}

MPCResult MPCController::compute(
    const auto_aim_interfaces::msg::Target & target_msg,
    double bullet_speed, double fly_time)
{
    MPCResult result;
    result.is_valid = false;
    result.is_fire = false;


    if (bullet_speed < 10 || bullet_speed > 35 || DT <= 1e-6) {
        bullet_speed = 22; // Fallback default
        RCLCPP_WARN(rclcpp::get_logger("MPCController"), "Invalid bullet speed, using default 22m/s");
    }
    Trajectory traj;
    
    try {
        traj = getTrajectory(target_msg, bullet_speed, fly_time);
    } catch (const std::exception & e) {
        RCLCPP_ERROR(rclcpp::get_logger("MPCController"), "Trajectory generation failed: %s", e.what());
        return result;
    }

    // Solve Yaw
    Eigen::VectorXd x0_yaw(2);
    x0_yaw << traj(0, 0), 0.0;
    if (x0_yaw.hasNaN() || x0_yaw.cwiseAbs().maxCoeff() > 1e6) {
        RCLCPP_ERROR(rclcpp::get_logger("MPCController"), "Invalid yaw initial state");
        return result;
    }
    tiny_set_x0(yaw_solver_, x0_yaw);

    Eigen::MatrixXd yaw_ref = traj.block(0, 0, 2, HORIZON);
    if (yaw_ref.hasNaN()) {
        RCLCPP_ERROR(rclcpp::get_logger("MPCController"), "NaN in yaw reference");
        return result;
    }
    yaw_solver_->work->Xref = yaw_ref;
    
    int solve_ret_yaw = tiny_solve(yaw_solver_);
    if (solve_ret_yaw != 0) {
        RCLCPP_WARN_THROTTLE(
            rclcpp::get_logger("MPCController"), g_mpc_log_clock, 2000,
            "Yaw MPC solve failed: %d", solve_ret_yaw);
    }

    // Solve Pitch
    Eigen::VectorXd x0_pitch(2);
    x0_pitch << traj(2, 0),  0.0;
    
    if (x0_pitch.hasNaN() || x0_pitch.cwiseAbs().maxCoeff() > 1e6) {
        RCLCPP_ERROR(rclcpp::get_logger("MPCController"), "Invalid pitch initial state");
        return result;
    }
    tiny_set_x0(pitch_solver_, x0_pitch);

    Eigen::MatrixXd pitch_ref = traj.block(2, 0, 2, HORIZON);
    if (pitch_ref.hasNaN()) {
         RCLCPP_ERROR(rclcpp::get_logger("MPCController"), "NaN in pitch reference");
         return result;
    }
    pitch_solver_->work->Xref = pitch_ref;

    // Solve Pitch
    int solve_ret_pitch = tiny_solve(pitch_solver_);
    if (solve_ret_pitch != 0) {
        RCLCPP_WARN_THROTTLE(
            rclcpp::get_logger("MPCController"), g_mpc_log_clock, 2000,
            "Pitch MPC solve failed: %d", solve_ret_pitch);
    }
    
    // Extract MPC results
    const int fire_delay_idx = static_cast<int>(fire_delay / DT);
    const int fire_idx = HALF_HORIZON + fire_delay_idx;
    const int aim_idx = HALF_HORIZON;    

    // 当前控制指令（发给云台执行，无fire_delay）
    double yaw_current = limit_rad(yaw_solver_->work->x.coeff(0, aim_idx));
    double yaw_vel_current = yaw_solver_->work->x.coeff(1, aim_idx);
    double yaw_acc_current = yaw_solver_->work->u.coeff(0, aim_idx);
    
    double pitch_current = limit_rad(pitch_solver_->work->x.coeff(0, aim_idx));
    double pitch_vel_current = pitch_solver_->work->x.coeff(1, aim_idx);
    double pitch_acc_current = pitch_solver_->work->u.coeff(0, aim_idx);

    // 开火时刻目标值（带fire_delay，用于开火判定）
    double yaw_at_fire = limit_rad(yaw_solver_->work->x.coeff(0, fire_idx));
    // double yaw_vel_at_fire = yaw_solver_->work->x.coeff(1, fire_idx);
    // double yaw_acc_at_fire = yaw_solver_->work->u.coeff(0, fire_idx);
    
    double pitch_at_fire = limit_rad(pitch_solver_->work->x.coeff(0, fire_idx));
    // double pitch_vel_at_fire = pitch_solver_->work->x.coeff(1, fire_idx);
    // double pitch_acc_at_fire = pitch_solver_->work->u.coeff(0, fire_idx);

    // 目标轨迹值
    double target_yaw_current = limit_rad(traj.coeff(0, aim_idx));
    double target_pitch_current = limit_rad(traj.coeff(2, aim_idx));
    double target_yaw_at_fire = limit_rad(traj.coeff(0, fire_idx));
    double target_pitch_at_fire = limit_rad(traj.coeff(2, fire_idx));

    result.is_valid = true;

    // 控制限幅和死区处理
    const double MAX_VEL = 5.0;  
    const double MAX_ACC = 8.0; 
    const double DEAD_ZONE_VEL = 0.01;
    const double DEAD_ZONE_ACC = 0.05;

    // 偏航角控制输出
    result.yaw = yaw_current;
    double yaw_vel_clamped = std::clamp(yaw_vel_current, -MAX_VEL, MAX_VEL);
    result.yaw_vel = std::abs(yaw_vel_clamped) < DEAD_ZONE_VEL ? 0.0 : yaw_vel_clamped;
    double yaw_acc_clamped = std::clamp(yaw_acc_current, -MAX_ACC, MAX_ACC);
    result.yaw_acc = std::abs(yaw_acc_clamped) < DEAD_ZONE_ACC ? 0.0 : yaw_acc_clamped;

    // 俯仰角控制输出
    result.pitch = pitch_current;
    double pitch_vel_clamped = std::clamp(pitch_vel_current, -MAX_VEL, MAX_VEL);
    result.pitch_vel = std::abs(pitch_vel_clamped) < DEAD_ZONE_VEL ? 0.0 : pitch_vel_clamped;
    double pitch_acc_clamped = std::clamp(pitch_acc_current, -MAX_ACC, MAX_ACC);
    result.pitch_acc = std::abs(pitch_acc_clamped) < DEAD_ZONE_ACC ? 0.0 : pitch_acc_clamped;

    // 检查NaN值
    if (std::isnan(result.yaw_vel) || std::isnan(result.pitch_vel)) {
        result.is_valid = false;
        return result;
    }

    // 开火判定逻辑：检查带fire_delay的目标角度和当前控制角度的偏差
    double yaw_error = std::abs(target_yaw_at_fire - yaw_at_fire);
    double pitch_error = std::abs(target_pitch_at_fire - pitch_at_fire);

    if (yaw_error <= iffire_ && pitch_error <= iffire_) {
        result.is_fire = true; // 处于非阶跃期
    }

    result.target_yaw = target_yaw_current;
    result.target_pitch = target_pitch_current;

    // 优化前
    Eigen::Vector3d target_pos_vec(target_msg.position.x, target_msg.position.y, target_msg.position.z);
    Eigen::Vector2d target_vel_vec(target_msg.velocity.x, target_msg.velocity.y);


    // Plot pitch reference (from traj) and optimized pitch (from solver) in a single image

    // try {
    //     if (HORIZON > 1 && pitch_solver_ && pitch_solver_->work) {
    //         const int img_w = 900;
    //         const int img_h = 360;
    //         const int margin = 40;
    //         cv::Mat img(img_h, img_w, CV_8UC3, cv::Scalar(255, 255, 255));

    //         std::vector<cv::Point> pts_ref;
    //         std::vector<cv::Point> pts_opt;

    //         double ang_min = -M_PI/4.0;
    //         double ang_max = M_PI/4.0;

    //         auto ang_to_y = [&](double ang)->int {
    //             double v = (ang - ang_min) / (ang_max - ang_min);
    //             v = std::clamp(v, 0.0, 1.0);
    //             return static_cast<int>((1.0 - v) * (img_h - 2 * margin) + margin);
    //         };

    //         for (int i = 0; i < HORIZON; ++i) {
    //             double pr = limit_rad(traj(2, i));
    //             double po = limit_rad(pitch_solver_->work->x(2, i));
    //             int xpix = static_cast<int>(margin + (double)i * (img_w - 2 * margin) / (HORIZON - 1));
    //             pts_ref.emplace_back(cv::Point(xpix, ang_to_y(pr)));
    //             pts_opt.emplace_back(cv::Point(xpix, ang_to_y(po)));
    //         }

    //         if (!pts_ref.empty()) cv::polylines(img, pts_ref, false, cv::Scalar(200, 50, 50), 2, cv::LINE_AA);
    //         if (!pts_opt.empty()) cv::polylines(img, pts_opt, false, cv::Scalar(50, 50, 200), 2, cv::LINE_AA);

    //         // Draw Y-axis ticks and labels (radian scale)
    //         std::vector<std::pair<double, std::string>> y_ticks = {
    //             {-M_PI/4.0, "-pi/4"}, {-M_PI/8.0, "-pi/8"}, {0.0, "0"}, {M_PI/8.0, "pi/8"}, {M_PI/4.0, "pi/4"}
    //         };
    //         for (const auto &tk : y_ticks) {
    //             int yy = ang_to_y(tk.first);
    //             cv::line(img, cv::Point(margin - 8, yy), cv::Point(margin, yy), cv::Scalar(80, 80, 80), 1);
    //             cv::putText(img, tk.second, cv::Point(2, yy + 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(80, 80, 80), 1);
    //         }

    //         // Draw X-axis ticks and labels (time in seconds relative to current)
    //         const int num_xticks = 5;
    //         for (int j = 0; j < num_xticks; ++j) {
    //             int idx = (j * (HORIZON - 1)) / (num_xticks - 1);
    //             int xpix = static_cast<int>(margin + (double)idx * (img_w - 2 * margin) / (HORIZON - 1));
    //             int ytick_top = img_h - margin;
    //             cv::line(img, cv::Point(xpix, ytick_top), cv::Point(xpix, ytick_top + 6), cv::Scalar(80, 80, 80), 1);
    //             double t_off = (idx - HALF_HORIZON) * DT; // seconds offset
    //             char buf[64];
    //             std::snprintf(buf, sizeof(buf), "%.2fs", t_off);
    //             cv::putText(img, buf, cv::Point(xpix - 20, img_h - 6), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(80, 80, 80), 1);
    //         }

    //         // mark current (HALF_HORIZON) column
    //         int curx = static_cast<int>(margin + (double)HALF_HORIZON * (img_w - 2 * margin) / (HORIZON - 1));
    //         cv::line(img, cv::Point(curx, margin), cv::Point(curx, img_h - margin), cv::Scalar(50, 180, 50), 1);

    //         cv::putText(img, "pitch_ref", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 50, 50), 2);
    //         cv::putText(img, "pitch_opt", cv::Point(120, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(50, 50, 200), 2);

    //         cv::imshow("pitch_traj", img);
    //         cv::waitKey(1);
    //     }
    // } catch (const std::exception & e) {
    //     RCLCPP_WARN(rclcpp::get_logger("MPCController"), "Plotting pitch trajectories failed: %s", e.what());
    // }

    // Plot yaw reference (from traj) and optimized yaw (from solver) in a single image
    // try {
    //     if (HORIZON > 1 && yaw_solver_ && yaw_solver_->work) {
    //         const int img_w = 900;
    //         const int img_h = 360;
    //         const int margin = 40;
    //         cv::Mat img(img_h, img_w, CV_8UC3, cv::Scalar(255, 255, 255));

    //         std::vector<cv::Point> pts_ref;
    //         std::vector<cv::Point> pts_opt;

    //         double ang_min = -M_PI/8.0;
    //         double ang_max = M_PI/8.0;

    //         auto ang_to_y = [&](double ang)->int {
    //             double v = (ang - ang_min) / (ang_max - ang_min);
    //             v = std::clamp(v, 0.0, 1.0);
    //             return static_cast<int>((1.0 - v) * (img_h - 2 * margin) + margin);
    //         };

    //         for (int i = 0; i < HORIZON; ++i) {
    //             double xr = limit_rad(traj(0, i));
    //             double xo = limit_rad(yaw_solver_->work->x(0, i));
    //             int xpix = static_cast<int>(margin + (double)i * (img_w - 2 * margin) / (HORIZON - 1));
    //             pts_ref.emplace_back(cv::Point(xpix, ang_to_y(xr)));
    //             pts_opt.emplace_back(cv::Point(xpix, ang_to_y(xo)));
    //         }

    //         if (!pts_ref.empty()) cv::polylines(img, pts_ref, false, cv::Scalar(200, 50, 50), 1, cv::LINE_AA);
    //         if (!pts_opt.empty()) cv::polylines(img, pts_opt, false, cv::Scalar(50, 50, 200), 1, cv::LINE_AA);

    //         // Draw Y-axis ticks and labels (radian scale)
    //         std::vector<std::pair<double, std::string>> y_ticks = {
    //             {-M_PI/4.0, "-pi/4"}, {-M_PI/8.0, "-pi/8"}, {0.0, "0"}, {M_PI/8.0, "pi/8"}, {M_PI/4.0, "pi/4"}
    //         };
    //         for (const auto &tk : y_ticks) {
    //             int yy = ang_to_y(tk.first);
    //             cv::line(img, cv::Point(margin - 8, yy), cv::Point(margin, yy), cv::Scalar(80, 80, 80), 1);
    //             cv::putText(img, tk.second, cv::Point(2, yy + 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(80, 80, 80), 1);
    //         }

    //         // Draw X-axis ticks and labels (time in seconds relative to current)
    //         const int num_xticks = 5;
    //         for (int j = 0; j < num_xticks; ++j) {
    //             int idx = (j * (HORIZON - 1)) / (num_xticks - 1);
    //             int xpix = static_cast<int>(margin + (double)idx * (img_w - 2 * margin) / (HORIZON - 1));
    //             int ytick_top = img_h - margin;
    //             cv::line(img, cv::Point(xpix, ytick_top), cv::Point(xpix, ytick_top + 6), cv::Scalar(80, 80, 80), 1);
    //             double t_off = (idx - HALF_HORIZON) * DT; // seconds offset
    //             char buf[64];
    //             std::snprintf(buf, sizeof(buf), "%.2fs", t_off);
    //             cv::putText(img, buf, cv::Point(xpix - 20, img_h - 6), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(80, 80, 80), 1);
    //         }

    //         // mark current (HALF_HORIZON) column
    //         int curx = static_cast<int>(margin + (double)HALF_HORIZON * (img_w - 2 * margin) / (HORIZON - 1));
    //         cv::line(img, cv::Point(curx, margin), cv::Point(curx, img_h - margin), cv::Scalar(50, 180, 50), 1);

    //         cv::putText(img, "yaw_ref", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 50, 50), 2);
    //         cv::putText(img, "yaw_opt", cv::Point(100, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(50, 50, 200), 2);

    //         cv::imshow("yaw_traj", img);
    //         cv::waitKey(1);
    //     }
    // } catch (const std::exception & e) {
    //     RCLCPP_WARN(rclcpp::get_logger("MPCController"), "Plotting yaw trajectories failed: %s", e.what());
    // }

    // 缓存 0 时刻（now + T）优化结果
    try {
        CachedState cached{};
        const double stamp = nowSeconds() + fly_time;
        cached.stamp = stamp;
        cached.yaw = yaw_current;
        cached.yaw_vel = yaw_vel_current;
        cached.yaw_acc = yaw_acc_current;
        cached.pitch = pitch_current;
        cached.pitch_vel = pitch_vel_current;
        cached.pitch_acc = pitch_acc_current;
        pruneCache(stamp - HALF_HORIZON * DT - DT);
        storeOptimizedState(cached);
    } catch (const std::exception & e) {
        RCLCPP_WARN(rclcpp::get_logger("MPCController"), "Cache store failed: %s", e.what());
    }

    return result;
}

// Compute odometry-based target position for a given time
Eigen::Vector3d MPCController::getOdomTarget(const auto_aim_interfaces::msg::Target & target_msg, double time)
{
    armor_selector_.updateTarget(target_msg);
    std::vector<double> init_armor;
    try{
        init_armor = armor_selector_.predictInfantryBestArmor(time, max_switch_speed_, max_switch_speed_, 5.0); 
    } catch (const std::exception & e) {
        RCLCPP_ERROR(rclcpp::get_logger("MPCController"), "Armor prediction failed: %s", e.what());
        throw;
    }
    if (init_armor.size() < 3) {
        RCLCPP_ERROR(rclcpp::get_logger("MPCController"), "predictInfantryBestArmor returned too few elements: %zu", init_armor.size());
        throw std::runtime_error("predictInfantryBestArmor returned insufficient data");
    }
    
    // std::cerr << "init_armor size: " << init_armor.size() << "\n";
    double r0 = init_armor[2];
    double z0 = init_armor[1];
    double theta0 = init_armor[0];

    return Eigen::Vector3d(
        target_msg.position.x + target_msg.velocity.x * time - r0 * std::cos(theta0 + target_msg.v_yaw * time),
        target_msg.position.y + target_msg.velocity.y * time - r0 * std::sin(theta0 + target_msg.v_yaw * time),
        z0 + target_msg.velocity.z * time
    );
}

Trajectory MPCController::getTrajectory(
    const auto_aim_interfaces::msg::Target & target_msg, 
    double bullet_speed, double fly_time)
{
    Trajectory traj;
    traj.setZero();
    
    Eigen::Vector3d current_pos(target_msg.position.x, target_msg.position.y, target_msg.position.z);
    
    Eigen::Vector2d last_yaw_pitch;
    bool has_last = false;

    const double now_stamp = nowSeconds();
    const double base_stamp = now_stamp + fly_time;
    const double hist_start = base_stamp - HALF_HORIZON * DT;

    // 清理过旧缓存，基准：当前预测窗口起点往前 1 个 DT
    pruneCache(hist_start - DT);
    // latest_cached: 截止到 base_stamp(=now+fly_time) 的最新缓存点，用于判定历史可用区间上界
    auto latest_cached = queryLatest(base_stamp);

    for (int i = -HALF_HORIZON; i < HALF_HORIZON; i++) {
        int idx = i + HALF_HORIZON; // column index in traj (0..HORIZON-1)
        if (idx < 0 || idx >= HORIZON) continue;

        double t_pred = DT * (i + 1);
        double abs_time = base_stamp + t_pred;

        // 使用历史插值的条件：窗口内存在数据且 abs_time 位于缓存范围 [hist_start, latest_cached->stamp]
        bool use_cache = latest_cached.has_value() &&
                         abs_time <= latest_cached->stamp + 1e-6 &&
                         abs_time >= hist_start - 1e-6;
        if (use_cache) {
            // 找到 abs_time 前后的最近两帧，做线性插值（时间加权）
            CachedState prev;
            CachedState next;
            bool has_prev = false;
            bool has_next = false;

            // prev: <= abs_time 的最近一帧
            for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
                if (it->stamp <= abs_time + 1e-9) {
                    prev = *it;
                    has_prev = true;
                    break;
                }
            }
            // next: > abs_time 的最近一帧
            for (const auto & st : history_) {
                if (st.stamp > abs_time - 1e-9) {
                    next = st;
                    has_next = true;
                    break;
                }
            }

            if (has_prev && has_next) {
                double t0 = prev.stamp;
                double t1 = next.stamp;
                if (std::abs(t1 - t0) < 1e-6) {
                    t1 = t0 + 1e-6; // 避免除零
                }
                double w1 = (abs_time - t0) / (t1 - t0);
                w1 = std::clamp(w1, 0.0, 1.0);
                double w0 = 1.0 - w1;

                auto lerp = [&](double a, double b) { return a * w0 + b * w1; };

                double yaw_interp = limit_rad(lerp(prev.yaw, next.yaw));
                double pitch_interp = limit_rad(lerp(prev.pitch, next.pitch));
                double yaw_vel_interp = lerp(prev.yaw_vel, next.yaw_vel);
                double pitch_vel_interp = lerp(prev.pitch_vel, next.pitch_vel);

                traj.col(idx) << yaw_interp, yaw_vel_interp,
                                 pitch_interp, pitch_vel_interp;
                last_yaw_pitch = {yaw_interp, pitch_interp};
                has_last = true;
                continue;
            } else if (has_prev) {
                traj.col(idx) << limit_rad(prev.yaw), prev.yaw_vel,
                                 limit_rad(prev.pitch), prev.pitch_vel;
                last_yaw_pitch = {limit_rad(prev.yaw), limit_rad(prev.pitch)};
                has_last = true;
                continue;
            } else if (has_next) {
                traj.col(idx) << limit_rad(next.yaw), next.yaw_vel,
                                 limit_rad(next.pitch), next.pitch_vel;
                last_yaw_pitch = {limit_rad(next.yaw), limit_rad(next.pitch)};
                has_last = true;
                continue;
            }
        }

        if (!has_last) {
            Eigen::Vector3d init_armor_pos = getOdomTarget(target_msg, fly_time - HALF_HORIZON * DT);
            last_yaw_pitch = aim(init_armor_pos, bullet_speed);
            has_last = true;
        }

        Eigen::Vector3d target_pos_pred = getOdomTarget(target_msg, fly_time + t_pred);
        Eigen::Vector2d curr_yaw_pitch = aim(target_pos_pred, bullet_speed);

        Eigen::Vector3d next_armor_pos = getOdomTarget(target_msg, fly_time + t_pred + DT);
        Eigen::Vector2d next_yaw_pitch = aim(next_armor_pos, bullet_speed);

        // central difference using (next - last) / (2*DT) gives velocity at curr
        double yaw_diff = limit_rad(next_yaw_pitch(0) - last_yaw_pitch(0));
        double yaw_vel = std::clamp(yaw_diff / (2.0 * DT), -8.0, 8.0);

        double pitch_diff = limit_rad(next_yaw_pitch(1) - last_yaw_pitch(1));
        double pitch_vel = std::clamp(pitch_diff / (2.0 * DT), -5.0, 5.0);
        
        // write into trajectory at proper column
        traj.col(idx) << limit_rad(curr_yaw_pitch(0)), yaw_vel,
                         limit_rad(curr_yaw_pitch(1)), pitch_vel;

        last_yaw_pitch = curr_yaw_pitch;
    }

    return traj;
}

Eigen::Matrix<double, 2, 1> MPCController::aim(
    const Eigen::Vector3d & target_odom, double bullet_speed)
{
    double dist = target_odom.head<2>().norm();
    // std::cerr << "target_odom.y(): "<< target_odom.y() << "\n";
    // std::cerr << "target_odom.x(): "<< target_odom.x() << "\n";
    double azim = std::atan2(target_odom.y(), target_odom.x());
    Ballistic ballistic(air_resistence_, bullet_speed);
    double horizon_dis = dist;
    double height = target_odom.z();
    auto [pitch, _] = ballistic.fixTiteratPitch(horizon_dis, height);
    return {limit_rad(azim), limit_rad(pitch)};
}

void MPCController::setupYawSolver(rclcpp::Node *node)
{
    double max_yaw_acc = node->declare_parameter("max_yaw_acc", 10.0);
    std::vector<double> Q_yaw = node->declare_parameter("Q_yaw", std::vector<double>{9e6, 0.0});
    std::vector<double> R_yaw = node->declare_parameter("R_yaw", std::vector<double>{1.0});

    Eigen::MatrixXd A{{1, DT}, {0, 1}};
    Eigen::MatrixXd B{{0}, {DT}};
    Eigen::VectorXd f{{0, 0}};
    Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
    Eigen::Matrix<double, 1, 1> R(R_yaw.data());
    tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

    Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
    Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
    tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

    yaw_solver_->settings->max_iter = 100;
}

void MPCController::setupPitchSolver(rclcpp::Node *node)
{
    double max_pitch_acc = node->declare_parameter("max_pitch_acc", 100.0);
    std::vector<double> Q_pitch = node->declare_parameter("Q_pitch", std::vector<double>{9e6, 0.0});
    std::vector<double> R_pitch = node->declare_parameter("R_pitch", std::vector<double>{1.0});

    Eigen::MatrixXd A{{1, DT}, {0, 1}};
    Eigen::MatrixXd B{{0}, {DT}};
    Eigen::VectorXd f{{0, 0}};
    Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
    Eigen::Matrix<double, 1, 1> R(R_pitch.data());
    tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

    Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
    Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
    tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

    pitch_solver_->settings->max_iter = 100;
}

}  // namespace rm_auto_aim