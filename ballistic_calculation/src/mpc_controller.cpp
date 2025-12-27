#include "ballistic_calculation/mpc_controller.hpp"
#include <Eigen/src/Core/Matrix.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "ballistic_calculation/math_util.hpp"
#include "yaml-cpp/yaml.h"
#include "rclcpp/rclcpp.hpp"

namespace rm_auto_aim
{
MPCController::MPCController(const std::string & config_path)
{
    YAML::Node yaml = YAML::LoadFile(config_path);
    yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;
    pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;
    fire_thresh_ = yaml["fire_thresh"].as<double>();
    decision_speed_ = yaml["decision_speed"].as<double>();
    high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
    low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
    min_switch_speed_ = yaml["min_switch_speed"].as<double>(); 
    max_switch_speed_ = yaml["max_switch_speed"].as<double>();

    setupYawSolver(config_path);
    setupPitchSolver(config_path);
}

MPCController::~MPCController() {
    if (yaw_solver_) free(yaw_solver_);
    if (pitch_solver_) free(pitch_solver_);
}

MPCResult MPCController::compute(
    const auto_aim_interfaces::msg::Target & target_msg,
    double bullet_speed, double T)
{
    MPCResult result;
    result.is_valid = false;


    if (bullet_speed < 10 || bullet_speed > 35 || DT <= 1e-6) {
        bullet_speed = 22; // Fallback default
        RCLCPP_WARN(rclcpp::get_logger("MPCController"), "Invalid bullet speed, using default 22m/s");
    }
    Trajectory traj;
    
    try {
        traj = getTrajectory(target_msg, bullet_speed, T);
    } catch (const std::exception & e) {
        RCLCPP_ERROR(rclcpp::get_logger("MPCController"), "Trajectory generation failed: %s", e.what());
        return result;
    }

    Eigen::VectorXd x0(2);
    x0 << traj(0, 0), traj(1, 0);
    if (x0.hasNaN() || x0.cwiseAbs().maxCoeff() > 1e6) {
        RCLCPP_WARN(rclcpp::get_logger("MPCController"), "Invalid yaw initial state");
        return result;
    }
    tiny_set_x0(yaw_solver_, x0);

    Eigen::MatrixXd yaw_ref = traj.block(0, 0, 2, HORIZON);
    if (yaw_ref.hasNaN()) {
        RCLCPP_WARN(rclcpp::get_logger("MPCController"), "NaN in yaw reference");
        return result;
    }
    yaw_solver_->work->Xref = yaw_ref;
    
    int solve_ret = tiny_solve(yaw_solver_);
    if (solve_ret != 0) {
        RCLCPP_WARN(rclcpp::get_logger("MPCController"), "Yaw MPC solve failed: %d", solve_ret);
        return result;
    }
    x0 << traj(2, 0),  traj(3, 0);
    
    if (x0.hasNaN() || x0.cwiseAbs().maxCoeff() > 1e6) {
        RCLCPP_WARN(rclcpp::get_logger("MPCController"), "Invalid pitch initial state");
        return result;
    }
    tiny_set_x0(pitch_solver_, x0);

    Eigen::MatrixXd pitch_ref = traj.block(2, 0, 2, HORIZON);
    if (pitch_ref.hasNaN()) {
         RCLCPP_WARN(rclcpp::get_logger("MPCController"), "NaN in pitch reference");
         return result;
    }
    pitch_solver_->work->Xref = pitch_ref;

    // Solve Pitch
    solve_ret = tiny_solve(pitch_solver_);
    if (solve_ret != 0) {
        RCLCPP_WARN(rclcpp::get_logger("MPCController"), "Pitch MPC solve failed: %d", solve_ret);
        return result;
    }
    result.is_valid = true;

    double yaw = limit_rad(yaw_solver_->work->x(0, HALF_HORIZON));
    double yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
    double yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);
    
    double pitch = limit_rad(pitch_solver_->work->x(0, HALF_HORIZON));
    double pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
    double pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

    const double MAX_VEL = 5.0;  
    const double MAX_ACC = 8.0; 
    const double DEAD_ZONE_VEL = 0.01;
    const double DEAD_ZONE_ACC = 0.05;

    result.yaw = yaw;
    double yaw_vel_clamped = std::clamp(yaw_vel, -MAX_VEL, MAX_VEL);
    result.yaw_vel = std::abs(yaw_vel_clamped) < DEAD_ZONE_VEL ? 0.0 : yaw_vel_clamped;
    double yaw_acc_clamped = std::clamp(yaw_acc, -MAX_ACC, MAX_ACC);
    result.yaw_acc = std::abs(yaw_acc_clamped) < DEAD_ZONE_ACC ? 0.0 : yaw_acc_clamped;

    result.pitch = pitch;
    double pitch_vel_clamped = std::clamp(pitch_vel, -MAX_VEL, MAX_VEL);
    result.pitch_vel = std::abs(pitch_vel_clamped) < DEAD_ZONE_VEL ? 0.0 : pitch_vel_clamped;
    double pitch_acc_clamped = std::clamp(pitch_acc, -MAX_ACC, MAX_ACC);
    result.pitch_acc = std::abs(pitch_acc_clamped) < DEAD_ZONE_ACC ? 0.0 : pitch_acc_clamped;

    if (std::isnan(result.yaw_vel) || std::isnan(result.pitch_vel)) {
        result.is_valid = false;
        return result;
    }

    // 优化前
    Eigen::Vector3d target_pos_vec(target_msg.position.x, target_msg.position.y, target_msg.position.z);
    Eigen::Vector2d target_vel_vec(target_msg.velocity.x, target_msg.velocity.y);

    result.target_yaw = limit_rad(traj(0, HALF_HORIZON));
    result.target_pitch = limit_rad(traj(2, HALF_HORIZON));

    // 静止死区处理
    // if (target_vel_vec.norm() < 0.05 && std::abs(target_msg.v_yaw) < 0.1) {
    //     const double ERROR_THRESH = 0.02;
    //     if (std::abs(limit_rad(result.yaw - result.target_yaw)) < ERROR_THRESH) {
    //         result.yaw_vel = 0.0; result.yaw_acc = 0.0;
    //         result.yaw = current_gimbal_state(0);
    //     }
    //     if (std::abs(result.pitch - result.target_pitch) < ERROR_THRESH) {
    //         result.pitch_vel = 0.0; result.pitch_acc = 0.0;
    //         result.pitch = current_gimbal_state(2);
    //     }
    // }

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
    double bullet_speed, double T)
{
    Trajectory traj;
    traj.setZero();
    
    Eigen::Vector3d current_pos(target_msg.position.x, target_msg.position.y, target_msg.position.z);
    
    Eigen::Vector2d last_yaw_pitch;
    // Use time in seconds: HALF_HORIZON steps before T (each step = DT)
    Eigen::Vector3d init_armor_pos = getOdomTarget(target_msg, T - HALF_HORIZON * DT);

    last_yaw_pitch = aim(init_armor_pos, bullet_speed);

    for (int i = -HALF_HORIZON; i < HALF_HORIZON; i++) {
        int idx = i + HALF_HORIZON; // column index in traj (0..HORIZON-1)
        if (idx < 0 || idx >= HORIZON) continue;

        double t_pred = DT * (i + 1);

        Eigen::Vector3d target_pos_pred = getOdomTarget(target_msg, T + t_pred);
        Eigen::Vector2d curr_yaw_pitch = aim(target_pos_pred, bullet_speed);

        Eigen::Vector3d next_armor_pos = getOdomTarget(target_msg, T + t_pred + DT);
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
    Ballistic ballistic(0.1, bullet_speed, 0.0);
    double horizon_dis = dist;
    double height = target_odom.z();
    auto [pitch, _] = ballistic.fixTiteratPitch(horizon_dis, height);
    return {limit_rad(azim + yaw_offset_), limit_rad(pitch + pitch_offset_)}; // 补偿这块考虑是不是可以删了，电控那边已经补偿过了
}

void MPCController::setupYawSolver(const std::string & config_path)
{
    YAML::Node yaml = YAML::LoadFile(config_path);
    double max_yaw_acc = yaml["max_yaw_acc"].as<double>(); // 有待确定
    std::vector<double> Q_yaw = yaml["Q_yaw"].as<std::vector<double>>();
    std::vector<double> R_yaw = yaml["R_yaw"].as<std::vector<double>>();

    Eigen::MatrixXd A{{1, DT}, {0, 1}};
    Eigen::MatrixXd B{{0}, {DT}};
    Eigen::VectorXd f{{0, 0}};
    Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
    Eigen::Matrix<double, 1, 1> R(R_yaw.data());
    tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

    // 添加合理的状态约束
    Eigen::MatrixXd x_min(2, HORIZON);
    Eigen::MatrixXd x_max(2, HORIZON);
    // 角度约束：±π rad
    x_min.row(0) = Eigen::VectorXd::Constant(HORIZON, -M_PI);
    x_max.row(0) = Eigen::VectorXd::Constant(HORIZON, M_PI);
    // 角速度约束：±3 rad/s（根据云台实际性能调整）
    x_min.row(1) = Eigen::VectorXd::Constant(HORIZON, -3.0);
    x_max.row(1) = Eigen::VectorXd::Constant(HORIZON, 3.0);
    // 控制输入约束：±max_yaw_acc
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
    
    tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

    yaw_solver_->settings->max_iter = 20;
}

void MPCController::setupPitchSolver(const std::string & config_path)
{
    YAML::Node yaml = YAML::LoadFile(config_path);
    double max_pitch_acc = yaml["max_pitch_acc"].as<double>();
    std::vector<double> Q_pitch = yaml["Q_pitch"].as<std::vector<double>>();
    std::vector<double> R_pitch = yaml["R_pitch"].as<std::vector<double>>();

    Eigen::MatrixXd A{{1, DT}, {0, 1}};
    Eigen::MatrixXd B{{0}, {DT}};
    Eigen::VectorXd f{{0, 0}};
    Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
    Eigen::Matrix<double, 1, 1> R(R_pitch.data());
    tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

    // 添加合理的状态约束
    Eigen::MatrixXd x_min(2, HORIZON);
    Eigen::MatrixXd x_max(2, HORIZON);
    // 角度约束：±π rad
    x_min.row(0) = Eigen::VectorXd::Constant(HORIZON, -M_PI);
    x_max.row(0) = Eigen::VectorXd::Constant(HORIZON, M_PI);
    // 角速度约束：±1 rad/s（根据云台实际性能调整）
    x_min.row(1) = Eigen::VectorXd::Constant(HORIZON, -1.0);
    x_max.row(1) = Eigen::VectorXd::Constant(HORIZON, 1.0);
    // 控制输入约束：±max_yaw_acc
    Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
    Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);

    tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

    pitch_solver_->settings->max_iter = 20;
}

}  // namespace rm_auto_aim