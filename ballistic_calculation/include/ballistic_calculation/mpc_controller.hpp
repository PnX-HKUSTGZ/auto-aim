#ifndef BALLISTIC_CALCULATION_MPC_CONTROLLER_HPP_
#define BALLISTIC_CALCULATION_MPC_CONTROLLER_HPP_

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <string>

#include "../tinympc/tiny_api.hpp"
#include "ballistic_calculator.hpp"
#include "math_util.hpp"
#include "armor_selector.hpp"
#include "aim_info.hpp"

namespace rm_auto_aim
{
constexpr double DT = 0.02;
constexpr int HALF_HORIZON = 10;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct MPCResult
{
    bool is_valid{false};
    bool is_fire{false};
    double target_yaw{0.0};
    double target_pitch{0.0};
    double yaw;
    double pitch;
    double yaw_vel{0.0};
    double yaw_acc{0.0};
    double pitch_vel{0.0};
    double pitch_acc{0.0};
};

class MPCController
{
public:
    MPCController(const std::string & config_path);
    ~MPCController();

    MPCResult compute(
        const auto_aim_interfaces::msg::Target & target_msg,
        const Eigen::Vector4d & current_gimbal_state,
        double bullet_speed,
        double T
    );

private:
    double yaw_offset_{0.0};
    double pitch_offset_{0.0};
    double fire_thresh_{0.0}; // 有待确定
    double low_speed_delay_time_{0.0}, high_speed_delay_time_{0.0}, decision_speed_{0.0};
    double min_switch_speed_{0.0};
    double max_switch_speed_{0.0};

    ArmorSelector armor_selector_;

    TinySolver * yaw_solver_{nullptr};
    TinySolver * pitch_solver_{nullptr};

    void setupYawSolver(const std::string & config_path);
    void setupPitchSolver(const std::string & config_path);

    Eigen::Matrix<double, 2, 1> aim(const Eigen::Vector3d & target_odom, double bullet_speed);
    Trajectory getTrajectory(
        const auto_aim_interfaces::msg::Target & target_msg, 
        double yaw0, 
        double bullet_speed, double T);
};

}  // namespace rm_auto_aim

#endif  // BALLISTIC_CALCULATION_MPC_CONTROLLER_HPP_