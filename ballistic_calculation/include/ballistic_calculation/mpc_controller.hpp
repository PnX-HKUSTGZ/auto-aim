#ifndef BALLISTIC_CALCULATION_MPC_CONTROLLER_HPP_
#define BALLISTIC_CALCULATION_MPC_CONTROLLER_HPP_

#include <Eigen/Dense>
#include <deque>
#include <optional>
#include <string>

#include "../tinympc/tiny_api.hpp"
#include "ballistic_calculator.hpp"
#include "math_util.hpp"
#include "armor_selector.hpp"
#include "aim_info.hpp"
#include "rclcpp/rclcpp.hpp"

namespace rm_auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
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
    MPCController(rclcpp::Node *node);
    ~MPCController();

    MPCResult compute(
        const auto_aim_interfaces::msg::Target & target_msg,
        double bullet_speed,
        double T
    );

private:
    struct CachedState
    {
        double stamp{0.0};          // 绝对时间戳（秒） = now + T
        double yaw{0.0};
        double pitch{0.0};
        double yaw_vel{0.0};
        double pitch_vel{0.0};
        double yaw_acc{0.0};
        double pitch_acc{0.0};
    };

    static constexpr size_t MAX_CACHE_SIZE = 600; // ~6s at 100Hz

    double fire_delay;
    double min_switch_speed_;
    double max_switch_speed_;
    double max_yaw_acc_;       // 偏航最大加速度
    double max_pitch_acc_;     // 俯仰最大加速度
    std::vector<double> Q_yaw_;// 偏航Q矩阵
    std::vector<double> R_yaw_;// 偏航R矩阵
    std::vector<double> Q_pitch_;// 俯仰Q矩阵
    std::vector<double> R_pitch_;// 俯仰R矩阵

    ArmorSelector armor_selector_;

    std::deque<CachedState> history_;

    TinySolver * yaw_solver_{nullptr};
    TinySolver * pitch_solver_{nullptr};

    void setupYawSolver(rclcpp::Node *node);
    void setupPitchSolver(rclcpp::Node *node);

    Eigen::Matrix<double, 2, 1> aim(const Eigen::Vector3d & target_odom, double bullet_speed);
    Trajectory getTrajectory(
        const auto_aim_interfaces::msg::Target & target_msg, 
        double bullet_speed, double T);

    void storeOptimizedState(const CachedState & state);
    void pruneCache(double min_stamp);
    std::optional<CachedState> queryLatest(double stamp) const;
    double nowSeconds() const;

    Eigen::Vector3d getOdomTarget(const auto_aim_interfaces::msg::Target & target_msg, double time);
};

}  // namespace rm_auto_aim

#endif  // BALLISTIC_CALCULATION_MPC_CONTROLLER_HPP_