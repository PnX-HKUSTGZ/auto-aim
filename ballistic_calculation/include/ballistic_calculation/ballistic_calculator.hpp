#ifndef RUNE_BALLISTIC_CALCULATOR_HPP_
#define RUNE_BALLISTIC_CALCULATOR_HPP_

#include <ceres/ceres.h>
#include <ceres/jet.h>

#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <auto_aim_interfaces/msg/rune_target.hpp>
#include <geometry_msgs/msg/detail/point__struct.hpp>
#include <geometry_msgs/msg/detail/vector3__struct.hpp>
#include <rclcpp/rclcpp.hpp>

#include "ballistic_calculation/aim_info.hpp"

//STD
#include <type_traits>
#include <utility>
#include <vector>

namespace rm_auto_aim
{

/**
 * @brief 弹道计算类
 * 
 * 该类实现了考虑空气阻力的弹道计算功能，支持迭代优化求解最佳射击角度。
 * 使用 Ceres 优化库进行非线性优化求解。
 */
class Ballistic
{
private:
    // parameter
    double k;           // 空气阻力系数，需要parameter_declare来调整参数
    double bulletV;     // 子弹速度，需要parameter_declare来调整参数
    bool has_last_valid_solution_ = false;
    double last_valid_pitch_ = 0.0;
    double last_valid_t_ = 0.05;
    static constexpr double MIN_T = 1e-3;
    static constexpr double MIN_DENOM = 1e-6;
    static constexpr double MAX_EXP_ARG = 50.0;

    double sanitizeTimeGuess(double guess) const
    {
        if (!std::isfinite(guess) || guess < MIN_T) {
            return 0.05;
        }
        return std::clamp(guess, MIN_T, 3.0);
    }

    std::pair<double, double> fallbackToLastValidOrSafe(double safe_pitch, double safe_t)
    {
        if (has_last_valid_solution_ && std::isfinite(last_valid_pitch_) && std::isfinite(last_valid_t_)) {
            return std::make_pair(last_valid_pitch_, sanitizeTimeGuess(last_valid_t_));
        }
        return std::make_pair(safe_pitch, sanitizeTimeGuess(safe_t));
    }

    void updateLastValidSolution(double pitch, double t)
    {
        if (!std::isfinite(pitch)) {
            return;
        }
        double safe_t = sanitizeTimeGuess(t);
        if (!std::isfinite(safe_t)) {
            return;
        }
        last_valid_pitch_ = pitch;
        last_valid_t_ = safe_t;
        has_last_valid_solution_ = true;
    }

    /**
     * @brief 使用 Ceres 优化器优化飞行时间
     * 
     * @tparam T 目标信息类型模板
     * @param initial_guess 初始时间猜测值
     * @param state_info 目标状态信息
     * @param temp_pitch 临时俯仰角
     * @return double 优化后的飞行时间
     */
    template <typename T>
    double optimizeTime(double initial_guess, T & state_info, double & temp_pitch)
    {
        double safe_initial_guess = sanitizeTimeGuess(initial_guess);
        if (!std::isfinite(temp_pitch) || !std::isfinite(k) || !std::isfinite(bulletV) ||
            std::abs(k) < MIN_DENOM || bulletV <= MIN_DENOM)
        {
            return safe_initial_guess;
        }
        double t = safe_initial_guess;  // 时间初值

        // 构建 Ceres 优化问题
        ceres::Problem problem;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<Ballistic::CostFunctor<T>, 1, 1>(
                new Ballistic::CostFunctor<T>(*this, state_info, temp_pitch)),
            nullptr, &t);

        // Constrain time to non-negative to avoid invalid log arguments
        problem.SetParameterLowerBound(&t, 0, 0.001);

        // 配置求解器选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        //options.minimizer_progress_to_stdout = true;

        // 执行优化
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (!summary.IsSolutionUsable() || !std::isfinite(t) || t <= MIN_T) {
            RCLCPP_WARN(
                rclcpp::get_logger("Ballistic"),
                "Time optimization failed, use safe initial guess: %.3f (raw: %.3f)",
                safe_initial_guess, initial_guess);
            return safe_initial_guess;
        }

        return t;  // 返回优化后的时间值
    }

    /**
     * @brief Ceres 优化的代价函数结构体
     * 
     * @tparam T 目标信息类型模板
     */
    template <typename T>
    struct CostFunctor
    {
        const Ballistic & ballistic_ref;  // 弹道计算器引用
        T & state_info;                   // 目标状态信息
        double temp_pitch;                // 临时俯仰角

        /**
         * @brief 构造函数
         * 
         * @param ballistic 弹道计算器对象
         * @param state_info 目标状态信息
         * @param temp_pitch 临时俯仰角
         */
        explicit CostFunctor(const Ballistic & ballistic, T & state_info, double temp_pitch)
        : ballistic_ref(ballistic), state_info(state_info), temp_pitch(temp_pitch)
        {
        }

        /**
         * @brief 代价函数计算操作符
         * 
         * @tparam U 数值类型（支持自动微分）
         * @param t 时间变量
         * @param residual 残差输出
         * @return true 计算成功
         */
        template <typename U>
        bool operator()(const U * const t, U * residual) const
        {
            // U v0 = U(ballistic_ref.bulletV);

            // 提取时间值的标量部分来调用 getHorizontalDistance
            double t_value;
            if constexpr (std::is_same_v<U, double>) {
                t_value = *t;
            } else {
                t_value = t->a;  // Jet 类型的标量部分
            }

            double distance_scalar = state_info.getHorizontalDistance(t_value);
            double log_arg_scalar = ballistic_ref.k * std::cos(temp_pitch) * ballistic_ref.bulletV * t_value + 1.0;

            // 保护：任何非法输入直接给大残差，避免 NaN 进入 Ceres
            if (!std::isfinite(distance_scalar) || !std::isfinite(log_arg_scalar) || log_arg_scalar <= 0.001) {
                residual[0] = U(1e8);
                if constexpr (std::is_same_v<U, double>) {
                    RCLCPP_ERROR(
                        rclcpp::get_logger("BallisticCostFunctor"),
                        "Invalid ballistic input (log_arg=%.6f, distance=%.6f, t=%.6f, k=%.6f, cos(pitch)=%.6f, v0=%.6f)",
                        log_arg_scalar,
                        distance_scalar,
                        t_value,
                        ballistic_ref.k,
                        std::cos(temp_pitch),
                        ballistic_ref.bulletV
                    );
                }
                return true;
            }

            U log_arg = U(log_arg_scalar);
            U distance_aim = U(distance_scalar);

            residual[0] = (U(1.0) / U(ballistic_ref.k)) * ceres::log(log_arg) - distance_aim;

            return true;
        }
    };

public:
    /**
     * @brief 构造函数
     * 
     * @param k 空气阻力系数，默认值 0.1
     * @param bulletV 子弹速度，默认值 22 m/s
     */
    Ballistic(double k = 0.1, double bulletV = 22)
    : k(k), bulletV(bulletV){};

    /**
     * @brief 主迭代函数，计算最佳射击角度
     * 
     * 该函数通过迭代优化的方式，综合考虑目标运动和弹道特性，
     * 计算出最佳的俯仰角和偏航角。
     * 
     * @tparam T 目标信息类型模板
     * @param thres 迭代收敛阈值
     * @param init_pitch 初始俯仰角
     * @param init_t 初始飞行时间
     * @param target_info 目标信息对象
     * @return std::pair<double, double> 返回最佳俯仰角和偏航角
     */
    template <typename T>
    std::pair<double, double> iteration(
        const double & thres, double & init_pitch, double & init_t, T & target_info, double & t_out)
    {
        double pitch = init_pitch, t = sanitizeTimeGuess(init_t);  // 初始化pitch和t
        if (!std::isfinite(pitch)) {
            Eigen::Vector3d target0 = target_info.getGunTarget(0.0);
            double dist0 = std::hypot(target0[0], target0[1]);
            pitch = std::atan2(target0[2], std::max(dist0, MIN_T));
        }
        if (!std::isfinite(pitch)) {
            auto fallback = fallbackToLastValidOrSafe(0.0, t);
            t_out = fallback.second;
            return std::make_pair(fallback.first, 0.0);
        }

        double differ;  // 角度差值
        std::pair<double, double> update_tmp_pitch_t;

        // 主迭代循环，最多100次
        for (int i = 0; i < 100; i++) {
            // 第一步：优化飞行时间
            t = optimizeTime(t, target_info, pitch);
            
            // 第二步：获取预测目标位置
            Eigen::Vector3d new_target = target_info.getGunTarget(t);
            if (!new_target.allFinite()) {
                t = sanitizeTimeGuess(t);
                break;
            }

            // 计算水平距离和高度
            double preddist = std::hypot(new_target[0], new_target[1]);
            double predheight = new_target[2];

            // 第三步：修正俯仰角
            update_tmp_pitch_t = fixTiteratPitch(preddist, predheight);

            // 检查收敛性
            if (std::isfinite(update_tmp_pitch_t.first)) {
                differ = pitch - update_tmp_pitch_t.first;
                pitch = update_tmp_pitch_t.first;
            } else {
                differ = thres + 1.0;
            }
            t = sanitizeTimeGuess(update_tmp_pitch_t.second);

            if (abs(differ) < thres) {
                break;  // 达到收敛条件，退出迭代
            }
        }
        t_out = sanitizeTimeGuess(t);
        // 计算最终目标位置和偏航角
        Eigen::Vector3d last_target = target_info.getGunTarget(t_out);
        if (!last_target.allFinite()) {
            last_target = target_info.getGunTarget(0.0);
        }
        if (!last_target.allFinite()) {
            auto fallback = fallbackToLastValidOrSafe(pitch, t_out);
            t_out = fallback.second;
            updateLastValidSolution(fallback.first, t_out);
            return std::make_pair(fallback.first, 0.0);
        }
        double predyaw = atan2(last_target[1], last_target[0]);
        updateLastValidSolution(pitch, t_out);
        
        return std::make_pair(pitch, predyaw);
    }
    
    /**
     * @brief 获取子弹速度
     * 
     * @return double 子弹速度值
     */
    double getBulletV() const
    {
        return bulletV;
    }

    /**
     * @brief 固定迭代法计算俯仰角
     * 
     * @param horizon_dis 水平距离（引用传递）
     * @param height 垂直高度（引用传递）
     * @return std::pair<double, double> 返回计算得到的俯仰角和飞行时间
     */
    std::pair<double, double> fixTiteratPitch(double & horizon_dis, double & height)
    {
        double dist_horizon = horizon_dis;  // 和目标在水平方向上的距离
        double target_height = height;      // 和目标在垂直方向上的距离

        if (!std::isfinite(dist_horizon) || !std::isfinite(target_height) || !std::isfinite(k) ||
            !std::isfinite(bulletV) || std::abs(k) < MIN_DENOM || bulletV <= MIN_DENOM)
        {
            return fallbackToLastValidOrSafe(0.0, 0.05);
        }
        dist_horizon = std::max(dist_horizon, MIN_T);

        // 迭代参数初始化
        double vx, vy, fly_time, tmp_height = target_height, delta_height = 0, tmp_pitch,
                                 real_height;
        
        // 进行10次迭代优化
        for (size_t i = 0; i < 10; i++) {
            // 计算当前俯仰角
            tmp_pitch = std::atan2(tmp_height, dist_horizon);
            
            // 分解初始速度
            vx = bulletV * cos(tmp_pitch);
            vy = bulletV * sin(tmp_pitch);
            if (!std::isfinite(vx) || !std::isfinite(vy) || std::abs(vx) < MIN_DENOM) {
                fly_time = std::max(MIN_T, dist_horizon / std::max(bulletV, MIN_DENOM));
                break;
            }

            // 计算飞行时间（考虑空气阻力）
            double exp_arg = std::clamp(k * dist_horizon, -MAX_EXP_ARG, MAX_EXP_ARG);
            fly_time = (std::exp(exp_arg) - 1) / (k * vx);
            if (!std::isfinite(fly_time) || fly_time <= MIN_T) {
                fly_time = std::max(MIN_T, dist_horizon / std::max(bulletV, MIN_DENOM));
                break;
            }
            
            // 计算实际高度（考虑重力和空气阻力）
            double term = vy + 9.8 / k;
            double decay_arg = std::clamp(-k * fly_time, -MAX_EXP_ARG, MAX_EXP_ARG);
            real_height = term * (1.0 - std::exp(decay_arg)) / k - (9.8 * fly_time) / k;
            if (!std::isfinite(real_height)) {
                fly_time = std::max(MIN_T, dist_horizon / std::max(bulletV, MIN_DENOM));
                break;
            }
            
            // 计算高度误差并修正
            delta_height = target_height - real_height;
            tmp_height += delta_height;
        }
        if (!std::isfinite(tmp_pitch)) {
            tmp_pitch = std::atan2(target_height, dist_horizon);
        }
        if (!std::isfinite(fly_time) || fly_time <= MIN_T) {
            fly_time = std::max(MIN_T, dist_horizon / std::max(bulletV, MIN_DENOM));
        }
        return std::make_pair(tmp_pitch, fly_time);
    };

};

}  //namespace rm_auto_aim

#endif  // RUNE_BALLISTIC_CALCULATOR_HPP_
