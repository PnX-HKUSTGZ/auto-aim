#ifndef RUNE_BALLISTIC_CALCULATOR_HPP_
#define RUNE_BALLISTIC_CALCULATOR_HPP_

#include <ceres/ceres.h>
#include <ceres/jet.h>

#include <Eigen/Dense>
#include <auto_aim_interfaces/msg/rune_target.hpp>
#include <geometry_msgs/msg/detail/point__struct.hpp>
#include <geometry_msgs/msg/detail/vector3__struct.hpp>
#include <rclcpp/rclcpp.hpp>

#include "ballistic_calculation/types.hpp"

//STD
#include <type_traits>
#include <utility>
#include <vector>

namespace rm_auto_aim
{

class Ballistic
{
private:
    // parameter
    double k;           //空气阻力系数，需要parameter_declare来调整参数
    double bulletV;     //子弹速度，需要parameter_declare来调整参数
    double fire_delay;  //开火延迟，需要parameter_declare来调整参数

    std::pair<double, double> fixTiteratPitch(double & horizon_dis, double & height)
    {
        double dist_horizon = horizon_dis;  // 和目标在水平方向上的距离
        double target_height = height;      // 和目标在垂直方向上的距离

        // 迭代参数
        double vx, vy, fly_time, tmp_height = target_height, delta_height = 0, tmp_pitch,
                                 real_height;
        for (size_t i = 0; i < 10; i++) {
            tmp_pitch = atan((tmp_height) / dist_horizon);
            vx = bulletV * cos(tmp_pitch);
            vy = bulletV * sin(tmp_pitch);

            fly_time = (exp(k * dist_horizon) - 1) / (k * vx);
            double term = vy + 9.8 / k;
            real_height = term * (1.0 - std::exp(-k * fly_time)) / k - (9.8 * fly_time) / k;
            delta_height = target_height - real_height;
            tmp_height += delta_height;
        }
        return std::make_pair(tmp_pitch, fly_time + fire_delay);
    };

    template <typename T>
    double optimizeTime(double initial_guess, T & state_info, double & temp_pitch)
    {
        double t = initial_guess;  // Initial guess for time t

        ceres::Problem problem;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<Ballistic::CostFunctor<T>, 1, 1>(
                new Ballistic::CostFunctor<T>(*this, state_info, temp_pitch)),
            nullptr, &t);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        //options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Optional: Output a brief report to std::cout
        //std::cout << summary.BriefReport() << "\n";

        return t;  // Return the optimized t value
    }

    template <typename T>
    struct CostFunctor
    {
        const Ballistic & ballistic_ref;
        T & state_info;
        double temp_pitch;

        explicit CostFunctor(const Ballistic & ballistic, T & state_info, double temp_pitch)
        : ballistic_ref(ballistic), state_info(state_info), temp_pitch(temp_pitch)
        {
        }

        template <typename U>
        bool operator()(const U * const t, U * residual) const
        {
            U v0 = U(ballistic_ref.bulletV);
            // 提取时间值的标量部分来调用 getHorizontalDistance
            double t_value; 
            if constexpr (std::is_same_v<U, double>) {
                t_value = *t;
            } else {
                t_value = t->a;
            } 
            U distance_aim = U(state_info.getHorizontalDistance(t_value));

            residual[0] =
                (U(1.0) / U(ballistic_ref.k)) *
                    ceres::log(U(ballistic_ref.k) * ceres::cos(U(temp_pitch)) * v0 * (*t) + U(1.0)) -
                distance_aim;
            return true;
        }
    };

public:
    Ballistic(double k = 0.1, double bulletV = 30, double fire_delay = 0.0)
    : k(k), bulletV(bulletV), fire_delay(fire_delay){};  //构造函数

    template <typename T>
    std::pair<double, double> iteration(
        double & thres, double & init_pitch, double & init_t, T & target_info)
    {
        double pitch = init_pitch, t = init_t;  // 初始化pitch和t
        double differ;
        std::pair<double, double> update_tmp_pitch_t;

        for (int i = 0; i < 100; i++) {
            t = optimizeTime(t, target_info, pitch);
            Eigen::Vector3d new_target = target_info.getGunTarget(t);

            double preddist = sqrt(pow(new_target[0], 2) + pow(new_target[1], 2));
            double predheight = new_target[2];

            update_tmp_pitch_t = fixTiteratPitch(preddist, predheight);

            differ = pitch - update_tmp_pitch_t.first;
            pitch = update_tmp_pitch_t.first;
            t = update_tmp_pitch_t.second;

            if (abs(differ) < thres) {
                break;
            }
        }
        Eigen::Vector3d last_target = target_info.getGunTarget(t);
        double predyaw = atan2(last_target[1], last_target[0]);
        return std::make_pair(pitch, predyaw);
    }
    double getBulletV() const
    {
        return bulletV;
    }
};

}  //namespace rm_auto_aim

#endif  // RUNE_BALLISTIC_CALCULATOR_HPP_