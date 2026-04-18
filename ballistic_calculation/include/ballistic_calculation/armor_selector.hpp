#ifndef BALLISTIC_CALCULATION_ARMOR_SELECTOR_HPP_
#define BALLISTIC_CALCULATION_ARMOR_SELECTOR_HPP_

#include <angles/angles.h>
#include <ceres/ceres.h>
#include "rclcpp/rclcpp.hpp"
#include "math_util.hpp"

namespace rm_auto_aim
{
/**
 * @brief 装甲板选择器类
 * 
 * 该类实现了针对步兵机器人的最优装甲板选择策略。
 * 包含三级选择策略：最优装甲板选择、引入放弃角度、高速瞄准中心。
 */
class ArmorSelector
{
using target = auto_aim_interfaces::msg::Target;

public:
    const double COST_DIFF_THRESHOLD = M_PI/18;

    double getTargetPositionZ() const {
        return target_msg.position.z;
    }

    /**
     * @brief 更新目标信息
     * 
     * @param new_target_msg 新的目标消息
     */
    void updateTarget(const target & new_target_msg) {
        auto msg_delay = (this->now() - rclcpp::Time(new_target_msg.header.stamp)).seconds();
        target_msg = new_target_msg;
        target_msg.yaw = new_target_msg.yaw + msg_delay * new_target_msg.v_yaw;
        target_msg.position.x = new_target_msg.position.x + msg_delay * new_target_msg.velocity.x;
        target_msg.position.y = new_target_msg.position.y + msg_delay * new_target_msg.velocity.y;
        target_msg.position.z = new_target_msg.position.z + msg_delay * new_target_msg.velocity.z;
    }
    
    /**
     * @brief 预测步兵最佳装甲板
     * 
     * 实现三级火控策略：
     * 一级策略：通过点乘算出最优装甲板
     * 二级策略：引入放弃角度，选择更容易击中的装甲板
     * 三级策略：高速瞄准中心点
     * 
     * @param T 预测时间
     * @param min_v 一级策略切换二级策略的速度临界值
     * @param max_v 二级策略切换三级策略的速度临界值
     * @param v_yaw_gimble 云台最大偏航角速度
     * @return std::vector<double> 返回 [偏航角, 高度, 半径]
     */
    std::vector<double> predictInfantryBestArmor(
        double T, double min_v, double max_v, double v_yaw_gimble)
    {
        const bool is_outpost = (target_msg.id == "outpost");

        // 三级策略：目标角速度过快时，瞄准中心点
        if (abs(target_msg.v_yaw) > max_v) {
            return {0.0, is_outpost ? target_msg.position.z : (target_msg.position.z + 0.5 * target_msg.dz), 0.0};
        }
        
        int a_n = target_msg.armors_num;  // 装甲板数量
        std::vector<Armor> armors(a_n);   // 装甲板容器

        // 计算未来 T 时间的目标中心位置
        double newyaw = target_msg.yaw + target_msg.v_yaw * T;        // 预测偏航角
        double newxc = target_msg.position.x + target_msg.velocity.x * T;  // 预测X坐标
        double newyc = target_msg.position.y + target_msg.velocity.y * T;  // 预测Y坐标

        // 计算枪口到目标中心的角度
        double gun_to_center_angle = atan2(newyc, newxc);

        // 按顺时针方向计算每个装甲板的位置和属性
        for (int i = 0; i < a_n; ++i) {
            // 计算装甲板的偏航角
            armors[i].yaw = i == 0 ? newyaw : armors[i - 1].yaw + 2 * M_PI / a_n;

            // 设置装甲板半径（奇偶装甲板可能有不同半径）
            armors[i].r = (i % 2 == 0) ? target_msg.radius_1 : target_msg.radius_2;
            
            // 设置装甲板高度（奇偶装甲板可能有不同高度）
            armors[i].z =
                (i % 2 == 0) ? target_msg.position.z : target_msg.position.z + target_msg.dz;
            

            // 计算装甲板在世界坐标系中的位置
            armors[i].x = newxc - armors[i].r * cos(armors[i].yaw);
            armors[i].y = newyc - armors[i].r * sin(armors[i].yaw);
            
            // 计算装甲板到枪口的最短角度距离（代价函数）
            armors[i].cost = angles::shortest_angular_distance(gun_to_center_angle, armors[i].yaw);
        }
        if (is_outpost) {
                armors[0].z = target_msg.position.z;
                armors[1].z = target_msg.z2;
                armors[2].z = target_msg.z3;
                // if (target_msg.v_yaw >= 0){
                //     armors[1].z = target_msg.position.z + target_msg.dz;
                // }
                // else{
                //     armors[2].z = target_msg.position.z + target_msg.dz;
                // }

            }

        // 按角度距离排序，找到最容易击中的装甲板
        std::sort(armors.begin(), armors.end(), [](const Armor & a, const Armor & b) {
            return std::abs(a.cost) < std::abs(b.cost);
        });
        
        Armor chosen_armor = armors[0];  // 选择角度距离最小的装甲板
        // 一级策略：低速或MPC控制或最优装甲板角度小于放弃角度时，直接选择最优装甲板        
        if (abs(target_msg.v_yaw) < min_v) {
            if(abs(armors[0].cost - armors[1].cost) < COST_DIFF_THRESHOLD){
                chosen_armor = armors[0].cost < 0 ? armors[0] : armors[1];
            }
            return {chosen_armor.yaw - target_msg.v_yaw * T, chosen_armor.z, chosen_armor.r};
        }
        
        // 计算放弃角度（用于二级策略判断）
        double yaw = findYaw(
            abs(target_msg.v_yaw), v_yaw_gimble, sqrt(pow(newxc, 2) + pow(newyc, 2)),
            chosen_armor.r, target_msg.armors_num);
            
        if (std::isnan(yaw) || abs(armors[0].cost) <= yaw) {
            return {chosen_armor.yaw - target_msg.v_yaw * T, chosen_armor.z, chosen_armor.r};
        }
        
        // 二级策略：考虑运动方向，选择更容易击中的装甲板
        if ((armors[0].cost < 0 && target_msg.v_yaw < 0) ||
            (armors[0].cost > 0 && target_msg.v_yaw > 0)) {
            chosen_armor = armors[1];  // 选择次优装甲板
        }
        
        // 根据运动方向调整放弃角度的符号
        yaw = chosen_armor.cost > 0 ? abs(yaw) : -abs(yaw);
        double set_yaw = gun_to_center_angle + yaw;
        return {set_yaw - target_msg.v_yaw * T, chosen_armor.z, chosen_armor.r};
    }

private: 
    target target_msg;  // 目标信息缓存(被设计为thin->now()时刻)

    /**
     * @brief 装甲板结构体
     * 
     * 存储单个装甲板的位置和属性信息
     */
    struct Armor
    {
        double x;     // X坐标
        double y;     // Y坐标
        double z;     // Z坐标（高度）
        double yaw;   // 偏航角
        double r;     // 半径
        double cost;  // 装甲板和枪口的角度差（代价）
    };
    
    /**
     * @brief 通过优化求解放弃角度
     * 
     * 使用Ceres优化器求解满足几何约束的最优放弃角度
     * 
     * @param v_yaw 目标偏航角速度
     * @param v_yaw_gimble 云台最大偏航角速度
     * @param distance 到目标中心的距离
     * @param radius 装甲板半径
     * @param armors_num 装甲板数量
     * @return double 最优放弃角度，失败时返回NaN
     */
    double findYaw(double v_yaw, double v_yaw_gimble, double distance, double radius, int armors_num)
    {
        // 初始猜测值
        double yaw = M_PI / 4;

        // 输入合法性检查，避免除零或无效几何
        if (armors_num <= 0 || v_yaw <= 1e-6 || v_yaw_gimble <= 0.0 || distance <= 1e-6 || radius <= 1e-6) {
            return std::nan("");
        }

        // 配置Ceres求解器选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 100;
        options.function_tolerance = 1e-6;
        options.parameter_tolerance = 1e-6;

        // 构建优化问题
        ceres::Problem problem;

        // 添加残差块
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<YawResidual, 1, 1>(
                new YawResidual(v_yaw, v_yaw_gimble, distance, radius, armors_num)),
            nullptr, &yaw);

        // yaw 应在 (0, 2*pi/armors_num) 范围内，避免求解跑飞
        const double yaw_upper = std::max(1e-3, 2 * M_PI / std::max(1, armors_num));
        problem.SetParameterLowerBound(&yaw, 0, 0.0);
        problem.SetParameterUpperBound(&yaw, 0, yaw_upper);

        // 执行优化求解
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 检查求解结果
        if (summary.termination_type == ceres::CONVERGENCE) {
            return yaw;  // 收敛成功，返回优化结果
        } else {
            return std::nan("");  // 求解失败，返回NaN
        }
    }

protected:
    rclcpp::Time now() const {
        // 使用 ROS 时间源，保证与消息时间戳一致，避免 time source mismatch
        static rclcpp::Clock clock(RCL_ROS_TIME);
        return clock.now();
    }
}; 

}  // namespace rm_auto_aim
#endif  // BALLISTIC_CALCULATION_ARMOR_SELECTOR_HPP_