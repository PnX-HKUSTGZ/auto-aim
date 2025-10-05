#ifndef BALLISTIC_CALCULATION_ARMOR_SELECTOR_HPP_
#define BALLISTIC_CALCULATION_ARMOR_SELECTOR_HPP_

#include <angles/angles.h>
#include <ceres/ceres.h>
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
    ArmorSelector(): lock_id(-1), coming_angle_(70.0 / 57.3), leaving_angle_(30.0 / 57.3) {}
    /**
     * @brief 更新目标信息
     * 
     * @param new_target_msg 新的目标消息
     */
    void updateTarget(const target & new_target_msg)
    {
        target_msg = new_target_msg;
    }
    
    /**
     * @brief 预测步兵最佳装甲板
     * 
     * 实现三级火控策略：
     * 一级策略：通过点乘算出最优装甲板
     * 二级策略：引入放弃角度，选择更容易击中的装甲板
     * 三级策略：高速瞄准中心点
     * 
     * 保持原有接口：输入预测时间 T，以及速度阈值 min_v/max_v。
     * 内部使用类似 auto_aim::Aimer 中的装甲板选择策略：
     * - 若目标角速度过快（> max_v），直接瞄向中心点（三级策略）
     * - 若目标不“快速旋转”（小角速度），选择在可射击视野内且角度差最小的装甲（锁定机制）
     * - 在“旋转”场景下，使用 coming/leaving 角度判断优先出现的装甲板
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
        (void)min_v;
        (void)v_yaw_gimble;
        // 三级策略：目标角速度过快时，瞄准中心点
        if (std::abs(target_msg.v_yaw) > max_v) {
            return {0.0, target_msg.position.z + 0.5 * target_msg.dz, 0.0};
        }
        
        int a_n = target_msg.armors_num;  // 装甲板数量
        std::vector<Armor> armors(a_n);   // 装甲板容器

        // 计算未来 T 时间的目标中心位置（线性运动预测）
        double newyaw = target_msg.yaw + target_msg.v_yaw * T;        // 预测偏航角
        double newxc = target_msg.position.x + target_msg.velocity.x * T;  // 预测X坐标
        double newyc = target_msg.position.y + target_msg.velocity.y * T;  // 预测Y坐标

        // 计算枪口到目标中心的角度
        double gun_to_center_angle = atan2(newyc, newxc);

        // 按顺时针方向计算每个装甲板的位置和属性
        for (int i = 0; i < a_n; ++i) {
            // 计算装甲板的偏航角
            armors[i].yaw = (i == 0) ? newyaw : armors[i - 1].yaw + 2 * M_PI / a_n;

            // 设置装甲板半径（奇偶装甲板可能有不同半径）
            armors[i].r = (i % 2 == 0) ? target_msg.radius_1 : target_msg.radius_2;
            
            // 设置装甲板高度（奇偶装甲板可能有不同高度）
            armors[i].z =
                (i % 2 == 0) ? target_msg.position.z : target_msg.position.z + target_msg.dz;

            // 计算装甲板在世界坐标系中的位置
            armors[i].x = newxc - armors[i].r * cos(armors[i].yaw);
            armors[i].y = newyc - armors[i].r * sin(armors[i].yaw);
            
            // 计算装甲板到枪口的最短角度距离（代价函数）
            armors[i].cost = shortest_angular_distance(gun_to_center_angle, armors[i].yaw);
        }

        // 当角速度较小，选在可射击范围（±60°）内代价最小的装甲板，并提供锁定机制
        const double small_spin_threshold = 2.0; // 阈值选择与Aimer中的保持一致
        if (std::abs(target_msg.v_yaw) <= small_spin_threshold) {
            // 收集在可射击视野内的装甲（±60度）
            std::vector<int> id_list;
            const double visible_limit = 60.0 / 57.3;
            for (int i = 0; i < static_cast<int>(armors.size()); ++i) {
                if (std::abs(shortest_angular_distance(armors[i].yaw, gun_to_center_angle)) > visible_limit)
                    continue;
                id_list.push_back(i);
            }
        
            // 若没有装甲在视野内，退回到代价最小的那个装甲
            if (id_list.empty()) {
                // 选代价最小的装甲
                auto it = std::min_element(armors.begin(), armors.end(), [](const Armor & A, const Armor & B){
                    return std::abs(A.cost) < std::abs(B.cost);
                });
                Armor chosen = *it;
                double set_yaw = chosen.yaw;
                // 返回补偿旋转后的偏航
                return {set_yaw - target_msg.v_yaw * T, chosen.z, chosen.r};
            } 

            // 如果多个装甲在视野内，使用锁定（lock_id_）避免在两个装甲间来回切换
            if (id_list.size() > 1) {
                int id0 = id_list[0];
                int id1 = id_list[1];
                if (lock_id != id0 && lock_id != id1) {
                    lock_id = (std::abs(armors[id0].cost) < std::abs(armors[id1].cost)) ? id0 : id1;
                }
                Armor chosen = armors[lock_id];
                double set_yaw = chosen.yaw;
                return {set_yaw - target_msg.v_yaw * T, chosen.z, chosen.r};
            }

            // 只有一个装甲在可射击范围内，退出锁定并选它
            lock_id = -1;
            Armor chosen = armors[id_list[0]];
            double set_yaw = chosen.yaw;
            return {set_yaw - target_msg.v_yaw * T, chosen.z, chosen.r};
        }


        // 处理“陀螺/快速旋转”情形（coming/leaving逻辑）
        double coming_angle = coming_angle_;
        double leaving_angle = leaving_angle_;

        for (int i = 0; i < static_cast<int>(armors.size()); ++i) {
            double delta_angle = shortest_angular_distance(armors[i].yaw, std::atan2(newyc, newxc));
            if (std::abs(delta_angle) > coming_angle) continue;
            if (target_msg.v_yaw > 0 && delta_angle < leaving_angle) {
                Armor chosen = armors[i];
                double set_yaw = chosen.yaw;
                return {set_yaw - target_msg.v_yaw * T, chosen.z, chosen.r};
            }
            if (target_msg.v_yaw < 0 && delta_angle > -leaving_angle) {
                Armor chosen = armors[i];
                double set_yaw = chosen.yaw;
                return {set_yaw - target_msg.v_yaw * T, chosen.z, chosen.r};
            }
        }

        // 若以上规则都没命中，则回退到代价最小的装甲（与点乘策略相似）
        auto it = std::min_element(armors.begin(), armors.end(), [](const Armor & A, const Armor & B) {
            return std::abs(A.cost) < std::abs(B.cost);
        });
        Armor chosen = *it;
        double set_yaw = chosen.yaw;
        return {set_yaw - target_msg.v_yaw * T, chosen.z, chosen.r};
    }

private: 
    target target_msg;          // 目标信息缓存
    int lock_id;                // 锁定的装甲索引，避免来回切换
    double coming_angle_;       // coming angle (rad)
    double leaving_angle_;      // leaving angle (rad)

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
     * @brief 通过优化求解放弃角度（保留原接口实现）
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
}; 

}
#endif  // BALLISTIC_CALCULATION_ARMOR_SELECTOR_HPP_