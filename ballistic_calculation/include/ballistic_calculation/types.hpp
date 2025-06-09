#ifndef BALLISTIC_CALCULATION_TYPES_HPP_
#define BALLISTIC_CALCULATION_TYPES_HPP_
#include <angles/angles.h>
#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <auto_aim_interfaces/msg/detail/target__struct.hpp>
#include <auto_aim_interfaces/msg/rune_target.hpp>
#include <map>
namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::Target;
using rune_target = auto_aim_interfaces::msg::RuneTarget;
enum class EulerOrder { XYZ, XZY, YXZ, YZX, ZXY, ZYX };
Eigen::Matrix3d eulerToMatrix(const Eigen::Vector3d & euler, EulerOrder order = EulerOrder::XYZ)
{
    auto r = Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX());
    auto p = Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY());
    auto y = Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ());
    switch (order) {
        case EulerOrder::XYZ:
            return (y * p * r).matrix();
        case EulerOrder::XZY:
            return (p * y * r).matrix();
        case EulerOrder::YXZ:
            return (y * r * p).matrix();
        case EulerOrder::YZX:
            return (r * y * p).matrix();
        case EulerOrder::ZXY:
            return (p * r * y).matrix();
        case EulerOrder::ZYX:
            return (r * p * y).matrix();
        default:
            return Eigen::Matrix3d::Identity();
    }
}
class YawResidual
{
public:
    YawResidual(double v_yaw, double v_yaw_gimble, double distance, double radius, int armors_num)
    : v_yaw_(v_yaw),
      v_yaw_gimble_(v_yaw_gimble),
      distance_(distance),
      radius_(radius),
      armors_num_(armors_num)
    {
    }

    template <typename T>
    bool operator()(const T * const yaw, T * residual) const
    {
        // 计算 yaw_gimble
        T numerator = ((T(2.0) * T(M_PI) / T(armors_num_)) - T(2.0) * yaw[0]) * T(v_yaw_gimble_);
        T denominator = T(2.0) * T(v_yaw_);
        T yaw_gimble = numerator / denominator;

        // 计算方程 sin(yaw + yaw_gimble) / distance - sin(yaw_gimble) / radius = 0
        residual[0] =
            ceres::sin(yaw[0] + yaw_gimble) / T(distance_) - ceres::sin(yaw_gimble) / T(radius_);

        return true;
    }

private:
    const double v_yaw_;
    const double v_yaw_gimble_;
    const double distance_;
    const double radius_;
    const int armors_num_;
};

class RuneInfo
{
private:
    rune_target target_msg;       //目标信息
    Eigen::Vector3d odom2gunxyz;  //枪口坐标, 需要parameter_declare来调整参数
    Eigen::Vector3d odom2gunrpy;  //枪口坐标, 需要parameter_declare来调整参数
private: 
    // Fitting Curve
#define BIG_RUNE_CURVE(x, a, omega, b, c, d, sign)                                             \
    (((-((a) / (omega)*std::cos((omega) * ((x) + (d)))) + (b) * ((x) + (d)) + (c)) * (sign)) - \
     ((-((a) / (omega)*std::cos((omega) * ((d)))) + (b) * ((d)) + (c)) * (sign)))
#define SMALL_RUNE_CURVE(x, a, b, c, sign) \
    ((((a) * ((x) + (b)) + (c)) * (sign)) - (((a) * (b) + (c)) * (sign)))

    // Rune arm length, Unit: m
    const double ARM_LENGTH = 0.700;

public:
    RuneInfo(const Eigen::Vector3d & odom2gunxyz, const Eigen::Vector3d & odom2gunrpy)
    : odom2gunxyz(odom2gunxyz), odom2gunrpy(odom2gunrpy)
    {
    }
    void updateTarget(const rune_target & new_target_msg) { target_msg = new_target_msg; }
    Eigen::Vector3d getOdomTarget(double time)
    {
        double angle_diff =
            target_msg.is_big ? BIG_RUNE_CURVE(
                                    time, target_msg.fitting_curve[0], target_msg.fitting_curve[1],
                                    target_msg.fitting_curve[2], target_msg.fitting_curve[3],
                                    target_msg.fitting_curve[4], target_msg.direction)
                              : SMALL_RUNE_CURVE(
                                    time, target_msg.fitting_curve[0], target_msg.fitting_curve[1],
                                    target_msg.fitting_curve[2], target_msg.direction);

        Eigen::Vector3d t_odom_2_rune =
            Eigen::Vector3d(target_msg.center.x, target_msg.center.y, target_msg.center.z);

        // Calculate the position of the armor in rune frame
        Eigen::Vector3d p_rune =
            eulerToMatrix(Eigen::Vector3d{-(angle_diff + target_msg.roll), 0, target_msg.yaw}) *
            Eigen::Vector3d(0, -ARM_LENGTH, 0);

        // Transform to odom frame
        Eigen::Vector3d p_odom = p_rune + t_odom_2_rune;

        return p_odom;
    }
    Eigen::Vector3d getGunTarget(double time)
    {
        // Get the predicted position
        Eigen::Vector3d target_position_odom = getOdomTarget(time);
        // Transform to gun frame
        // The gun frame is rotated by odom2gunrpy and translated by odom2gunxyz
        Eigen::Matrix3d R_odom2gun = eulerToMatrix(odom2gunrpy);
        Eigen::Vector3d target_position_gun = R_odom2gun * target_position_odom + odom2gunxyz;

        return target_position_gun;
    }
    double getHorizontalDistance(double time)
    {
        Eigen::Vector3d target_position_odom = getGunTarget(time);
        return sqrt(pow(target_position_odom.x(), 2) + pow(target_position_odom.y(), 2));
    }

};

class CarInfo
{
private:
    target target_msg;            //目标信息
    Eigen::Vector3d odom2gunxyz;  //枪口坐标, 需要parameter_declare来调整参数
    Eigen::Vector3d odom2gunrpy;  //枪口坐标, 需要parameter_declare来调整参数

public:
    CarInfo(const Eigen::Vector3d & odom2gunxyz, const Eigen::Vector3d & odom2gunrpy)
    : odom2gunxyz(odom2gunxyz), odom2gunrpy(odom2gunrpy)
    {
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 100;
        options.function_tolerance = 1e-6;
        options.parameter_tolerance = 1e-6;
    }
    void updateTarget(const target & new_target_msg) { target_msg = new_target_msg; }
    Eigen::Vector3d getOdomTarget(double time)
    {
        // Get the predicted position
        Eigen::Vector3d target_position_odom = Eigen::Vector3d(
            target_msg.position.x + target_msg.velocity.x * time,
            target_msg.position.y + target_msg.velocity.y * time,
            target_msg.position.z + target_msg.velocity.z * time);
        return target_position_odom;
    }
    Eigen::Vector3d getGunTarget(double time)
    {
        // Get the predicted position
        Eigen::Vector3d target_position_odom = getOdomTarget(time); 
        // Transform to gun frame
        // The gun frame is rotated by odom2gunrpy and translated by odom2gunxyz
        Eigen::Matrix3d R_odom2gun = eulerToMatrix(odom2gunrpy);
        Eigen::Vector3d target_position_gun = R_odom2gun * target_position_odom + odom2gunxyz;

        return target_position_gun;
    }
    double getHorizontalDistance(double time)
    {
        Eigen::Vector3d target_position_odom = getGunTarget(time);
        return sqrt(pow(target_position_odom.x(), 2) + pow(target_position_odom.y(), 2));
    }

public:
    //火控代码
    /*
    一级策略：通过点乘算出最优装甲板
    二级策略：引入放弃角度
    三级策略：高速瞄中心
    */
    std::vector<double> predictInfantryBestArmor(
        double T, double min_v, double max_v, double v_yaw_gimble)
    {
        if (abs(target_msg.v_yaw) > max_v) {
            return {0.0, target_msg.position.z + 0.5 * target_msg.dz, 0.0};
        }
        int a_n = target_msg.armors_num;
        std::vector<Armor> armors(a_n);

        // 计算未来 T 时间的中心位置
        double newyaw = target_msg.yaw + target_msg.v_yaw * T;
        double newxc = target_msg.position.x + target_msg.velocity.x * T;
        double newyc = target_msg.position.y + target_msg.velocity.y * T;

        // 计算枪口到目标中心的角度
        double gun_to_center_angle = atan2(newyc, newxc);

        // 按顺时针方向计算每个装甲板的 yaw，并计算其坐标
        for (int i = 0; i < a_n; ++i) {
            // 设置装甲板的角度,高度,半径
            armors[i].yaw = i == 0 ? newyaw : armors[i - 1].yaw + 2 * M_PI / a_n;

            armors[i].r = (i % 2 == 0) ? target_msg.radius_1 : target_msg.radius_2;
            armors[i].z =
                (i % 2 == 0) ? target_msg.position.z : target_msg.position.z + target_msg.dz;

            armors[i].x = newxc - armors[i].r * cos(armors[i].yaw);
            armors[i].y = newyc - armors[i].r * sin(armors[i].yaw);
            // 计算最短角度距离
            armors[i].cost = angles::shortest_angular_distance(gun_to_center_angle, armors[i].yaw);
        }

        // 找到最小的角度对应的装甲板
        std::sort(armors.begin(), armors.end(), [](const Armor & a, const Armor & b) {
            return std::abs(a.cost) < std::abs(b.cost);
        });
        Armor chosen_armor = armors[0];
        /*
        yaw, yaw_gimble未知
        t = (pi/2 - 2*yaw)/v_yaw
        2 * yaw_gimble = t * v_yaw_gimble
        so:
        2 * yaw_gimble = (pi/2 - 2*yaw) / v_yaw * v_yaw_gimble
        sin(pi - yaw - yaw_gimble) / distance = sin(yaw_gimble) / radius
        联立解得yaw
        */
        double yaw = findYaw(
            abs(target_msg.v_yaw), v_yaw_gimble, sqrt(pow(newxc, 2) + pow(newyc, 2)),
            chosen_armor.r);  // 通过迭代计算得到的放弃角
        // 策略1
        if (std::isnan(yaw) || abs(target_msg.v_yaw) < min_v || abs(armors[0].cost) <= yaw) {
            return {chosen_armor.yaw - target_msg.v_yaw * T, chosen_armor.z, chosen_armor.r};
        }
        // 策略2
        if ((armors[0].cost < 0 && target_msg.v_yaw < 0) ||
            (armors[0].cost > 0 && target_msg.v_yaw > 0)) {
            chosen_armor = armors[1];
        }
        yaw = chosen_armor.cost > 0 ? abs(yaw) : -abs(yaw);
        double set_yaw = gun_to_center_angle + yaw;
        return {set_yaw - target_msg.v_yaw * T, chosen_armor.z, chosen_armor.r};
    }
private: 
    // 配置求解器选项
    ceres::Solver::Options options;

    struct Armor
    {
        double x;
        double y;
        double z;
        double yaw;
        double r;
        double cost;  //装甲板和枪口的角度差
    };
    double findYaw(double v_yaw, double v_yaw_gimble, double distance, double radius)
    {
        // 初始猜测值
        double yaw = M_PI / 4;

        // 构建优化问题
        ceres::Problem problem;

        // 添加残差块
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<YawResidual, 1, 1>(
                new YawResidual(v_yaw, v_yaw_gimble, distance, radius, target_msg.armors_num)),
            nullptr, &yaw);

        // 求解
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 检查求解是否成功
        if (summary.termination_type == ceres::CONVERGENCE) {
            return yaw;
        } else {
            return std::nan("");  // 返回NaN表示求解失败
        }
    }
};

class ArmorInfo
{
private:
    target target_msg;            //目标信息
    double yaw, z, r;             //装甲板的偏航角、z轴坐标和半径
    Eigen::Vector3d odom2gunxyz;  //枪口坐标, 需要parameter_declare来调整参数
    Eigen::Vector3d odom2gunrpy;  //枪口坐标, 需要parameter_declare来调整参数
public:
    ArmorInfo(const Eigen::Vector3d & odom2gunxyz, const Eigen::Vector3d & odom2gunrpy)
    : odom2gunxyz(odom2gunxyz), odom2gunrpy(odom2gunrpy)
    {
    }
    void updateTarget(const target & new_target_msg, double new_yaw, double new_z, double new_r)
    {
        target_msg = new_target_msg;
        yaw = new_yaw;
        z = new_z;
        r = new_r;
    }
    Eigen::Vector3d getOdomTarget(double time)
    {
        // Get the predicted position
        Eigen::Vector3d target_position_odom = Eigen::Vector3d(
            target_msg.position.x + target_msg.velocity.x * time -
                r * cos(yaw + target_msg.v_yaw * time),
            target_msg.position.y + target_msg.velocity.y * time -
                r * sin(yaw + target_msg.v_yaw * time),
            z + target_msg.velocity.z * time);
        return target_position_odom;
    }
    Eigen::Vector3d getGunTarget(double time)
    {
        Eigen::Vector3d target_position_odom = getOdomTarget(time);
        // Transform to gun frame
        // The gun frame is rotated by odom2gunrpy and translated by odom2gunxyz
        Eigen::Matrix3d R_odom2gun = eulerToMatrix(odom2gunrpy);
        Eigen::Vector3d target_position_gun = R_odom2gun * target_position_odom + odom2gunxyz;
        return target_position_gun;
    }
    double getHorizontalDistance(double time)
    {
        Eigen::Vector3d target_position_odom = getGunTarget(time);
        return sqrt(pow(target_position_odom.x(), 2) + pow(target_position_odom.y(), 2));
    }
};

}  // namespace rm_auto_aim
#endif  // BALLISTIC_CALCULATION_TYPES_HPP_