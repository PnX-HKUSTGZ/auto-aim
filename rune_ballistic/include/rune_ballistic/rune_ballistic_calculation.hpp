#ifndef RUNE_BALLISTIC_CALCULATION_HPP_
#define RUNE_BALLISTIC_CALCULATION_HPP_

#include <auto_aim_interfaces/msg/rune_target.hpp>
#include <geometry_msgs/msg/detail/vector3__struct.hpp>
#include <geometry_msgs/msg/detail/point__struct.hpp>
#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <rclcpp/rclcpp.hpp>

//STD
#include <vector>
#include <utility>




namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::RuneTarget;

class Ballistic
{
public:
    Ballistic(double k = 0.1 , double K = 0.3, double bulletV = 30, Eigen::Vector3d odom2gun = Eigen::Vector3d(0,0,0)); //构造函数
    target target_msg;
    // 迭代,返回pitch和T，传入T和pitch的初始值
    std::pair<double,double> iteration(double &thres , double &init_pitch , double &initT);
    Eigen::Vector3d getTarget(double time); 
private:
    // Fitting Curve
#define BIG_RUNE_CURVE(x, a, omega, b, c, d, sign) \
    ((-((a) / (omega) * ceres::cos((omega) * ((x) + (d)))) + (b) * ((x) + (d)) + (c)) * (sign))
#define SMALL_RUNE_CURVE(x, a, b, c, sign) (((a) * ((x) + (b)) + (c)) * (sign))
    // Rune arm length, Unit: m
    const double ARM_LENGTH = 0.700;

    // parameter
    double bulletV;
    double K;//迭代时的步长，需要parameter_declare来调整参数
    double k; //空气阻力系数，需要parameter_declare来调整参数
    Eigen::Vector3d odom2gun; //枪口坐标, 需要parameter_declare来调整参数

    //预测角度
    double predictAngle(double time); 
    //预测位置
    Eigen::Vector3d getTargetPosition(double angle_diff); 
    //欧拉角转换为旋转矩阵
    Eigen::Matrix3d eulerToMatrix(const Eigen::Vector3d &euler, const std::string &order); 

    //ceres计算出一个T后进入该函数，去迭代出一个对准固定T预测目标(固定dist)的theta和T
    std::pair<double , double> fixTiteratPitch(double& horizon_dis , double& height); 
    double optimizeTime(double initial_guess , double& l , double& h, double& theta); 
    struct CostFunctor {
        Ballistic& ballistic_ref;
        double distance;
        double height;
        double theta;
        explicit CostFunctor( Ballistic& ballistic , double& distance , double& height, double& theta) : ballistic_ref(ballistic) , distance(distance) , height(height), theta(theta) {}

        template <typename T>
        bool operator()(const T* const t, T* residual) const {
            T v0 = T(ballistic_ref.bulletV);
            
            T distance_aim = T(distance);
            T height_aim = T(height);
            residual[0] =T(1 / ballistic_ref.k)*ceres::log(T(ballistic_ref.k)*ceres::cos(theta)*v0 * (*t) + T(1.0)) - T(ceres::sqrt(distance_aim * distance_aim + height_aim * height_aim)) ;
            return true;
        }
    };
};

}//namespace rm_auto_aim


#endif // RUNE_BALLISTIC_CALCULATION_HPP_