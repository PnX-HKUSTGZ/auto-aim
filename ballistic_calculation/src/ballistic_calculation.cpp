#include "ballistic_calculation/ballistic_calculation.hpp"
#include <auto_aim_interfaces/msg/target.hpp>
#include <auto_aim_interfaces/msg/firecontrol.hpp>
#include<cmath>
#include <cstddef>
#include<vector>
#include <memory>
#include<numeric>
#include<map>
#include <algorithm>

#include <ceres/ceres.h>

namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::Target;

Ballistic::Ballistic(double bulletV , target target_msg ,double k , double K1 , double K2)    
{
    this->target_msg = target_msg;
    this->bulletV = bulletV;
    this->K1 = K1;
    this->K2 = K2;
    this->k = k;
    
    
}

std::pair<double,double> Ballistic::iteration1(double &thres , double &init_pitch , double &initT )
{
    theta = init_pitch;
    double differ;
    double t;
    for(int i = 0 ; i < 100 ; i++){

        double t = static_cast<double>(optimizeTime1(static_cast<double>(initT)));
        double hk = bulletV * static_cast<double>(sin(static_cast<double>(theta))) * t - 0.5 * 9.8 * t * t;//子弹击打高度
        double hr = robotcenter.z + velocity.z*t;//预测高度
        differ = hk - hr;
        if(differ < thres){
            break;
        }
        theta = theta + differ*K1;
    }
    return std::make_pair(theta , t);
    
}



double Ballistic::optimizeTime1(double initial_guess) {
    double t = initial_guess; // Initial guess for time t

    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Ballistic::CostFunctor1, 1, 1>(new Ballistic::CostFunctor1(*this)),
        nullptr,
        &t
    );

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Optional: Output a brief report to std::cout
    std::cout << summary.BriefReport() << "\n";

    return t; // Return the optimized t value
}


double Ballistic::optimizeTime2(double initial_guess) {
    double t = initial_guess; // Initial guess for time t

    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Ballistic::CostFunctor2, 1, 1>(new Ballistic::CostFunctor2(*this)),
        nullptr,
        &t
    );

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Optional: Output a brief report to std::cout
    std::cout << summary.BriefReport() << "\n";

    return t; // Return the optimized t value
}


std::vector<double> Ballistic::predictBestArmor(double T)
{
    const double pi = 3.1415926;
    
    double newyaw = target_msg.yaw + target_msg.v_yaw * T;
    double newxc = target_msg.position.x + target_msg.velocity.x * T;//计算未来T时间的中心位置
    double newyc = target_msg.position.y + target_msg.velocity.y * T;

    double newx1 = newxc + target_msg.radius_1 * cos(static_cast<double>(newyaw));
    double newy1 = newyc + target_msg.radius_1 * sin(static_cast<double>(newyaw));
    

    double yaw2 = newyaw - pi/2;
    double yaw3 = yaw2 - pi/2;
    double yaw4 = yaw3 - pi/2;  //顺时针填入1234装甲板

    double newx2 = newxc + target_msg.radius_2 * cos(static_cast<double>(yaw2));
    double newy2 = newyc + target_msg.radius_2 * sin(static_cast<double>(yaw2));

    double newx3 = newxc + target_msg.radius_1 * cos(static_cast<double>(yaw3));
    double newy3 = newyc + target_msg.radius_1 * sin(static_cast<double>(yaw3));

    double newx4 = newxc + target_msg.radius_2 * cos(static_cast<double>(yaw4));
    double newy4 = newyc + target_msg.radius_2 * sin(static_cast<double>(yaw4));

    std::vector<double>vec1 = {newxc - newx1 , newyc - newy1};
    std::vector<double>vec2 = {newxc - newx2 , newyc - newy2};
    std::vector<double>vec3 = {newxc - newx3 , newyc - newy3};
    std::vector<double>vec4 = {newxc - newx4 , newyc - newy4};

    std::vector<double>vecto_odom1 = {newx1 , newy1};
    std::vector<double>vecto_odom2 = {newx2 , newy2};
    std::vector<double>vecto_odom3 = {newx3 , newy3};
    std::vector<double>vecto_odom4 = {newx4 , newy4};

    std::map<double,std::vector<double>>armorlist_map;

    double a = angleBetweenVectors(vec1,vecto_odom1);
    double b = angleBetweenVectors(vec2,vecto_odom2);
    double c = angleBetweenVectors(vec3,vecto_odom3);
    double d = angleBetweenVectors(vec4,vecto_odom4);

    armorlist_map[a] = vecto_odom1;
    armorlist_map[b] = vecto_odom2;
    armorlist_map[c] = vecto_odom3;
    armorlist_map[d] = vecto_odom4;

    return armorlist_map[sortFourdoubles(a, b, c, d)];
}


// 计算两个向量的点积
double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

// 计算向量的模长
double magnitude(const std::vector<double>& v) {
    double sum = 0.0;
    for (auto& element : v) {
        sum += element * element;
    }
    return std::sqrt(sum);
}

double angleBetweenVectors(const std::vector<double>& v1, const std::vector<double>& v2) {
    double dot = dotProduct(v1, v2);
    double magV1 = magnitude(v1);
    double magV2 = magnitude(v2);
    
    // 计算余弦值
    double cosAngle = dot / (magV1 * magV2);
    
    // 将余弦值转换为弧度
    double angleRadians = std::acos(cosAngle);

    return angleRadians;
}

double sortFourdoubles(double a, double b, double c, double d) {
    std::vector<double> vec = {a, b, c, d};
    std::sort(vec.begin(), vec.end());
    return vec[0];
}


}//namespace rm_auto_aim