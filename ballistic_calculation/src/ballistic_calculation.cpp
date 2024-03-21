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

Ballistic::Ballistic( target target_msg ,double k , double K1 , double K2 , double bulletV)    
{
    this->target_msg = target_msg;
    robotcenter = target_msg.position;
    velocity = target_msg.velocity;
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
    std::pair<double, double> updateTmpThetaT;
    for(int i = 0 ; i < 100 ; i++){

        t = optimizeTime1(initT);
        double fx = target_msg.position.x + target_msg.velocity.x * t;
        double fy = target_msg.position.y + target_msg.velocity.y * t;
        double preddist = sqrt(fx * fx + fy * fy);
        double predheight = target_msg.position.z + target_msg.velocity.z * t;

        updateTmpThetaT = fixTiteratPitch(preddist , predheight);

        
        double temp = theta;
        theta = updateTmpThetaT.first;
        initT = updateTmpThetaT.second;
        differ = temp - theta;
        
        if(abs(differ) < thres){
            break;
        }
    }
    return std::make_pair( updateTmpThetaT.first , updateTmpThetaT.second);
    
}

std::pair<double,double> Ballistic::iteration2(double &thres , double &init_pitch , double &initT , double& yaw , double& z , double& r)
{
    theta = init_pitch;
    double differ;
    double t;
    std::pair<double , double>updateTmpThetaT;
    double x = target_msg.position.x + target_msg.velocity.x * initT + r * cos(yaw + target_msg.v_yaw * initT);
    double y = target_msg.position.y + target_msg.velocity.y * initT + r * sin(yaw + target_msg.v_yaw * initT);


    for(int i = 0 ; i < 100 ; i++){

        t = optimizeTime2(initT , x , y);
        double newyaw = yaw + target_msg.v_yaw * t;
        double fx = target_msg.position.x + target_msg.velocity.x * t + r * cos(newyaw);
        double fy = target_msg.position.x + target_msg.velocity.x * t + r * cos(newyaw);
        
        double preddist = sqrt(fx * fx + fy * fy);
        double predheight = z + target_msg.velocity.z * t;
        
        updateTmpThetaT = fixTiteratPitch(preddist , predheight);
        
        double temp = theta;
        theta = updateTmpThetaT.first;
        initT = updateTmpThetaT.second;
        differ = temp - theta;
        
        if(abs(differ) < thres){
            break;
        }

    }
    return std::make_pair( updateTmpThetaT.first , updateTmpThetaT.second);
    
    
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
    //options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Optional: Output a brief report to std::cout
    //std::cout << summary.BriefReport() << "\n";

    return t; // Return the optimized t value
}


double Ballistic::optimizeTime2(double initial_guess , double& x , double& y) {
    double t = initial_guess; // Initial guess for time t

    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Ballistic::CostFunctor2, 1, 1>(new Ballistic::CostFunctor2(*this, x , y)),
        nullptr,
        &t
    );

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    //options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Optional: Output a brief report to std::cout
    //std::cout << summary.BriefReport() << "\n";

    return t; // Return the optimized t value
}



std::pair<double , double> Ballistic::fixTiteratPitch(double& horizon_dis , double& height)
{
        
        

        double dist_horizon =  horizon_dis;// 和目标在水平方向上的距离
        double target_height = height;                    // 和目标在垂直方向上的距离

        // 迭代参数
        double vx, vy, fly_time, tmp_height = target_height, delta_height = 0, tmp_pitch, real_height;
#ifdef DEBUG_COMPENSATION
        cout << "init pitch: " << atan2(target_height, dist_horizon) * 180 / M_PI << endl;
#endif// DEBUG_COMPENSATION
        for (size_t i = 0; i < 10; i++)
        {
            tmp_pitch = atan((tmp_height) / dist_horizon);
            vx = bulletV * cos(tmp_pitch);
            vy = bulletV * sin(tmp_pitch);

            fly_time = (exp(k * dist_horizon) - 1) / (k * vx);
            real_height = vy * fly_time - 0.5 * 9.8 * pow(fly_time, 2);
            delta_height = target_height - real_height;
            tmp_height += delta_height;
#ifdef DEBUG_COMPENSATION
            cout << "iter: " << i + 1 << " " << delta_height << endl;
#endif// DEBUG_COMPENSATION
        }
#ifdef DEBUG_COMPENSATION
        cout << "res:" << tmp_pitch * 180 / 3.141592 << endl;
        cout << "fly_time:" << fly_time << endl;
#endif// DEBUG_COMPENSATION
        return std::make_pair(tmp_pitch , fly_time);
}





std::vector<double> Ballistic::predictInfantryBestArmor(double T)
{
    const double pi = 3.1415926;
    struct armor_info{
        double x;
        double y;
        double z;
        std::vector<double>vec;
        std::vector<double>vecto_odom;
        double yaw;
        double r;
    };

    armor_info armor1;
    armor_info armor2;
    armor_info armor3;
    armor_info armor4;

    armor1.r =armor3.r= target_msg.radius_1;
    armor2.r =armor4.r= target_msg.radius_2;
    
    //计算未来T时间的中心位置
    double newyaw = target_msg.yaw + target_msg.v_yaw * T;
    double newxc = target_msg.position.x + target_msg.velocity.x * T;
    double newyc = target_msg.position.y + target_msg.velocity.y * T;

    armor1.x = newxc + target_msg.radius_1 * cos(newyaw);
    armor1.y = newyc + target_msg.radius_1 * sin(newyaw);
    
    //顺时针填入1234装甲板(初始静止调试发现为逆时针，这里逻辑和预想不对，但静止时发现最终返回值正常)
    armor2.yaw = newyaw - pi/2;
    armor3.yaw = armor2.yaw - pi/2;
    armor4.yaw = armor3.yaw - pi/2;  
    armor2.x = newxc + target_msg.radius_2 * cos(armor2.yaw);
    armor2.y = newyc + target_msg.radius_2 * sin(armor2.yaw);
    armor3.x = newxc + target_msg.radius_1 * cos(armor3.yaw);
    armor3.y = newyc + target_msg.radius_1 * sin(armor3.yaw);
    armor4.x = newxc + target_msg.radius_2 * cos(armor4.yaw);
    armor4.y = newyc + target_msg.radius_2 * sin(armor4.yaw);


    armor1.vec = {newxc - armor1.x , newyc - armor1.y};
    armor2.vec = {newxc - armor2.x , newyc - armor2.y};
    armor3.vec = {newxc - armor3.x , newyc - armor3.y};
    armor4.vec = {newxc - armor4.x , newyc - armor4.y};

    armor1.vecto_odom = {armor1.x , armor1.y};
    armor2.vecto_odom = {armor2.x , armor2.y};
    armor3.vecto_odom = {armor3.x , armor3.y};
    armor4.vecto_odom = {armor4.x , armor4.y};

    armor1.z  = armor3.z = target_msg.position.z;
    armor2.z  = armor4.z = target_msg.position.z + target_msg.dz;

    std::map<double,armor_info>armorlist_map;

    double a = angleBetweenVectors(armor1.vec, armor1.vecto_odom);
    double b = angleBetweenVectors(armor2.vec, armor2.vecto_odom);
    double c = angleBetweenVectors(armor3.vec, armor3.vecto_odom);
    double d = angleBetweenVectors(armor4.vec, armor4.vecto_odom);


    armorlist_map[a] = armor1;
    armorlist_map[b] = armor2;
    armorlist_map[c] = armor3;
    armorlist_map[d] = armor4;

    
    armor_info chosen_armor =  armorlist_map[sortFourdoubles(a, b, c, d)];
    
    return {chosen_armor.yaw - target_msg.v_yaw * T , chosen_armor.z, chosen_armor.r};
}


std::vector<double> Ballistic::predictBalanceBestArmor(double T)
{
    const double pi = 3.1415926;
    struct armor_info{
        double x;
        double y;
        double z;
        std::vector<double>vec;
        std::vector<double>vecto_odom;
        double yaw;
        double r;
    };

    armor_info armor1;
    armor_info armor2;
   

    armor1.r =armor2.r= target_msg.radius_1;
    
    
    //计算未来T时间的中心位置
    double newyaw = target_msg.yaw + target_msg.v_yaw * T;
    double newxc = target_msg.position.x + target_msg.velocity.x * T;
    double newyc = target_msg.position.y + target_msg.velocity.y * T;

    armor1.x = newxc + target_msg.radius_1 * cos(newyaw);
    armor1.y = newyc + target_msg.radius_1 * sin(newyaw);
    
    //顺时针填入1234装甲板
    armor2.yaw = newyaw - pi;
  
    armor2.x = newxc + target_msg.radius_1 * cos(armor2.yaw);
    armor2.y = newyc + target_msg.radius_1 * sin(armor2.yaw);



    armor1.vec = {newxc - armor1.x , newyc - armor1.y};
    armor2.vec = {newxc - armor2.x , newyc - armor2.y};


    armor1.vecto_odom = {armor1.x , armor1.y};
    armor2.vecto_odom = {armor2.x , armor2.y};

    armor1.z  = armor2.z = target_msg.position.z;


    std::map<double,armor_info>armorlist_map;

    double a = angleBetweenVectors(armor1.vec, armor1.vecto_odom);
    double b = angleBetweenVectors(armor2.vec, armor2.vecto_odom);



    armorlist_map[a] = armor1;
    armorlist_map[b] = armor2;

    double chosen = (b < a) ? b : a;
    
    armor_info chosen_armor =  armorlist_map[chosen];
    
    return {chosen_armor.yaw - target_msg.v_yaw * T , chosen_armor.z, chosen_armor.r};
}



// 计算两个向量的点积
double Ballistic::dotProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

// 计算向量的模长
double Ballistic::magnitude(const std::vector<double>& v) {
    double sum = 0.0;
    for (auto& element : v) {
        sum += element * element;
    }
    return std::sqrt(sum);
}

double Ballistic::angleBetweenVectors(const std::vector<double>& v1, const std::vector<double>& v2) {
    double dot = dotProduct(v1, v2);
    double magV1 = magnitude(v1);
    double magV2 = magnitude(v2);
    
    // 计算余弦值
    double cosAngle = dot / (magV1 * magV2);
    
    // 将余弦值转换为弧度
    double angleRadians = std::acos(cosAngle);

    return angleRadians;
}

double Ballistic::sortFourdoubles(double a, double b, double c, double d) {
    std::vector<double> vec = {a, b, c, d};
    std::sort(vec.begin(), vec.end());
    return vec[0];
}





}//namespace rm_auto_aim