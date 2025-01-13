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

Ballistic::Ballistic(double k , double K1 , double K2 , double bulletV)    
{
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
        double fy = target_msg.position.y + target_msg.velocity.y * t + r * sin(newyaw);
        
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
    double newyaw = yaw + target_msg.v_yaw * updateTmpThetaT.second;
    double fx = target_msg.position.x + target_msg.velocity.x * updateTmpThetaT.second + r * cos(newyaw);
    double fy = target_msg.position.y + target_msg.velocity.y * updateTmpThetaT.second + r * sin(newyaw);
    double predyaw = atan2(fy , fx);
    return std::make_pair( updateTmpThetaT.first , predyaw);
    
    
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




//重点修改
//火控代码
/*
一级策略：通过点乘算出最优装甲板
二级策略：引入放弃角度
三级策略：高速瞄中心
*/
std::vector<double> Ballistic::predictInfantryBestArmor(double T, double min_v, double max_v, double v_yaw_PTZ)
{
    if(abs(target_msg.v_yaw) < min_v){
        return stategy_1(T);
    }
    else if(abs(target_msg.v_yaw) > max_v){
        return {0.0, target_msg.position.z + 0.5 * target_msg.dz, 0.0, -1.0}; 
    }
    else{
        return stategy_2(T, v_yaw_PTZ);
    }
}
std::vector<double> Ballistic::predictOutpostBestArmor(double T, double min_v, double max_v, double v_yaw_PTZ)
{
    if(abs(target_msg.v_yaw) < min_v){
        return stategy_1_Outpost(T);
    }
    else if(abs(target_msg.v_yaw) > max_v){
        return {0.0, target_msg.position.z + 0.5 * target_msg.dz, 0.0, -1.0}; 
    }
    else{
        return stategy_2_Outpost(T, v_yaw_PTZ);
    }
}
std::vector<double> Ballistic::stategy_1(double T)
{
    // C++
    const double pi = 3.1415926;
    std::vector<Ballistic::armor_info> armors(4);

    // 计算未来 T 时间的中心位置
    double newyaw = target_msg.yaw + target_msg.v_yaw * T;
    double newxc = target_msg.position.x + target_msg.velocity.x * T;
    double newyc = target_msg.position.y + target_msg.velocity.y * T;

    // 初始化第一个装甲板的偏航角
    armors[0].yaw = newyaw;

    // 按顺时针方向计算每个装甲板的 yaw，并计算其坐标
    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            armors[i].yaw = armors[i - 1].yaw - pi / 2;
        }
        // 设置装甲板的高度,半径
        armors[i].r = (i % 2 == 0) ? target_msg.radius_1 : target_msg.radius_2;
        armors[i].z = (i % 2 == 0) ? target_msg.position.z : target_msg.position.z + target_msg.dz;

        armors[i].x = newxc + armors[i].r * cos(armors[i].yaw);
        armors[i].y = newyc + armors[i].r * sin(armors[i].yaw);

        armors[i].vec = {newxc - armors[i].x, newyc - armors[i].y};
        armors[i].vecto_odom = {armors[i].x, armors[i].y};
    }

    // 计算每个装甲板的角度，并存入 map
    std::map<double, armor_info> armorlist_map;
    std::vector<double> angles(4);
    for (int i = 0; i < 4; ++i) {
        angles[i] = angleBetweenVectors(armors[i].vec, armors[i].vecto_odom);
        armorlist_map[angles[i]] = armors[i];
    }

    // 找到最小的角度对应的装甲板
    sort(angles.begin(), angles.end());
    armor_info chosen_armor; 
    chosen_armor = armorlist_map[angles[0]];
    if(abs(angles[1]) - abs(angles[0]) < 15.0 * pi / 180.0  && abs(chosen_armor.yaw - last_yaw) > abs(armorlist_map[angles[1]].yaw - last_yaw)){
        chosen_armor = armorlist_map[angles[1]];
    }
    last_yaw = chosen_armor.yaw;

    return {chosen_armor.yaw - target_msg.v_yaw * T , chosen_armor.z, chosen_armor.r};
}
std::vector<double> Ballistic::stategy_2(double T, double v_yaw_PTZ)
{
    // C++
    const double pi = 3.1415926;
    std::vector<Ballistic::armor_info> armors(4);

    // 计算未来 T 时间的中心位置
    double newyaw = target_msg.yaw + target_msg.v_yaw * T;
    double newxc = target_msg.position.x + target_msg.velocity.x * T;
    double newyc = target_msg.position.y + target_msg.velocity.y * T;

    // 初始化第一个装甲板的偏航角
    armors[0].yaw = newyaw;

    // 按顺时针方向计算每个装甲板的 yaw，并计算其坐标
    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            armors[i].yaw = armors[i - 1].yaw - pi / 2;
        }
        // 设置装甲板的高度,半径
        armors[i].r = (i % 2 == 0) ? target_msg.radius_1 : target_msg.radius_2;
        armors[i].z = (i % 2 == 0) ? target_msg.position.z : target_msg.position.z + target_msg.dz;

        armors[i].x = newxc + armors[i].r * cos(armors[i].yaw);
        armors[i].y = newyc + armors[i].r * sin(armors[i].yaw);

        armors[i].vec = {newxc - armors[i].x, newyc - armors[i].y};
        armors[i].vecto_odom = {armors[i].x, armors[i].y};
    }

    // 计算每个装甲板的角度，并存入 map
    std::map<double, armor_info> armorlist_map;
    std::vector<double> angles(4);
    for (int i = 0; i < 4; ++i) {
        angles[i] = angleBetweenVectors(armors[i].vec, armors[i].vecto_odom);
        armorlist_map[angles[i]] = armors[i];
    }

    // 找到最小的角度对应的装甲板
    sort(angles.begin(), angles.end());
    armor_info chosen_armor = armorlist_map[angles[0]];
    /*
    yaw, yaw_PTZ未知
    t = (pi/2 - 2*yaw)/v_yaw
    2 * yaw_PTZ = t * v_yaw_PTZ
    so:
    2 * yaw_PTZ = (pi/2 - 2*yaw) / v_yaw * v_yaw_PTZ
    sin(pi - yaw - yaw_PTZ) / distance = sin(yaw) / radius
    联立解得yaw
    */
    double yaw = findYaw(target_msg.v_yaw, v_yaw_PTZ, sqrt(pow(newxc, 2) + pow(newyc, 2)), chosen_armor.r); // 通过迭代计算得到的 yaw
    //std::cout << "giveup_yaw: " << yaw << std::endl;
    if (chosen_armor.yaw > yaw) {
        chosen_armor = armorlist_map[angles[1]]; 
        return {chosen_armor.yaw - target_msg.v_yaw * T, chosen_armor.z, chosen_armor.r};
    }
    else {
        return {chosen_armor.yaw - target_msg.v_yaw * T, chosen_armor.z, chosen_armor.r};
    }
}
std::vector<double> Ballistic::stategy_1_Outpost(double T)//4 armors -> 3 armors
{
    // C++
    const double pi = 3.1415926;
    std::vector<Ballistic::armor_info> armors(3);
     double rr =0.325; // 前哨站专属半径
    // 计算未来 T 时间的中心位置
    double newyaw = target_msg.yaw + target_msg.v_yaw * T;
    double newxc = target_msg.position.x + target_msg.velocity.x * T;
    double newyc = target_msg.position.y + target_msg.velocity.y * T;

    // 初始化第一个装甲板的偏航角
    armors[0].yaw = newyaw;

    // 按顺时针方向计算每个装甲板的 yaw，并计算其坐标
    for (int i = 0; i < 3; ++i) {
        if (i > 0) {
            armors[i].yaw = armors[i - 1].yaw - (2*pi / 3);
        }
        // 设置装甲板的高度,半径
        armors[i].r = rr;
        armors[i].z = (i % 2 == 0) ? target_msg.position.z : target_msg.position.z + target_msg.dz;

        armors[i].x = newxc + armors[i].r * cos(armors[i].yaw);
        armors[i].y = newyc + armors[i].r * sin(armors[i].yaw);

        armors[i].vec = {newxc - armors[i].x, newyc - armors[i].y};
        armors[i].vecto_odom = {armors[i].x, armors[i].y};
    }

    // 计算每个装甲板的角度，并存入 map
    std::map<double, armor_info> armorlist_map;
    std::vector<double> angles(3);
    for (int i = 0; i < 3; ++i) {
        angles[i] = angleBetweenVectors(armors[i].vec, armors[i].vecto_odom);
        armorlist_map[angles[i]] = armors[i];
    }

    // 找到最小的角度对应的装甲板
    sort(angles.begin(), angles.end());
    armor_info chosen_armor; 
    chosen_armor = armorlist_map[angles[0]];
    if(abs(angles[1]) - abs(angles[0]) < 15.0 * pi / 180.0  && abs(chosen_armor.yaw - last_yaw) > abs(armorlist_map[angles[1]].yaw - last_yaw)){
        chosen_armor = armorlist_map[angles[1]];
    }
    last_yaw = chosen_armor.yaw;

    return {chosen_armor.yaw - target_msg.v_yaw * T , chosen_armor.z, chosen_armor.r};
}
std::vector<double> Ballistic::stategy_2_Outpost(double T, double v_yaw_PTZ)
{
    // C++
    const double pi = 3.1415926;
    std::vector<Ballistic::armor_info> armors(3);
     double rr =0.325; 
    // 计算未来 T 时间的中心位置
    double newyaw = target_msg.yaw + target_msg.v_yaw * T;
    double newxc = target_msg.position.x + target_msg.velocity.x * T;
    double newyc = target_msg.position.y + target_msg.velocity.y * T;

    // 初始化第一个装甲板的偏航角
    armors[0].yaw = newyaw;

    // 按顺时针方向计算每个装甲板的 yaw，并计算其坐标
    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            armors[i].yaw = armors[i - 1].yaw - pi / 2;
        }
        // 设置装甲板的高度,半径
        armors[i].r = rr;
        armors[i].z = (i % 2 == 0) ? target_msg.position.z : target_msg.position.z + target_msg.dz;

        armors[i].x = newxc + armors[i].r * cos(armors[i].yaw);
        armors[i].y = newyc + armors[i].r * sin(armors[i].yaw);

        armors[i].vec = {newxc - armors[i].x, newyc - armors[i].y};
        armors[i].vecto_odom = {armors[i].x, armors[i].y};
    }

    // 计算每个装甲板的角度，并存入 map
    std::map<double, armor_info> armorlist_map;
    std::vector<double> angles(4);
    for (int i = 0; i < 4; ++i) {
        angles[i] = angleBetweenVectors(armors[i].vec, armors[i].vecto_odom);
        armorlist_map[angles[i]] = armors[i];
    }

    // 找到最小的角度对应的装甲板
    sort(angles.begin(), angles.end());
    armor_info chosen_armor = armorlist_map[angles[0]];
    /*
    yaw, yaw_PTZ未知
    t = (pi/2 - 2*yaw)/v_yaw
    2 * yaw_PTZ = t * v_yaw_PTZ
    so:
    2 * yaw_PTZ = (pi/2 - 2*yaw) / v_yaw * v_yaw_PTZ
    sin(pi - yaw - yaw_PTZ) / distance = sin(yaw) / radius
    联立解得yaw
    */
    double yaw = findYaw(target_msg.v_yaw, v_yaw_PTZ, sqrt(pow(newxc, 2) + pow(newyc, 2)), chosen_armor.r); // 通过迭代计算得到的 yaw
    //std::cout << "giveup_yaw: " << yaw << std::endl;
    if (chosen_armor.yaw > yaw) {
        chosen_armor = armorlist_map[angles[1]]; 
        return {chosen_armor.yaw - target_msg.v_yaw * T, chosen_armor.z, chosen_armor.r};
    }
    else {
        return {chosen_armor.yaw - target_msg.v_yaw * T, chosen_armor.z, chosen_armor.r};
    }
}
double Ballistic::findYaw(double v_yaw, double v_yaw_PTZ, double distance, double radius) {
    // 常量定义
    const double epsilon = 1e-3; // 迭代精度
    const int max_iter = 100;    // 最大迭代次数

    // 初始猜测值（根据实际情况调整）
    double yaw = M_PI / 4;

    for (int i = 0; i < max_iter; ++i) {
        // 计算 yaw_PTZ
        double numerator = (M_PI / 2 - 2 * yaw) * v_yaw_PTZ;
        double denominator = 2 * v_yaw;
        double yaw_PTZ = numerator / denominator;

        // 计算方程左侧和右侧
        double left = sin(yaw + yaw_PTZ) / distance;
        double right = sin(yaw) / radius;
        double f = left - right;

        // 判断是否收敛
        if (fabs(f) < epsilon) {
            break;
        }

        // 计算导数 f'(yaw)
        double dyaw_PTZ = (-2 * v_yaw_PTZ) / denominator;
        double df = (cos(yaw + yaw_PTZ) * (1 + dyaw_PTZ)) / distance - (cos(yaw) / radius);

        // 更新 yaw 值
        yaw -= f / df;
    }

    return yaw;
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





}//namespace rm_auto_aim