#include "rune_ballistic/rune_ballistic_calculation.hpp"
#include <auto_aim_interfaces/msg/rune_target.hpp>
#include <auto_aim_interfaces/msg/firecontrol.hpp>
#include <cmath>
#include <cstddef>
#include <vector>
#include <memory>
#include <numeric>
#include <map>
#include <algorithm>

#include <Eigen/src/Core/Matrix.h>
#include <ceres/ceres.h>


namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::RuneTarget;

Ballistic::Ballistic(double k , double K , double bulletV, Eigen::Vector3d odom2gun) {
    this->bulletV = bulletV;
    this->K = K;
    this->k = k;
    this->odom2gun = odom2gun;
}

Eigen::Vector3d Ballistic::getTarget(double time) {
    double predict_angle_diff = predictAngle(time) - predictAngle(target_msg.header.stamp.sec); 

    // Get the predicted position
    Eigen::Vector3d target_position_odom = getTargetPosition(predict_angle_diff);
    Eigen::Vector3d target_position_gun = target_position_odom - odom2gun;
    
    return target_position_gun;
}
std::pair<double,double> Ballistic::iteration(double &thres , double &init_pitch , double &initT) {
    double theta = init_pitch;
    double differ;
    double t;
    std::pair<double , double>update_tmp_theta_t;
    Eigen::Vector3d target = getTarget(initT + target_msg.header.stamp.sec);
    double distance = sqrt(pow(target[0] , 2) + pow(target[1], 2));
    double height = target[2];

    for(int i = 0 ; i < 100 ; i++){

        t = optimizeTime(initT , distance , height, theta);
        Eigen::Vector3d new_target = getTarget(t + target_msg.header.stamp.sec);
        
        double preddist = sqrt(pow(new_target[0] , 2) + pow(new_target[1], 2));
        double predheight = new_target[2]; 
        
        update_tmp_theta_t = fixTiteratPitch(preddist , predheight);
        
        differ = theta - update_tmp_theta_t.first;
        theta = update_tmp_theta_t.first;
        initT = update_tmp_theta_t.second;
        
        if(abs(differ) < thres){
            break;
        }
    }
    Eigen::Vector3d last_target = getTarget(update_tmp_theta_t.second + target_msg.header.stamp.sec);
    double predyaw = atan2(last_target[1] , last_target[0]);
    return std::make_pair( update_tmp_theta_t.first , predyaw);
}
double Ballistic::predictAngle(double time) {
    double current_time = time - target_msg.start_time;
    double pred_angle = 0;
    if (target_msg.is_big) {
        pred_angle = BIG_RUNE_CURVE(current_time,
                                    target_msg.fitting_curve[0],
                                    target_msg.fitting_curve[1],
                                    target_msg.fitting_curve[2],
                                    target_msg.fitting_curve[3],
                                    target_msg.fitting_curve[4],
                                    target_msg.direction);
    }
    else {
        pred_angle = SMALL_RUNE_CURVE(
        current_time, target_msg.fitting_curve[0], target_msg.fitting_curve[1], target_msg.fitting_curve[2], target_msg.direction);
    }
    return pred_angle; 
}
Eigen::Vector3d Ballistic::getTargetPosition(double angle_diff) {
    Eigen::Vector3d t_odom_2_rune = Eigen::Vector3d(target_msg.center.x, target_msg.center.y, target_msg.center.z);

    // Considering the large error and jitter(抖动) in the orientation obtained from PnP,
    // and the fact that the position of the Rune are static in the odom frame,
    // it is advisable to reconstruct the rotation matrix using geometric information
    double yaw = target_msg.yaw;
    double pitch = 0;
    double roll = target_msg.roll;
    Eigen::Matrix3d R_odom_2_rune =
        eulerToMatrix(Eigen::Vector3d{roll, pitch, yaw}, "XYZ");

    // Calculate the position of the armor in rune frame
    Eigen::Vector3d p_rune = Eigen::AngleAxisd(-angle_diff * 0, Eigen::Vector3d::UnitX()).matrix() *
                            Eigen::Vector3d(0, -ARM_LENGTH, 0);

    // Transform to odom frame
    Eigen::Vector3d p_odom = R_odom_2_rune * p_rune + t_odom_2_rune;

    return p_odom;
}
Eigen::Matrix3d Ballistic::eulerToMatrix(const Eigen::Vector3d &euler, const std::string &order) {
  Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
  if (order == "XYZ") {
    rotation_matrix = Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX()).toRotationMatrix() *
                      Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()).toRotationMatrix() *
                      Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ()).toRotationMatrix();
  } else if (order == "ZYX") {
    rotation_matrix = Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ()).toRotationMatrix() *
                      Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()).toRotationMatrix() *
                      Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX()).toRotationMatrix();
  }
  // 可以根据需要添加其他顺序
  return rotation_matrix;
}
std::pair<double , double> Ballistic::fixTiteratPitch(double& horizon_dis , double& height)
{
    double dist_horizon =  horizon_dis;// 和目标在水平方向上的距离
    double target_height = height;     // 和目标在垂直方向上的距离

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
double Ballistic::optimizeTime(double initial_guess , double& l , double& h, double& theta) {
    double t = initial_guess; // Initial guess for time t

    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Ballistic::CostFunctor, 1, 1>(new Ballistic::CostFunctor(*this, l , h, theta)),
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


}//namespace rm_auto_aim