#include "ballistic/ballistic.hpp"

#include <cmath>

namespace rm_auto_aim
{

/**
 * @brief API call
 * 
 * @todo
 * @param info 目标状态信息,包括位置[角]速度,底盘类型以及id
 * @return true 解算成功
 * @return false 失败,迭代无法收敛或目标状态异常
 */
bool Ballistic::solveBallistic(Ballistic::TargetInfo & info)
{
  // 1. solveCVModel(); to get initial pitch and T to predict the pos of each armor

  // 2. predictBestTarget(); depending on the type(balance,normal,outpost)

  // for low speed armor, stop here and go to 4.
  // some logics to determine whether to toggle to holonomic model

  // 3. for high anglar vel target
  // use solveHolonomicModel(); to get final T and pitch

  // 4. calculate yaw angle using T

  // 5. publish the firecontrol message(to serial & debug)
}
// @todo
void Ballistic::solveCVModel()
{
  // loop begins

  // 1. init **pitch** by align gimbal to the target center, calc ``T``

  // 2. given the ``T``, calc dist then use fixTimeIterPitch() to get **pitch**

  // loop until converging
}
// @todo
void Ballistic::predictBestTarget()
{
  // 1. with the initial T given by solveCVModel(),
  // predict the pos of each armor
  // outpost, balance and normal

  // 2. choose the best armor to hit,
  // the metric is the angle between the armor and origin
}

// @todo
void Ballistic::solveHolonomicModel()
{
  // given the best target, consider rotation of chassis
  // solve the final time T and pitch
}

/**
 * @brief 固定时间迭代求解pitch,方法来自RoboRTS-Tutorial的弹道解算模型:
 *        https://github.com/RoboMaster/RoboRTS-Tutorial/tree/master/pdf
 * 
 * @todo 1.考虑IMU到摩擦轮/枪管测速模块的偏置以获得更精确的弹道
 *       2.改为ceres优化求解
 * 
 * @param horizon_dis 
 * @param height 
 */
std::pair<double, double> Ballistic::fixTimeIterPitch(double & horizon_dis, double & height)
{
  double dist_horizon = horizon_dis;  // 和目标在水平方向上的距离
  double target_height = height;      // 和目标在垂直方向上的距离

  // 迭代参数
  double vx, vh, fly_time, tmp_height = target_height, delta_height = 0, tmp_pitch, real_height;
#ifdef DEBUG_COMPENSATION
  cout << "init pitch: " << atan2(target_height, dist_horizon) * 180 / M_PI << endl;
#endif  // DEBUG_COMPENSATION
  for (size_t i = 0; i < 10; i++) {
    tmp_pitch = std::atan((tmp_height) / dist_horizon);
    vx = param_.bullet_speed * std::cos(tmp_pitch);
    vh = param_.bullet_speed * std::sin(tmp_pitch);

    fly_time =
      (std::exp(param_.air_resistance_k * dist_horizon) - 1) / (param_.air_resistance_k * vx);
    real_height = vh * fly_time - 0.5 * 9.8 * std::pow(fly_time, 2);
    delta_height = target_height - real_height;
    tmp_height += delta_height;
#ifdef DEBUG_COMPENSATION
    cout << "iter: " << i + 1 << " " << delta_height << endl;
#endif  // DEBUG_COMPENSATION
  }
#ifdef DEBUG_COMPENSATION
  cout << "res:" << tmp_pitch * 180 / 3.141592 << endl;
  cout << "fly_time:" << fly_time << endl;
#endif  // DEBUG_COMPENSATION
  return std::make_pair(tmp_pitch, fly_time);
}

}  // namespace rm_auto_aim