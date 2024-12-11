// Copyright 2022 Chen Jun

#include "armor_tracker/extended_kalman_filter.hpp"

namespace rm_auto_aim
{
ExtendedKalmanFilter::ExtendedKalmanFilter(
    const VecVecFunc & f, const VecVecFunc & h1, const VecVecFunc & h2, const VecVecFunc & h_two, 
    const VecMatFunc & j_f, const VecMatFunc & j_h1, const VecMatFunc & j_h2, const VecMatFunc & j_h_two,
    const VoidMatFunc & u_q, const VecMatFunc & u_r, const VecMatFunc u_r_two, const Eigen::MatrixXd & P0)
: f(f),
  h1(h1),
  h2(h2),
  h_two(h_two),
  jacobian_f(j_f),
  jacobian_h1(j_h1),
  jacobian_h2(j_h2),
  jacobian_h_two(j_h_two),
  update_Q(u_q),
  update_R(u_r),
  update_R_two(u_r_two),
  P_post(P0),
  n(P0.rows()),
  I(Eigen::MatrixXd::Identity(n, n)),
  x_pri(n),
  x_post(n)
{
}

void ExtendedKalmanFilter::setState(const Eigen::VectorXd & x0) { x_post = x0; }

Eigen::MatrixXd ExtendedKalmanFilter::predict()
{
  F = jacobian_f(x_post), Q = update_Q(); 

  x_pri = f(x_post);
  P_pri = F * P_post * F.transpose() + Q;

  // handle the case when there will be no measurement before the next predict
  x_post = x_pri;
  //如果没有观测值（后验状态），就把先验状态作为后验状态（预测值当作真实值）去计算下一个时刻的状态
  //如果有观测值，x_post就会被覆盖
  P_post = P_pri;

  return x_pri;
}

Eigen::MatrixXd ExtendedKalmanFilter::update1(const Eigen::VectorXd & z)
{
  H1 = jacobian_h1(x_pri), R = update_R(z);

  K = P_pri * H1.transpose() * (H1 * P_pri * H1.transpose() + R).inverse();
  x_post = x_pri + K * (z - h1(x_pri));
  P_post = (I - K * H1) * P_pri;

  return x_post;
}

Eigen::MatrixXd ExtendedKalmanFilter::update2(const Eigen::VectorXd & z)
{
  H2 = jacobian_h2(x_pri), R = update_R(z);

  K = P_pri * H2.transpose() * (H2 * P_pri * H2.transpose() + R).inverse();
  x_post = x_pri + K * (z - h2(x_pri));
  P_post = (I - K * H2) * P_pri;

  return x_post;
}
Eigen::MatrixXd ExtendedKalmanFilter::updateTwo(const Eigen::VectorXd & z)
{
  H_two = jacobian_h_two(x_pri), R_two = update_R_two(z);

  K = P_pri * H_two.transpose() * (H_two * P_pri * H_two.transpose() + R_two).inverse();
  x_post = x_pri + K * (z - h_two(x_pri));
  P_post = (I - K * H_two) * P_pri;

  return x_post;
}

}  // namespace rm_auto_aim
