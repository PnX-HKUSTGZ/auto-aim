// Copyright 2022 Chen Jun

#ifndef ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
#define ARMOR_PROCESSOR__KALMAN_FILTER_HPP_

#include <Eigen/Dense>
#include <functional>

namespace rm_auto_aim
{

class ExtendedKalmanFilter
{
public:
  ExtendedKalmanFilter() = default;

  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;

  explicit ExtendedKalmanFilter(
    const VecVecFunc & f, const VecVecFunc & h1, const VecVecFunc & h2, const VecVecFunc & h_two, 
    const VecMatFunc & j_f, const VecMatFunc & j_h1, const VecMatFunc & j_h2, const VecMatFunc & j_h_two,
    const VoidMatFunc & u_q, const VecMatFunc & u_r, const VecMatFunc u_r_two, const Eigen::MatrixXd & P0);

  // Set the initial state
  void setState(const Eigen::VectorXd & x0);

  // Compute a predicted state
  Eigen::MatrixXd predict();

  // Update the estimated state based on measurement
  Eigen::MatrixXd update1(const Eigen::VectorXd & z);
  Eigen::MatrixXd update2(const Eigen::VectorXd & z);
  // Update the estimated state based on measurement2 with two armors
  Eigen::MatrixXd updateTwo(const Eigen::VectorXd & z);

private:
  // Process nonlinear vector function
  VecVecFunc f;
  // Observation nonlinear vector function
  VecVecFunc h1;
  VecVecFunc h2;
  VecVecFunc h_two;
  // Jacobian of f()
  VecMatFunc jacobian_f;
  Eigen::MatrixXd F;
  // Jacobian of h()
  VecMatFunc jacobian_h1;
  Eigen::MatrixXd H1;
  VecMatFunc jacobian_h2;
  Eigen::MatrixXd H2;
  VecMatFunc jacobian_h_two;
  Eigen::MatrixXd H_two;
  // Process noise covariance matrix
  VoidMatFunc update_Q;
  Eigen::MatrixXd Q;
  // Measurement noise covariance matrix
  VecMatFunc update_R;
  Eigen::MatrixXd R;
  VecMatFunc update_R_two;
  Eigen::MatrixXd R_two;

  // Priori error estimate covariance matrix
  Eigen::MatrixXd P_pri;
  // Posteriori error estimate covariance matrix
  Eigen::MatrixXd P_post;

  // Kalman gain
  Eigen::MatrixXd K;

  // System dimensions
  int n;

  // N-size identity
  Eigen::MatrixXd I;

  // Priori state
  Eigen::VectorXd x_pri;
  // Posteriori state
  Eigen::VectorXd x_post;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
