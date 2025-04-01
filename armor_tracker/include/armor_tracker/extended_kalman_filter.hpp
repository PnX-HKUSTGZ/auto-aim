#ifndef ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
#define ARMOR_PROCESSOR__KALMAN_FILTER_HPP_

#include <Eigen/Dense>
#include <functional>

namespace rm_auto_aim
{

class ExtendedKalmanFilter
{
public:
  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;

  ExtendedKalmanFilter() = default;

  explicit ExtendedKalmanFilter(
    const VecVecFunc & f, const VecVecFunc & h1, const VecVecFunc & h2, const VecVecFunc & h_two, 
    const VecMatFunc & j_f, const VecMatFunc & j_h1, const VecMatFunc & j_h2, const VecMatFunc & j_h_two,
    const VoidMatFunc & u_q, const VecMatFunc & u_r, const VecMatFunc u_r_two, const Eigen::MatrixXd & P0);

  void setState(const Eigen::VectorXd & x0);
  Eigen::VectorXd predict();
  Eigen::VectorXd update1(const Eigen::VectorXd & z);
  Eigen::VectorXd update2(const Eigen::VectorXd & z);
  Eigen::VectorXd updateTwo(const Eigen::VectorXd & z);

private:
  // Optimization: Cache commonly used values
  Eigen::MatrixXd compute_kalman_gain(const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);
  void update_state_and_covariance(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, 
                                 const VecVecFunc& h_func, const Eigen::MatrixXd& K);
  
  int n;  // System dimension

  VecVecFunc f, h1, h2, h_two;
  VecMatFunc jacobian_f, jacobian_h1, jacobian_h2, jacobian_h_two;
  VoidMatFunc update_Q;
  VecMatFunc update_R, update_R_two;

  Eigen::MatrixXd I;  // Identity matrix
  Eigen::VectorXd x_pri, x_post;
  Eigen::MatrixXd P_post, P_pri;
  
  // Cached matrices to avoid repeated allocation
  Eigen::MatrixXd F, H1, H2, H_two, Q, R, R_two, K;
  Eigen::MatrixXd temp_matrix;  // Reusable temporary matrix
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__KALMAN_FILTER_HPP_