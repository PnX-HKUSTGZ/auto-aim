#include "armor_tracker/extended_kalman_filter.hpp"

namespace rm_auto_aim
{

ExtendedKalmanFilter::ExtendedKalmanFilter(
    const VecVecFunc & f, const VecVecFunc & h1, const VecVecFunc & h2, const VecVecFunc & h_two, 
    const VecMatFunc & j_f, const VecMatFunc & j_h1, const VecMatFunc & j_h2, const VecMatFunc & j_h_two,
    const VoidMatFunc & u_q, const VecMatFunc & u_r, const VecMatFunc u_r_two, const Eigen::MatrixXd & P0)
: n(P0.rows()),  // 首先初始化系统维度
  f(f), h1(h1), h2(h2), h_two(h_two),
  jacobian_f(j_f), jacobian_h1(j_h1), jacobian_h2(j_h2), jacobian_h_two(j_h_two),
  update_Q(u_q), update_R(u_r), update_R_two(u_r_two),
  I(Eigen::MatrixXd::Identity(n, n)),
  x_pri(n), x_post(n),
  P_post(P0), P_pri(P0.rows(), P0.cols())
{
    // Pre-allocate matrices
    F.resize(n, n);
    H1.resize(n, n);
    H2.resize(n, n);
    H_two.resize(n, n);
    Q.resize(n, n);
    R.resize(n, n);
    R_two.resize(n, n);
    K.resize(n, n);
    temp_matrix.resize(n, n);
}

void ExtendedKalmanFilter::setState(const Eigen::VectorXd & x0) 
{ 
    x_post = x0; 
}

Eigen::VectorXd ExtendedKalmanFilter::predict()
{
    // Compute Jacobian and process noise
    F.noalias() = jacobian_f(x_post);
    Q.noalias() = update_Q();

    // State prediction
    x_pri.noalias() = f(x_post);
    
    // Covariance prediction using optimized matrix operations
    temp_matrix.noalias() = F * P_post;
    P_pri.noalias() = temp_matrix * F.transpose() + Q;

    // Handle no measurement case
    x_post = x_pri;
    P_post = P_pri;

    return x_pri;
}

Eigen::MatrixXd ExtendedKalmanFilter::compute_kalman_gain(
    const Eigen::MatrixXd& H, const Eigen::MatrixXd& R)
{
    temp_matrix.noalias() = H * P_pri * H.transpose() + R;
    return P_pri * H.transpose() * temp_matrix.inverse();
}

void ExtendedKalmanFilter::update_state_and_covariance(
    const Eigen::VectorXd& z, const Eigen::MatrixXd& H, 
    const VecVecFunc& h_func, const Eigen::MatrixXd& K)
{
    x_post.noalias() = x_pri + K * (z - h_func(x_pri));
    P_post.noalias() = (I - K * H) * P_pri;
}

Eigen::VectorXd ExtendedKalmanFilter::update1(const Eigen::VectorXd & z)
{
    H1.noalias() = jacobian_h1(x_pri);
    R.noalias() = update_R(z);
    K.noalias() = compute_kalman_gain(H1, R);
    update_state_and_covariance(z, H1, h1, K);
    return x_post;
}

Eigen::VectorXd ExtendedKalmanFilter::update2(const Eigen::VectorXd & z)
{
    H2.noalias() = jacobian_h2(x_pri);
    R.noalias() = update_R(z);
    K.noalias() = compute_kalman_gain(H2, R);
    update_state_and_covariance(z, H2, h2, K);
    return x_post;
}

Eigen::VectorXd ExtendedKalmanFilter::updateTwo(const Eigen::VectorXd & z)
{
    H_two.noalias() = jacobian_h_two(x_pri);
    R_two.noalias() = update_R_two(z);
    K.noalias() = compute_kalman_gain(H_two, R_two);
    update_state_and_covariance(z, H_two, h_two, K);
    return x_post;
}

}  // namespace rm_auto_aim