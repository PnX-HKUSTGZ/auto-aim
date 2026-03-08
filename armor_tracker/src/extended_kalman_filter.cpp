#include "armor_tracker/extended_kalman_filter.hpp"

namespace rm_auto_aim
{

ExtendedKalmanFilter::ExtendedKalmanFilter(
    const VecVecFunc & f, const VecVecFunc & h1, const VecVecFunc & h2, const VecVecFunc & h_two,
    const VecVecFunc & h_three, const VecMatFunc & j_f, const VecMatFunc & j_h1,
    const VecMatFunc & j_h2, const VecMatFunc & j_h_two, const VecMatFunc & j_h_three,
    const VoidMatFunc & u_q, const VecMatFunc & u_r, const VecMatFunc & u_r_two,
    const VecMatFunc & u_r_three, const Eigen::MatrixXd & P0)
: n(P0.rows()),  // 首先初始化系统维度
  f(f),
  h1(h1),
  h2(h2),
  h_two(h_two),
  h_three(h_three),
  jacobian_f(j_f),
  jacobian_h1(j_h1),
  jacobian_h2(j_h2),
  jacobian_h_two(j_h_two),
  jacobian_h_three(j_h_three),
  update_Q(u_q),
  update_R(u_r),
  update_R_two(u_r_two),
  update_R_three(u_r_three),
  I(Eigen::MatrixXd::Identity(n, n)),
  x_pri(n),
  x_post(n),
  P_post(P0),
  P_pri(P0.rows(), P0.cols())
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

void ExtendedKalmanFilter::setTimeInterval(double dt)
{
    // 更新状态转移函数，适配19维状态向量
    f = [dt](const Eigen::VectorXd & x) {
        Eigen::VectorXd x_new = x;
        // 更新位置: 位置 + 速度 * 时间
        x_new(0) += x(1) * dt;  // xc = xc + v_xc * dt
        x_new(2) += x(3) * dt;  // yc = yc + v_yc * dt
        x_new(4) += x(7) * dt;  // zc1 = zc1 + v_zc * dt
        x_new(5) += x(7) * dt;  // zc2 = zc2 + v_zc * dt
        x_new(6) += x(7) * dt;  // zc3 = zc3 + v_zc * dt

        // 更新偏航角: 偏航角 + 偏航角速度 * 时间
        x_new(12) += x(8) * dt;  // yaw1 = yaw1 + v_yaw * dt
        x_new(13) += x(8) * dt;  // yaw2 = yaw2 + v_yaw * dt
        x_new(14) += x(8) * dt;  // yaw3 = yaw3 + v_yaw * dt

        // 前哨站高差约束：Δz保持恒定（软约束）
        x_new(15) = x(5) - x(4);
        x_new(16) = x(6) - x(5);
        x_new(17) = x(4) - x(6);
        // 前哨站半径约束：固定基准半径
        x_new(18) = 0.2765;  // 前哨站固定半径值

        return x_new;
    };

    // 同时更新对应的雅可比矩阵函数
    jacobian_f = [dt](const Eigen::VectorXd &) {
        Eigen::MatrixXd f(19, 19);
        f.setZero();  // 先初始化全零，避免未初始化的随机值
        // clang-format off
        f<< 1,  dt,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // XC = XC + VXC*dt
            0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // VXC = VXC
            0,   0,   1,  dt,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // YC = YC + VYC*dt
            0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // VYC = VYC
            0,   0,   0,   0,   1,   0,   0,  dt,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // ZC1 = ZC1 + VZC*dt
            0,   0,   0,   0,   0,   1,   0,  dt,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // ZC2 = ZC2 + VZC*dt
            0,   0,   0,   0,   0,   0,   1,  dt,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // ZC3 = ZC3 + VZC*dt
            0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // VZC = VZC
            0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // VYAW = VYAW
            0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0, // R1 = R1
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0, // R2 = R2
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0, // R3 = R3
            0,   0,   0,   0,   0,   0,   0,   0,  dt,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0, // YAW1 = YAW1 + VYAW*dt
            0,   0,   0,   0,   0,   0,   0,   0,  dt,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0, // YAW2 = YAW2 + VYAW*dt
            0,   0,   0,   0,   0,   0,   0,   0,  dt,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0, // YAW3 = YAW3 + VYAW*dt
            0,   0,   0,   0,  -1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0, // DZ1 = ZC2 - ZC1 (Δz1=ZC2-ZC1)
            0,   0,   0,   0,   0,  -1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0, // DZ2 = ZC3 - ZC2 (Δz2=ZC3-ZC2)
            0,   0,   0,   0,   1,   0,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0, // DZ3 = ZC1 - ZC3 (Δz3=ZC1-ZC3)
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1; // R_OUTPOST = R_OUTPOST (固定值)
        // clang-format on
        return f;
    };
}
void ExtendedKalmanFilter::setState(const Eigen::VectorXd & x0) { x_post = x0; }

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

Eigen::MatrixXd ExtendedKalmanFilter::computeKalmanGain(
    const Eigen::MatrixXd & H, const Eigen::MatrixXd & R)
{
    temp_matrix.noalias() = H * P_pri * H.transpose() + R;
    return P_pri * H.transpose() * temp_matrix.inverse();
}

void ExtendedKalmanFilter::updateStateAndCovariance(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const VecVecFunc & h_func,
    const Eigen::MatrixXd & K)
{
    x_post.noalias() = x_pri + K * (z - h_func(x_pri));
    P_post.noalias() = (I - K * H) * P_pri;
}

Eigen::VectorXd ExtendedKalmanFilter::update1(const Eigen::VectorXd & z)
{
    H1.noalias() = jacobian_h1(x_pri);
    R.noalias() = update_R(z);
    K.noalias() = computeKalmanGain(H1, R);
    updateStateAndCovariance(z, H1, h1, K);
    return x_post;
}

Eigen::VectorXd ExtendedKalmanFilter::update2(const Eigen::VectorXd & z)
{
    H2.noalias() = jacobian_h2(x_pri);
    R.noalias() = update_R(z);
    K.noalias() = computeKalmanGain(H2, R);
    updateStateAndCovariance(z, H2, h2, K);
    return x_post;
}

Eigen::VectorXd ExtendedKalmanFilter::update3(const Eigen::VectorXd & z)
{
    // 复用update2的逻辑，仅替换H3/jacobian_h3
    static VecMatFunc jacobian_h3 = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(4, 19);
        h.setZero();
        double yaw = x(14), r = x(11);
        h(0, 0) = 1;
        h(0, 11) = -cos(yaw);
        h(0, 14) = r * sin(yaw);
        h(1, 2) = 1;
        h(1, 11) = -sin(yaw);
        h(1, 14) = -r * cos(yaw);
        h(2, 6) = 1;
        h(3, 14) = 1;
        return h;
    };
    Eigen::MatrixXd H3 = jacobian_h3(x_pri);
    R.noalias() = update_R(z);
    // 前哨站专属：给Δz/r设极小观测噪声（软约束）
    if (x_post(18) > 0) {
        R(2, 2) = 0.001;  // Z轴观测噪声极小
        R(3, 3) = 0.001;  // Yaw观测噪声极小
    }
    K.noalias() = computeKalmanGain(H3, R);
    updateStateAndCovariance(
        z, H3,
        [](const Eigen::VectorXd & x) {
            Eigen::VectorXd z(4);
            double xc = x(0), yc = x(2), yaw = x(14), r = x(11);
            z(0) = xc - r * cos(yaw);
            z(1) = yc - r * sin(yaw);
            z(2) = x(6);
            z(3) = yaw;
            return z;
        },
        K);
    return x_post;
}

Eigen::VectorXd ExtendedKalmanFilter::updateTwo(const Eigen::VectorXd & z)
{
    H_two.noalias() = jacobian_h_two(x_pri);
    R_two.noalias() = update_R_two(z);
    K.noalias() = computeKalmanGain(H_two, R_two);
    updateStateAndCovariance(z, H_two, h_two, K);
    return x_post;
}

Eigen::VectorXd ExtendedKalmanFilter::updateThreeArmors(const Eigen::VectorXd & z)
{
    H_two.noalias() = jacobian_h_three(x_pri);
    R_two.noalias() = update_R_three(z);
    // 前哨站约束：半径观测噪声设为极小
    if (R_two.rows() >= 16) {
        R_two.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 0.001;  // r1/r2/r3噪声极小
        R_two(15, 15) = 1e-6;                                             // r_outpost噪声极小
    }
    K.noalias() = computeKalmanGain(H_two, R_two);
    updateStateAndCovariance(z, H_two, h_three, K);
    return x_post;
}

}  // namespace rm_auto_aim