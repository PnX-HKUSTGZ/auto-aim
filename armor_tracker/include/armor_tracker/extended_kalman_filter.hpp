#ifndef ARMOR_PROCESSOR__KALMAN_FILTER_HPP_
#define ARMOR_PROCESSOR__KALMAN_FILTER_HPP_

#include <Eigen/Dense>
#include <functional>

namespace rm_auto_aim
{

/**
 * @brief 扩展卡尔曼滤波器类，用于跟踪装甲板目标的状态估计
 * 
 * 该类实现了一个扩展卡尔曼滤波器，用于估计和预测装甲板目标的状态，
 * 包括位置、速度、角度等信息。支持单装甲板和双装甲板的观测更新。
 */
class ExtendedKalmanFilter
{
public:
    using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
    using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
    using VoidMatFunc = std::function<Eigen::MatrixXd()>;

    /**
     * @brief 默认构造函数
     */
    ExtendedKalmanFilter() = default;

    /**
     * @brief 构造一个扩展卡尔曼滤波器
     * 
     * @param f 状态转移函数，描述系统的动态模型
     * @param h1 装甲板1的观测函数，将状态映射到观测空间
     * @param h2 装甲板2的观测函数，将状态映射到观测空间
     * @param h_two 双装甲板观测函数，同时观测两个装甲板
     * @param j_f 状态转移函数的雅可比矩阵
     * @param j_h1 装甲板1观测函数的雅可比矩阵
     * @param j_h2 装甲板2观测函数的雅可比矩阵
     * @param j_h_two 双装甲板观测函数的雅可比矩阵
     * @param u_q 过程噪声协方差矩阵更新函数
     * @param u_r 单装甲板观测噪声协方差矩阵更新函数
     * @param u_r_two 双装甲板观测噪声协方差矩阵更新函数
     * @param P0 初始误差协方差矩阵
     */
    explicit ExtendedKalmanFilter(
        const VecVecFunc & f, const VecVecFunc & h1, const VecVecFunc & h2,
        const VecVecFunc & h_two, const VecMatFunc & j_f, const VecMatFunc & j_h1,
        const VecMatFunc & j_h2, const VecMatFunc & j_h_two, const VoidMatFunc & u_q,
        const VecMatFunc & u_r, const VecMatFunc u_r_two, const Eigen::MatrixXd & P0);

    /**
     * @brief 设置时间间隔，更新状态转移函数中的时间参数
     * 
     * @param dt 时间间隔（秒），用于更新位置和角度的预测
     */
    void setTimeInterval(double dt);

    /**
     * @brief 设置系统的初始状态
     * 
     * @param x0 初始状态向量，包含位置、速度、角度等信息
     */
    void setState(const Eigen::VectorXd & x0);

    /**
     * @brief 执行预测步骤，基于动态模型预测下一时刻的状态
     * 
     * 使用状态转移函数和过程噪声协方差矩阵进行状态预测，
     * 同时更新误差协方差矩阵。
     * 
     * @return 预测的状态向量
     */
    Eigen::VectorXd predict();

    /**
     * @brief 使用装甲板1的观测数据更新状态估计
     * 
     * @param z 观测向量，包含装甲板1的位置和角度信息
     * @return 更新后的状态向量
     */
    Eigen::VectorXd update1(const Eigen::VectorXd & z);

    /**
     * @brief 使用装甲板2的观测数据更新状态估计
     * 
     * @param z 观测向量，包含装甲板2的位置和角度信息
     * @return 更新后的状态向量
     */
    Eigen::VectorXd update2(const Eigen::VectorXd & z);

    /**
     * @brief 使用两个装甲板的观测数据同时更新状态估计
     * 
     * 当同时观测到两个装甲板时，使用联合观测进行状态更新，
     * 可以提供更准确的状态估计。
     * 
     * @param z 联合观测向量，包含两个装甲板的位置和角度信息
     * @return 更新后的状态向量
     */
    Eigen::VectorXd updateTwo(const Eigen::VectorXd & z);

private:
    /**
     * @brief 计算卡尔曼增益矩阵
     * 
     * 基于观测矩阵H和观测噪声协方差矩阵R计算最优的卡尔曼增益，
     * 用于平衡预测值和观测值的权重。
     * 
     * @param H 观测函数的雅可比矩阵
     * @param R 观测噪声协方差矩阵
     * @return 计算得到的卡尔曼增益矩阵
     */
    Eigen::MatrixXd compute_kalman_gain(const Eigen::MatrixXd & H, const Eigen::MatrixXd & R);

    /**
     * @brief 更新状态向量和协方差矩阵
     * 
     * 使用观测数据和卡尔曼增益更新状态估计和误差协方差矩阵，
     * 这是卡尔曼滤波器更新步骤的核心计算。
     * 
     * @param z 观测向量
     * @param H 观测函数的雅可比矩阵
     * @param h_func 观测函数
     * @param K 卡尔曼增益矩阵
     */
    void update_state_and_covariance(
        const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const VecVecFunc & h_func,
        const Eigen::MatrixXd & K);

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