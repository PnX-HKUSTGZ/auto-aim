#include "ballistic_calculation/mpc_controller.hpp"

#include <vector>

#include "ballistic_calculation/math_util.hpp"
#include "yaml-cpp/yaml.h"

namespace rm_auto_aim
{
MPCController::MPCController(const std::string & config_path)
{
  YAML::Node yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;
  fire_thresh_ = yaml["fire_thresh"].as<double>();
  decision_speed_ = yaml["decision_speed"].as<double>();
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();

  setupYawSolver(config_path);
  setupPitchSolver(config_path);
}

MPCResult MPCController::compute(
  const Eigen::Vector3d & target_odom,
  const Eigen::Vector4d & current_gimbal_state,
  double bullet_speed)
{
  (void)current_gimbal_state;
  MPCResult result;
  result.is_valid = false;

  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target_odom, bullet_speed)(0);
    traj = getTrajectory(target_odom, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    return result;
  }

  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  result.is_valid = true;
  result.target_yaw = limit_rad(traj(0, HALF_HORIZON) + yaw0);
  result.target_pitch = traj(2, HALF_HORIZON);

  result.yaw = limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  result.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  result.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  result.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  result.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  result.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  int shoot_offset_ = 2;
  result.is_fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  return result;
}

void MPCController::setupYawSolver(const std::string & config_path)
{
  YAML::Node yaml = YAML::LoadFile(config_path);
  double max_yaw_acc = yaml["max_yaw_acc"].as<double>();
  std::vector<double> Q_yaw = yaml["Q_yaw"].as<std::vector<double>>();
  std::vector<double> R_yaw = yaml["R_yaw"].as<std::vector<double>>();

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void MPCController::setupPitchSolver(const std::string & config_path)
{
  YAML::Node yaml = YAML::LoadFile(config_path);
  double max_pitch_acc = yaml["max_pitch_acc"].as<double>();
  std::vector<double> Q_pitch = yaml["Q_pitch"].as<std::vector<double>>();
  std::vector<double> R_pitch = yaml["R_pitch"].as<std::vector<double>>();

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> MPCController::aim(const Eigen::Vector3d & target_odom, double bullet_speed)
{
  double dist = target_odom.head<2>().norm();
  double azim = std::atan2(target_odom.y(), target_odom.x());
  Ballistic ballistic(0.1, bullet_speed, 0.0);
  double horizon_dis = dist;
  double height = target_odom.z();
  auto [pitch, _] = ballistic.fixTiteratPitch(horizon_dis, height);
  return {limit_rad(azim + yaw_offset_), -pitch - pitch_offset_};
}

Trajectory MPCController::getTrajectory(const Eigen::Vector3d & target_odom, double yaw0, double bullet_speed)
{
  Trajectory traj;

  Eigen::Vector3d target_pred = target_odom;
  target_pred = target_pred - Eigen::Vector3d(0, 0, DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target_pred, bullet_speed);

  target_pred = target_pred + Eigen::Vector3d(0, 0, DT);
  auto yaw_pitch = aim(target_pred, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target_pred = target_pred + Eigen::Vector3d(0, 0, DT);
    auto yaw_pitch_next = aim(target_pred, bullet_speed);

    auto yaw_vel = limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

}  // namespace rm_auto_aim