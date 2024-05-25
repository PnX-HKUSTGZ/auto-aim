#ifndef __BALLISTIC_H__
#define __BALLISTIC_H__

#include <ceres/ceres.h>
#include <ceres/jet.h>

#include <cmath>
#include <memory>
#include <string>

namespace rm_auto_aim
{
enum class TargetType { NORMAL_4 = 4, BALANCE_2 = 2, OUTPOST_3 = 3 };

class Ballistic
{
public:
  struct TargetInfo
  {
    double x, y, z, yaw;
    double vx, vy, vz, w;
    double r1, r2, dz;
    TargetType type;
    std::string id;
  };

  struct BallisticParam
  {
    double bullet_speed;
    double gravity;
    double air_resistance_k;

    double step_cv;
    double thre_cv;
    double mx_iter_cv;

    double step_holonomic;
    double thre_holonomic;
    double mx_iter_holonomic;

    double step_calc_pitch;
    double thre_calc_pitch;
    double mx_iter_calc_pitch;

    double extra_compensation;
  };

  BallisticParam param_;

  Ballistic(const BallisticParam & param) : param_(param)
  {
    cv_cost_functor_ =
      std::make_unique<CVCostFunctor>(param_.air_resistance_k, param_.bullet_speed);
    holonomic_cost_functor_ =
      std::make_unique<HolonomicCostFunctor>(param_.air_resistance_k, param_.bullet_speed);
  };

  bool solveBallistic(Ballistic::TargetInfo & info);

private:
  void solveCVModel();

  void predictBestTarget();

  void solveHolonomicModel();

  std::pair<double, double> fixTimeIterPitch(double & horizon_dis, double & height);

private:
  struct CVCostFunctor
  {
    // update every iteration
    double x, y, vx, vy;
    double pitch_theta;

    // constant after initialization
    double air_resistance_k;
    double bullet_speed;

    CVCostFunctor(double air_resistance_k, double bullet_speed)
    : air_resistance_k(air_resistance_k), bullet_speed(bullet_speed)
    {
    }

    template <typename T>
    bool operator()(const T * const t, T * residual) const
    {
      T futurex2 = ceres::pow(x + vx * (*t), 2);
      T futurey2 = ceres::pow(y + vy * (*t), 2);

      residual[0] =
        1 / air_resistance_k *
          ceres::log(air_resistance_k * ceres::cos(pitch_theta) * bullet_speed * (*t) + 1) -
        ceres::sqrt(futurex2 + futurey2);

      return true;
    }
  };

  // @todo: finish residual calculation
  struct HolonomicCostFunctor
  {
    // update every iteration
    double x, y, vx, vy, w;
    double pitch_theta, r;

    // constant after initialization
    double air_resistance_k;
    double bullet_speed;

    HolonomicCostFunctor(double air_resistance_k, double bullet_speed)
    : air_resistance_k(air_resistance_k), bullet_speed(bullet_speed)
    {
    }

    template <typename T>
    bool operator()(const T * const t, T * residual) const
    {
      T futurex = x + vx * (*t) + w * (*t);
      //   T futurey = y + vy * (*t);

      residual[0] = 0;

      return true;
    }
  };

  std::unique_ptr<CVCostFunctor> cv_cost_functor_;
  std::unique_ptr<HolonomicCostFunctor> holonomic_cost_functor_;
};

}  // namespace rm_auto_aim

#endif  // !__BALLISTIC_H__