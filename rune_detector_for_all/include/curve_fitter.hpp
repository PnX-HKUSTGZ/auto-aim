#ifndef RUNE_DETECTOR_CURVE_FITTER_HPP_
#define RUNE_DETECTOR_CURVE_FITTER_HPP_

// std
#include <future>
#include <deque>
#include <array>
#include <memory>
#include <string>

// third party
#include <ceres/ceres.h>
#include <rclcpp/rclcpp.hpp>

// project
#include "types.hpp"

namespace rune_detector_for_all
{

class CurveFitter
{
public:
    explicit CurveFitter(const MotionType & t);

    // Update data to be fitted
    void update(double time, double angle);

    // Reset fitter
    void reset();

    // Check the state of fitter
    bool statusVerified();

    // Set the type of fitter
    void setType(const MotionType & t);

    void setAutoTypeDetermined(bool auto_type_determined);

    MotionType getType() const;

    // Get the string of the fitting result
    std::string getDebugText();

    // Get the fitting parameters
    std::array<double, 5> getFittingParam() const;

    // Get the direction of the fitting curve
    bool getDirection() const;

    double start_time;

private:
    // Perform double curve fitting
    // automated determination of the type of curve
    void fitDoubleCurve();

    // Perform curve fitting
    void fitCurve();

    // Status value
    MotionType type_;
    bool is_static_ = false;
    bool auto_type_determined_ = false;
    int direction_;

    // Data to be fitted
    static constexpr int QUEUE_UPPER_LIMIT = 500;
    static constexpr int QUEUE_LOWER_LIMIT = 50;
    struct Data
    {
        double time;
        double angle;
    };
    std::deque<Data> data_history_queue_;

    // Parameters to be fitted
    std::array<double, 5> fitting_param_;
    std::unique_ptr<std::future<void>> fitting_future_;

private:
    // Fitting Curve
#define BIG_RUNE_CURVE(x, a, omega, b, c, d, sign) \
    ((-((a) / (omega)*ceres::cos((omega) * ((x) + (d)))) + (b) * ((x) + (d)) + (c)) * (sign))

#define SMALL_RUNE_CURVE(x, a, b, c, sign) (((a) * ((x) + (b)) + (c)) * (sign))

    // Fitting Cost
    struct BigRuneFittingCost
    {
        BigRuneFittingCost(double x, double y, int m) : x_(x), y_(y), mov_(static_cast<double>(m))
        {
        }
        template <typename T>
        bool operator()(const T * const p, T * residual) const
        {
            residual[0] = y_ - BIG_RUNE_CURVE(x_, p[0], p[1], p[2], p[3], p[4], mov_);
            return true;
        }

        const double x_, y_;
        const double mov_;
    };

    struct SmallRuneFittingCost
    {
        SmallRuneFittingCost(double x, double y, int m) : x_(x), y_(y), mov_(static_cast<double>(m))
        {
        }
        template <typename T>
        bool operator()(const T * const p, T * residual) const
        {
            residual[0] = y_ - SMALL_RUNE_CURVE(x_, p[0], p[1], p[2], mov_);
            return true;
        }

        const double x_, y_;
        const double mov_;
    };
};
}  //namespace rune_detector_for_all
#endif  // RUNE_DETECTOR_CURVE_FITTER_HPP_
