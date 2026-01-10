#ifndef RUNE_DETECTOR_TYPES_HPP_
#define RUNE_DETECTOR_TYPES_HPP_

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

namespace rune_detector_for_all
{

// Motion type
enum class MotionType { SMALL, BIG, UNKNOWN };

// Moving direction
enum Direction { CLOCKWISE = -1, ANTI_CLOCKWISE = 1, UNKNOWN = 0 };

constexpr double DEG_72 = 0.4 * CV_PI;

} // namespace rune_detector_for_all

#endif // RUNE_DETECTOR_TYPES_HPP_
