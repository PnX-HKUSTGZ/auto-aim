#ifndef ARMOR_DETECTOR_LIGHT_CORNER_CORRECTOR_HPP_
#define ARMOR_DETECTOR_LIGHT_CORNER_CORRECTOR_HPP_

// OpenCV 头文件
#include <opencv2/opencv.hpp>
// 项目头文件
#include "armor_detector/types.hpp"

namespace rm_auto_aim
{

/**
 * @brief 灯条对称轴结构体
 * 
 * 包含灯条对称轴的质心、方向向量和平均亮度值
 */
struct SymmetryAxis
{
    cv::Point2f centroid;   ///< 对称轴的质心
    cv::Point2f direction;  ///< 对称轴的方向向量，已归一化
    float mean_val;         ///< 灯条区域的平均亮度值
};

/**
 * @brief 灯条角点校正器
 * 
 * 该类用于提高灯条角点的精度。首先使用 PCA 算法找到灯条的对称轴，
 * 然后沿着对称轴根据亮度梯度寻找灯条的精确角点位置。
 * 精确的灯条角点对于后续的 PnP 解算和姿态估计至关重要。
 */
class LightCornerCorrector
{
public:
    /**
     * @brief 构造一个灯条角点校正器
     */
    explicit LightCornerCorrector() noexcept {}

    /**
     * @brief 修正装甲板的灯条角点
     * 
     * 对装甲板两侧灯条的四个角点进行精确定位和修正，
     * 以提高后续 PnP 解算的精度。
     * 
     * @param armor 需要修正角点的装甲板
     * @param gray_img 灰度图像，用于分析亮度梯度
     */
    void correctCorners(Armor & armor, const cv::Mat & gray_img);

private:
    /**
     * @brief 寻找灯条的对称轴
     * 
     * 使用 PCA 算法分析灯条区域的像素分布，找到灯条的主方向，
     * 即对称轴。
     * 
     * @param gray_img 灰度图像
     * @param light 待分析的灯条
     * @return SymmetryAxis 灯条的对称轴信息
     */
    SymmetryAxis findSymmetryAxis(const cv::Mat & gray_img, const Light & light);

    /**
     * @brief 寻找灯条的角点
     * 
     * 沿着对称轴，根据亮度梯度变化，精确定位灯条的角点位置。
     * 
     * @param gray_img 灰度图像
     * @param light 待分析的灯条
     * @param axis 灯条的对称轴
     * @param order 角点位置标识，如 "top" 或 "bottom"
     * @return cv::Point2f 修正后的角点坐标
     */
    cv::Point2f findCorner(
        const cv::Mat & gray_img, const Light & light, const SymmetryAxis & axis,
        std::string order);
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_LIGHT_CORNER_CORRECTOR_HPP_