#ifndef RUNE_DETECTOR_NODE_HPP_
#define RUNE_DETECTOR_NODE_HPP_

// ROS
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

// Tiger Core
#include "vc/core/type_expansion.hpp"   // PixChannel
#include "vc/dataio/dataio.h"           // GyroData
#include "vc/detector/detector.h"       // DetectorInput/Output
#include "vc/detector/rune_detector.h"  // RuneDetector
#include "vc/feature/feature_node.h"    // FeatureNode_ptr
#include "vc/feature/rune_group.h"      // RuneGroup
#include "vc/math/pose_node.hpp"        // PoseNode
#include "vc/camera/camera_param.h"     // camera_param
#include "vc/core/debug_tools.h"        // DebugTools

// Project
#include "curve_fitter.hpp"

// Interfaces
#include "auto_aim_interfaces/msg/rune_target.hpp" // For output
#include "auto_aim_interfaces/srv/set_mode.hpp"

namespace rune_detector_for_all
{

class RuneDetectorNode : public rclcpp::Node
{
public:
    explicit RuneDetectorNode(const rclcpp::NodeOptions & options);

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    
    // Helper to get GyroData from TF
    GyroData getGyroData(const rclcpp::Time& stamp);

    // Initializer for Tiger Core params from ROS Params
    void initTigerParams();
    
    // Service callback
    void setModeCallback(
        const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
        std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response);

    // Persist rune groups across frames for tracking
    std::vector<FeatureNode_ptr> rune_groups_;

    // Curve Fitter
    std::unique_ptr<CurveFitter> curve_fitter_;
    
    // Angle tracking variables
    double last_angle_ = 0.0;
    double last_observed_angle_ = 0.0;
    bool tracker_init_ = false;
    double angle_offset_thres_ = 0.78; // Approx 45 degrees

    double getObservedAngle(double normal_angle);

    // ROS
    image_transport::Publisher result_img_pub_;
    rclcpp::Publisher<auto_aim_interfaces::msg::RuneTarget>::SharedPtr rune_target_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Service<auto_aim_interfaces::srv::SetMode>::SharedPtr set_mode_srv_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Tiger Core
    std::shared_ptr<RuneDetector> tiger_detector_;
    
    // Configs
    int detect_color_; // 0: RED, 1: BLUE
    bool is_rune_ = false;
    
    // Camera params
    sensor_msgs::msg::CameraInfo camera_info_;
    bool has_camera_info_ = false;
};

} // namespace rune_detector_for_all

#endif // RUNE_DETECTOR_NODE_HPP_
