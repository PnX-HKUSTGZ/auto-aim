#include "rune_detector_node.hpp"
#include <opencv2/opencv.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <angles/angles.h>

// Tiger Core
#include "vc/core/type_expansion.hpp"   // PixChannel
#include "vc/dataio/dataio.h"
#include "vc/detector/detector.h"
#include "vc/detector/rune_detector.h"
#include "vc/feature/feature_node.h"
#include "vc/feature/rune_group.h"
#include "vc/math/pose_node.hpp"
#include "vc/camera/camera_param.h"
#include "vc/core/debug_tools.h"

namespace rune_detector_for_all
{

RuneDetectorNode::RuneDetectorNode(const rclcpp::NodeOptions & options)
: Node("rune_detector", options)
{
    RCLCPP_INFO(this->get_logger(), "Starting RuneDetectorNode (Tiger Core Wrapper)!");

    // Initialize Tiger Core Detector
    tiger_detector_ = std::make_shared<RuneDetector>();

    // Parameters
    detect_color_ = this->declare_parameter("detect_color", 0); // 0: RED, 1: BLUE
    this->declare_parameter("min_lightness", 100); 
    angle_offset_thres_ = this->declare_parameter("angle_offset_thres", 0.3);
    
    // Init Curve Fitter (Default to SMALL (Linear))
    curve_fitter_ = std::make_unique<CurveFitter>(MotionType::SMALL);
    last_angle_ = 1000.0; // Invalid init value

    // Init TF
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Subscriptions
    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "image_raw", rclcpp::SensorDataQoS(),
        std::bind(&RuneDetectorNode::imageCallback, this, std::placeholders::_1));
        
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
            camera_info_ = *camera_info;
            has_camera_info_ = true;
            cv::Matx33f camera_matrix(
                static_cast<float>(camera_info_.k[0]), static_cast<float>(camera_info_.k[1]), static_cast<float>(camera_info_.k[2]),
                static_cast<float>(camera_info_.k[3]), static_cast<float>(camera_info_.k[4]), static_cast<float>(camera_info_.k[5]),
                static_cast<float>(camera_info_.k[6]), static_cast<float>(camera_info_.k[7]), static_cast<float>(camera_info_.k[8]));

            cv::Matx<float, 5, 1> dist_coeffs(
                static_cast<float>(camera_info_.d[0]), static_cast<float>(camera_info_.d[1]), static_cast<float>(camera_info_.d[2]),
                static_cast<float>(camera_info_.d[3]), static_cast<float>(camera_info_.d[4]));

            camera_param.cameraMatrix = camera_matrix;
            camera_param.distCoeff = dist_coeffs;
            camera_param.image_width = static_cast<int>(camera_info_.width);
            camera_param.image_height = static_cast<int>(camera_info_.height);
            cam_info_sub_.reset(); 
        });

    // Publishers
    rune_target_pub_ = this->create_publisher<auto_aim_interfaces::msg::RuneTarget>(
        "/tracker/rune_target", 10);
    result_img_pub_ = image_transport::create_publisher(this, "rune_detector/result_img");

    // Services
    set_mode_srv_ = this->create_service<auto_aim_interfaces::srv::SetMode>(
        "rune_detector/set_mode",
        std::bind(&RuneDetectorNode::setModeCallback, this, std::placeholders::_1, std::placeholders::_2));
}

void RuneDetectorNode::setModeCallback(
    const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
    std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response)
{
    response->success = true;
    if (request->mode == 8) { // VisionMode::RUNE = 8
        is_rune_ = true;
        if (!img_sub_) {
             img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "image_raw", rclcpp::SensorDataQoS(),
                std::bind(&RuneDetectorNode::imageCallback, this, std::placeholders::_1));
        }
        RCLCPP_INFO(this->get_logger(), "Switched to RUNE mode");
    } else {
        is_rune_ = false;
        img_sub_.reset(); 
        RCLCPP_INFO(this->get_logger(), "Switched out of RUNE mode");
    }
}

GyroData RuneDetectorNode::getGyroData(const rclcpp::Time& stamp)
{
    GyroData gyro;

    gyro.rotation.yaw = 0;
    gyro.rotation.pitch = 0;
    gyro.rotation.roll = 0;

    try {
        geometry_msgs::msg::TransformStamped t = 
            tf_buffer_->lookupTransform("odom", "gimbal_link", stamp); 
        
        tf2::Quaternion q(
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w);
        double r, p, y;
        tf2::Matrix3x3(q).getRPY(r, p, y);
        
        gyro.rotation.yaw = static_cast<float>(y);
        gyro.rotation.pitch = static_cast<float>(p);
        gyro.rotation.roll = static_cast<float>(r); 
        
    } catch (tf2::TransformException & ex) {
    }
    return gyro;
}

double RuneDetectorNode::getObservedAngle(double angle)
{
    if (last_angle_ > 900.0) { // Check invalid
        last_angle_ = angle;
        return angle;
    }
    
    double out_angle = angle;
    double diff_a = out_angle - last_angle_;
    
    // Unwrap to ensure continuity
    while (diff_a > M_PI) {
        out_angle -= 2 * M_PI;
        diff_a = out_angle - last_angle_;
    }
    while (diff_a < -M_PI) {
        out_angle += 2 * M_PI;
        diff_a = out_angle - last_angle_;
    }
    
    last_angle_ = out_angle;
    return out_angle;
}

void RuneDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    if (!is_rune_ || !has_camera_info_) return;

    DetectorInput input;
    try {
        input.setImage(cv_bridge::toCvCopy(msg, "bgr8")->image);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    
    const auto detect_color = this->get_parameter("detect_color").as_int();

    input.setTick(rclcpp::Time(msg->header.stamp).nanoseconds());
    input.setGyroData(getGyroData(rclcpp::Time(msg->header.stamp)));
    input.setColor((detect_color == 0) ? PixChannel::RED : PixChannel::BLUE);
    
    int binary_thresh = this->get_parameter("min_lightness").as_int(); 
    input.setColorThresh(binary_thresh);
    
    input.setFeatureNodes(rune_groups_);

    DetectorOutput output;
    try {
        tiger_detector_->detect(input, output);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Tiger Core Exception: %s", e.what());
        return;
    }

    rune_groups_ = output.getFeatureNodes();

    auto_aim_interfaces::msg::RuneTarget rune_target_msg;
    rune_target_msg.header = msg->header;
    rune_target_msg.header.frame_id = "odom"; 
    rune_target_msg.tracking = false;

    std::shared_ptr<RuneGroup> target_group = nullptr;
    if (!rune_groups_.empty()) {
        target_group = std::dynamic_pointer_cast<RuneGroup>(rune_groups_.front());
    }

    if (target_group && !target_group->childFeatures().empty()) {
        PoseNode r_cam;
        bool success = target_group->getCamPnpDataFromFilter(r_cam);
        
        if (success) {
            try {
                geometry_msgs::msg::TransformStamped odom_T_cam_msg = 
                    tf_buffer_->lookupTransform("odom", "camera_optical_frame", msg->header.stamp);
                    
                tf2::Transform odom_T_cam;
                tf2::fromMsg(odom_T_cam_msg.transform, odom_T_cam);
                
                cv::Mat rmat_cv;
                cv::Rodrigues(r_cam.rvec(), rmat_cv);
                tf2::Matrix3x3 rmat_tf(
                    rmat_cv.at<float>(0,0), rmat_cv.at<float>(0,1), rmat_cv.at<float>(0,2),
                    rmat_cv.at<float>(1,0), rmat_cv.at<float>(1,1), rmat_cv.at<float>(1,2),
                    rmat_cv.at<float>(2,0), rmat_cv.at<float>(2,1), rmat_cv.at<float>(2,2)
                );
                tf2::Vector3 tvec_tf(r_cam.tvec()(0), r_cam.tvec()(1), r_cam.tvec()(2));
                tf2::Transform cam_T_rune(rmat_tf, tvec_tf);
                tf2::Transform odom_T_rune = odom_T_cam * cam_T_rune;
                
                rune_target_msg.center.x = odom_T_rune.getOrigin().x();
                rune_target_msg.center.y = odom_T_rune.getOrigin().y();
                rune_target_msg.center.z = odom_T_rune.getOrigin().z();
                
                double roll, pitch, yaw;
                odom_T_rune.getBasis().getRPY(roll, pitch, yaw);
                rune_target_msg.yaw = yaw;
                
                float current_roll_rad = 0;
                target_group->getCurrentRotateAngle(current_roll_rad);
                
                double observed_angle = getObservedAngle((double)current_roll_rad);
                rune_target_msg.roll = observed_angle; 
                
                rune_target_msg.tracking = true; 
                
                double t_now = rclcpp::Time(msg->header.stamp).seconds();
                
                curve_fitter_->update(t_now, observed_angle);
                
                rune_target_msg.is_big = (curve_fitter_->getType() == MotionType::BIG);
                auto params = curve_fitter_->getFittingParam();
                rune_target_msg.fitting_curve[0] = params[0];
                rune_target_msg.fitting_curve[1] = params[1];
                rune_target_msg.fitting_curve[2] = params[2];
                rune_target_msg.fitting_curve[3] = params[3];
                rune_target_msg.fitting_curve[4] = params[4];
                
                rune_target_msg.start_time = 0.0; 
                
                rune_target_msg.direction = curve_fitter_->getDirection();
                
            } catch (const std::exception& e) {
               rune_target_msg.tracking = false;
               RCLCPP_WARN(this->get_logger(), "PnP/Math error: %s", e.what());
            }
        }
    } else {
        curve_fitter_->reset();
        last_angle_ = 1000.0;
    }
    
    rune_target_pub_->publish(rune_target_msg);
    
    cv::Mat debug_img = DebugTools::get()->getImage();
    if (!debug_img.empty()) {
        try {
            sensor_msgs::msg::Image::SharedPtr dbg_msg = 
                cv_bridge::CvImage(msg->header, "bgr8", debug_img).toImageMsg();
            result_img_pub_.publish(dbg_msg);
        } catch (...) {}
    }
}

} // namespace rune_detector_for_all

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rune_detector_for_all::RuneDetectorNode)
