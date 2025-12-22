// Copyright 2022 Chen Jun

#ifndef ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_
#define ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_

// ROS
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include <image_transport/image_transport.hpp>
#include <opencv2/core/types.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// STD
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "armor_tracker/tracker.hpp"
#include "armor_tracker/tracker_manager.hpp"
#include "armor_tracker/types.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/tracker_info.hpp"
#include "auto_aim_interfaces/srv/set_mode.hpp"

// OpenCV
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core.hpp>

namespace rm_auto_aim
{
// 定义消息类型别名
using ArmorsMsg = auto_aim_interfaces::msg::Armors;

/**
 * @brief 装甲板追踪节点类
 * 
 * 该节点负责接收装甲板检测结果，进行多目标追踪，并发布最佳追踪目标信息。
 * 支持坐标变换、EKF状态估计、可视化标记发布等功能。
 */
class ArmorTrackerNode : public rclcpp::Node
{
public:
    /**
     * @brief 构造函数，初始化追踪节点
     * 
     * @param options ROS节点选项
     */
    explicit ArmorTrackerNode(const rclcpp::NodeOptions & options);

private:
    /**
     * @brief 初始化扩展卡尔曼滤波器
     * 
     * 设置EKF的状态转移函数、观测函数、雅可比矩阵和噪声协方差矩阵。
     */
    void initializeEKF();

    void mainArmorsCallback(const ArmorsMsg::SharedPtr armors_msg);
    void wideArmorsCallback(const ArmorsMsg::SharedPtr armors_msg);

    /**
     * @brief 核心处理函数
     * 
     * 接收选定的装甲板数据和相机参数，执行坐标变换、追踪、发布等所有核心逻辑。
     * 
     * @param armors_msg 使用的装甲板消息
     * @param cam_info 对应的相机内参
     * @param cam_center 对应的相机中心点
     * @param frame_id 坐标系ID
     * @param is_main_camera 是否为主相机
     */
    void processArmors(
        const ArmorsMsg::SharedPtr & armors_msg, const sensor_msgs::msg::CameraInfo & cam_info,
        const cv::Point2f & cam_center, const std::string & frame_id, bool is_main_camera);

    /**
     * @brief 定时发布回调
     */
    void publishCallback();
    
    // 发布定时器
    rclcpp::TimerBase::SharedPtr publish_timer_;

    /**
     * @brief 绘制可视化标记
     * 
     * 在RViz中绘制追踪目标的位置、速度、轨迹等可视化标记。
     * 
     * @param target_msg 目标消息
     * @param marker_array 标记数组
     */
    void drawMarkers(
        const auto_aim_interfaces::msg::Target & target_msg,
        visualization_msgs::msg::MarkerArray & marker_array);

    /**
     * @brief 设置模式服务回调函数
     * 
     * 处理来自其他节点的模式设置请求，切换追踪器的工作模式。
     * 
     * @param request 服务请求
     * @param response 服务响应
     */
    void setModeCallback(
        const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
        std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response);

    /**
     * @brief 在图像上绘制追踪结果
     * 
     * 在图像上绘制装甲板位置、追踪框、预测轨迹等信息，用于调试和可视化。
     * 
     * @param target_msg 目标消息
     * @param image 输入输出图像
     * @param is_primary_target 是否为主要目标
     * @param cam_info 相机内参
     * @param img_frame_id 图像对应的坐标系ID
     */
    void drawImgAll(
        const auto_aim_interfaces::msg::Target & target_msg, cv::Mat & image,
        bool is_primary_target, const sensor_msgs::msg::CameraInfo & cam_info,
        const std::string & img_frame_id);

    // Debug
    bool debug_;
    int last_sec = 0, current_sec = 0, frame_count = 0;

    // Maximum allowable armor distance in the XOY plane
    double max_armor_distance_;

    // The time when the last message was received
    rclcpp::Time last_time_ = rclcpp::Time(0);
    double dt_ = 0.01;


    // Armor tracker
    double s2qxy_, s2qz_, s2qyaw_, s2qr_;
    double r_xyz_factor, r_yaw, r_radius;
    double lost_time_thres_;
    std::unique_ptr<TrackerManager> tracker_manager_;

    // set_mode service
    rclcpp::Service<auto_aim_interfaces::srv::SetMode>::SharedPtr set_mode_srv_;

    // Subscriber with tf2 message_filter
    std::string target_frame_;
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;

    // Subscriptions
    rclcpp::Subscription<ArmorsMsg>::SharedPtr main_armors_sub_;
    rclcpp::Subscription<ArmorsMsg>::SharedPtr wide_armors_sub_;

    // Synchronization control
    std::mutex mutex_;
    rclcpp::Time last_main_update_time_;
    rclcpp::Time last_time_main_;
    rclcpp::Time last_time_wide_;

    // Tracker info publisher
    rclcpp::Publisher<auto_aim_interfaces::msg::TrackerInfo>::SharedPtr info_pub_;

    // Publisher
    rclcpp::Publisher<auto_aim_interfaces::msg::Target>::SharedPtr target_pub_;

    // Visualization marker publisher
    visualization_msgs::msg::Marker position_marker_;
    visualization_msgs::msg::Marker linear_v_marker_;
    visualization_msgs::msg::Marker angular_v_marker_;
    visualization_msgs::msg::Marker armor_marker_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // 相机参数
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_wide;
    sensor_msgs::msg::CameraInfo cam_info_;
    sensor_msgs::msg::CameraInfo cam_info_wide;
    cv::Point2f cam_center_;
    cv::Point2f cam_center_wide;

    // 发布图像
    image_transport::Publisher tracker_img_pub_;
    std_msgs::msg::Header_<std::allocator<void>>::_stamp_type last_img_time_;

    VisionMode mode_ = VisionMode::AUTO;  // 默认模式为AUTO
};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_
