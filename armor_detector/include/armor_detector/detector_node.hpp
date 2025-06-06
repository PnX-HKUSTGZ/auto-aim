// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_NODE_HPP_
#define ARMOR_DETECTOR__DETECTOR_NODE_HPP_

// ROS
#include <tf2_ros/buffer.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

// STD
#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

#include "armor_detector/ba_solver.hpp"
#include "armor_detector/detector.hpp"
#include "armor_detector/light_corner_corrector.hpp"
#include "armor_detector/number_classifier.hpp"
#include "armor_detector/pnp_solver.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/srv/set_mode.hpp"

namespace rm_auto_aim
{

class ArmorDetectorNode : public rclcpp::Node
{
public:
    /**
     * @brief 装甲板检测节点的构造函数
     * @param options ROS节点选项
     */
    ArmorDetectorNode(const rclcpp::NodeOptions & options);

private:
    // -------------------- 核心处理功能 --------------------
    /**
     * @brief 订阅图像的回调函数，处理图像并进行装甲板检测
     * @param img_msg 输入的图像消息
     */
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);

    /**
     * @brief 初始化装甲板检测器
     * @return 初始化好的检测器实例
     */
    std::unique_ptr<Detector> initDetector();

    /**
     * @brief 执行装甲板检测
     * @param img_msg 输入的图像消息
     * @param img 输出的OpenCV图像
     * @return 检测到的装甲板列表
     */
    std::vector<Armor> detectArmors(
        const sensor_msgs::msg::Image::ConstSharedPtr & img_msg, cv::Mat & img);

    // -------------------- 坐标变换和位姿处理 --------------------
    /**
     * @brief 更新从odom到相机的坐标变换
     * @param target_frame 目标坐标系
     * @param source_frame 源坐标系
     * @param timestamp 时间戳
     * @return 是否成功获取坐标变换
     */
    bool updateTransform(
        std::string target_frame, std::string source_frame, rclcpp::Time timestamp);

    /**
     * @brief 选择装甲板的最佳姿态，处理多解情况
     * @param armor 待处理的装甲板
     * @param rvec PnP解算得到的旋转向量
     * @param tvec PnP解算得到的平移向量
     */
    void chooseBestPose(Armor & armor, const cv::Mat & rvec, const cv::Mat & tvec);

    /**
     * @brief 处理同一车辆上的两块装甲板，提高解算精度
     * @param armor1 第一块装甲板
     * @param armor2 第二块装甲板
     */
    void fix_two_armors(Armor & armor1, Armor & armor2);

    // -------------------- 可视化和调试功能 --------------------
    /**
     * @brief 绘制检测结果到图像上
     * @param img_msg 原始图像消息
     * @param img 待绘制的图像
     * @param armors 检测到的装甲板列表
     */
    void drawResults(
        const sensor_msgs::msg::Image::ConstSharedPtr & img_msg, cv::Mat & img,
        const std::vector<Armor> & armors);

    /**
     * @brief 创建用于调试的发布器
     */
    void createDebugPublishers();

    /**
     * @brief 销毁调试用的发布器
     */
    void destroyDebugPublishers();

    /**
     * @brief 发布装甲板可视化标记
     */
    void publishMarkers();

    // -------------------- 服务回调 --------------------
    /**
     * @brief 设置工作模式的回调函数
     * @param request 服务请求
     * @param response 服务响应
     */
    void setModeCallback(
        const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
        std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response);
    // Light corner corrector
    LightCornerCorrector lcc;

    //dynamic parameter
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    rcl_interfaces::msg::SetParametersResult onParameterChanged(
        const std::vector<rclcpp::Parameter> & parameters);

    // Armor Detector
    std::unique_ptr<Detector> detector_;

    // set_mode service
    rclcpp::Service<auto_aim_interfaces::srv::SetMode>::SharedPtr set_mode_srv_;

    // Detected armors publisher
    auto_aim_interfaces::msg::Armors armors_msg_;
    rclcpp::Publisher<auto_aim_interfaces::msg::Armors>::SharedPtr armors_pub_;

    // Visualization marker publisher
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // Camera info part
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    cv::Point2f cam_center_;
    std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
    std::unique_ptr<PnPSolver> pnp_solver_;
    std::unique_ptr<BaSolver> ba_solver_;

    // Image subscrpition
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

    // tf2
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    Eigen::Matrix3d r_odom_to_camera;
    Eigen::Vector3d t_odom_to_camera;

    // Debug information
    bool debug_;
    std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;
    rclcpp::Publisher<auto_aim_interfaces::msg::DebugLights>::SharedPtr lights_data_pub_;
    rclcpp::Publisher<auto_aim_interfaces::msg::DebugArmors>::SharedPtr armors_data_pub_;
    image_transport::Publisher binary_img_pub_;
    image_transport::Publisher number_img_pub_;
    image_transport::Publisher result_img_pub_;

    bool enable_ = true;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_NODE_HPP_
