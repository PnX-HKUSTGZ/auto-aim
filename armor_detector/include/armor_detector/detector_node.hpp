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

#include "armor_detector/ai_detector.hpp"
#include "armor_detector/ba_solver.hpp"
#include "armor_detector/detector.hpp"
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
    // -------------------- 初始化功能 --------------------
    /**
     * @brief 初始化装甲板检测器
     * @return 初始化好的检测器实例
     */
    std::unique_ptr<Detector> initDetector();

    /**
     * @brief 初始化AI检测器
     * @return 初始化好的AI检测器实例
     */
    std::unique_ptr<AIDetector> initAIDetector();

    // -------------------- 核心处理功能 --------------------
    /**
     * @brief 订阅图像的回调函数，处理图像并进行装甲板检测
     * @param img_msg 输入的图像消息
     */
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);

    /**
     * @brief 执行装甲板检测
     * @param img_msg 输入的图像消息
     * @param img 输出的OpenCV图像
     * @return 检测到的装甲板列表
     */
    std::vector<Armor> detectArmors(
        const sensor_msgs::msg::Image::ConstSharedPtr & img_msg, cv::Mat & img);

    /**
     * @brief 使用AI检测器执行装甲板检测
     * @param img_msg 输入的图像消息
     * @param img 输出的OpenCV图像
     * @return 检测到的装甲板列表
     */
    std::vector<Armor> aiDetectArmors(
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

    // -------------------- 核心检测器 --------------------
    std::unique_ptr<Detector> detector_;
    std::unique_ptr<AIDetector> ai_detector_;
    bool use_ai_detector_ = false;

    // -------------------- 相机相关 --------------------
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    cv::Point2f cam_center_;
    std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
    std::unique_ptr<PnPSolver> pnp_solver_;
    std::unique_ptr<BaSolver> ba_solver_;

    // -------------------- 发布器 --------------------
    auto_aim_interfaces::msg::Armors armors_msg_;
    rclcpp::Publisher<auto_aim_interfaces::msg::Armors>::SharedPtr armors_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // -------------------- 服务 --------------------
    rclcpp::Service<auto_aim_interfaces::srv::SetMode>::SharedPtr set_mode_srv_;

    // -------------------- TF2坐标变换 --------------------
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    Eigen::Matrix3d r_odom_to_camera;
    Eigen::Vector3d t_odom_to_camera;

    // -------------------- 调试相关 --------------------
    bool debug_;
    std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;
    image_transport::Publisher binary_img_pub_;
    image_transport::Publisher number_img_pub_;
    image_transport::Publisher result_img_pub_;
    
    /**
    * @brief 装甲板可视化标记
    * 用于在RViz中显示检测到的装甲板
    */
    visualization_msgs::msg::Marker armor_marker_ = [] {
        visualization_msgs::msg::Marker marker;
        marker.ns = "armors";
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.scale.x = 0.05;
        marker.scale.z = 0.125;
        marker.color.a = 1.0;
        marker.color.g = 0.5;
        marker.color.b = 1.0;
        marker.lifetime = rclcpp::Duration::from_seconds(0.1);
        return marker;
    }();


    /**
    * @brief 文本可视化标记
    * 用于在RViz中显示装甲板的分类结果
    */
    visualization_msgs::msg::Marker text_marker_ = [] {
        visualization_msgs::msg::Marker marker;
        marker.ns = "classification";
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        marker.scale.z = 0.1;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        marker.lifetime = rclcpp::Duration::from_seconds(0.1);
        return marker;
    }();

    /**
    * @brief 标记数组
    * 用于批量管理可视化标记
    */
    visualization_msgs::msg::MarkerArray marker_array_;

    // -------------------- 状态控制 --------------------
    bool enable_ = true;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_NODE_HPP_
