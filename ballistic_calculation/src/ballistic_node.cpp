// STD
#include "ballistic_calculation/ballistic_node.hpp"

#include <Eigen/Eigen>
#include <auto_aim_interfaces/msg/detail/firecontrol__struct.hpp>
#include <auto_aim_interfaces/msg/firecontrol.hpp>
#include <auto_aim_interfaces/msg/target.hpp>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <string>
#include <vector>

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "ballistic_calculation/mpc_controller.hpp"
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::Target;
using firemsg = auto_aim_interfaces::msg::Firecontrol;

const double BallisticCalculateNode::THRES1 = 0.01;
const double BallisticCalculateNode::THRES2 = 0.005;

BallisticCalculateNode::BallisticCalculateNode(const rclcpp::NodeOptions & options)
: Node("ballistic_calculation", options)
{
    RCLCPP_INFO(this->get_logger(), "start ballistic calculation!");
    K1 = this->declare_parameter("iteration_coeffcient_first", 0.1);
    K2 = this->declare_parameter("iteration_coeffcient_second", 0.05);
    K = this->declare_parameter("air_resistence", 0.1);
    BULLET_V = this->declare_parameter("bullet_speed", 21.0);
    ifFireK_ = this->declare_parameter("ifFireK", 0.05);
    min_v = this->declare_parameter("switch_stategy_1", 5.0) * M_PI / 30;
    max_v = this->declare_parameter("switch_stategy_2", 30.0) * M_PI / 30;
    v_yaw_gimble = this->declare_parameter("max_v_yaw_gimble", 0.8);
    stop_fire_time = this->declare_parameter("stop_fire_time", 0.1);
    double fire_delay = this->declare_parameter("fire_delay", 0.0);
    xyz_vec = this->declare_parameter("xyz", std::vector<double>{0.0, 0.0, 0.0});
    rpy_vec = this->declare_parameter("rpy", std::vector<double>{0.0, 0.0, 0.0});
    // 添加MPC开关参数，默认false（不采用MPC结果）
    use_mpc_default = this->declare_parameter("use_mpc_default", true);
    Eigen::Vector3d odom2gunxyz(xyz_vec[0], xyz_vec[1], xyz_vec[2]);
    Eigen::Vector3d odom2gunrpy(rpy_vec[0], rpy_vec[1], rpy_vec[2]);

    calculator = std::make_unique<rm_auto_aim::Ballistic>(K, BULLET_V, fire_delay);
    car_info_ = std::make_shared<CarInfo>(odom2gunxyz, odom2gunrpy);
    armor_info_ = std::make_unique<ArmorInfo>(odom2gunxyz, odom2gunrpy);
    rune_info_ = std::make_unique<RuneInfo>(odom2gunxyz, odom2gunrpy);
    armor_selector_ = std::make_shared<ArmorSelector>();

    //初始化MPC控制器
    std::string mpc_config_path = this->declare_parameter<std::string>("mpc_config_path", "config/mpc_params.yaml");
    mpc_controller_ = std::make_unique<rm_auto_aim::MPCController>(mpc_config_path);
    RCLCPP_INFO(this->get_logger(), "MPC Controller initialized with config: %s", mpc_config_path.c_str());

    //创建监听器，监听云台位姿
    tfBuffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer, this);

    //创建订阅者
    car_target_sub_ = this->create_subscription<auto_aim_interfaces::msg::Target>(
        "/tracker/target", rclcpp::SensorDataQoS(),
        std::bind(&BallisticCalculateNode::carTargetCallback, this, std::placeholders::_1));
    rune_target_sub_ = this->create_subscription<auto_aim_interfaces::msg::RuneTarget>(
        "/tracker/rune_target", rclcpp::SensorDataQoS(),
        std::bind(&BallisticCalculateNode::runeTargetCallback, this, std::placeholders::_1));
    //创建发布者
    publisher_ = this->create_publisher<auto_aim_interfaces::msg::Firecontrol>("/firecontrol", 10);
    // marker publisher for visualization of aim point (yellow)
    aim_point_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/aim_point_marker", 10);
    //设置时间callback
    last_fire_time = this->now();

    // 在构造函数中添加相机信息订阅
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
            cam_info_ = *camera_info;
            cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
            cam_info_sub_.reset();
        });

    // 在构造函数内添加订阅
    gimbal_vel_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/gimbal_vel", rclcpp::SensorDataQoS(),
        [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
            if (msg->data.size() >= 2) {
                this->current_yaw_vel = msg->data[0];
                this->current_pitch_vel = msg->data[1];
            }
        });
    rclcpp::QoS point_qos = rclcpp::QoS(10).transient_local();
    // PointStamped发布器（rqt_plot绘制曲线）
    mpc_pre_point_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/mpc/pre_target_point", point_qos);
    mpc_post_point_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/mpc/post_target_point", point_qos);
    
}

bool BallisticCalculateNode::ifFire(double targetpitch, double targetyaw)
{
    geometry_msgs::msg::TransformStamped t;
    //获取当前云台位姿
    try {
        // 使用最新的可用变换，而不是当前时间
        t = tfBuffer->lookupTransform("odom", "gimbal_link", tf2::TimePointZero);
    } catch (tf2::TransformException & ex) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "%s", ex.what());
        return false;
    }

    tf2::Quaternion q(
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
        t.transform.rotation.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    //计算云台位姿和预测位置的差值,当差值小于某一个阈值时，返回true
    return std::abs(yaw - targetyaw) < ifFireK && std::abs(pitch + targetpitch) < ifFireK; // 注意检查符号适配
}

// 获取当前云台状态
Eigen::Vector4d BallisticCalculateNode::getCurrentGimbalState()
{
    Eigen::Vector4d state;
    state.setZero();
    try {
        auto t = tfBuffer->lookupTransform("odom", "gimbal_link", tf2::TimePointZero);
        tf2::Quaternion q(t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        state << yaw, current_yaw_vel, pitch, current_pitch_vel; 
    } catch (tf2::TransformException & ex) {
        RCLCPP_WARN(this->get_logger(), "Failed to get gimbal state: %s", ex.what());
    }
    return state;
}

void BallisticCalculateNode::carTargetCallback(
    auto_aim_interfaces::msg::Target::SharedPtr _target_msg)
{
    if (!_target_msg) {
        RCLCPP_WARN(this->get_logger(), "Received null target message");
        return;
    }
    this->car_target_msg = std::move(_target_msg);

    car_info_->updateTarget(*car_target_msg);
    armor_selector_->updateTarget(*car_target_msg);

    ifFireK = ifFireK_ + abs(car_target_msg->v_yaw) * 0.002;
    //进入第一次大迭代
    Eigen::Vector3d target = car_info_->getGunTarget(0.0);

    //进入迭代
    double init_pitch =
        std::atan(target[2] / std::sqrt(target[0] * target[0] + target[1] * target[1]));
    double init_t =
        std::sqrt(target[0] * target[0] + target[1] * target[1]) / (cos(init_pitch) * BULLET_V);
    
    double temp_t;
    std::pair<double, double> first_iteration_result =
        this->calculator->iteration(THRES1, init_pitch, init_t, *car_info_, temp_t);

    //预测并选择合适击打的装甲板
    double temp_theta = first_iteration_result.first;

    //预测平衡步兵的最佳装甲板
    std::pair<double, double> iffire_result, final_result;
    if (car_target_msg->armors_num == 4 or car_target_msg->armors_num == 3) {
        std::vector<double> hit_aim = armor_selector_->
            predictInfantryBestArmor(temp_t, min_v, max_v, v_yaw_gimble); // 瞄准用
        std::vector<double> hit_aim_fire = armor_selector_->
            predictInfantryBestArmor(temp_t, DBL_MAX, DBL_MAX, v_yaw_gimble); // 开火判定用
        
        //计算瞄准目标
        armor_info_->updateTarget(
            *car_target_msg, hit_aim[0], hit_aim[2] == 0 ? hit_aim_fire[1] : hit_aim[1],
            hit_aim[2]);
        final_result = calculator->iteration(THRES2, temp_theta, temp_t, *armor_info_, temp_t);
        //计算是否开火
        armor_info_->updateTarget(
            *car_target_msg, hit_aim_fire[0], hit_aim_fire[1], hit_aim_fire[2]);
        iffire_result = calculator->iteration(THRES2, temp_theta, temp_t, *armor_info_, temp_t);
        
        Eigen::Vector3d aim_target_pos = armor_info_->getOdomTarget(temp_t);  // 使用迭代后的temp_t

        // 发布可视化标记，展示选择出的最优装甲板在temp_t时刻的位置
        try {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "odom";
            marker.header.stamp = this->now();
            marker.ns = "aim_target";
            marker.id = 0;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            // 设置瞄准目标位置
            marker.pose.position.x = aim_target_pos.x();
            marker.pose.position.y = aim_target_pos.y();
            marker.pose.position.z = aim_target_pos.z();
            marker.pose.orientation.w = 1.0;  // 无旋转

            // 标记大小
            marker.scale.x = 0.05;
            marker.scale.y = 0.05;
            marker.scale.z = 0.05;
            marker.color.r = 1.0f;  // 红色分量
            marker.color.g = 1.0f;  // 绿色分量
            marker.color.b = 0.0f;  // 蓝色分量（黄色）
            marker.color.a = 1.0f;  // 不透明

            // 生命周期（0.1秒，避免残留）
            marker.lifetime = rclcpp::Duration::from_seconds(0.1);

            aim_point_pub_->publish(marker);
        } catch (...) {
            RCLCPP_WARN(this->get_logger(), "Failed to publish aim target marker");
        }
    } else {
        RCLCPP_ERROR(this->get_logger(), "The number of armors is not 4 or 3");
        return;
    }

    // MPC优化部分
    // 1. 获取MPC所需输入
    // Eigen::Vector3d target_odom = armor_info_->getOdomTarget(temp_t); // 使用预测时间t的目标位置
    double current_bullet_speed = BULLET_V;
    // 2. 调用MPC计算
    MPCResult mpc_result = mpc_controller_->compute(*car_target_msg, current_bullet_speed, temp_t);


    // MPC解算前可视化：发布当前云台yaw与yaw_vel到PointStamped
    try {
        geometry_msgs::msg::PointStamped pre_mpc_point;
        pre_mpc_point.header.frame_id = "odom";
        pre_mpc_point.header.stamp = this->now();
        // x: gimbal_yaw, y: gimbal_yaw_vel
        pre_mpc_point.point.x = mpc_result.target_yaw;
        pre_mpc_point.point.y = 0;
        if (mpc_pre_point_pub_) {
            mpc_pre_point_pub_->publish(pre_mpc_point);
        }
    } catch (...) {
        RCLCPP_WARN(this->get_logger(), "Failed to publish pre-MPC point");
    }

    // MPC解算后可视化：发布MPC计划的yaw与yaw_vel到PointStamped
    try {
        if (use_mpc_default && mpc_result.is_valid) {
            geometry_msgs::msg::PointStamped post_mpc_point;
            post_mpc_point.header.frame_id = "odom";
            post_mpc_point.header.stamp = this->now();
            // x: plan_yaw, y: plan_yaw_vel
            post_mpc_point.point.x = mpc_result.yaw;
            post_mpc_point.point.y = mpc_result.yaw_vel;
            if (mpc_post_point_pub_) {
                mpc_post_point_pub_->publish(post_mpc_point);
            }
        }
    } catch (...) {
        RCLCPP_WARN(this->get_logger(), "Failed to publish post-MPC point");
    }
    // 3. 结果融合：优先使用MPC结果，如果MPC求解失败，直接返回，不发送信息
    double final_pitch, final_yaw;
    double yaw_vel = 0.0, yaw_acc = 0.0, pitch_vel = 0.0, pitch_acc = 0.0;

    if (use_mpc_default && mpc_result.is_valid) {
        RCLCPP_DEBUG(this->get_logger(), "MPC solved successfully.");
        final_pitch = mpc_result.target_pitch + rpy_vec[1]; // 转换到云台坐标系
        final_yaw = mpc_result.target_yaw - rpy_vec[2];
        
        yaw_vel = mpc_result.yaw_vel;
        yaw_acc = mpc_result.yaw_acc;
        pitch_vel = mpc_result.pitch_vel;
        pitch_acc = mpc_result.pitch_acc;
    } else {
        // MPC 失败或未开启，直接返回
        // RCLCPP_WARN(this->get_logger(), "MPC invalid or disabled. Skipping publish.");
        return;
    }
    // 将 odom 坐标系中的点投影到图像上（使用计算出的瞄准时间 temp_t）
    cv::Point2f projected_point = projectPointToImage(armor_info_->getOdomTarget(temp_t));

    //发布消息
    firemsg fire_msg;
    fire_msg.header = car_target_msg->header;
    fire_msg.pitch = final_pitch;
    fire_msg.yaw = final_yaw;

    // 新增MPC输出的控制量
    fire_msg.yaw_vel = yaw_vel;
    fire_msg.yaw_acc = yaw_acc;
    fire_msg.pitch_vel = pitch_vel;
    fire_msg.pitch_acc = pitch_acc;

    fire_msg.tracking = car_target_msg->tracking;
    fire_msg.id = car_target_msg->id;
    fire_msg.projected_x = projected_point.x;
    fire_msg.projected_y = projected_point.y;
    if (this->now() - last_fire_time < rclcpp::Duration::from_seconds(stop_fire_time)) {
        ifFireK += abs(car_target_msg->v_yaw) * 0.004;
    }
    fire_msg.iffire = ifFire(fire_msg.pitch, fire_msg.yaw);

    if (fire_msg.iffire) last_fire_time = this->now();
    publisher_->publish(fire_msg);
}
void BallisticCalculateNode::runeTargetCallback(
    auto_aim_interfaces::msg::RuneTarget::SharedPtr _target_msg)
{
    if (!_target_msg) {
        RCLCPP_WARN(this->get_logger(), "Received null target message");
        return;
    }
    this->rune_target_msg = std::move(_target_msg);

    rune_info_->updateTarget(*rune_target_msg);

    Eigen::Vector3d target = rune_info_->getGunTarget(0.0);

    //进入迭代
    double init_pitch =
        std::atan(target[2] / std::sqrt(target[0] * target[0] + target[1] * target[1]));
    double init_t =
        std::sqrt(target[0] * target[0] + target[1] * target[1]) / (cos(init_pitch) * BULLET_V);

    double rune_t = init_t;
    // std::pair<double, double> iteration_result =
    //     this->calculator->iteration(THRES2, init_pitch, init_t, *rune_info_, rune_t);

    // MPC 优化部分
    Eigen::Vector3d target_odom = rune_info_->getOdomTarget(rune_t);
    double current_bullet_speed = BULLET_V;

    Eigen::Vector2d target_vel(0.0, 0.0); // 暂时先设为0.0
    RCLCPP_ERROR(this->get_logger(), "暂时没有开发完成，以后记得改\n");
    auto_aim_interfaces::msg::Target mpc_target;
    mpc_target.position.x = target_odom.x();
    mpc_target.position.y = target_odom.y();
    mpc_target.position.z = target_odom.z();
    mpc_target.velocity.x = target_vel.x();
    mpc_target.velocity.y = target_vel.y();
    MPCResult mpc_result = mpc_controller_->compute(mpc_target, current_bullet_speed, rune_t);

    double final_pitch, final_yaw;
    double yaw_vel = 0.0, yaw_acc = 0.0, pitch_vel = 0.0, pitch_acc = 0.0;

    if (use_mpc_default && mpc_result.is_valid) {
        RCLCPP_DEBUG(this->get_logger(), "MPC solved successfully for rune.");
        final_pitch = mpc_result.target_pitch;
        final_yaw = mpc_result.target_yaw;
        
        yaw_vel = mpc_result.yaw_vel;
        yaw_acc = mpc_result.yaw_acc;
        pitch_vel = mpc_result.pitch_vel;
        pitch_acc = mpc_result.pitch_acc;
    } else {
        RCLCPP_WARN(this->get_logger(), "MPC failed to solve for rune. Skipping publish.");
        return; // 直接返回，不发布
    }
    
    // 将 odom 坐标系中的点投影到图像上（使用计算出的瞄准时间 iteration_result.second）
    cv::Point2f projected_point = projectPointToImage(rune_info_->getOdomTarget(rune_t));

    //发布消息
    firemsg fire_msg;
    fire_msg.header = _target_msg->header;
    fire_msg.pitch = final_pitch;
    fire_msg.yaw = final_yaw;

    // 新增MPC输出的控制量
    fire_msg.yaw_vel = yaw_vel;
    fire_msg.yaw_acc = yaw_acc;
    fire_msg.pitch_vel = pitch_vel;
    fire_msg.pitch_acc = pitch_acc;

    fire_msg.tracking = _target_msg->tracking;
    fire_msg.projected_x = projected_point.x;
    fire_msg.projected_y = projected_point.y;
    fire_msg.id = "rune";
    fire_msg.iffire = 0; // 符文模式下暂不使用MPC开火决策
    publisher_->publish(fire_msg);
}
cv::Point2f BallisticCalculateNode::projectPointToImage(const Eigen::Vector3d & point_3d)
{
    // 获取相机内参矩阵
    cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << cam_info_.k[0], cam_info_.k[1], cam_info_.k[2], cam_info_.k[3],
         cam_info_.k[4], cam_info_.k[5], cam_info_.k[6], cam_info_.k[7], cam_info_.k[8]);

    // 获取相机畸变系数
    cv::Mat dist_coeffs =
        (cv::Mat_<double>(1, 5) << cam_info_.d[0], cam_info_.d[1], cam_info_.d[2], cam_info_.d[3],
         cam_info_.d[4]);

    // 世界坐标点
    std::vector<cv::Point3d> object_point = {cv::Point3d(point_3d.x(), point_3d.y(), point_3d.z())};

    // 输出的图像点
    std::vector<cv::Point2d> image_points;

    cv::Mat rvec, tvec;

    // ROS 坐标系到 OpenCV 坐标系的变换矩阵
    cv::Mat ros_to_cv = (cv::Mat_<double>(3, 3) << 0, -1, 0, 0, 0, -1, 1, 0, 0);

    try {
        // 获取从 odom 到 camera_link 的变换
        geometry_msgs::msg::TransformStamped transform_stamped =
            tfBuffer->lookupTransform("camera_link", "odom", tf2::TimePointZero);

        // 提取旋转部分
        tf2::Quaternion quat(
            transform_stamped.transform.rotation.x, transform_stamped.transform.rotation.y,
            transform_stamped.transform.rotation.z, transform_stamped.transform.rotation.w);

        tf2::Matrix3x3 mat(quat);
        // 转换旋转矩阵
        cv::Mat rotation_matrix =
            (cv::Mat_<double>(3, 3) << mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1],
             mat[1][2], mat[2][0], mat[2][1], mat[2][2]);

        // 调整旋转矩阵
        rotation_matrix = ros_to_cv * rotation_matrix;

        // 将旋转矩阵转换为旋转向量
        cv::Rodrigues(rotation_matrix, rvec);

        // 提取平移部分
        tvec =
            (cv::Mat_<double>(3, 1) << transform_stamped.transform.translation.x,
             transform_stamped.transform.translation.y, transform_stamped.transform.translation.z);

        // 调整平移向量
        tvec = ros_to_cv * tvec;

        // 执行投影
        cv::projectPoints(object_point, rvec, tvec, camera_matrix, dist_coeffs, image_points);

        return cv::Point2f(image_points[0].x / cam_info_.width, image_points[0].y / cam_info_.height);
    } catch (tf2::TransformException & ex) {
        RCLCPP_WARN(this->get_logger(), "Could not transform: %s", ex.what());
        return cv::Point2f(-1, -1);  // 返回无效点
    }
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::BallisticCalculateNode)
