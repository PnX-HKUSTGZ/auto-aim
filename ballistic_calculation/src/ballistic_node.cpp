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

namespace rm_auto_aim
{
using target = auto_aim_interfaces::msg::Target;
using firemsg = auto_aim_interfaces::msg::Firecontrol;

BallisticCalculateNode::BallisticCalculateNode(const rclcpp::NodeOptions & options)
: Node("ballistic_calculation", options)
{
    RCLCPP_INFO(this->get_logger(), "start ballistic calculation!");
    K1 = this->declare_parameter("iteration_coeffcient_first", 0.1);
    K2 = this->declare_parameter("iteration_coeffcient_second", 0.05);
    K = this->declare_parameter("air_resistence", 0.1);
    BULLET_V = this->declare_parameter("bullet_speed", 23.0);
    ifFireK_ = this->declare_parameter("ifFireK", 0.05);
    min_v = this->declare_parameter("swich_stategy_1", 5.0) * M_PI / 30;
    max_v = this->declare_parameter("swich_stategy_2", 30.0) * M_PI / 30;
    v_yaw_gimble = this->declare_parameter("max_v_yaw_gimble", 0.8);
    stop_fire_time = this->declare_parameter("stop_fire_time", 0.1);
    double fire_delay = this->declare_parameter("fire_delay", 0.0);
    xyz_vec = this->declare_parameter("xyz", std::vector<double>{0.0, 0.0, 0.0});
    rpy_vec = this->declare_parameter("rpy", std::vector<double>{0.0, 0.0, 0.0});
    Eigen::Vector3d odom2gunxyz(xyz_vec[0], xyz_vec[1], xyz_vec[2]);
    Eigen::Vector3d odom2gunrpy(rpy_vec[0], rpy_vec[1], rpy_vec[2]);

    calculator = std::make_unique<rm_auto_aim::Ballistic>(K, BULLET_V, fire_delay);
    car_info_ = std::make_shared<CarInfo>(odom2gunxyz, odom2gunrpy);
    armor_info_ = std::make_unique<ArmorInfo>(odom2gunxyz, odom2gunrpy);
    rune_info_ = std::make_unique<RuneInfo>(odom2gunxyz, odom2gunrpy);
    armor_selector_ = std::make_shared<ArmorSelector>();

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
    return std::abs(yaw - targetyaw) < ifFireK && std::abs(pitch + targetpitch) < ifFireK;
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
    Eigen::Vector3d target = rune_info_->getGunTarget(0.0);

    //进入迭代
    double init_pitch =
        std::atan(target[2] / std::sqrt(target[0] * target[0] + target[1] * target[1]));
    double init_t =
        std::sqrt(target[0] * target[0] + target[1] * target[1]) / (cos(init_pitch) * BULLET_V);

    std::pair<double, double> first_iteration_result =
        this->calculator->iteration(THRES1, init_pitch, init_t, *car_info_);

    //预测并选择合适击打的装甲板
    double temp_theta = first_iteration_result.first;
    double temp_t = first_iteration_result.second;

    //预测平衡步兵的最佳装甲板
    std::pair<double, double> iffire_result, final_result;
    if (car_target_msg->armors_num == 4 or car_target_msg->armors_num == 3) {
        std::vector<double> hit_aim = armor_selector_->
            predictInfantryBestArmor(temp_t, min_v, max_v, v_yaw_gimble);
        std::vector<double> hit_aim_fire = armor_selector_->
            predictInfantryBestArmor(temp_t, DBL_MAX, DBL_MAX, v_yaw_gimble);
        
        //计算瞄准目标
        armor_info_->updateTarget(
            *car_target_msg, hit_aim[0], hit_aim[2] == 0 ? hit_aim_fire[1] : hit_aim[1],
            hit_aim[2]);
        final_result = calculator->iteration(THRES2, temp_theta, temp_t, *armor_info_);
        //计算是否开火
        armor_info_->updateTarget(
            *car_target_msg, hit_aim_fire[0], hit_aim_fire[1], hit_aim_fire[2]);
        iffire_result = calculator->iteration(THRES2, temp_theta, temp_t, *armor_info_);
    } else {
        RCLCPP_ERROR(this->get_logger(), "The number of armors is not 4 or 3");
        return;
    }
    // 将 odom 坐标系中的点投影到图像上
    cv::Point2f projected_point = projectPointToImage(armor_info_->getOdomTarget(0.0));

    //发布消息
    firemsg fire_msg;
    fire_msg.header = car_target_msg->header;
    fire_msg.pitch = final_result.first + rpy_vec[1];
    fire_msg.yaw = final_result.second - rpy_vec[2];
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

    std::pair<double, double> iteration_result =
        this->calculator->iteration(THRES2, init_pitch, init_t, *rune_info_);

    // 将 odom 坐标系中的点投影到图像上
    cv::Point2f projected_point = projectPointToImage(rune_info_->getOdomTarget(0.0));

    //发布消息
    firemsg fire_msg;
    fire_msg.header = _target_msg->header;
    fire_msg.pitch = iteration_result.first;
    fire_msg.yaw = iteration_result.second;
    fire_msg.tracking = _target_msg->tracking;
    fire_msg.projected_x = projected_point.x;
    fire_msg.projected_y = projected_point.y;
    fire_msg.id = "rune";
    fire_msg.iffire = 0;
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
