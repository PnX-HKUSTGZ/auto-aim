// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <tf2_ros/create_timer_ros.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "armor_detector/detector_node.hpp"
#include "armor_detector/types.hpp"

namespace rm_auto_aim
{
// ==================== 构造函数 ====================
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions & options)
: Node("armor_detector", options)
{
    RCLCPP_INFO(this->get_logger(), "Starting DetectorNode!");

    // 是否使用 AI detector 参数
    use_ai_detector_ = this->declare_parameter("use_ai_detector", false);

    //设置需要探测的颜色
    declare_parameter("detect_color", RED);

    if (use_ai_detector_) {
        // 初始化 AI 检测器
        ai_detector_ = initAIDetector();
    } else {
        // 初始化Detector参数
        detector_ = initDetector();
    }

    //提取相机内参
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
            cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
            cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
            pnp_solver_ = std::make_unique<PnPSolver>(camera_info->k, camera_info->d);
            ba_solver_ = std::make_unique<BaSolver>(camera_info->k, camera_info->d);
            cam_info_sub_.reset();  //取消订阅
        });

    // 设置自瞄模式
    set_mode_srv_ = this->create_service<auto_aim_interfaces::srv::SetMode>(
        "armor_detector/set_mode", std::bind(
                                       &ArmorDetectorNode::setModeCallback, this,
                                       std::placeholders::_1, std::placeholders::_2));

    //收到图像信息后回调imageCallback函数
    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/image_raw", rclcpp::SensorDataQoS(),
        std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1));

    // 初始化Armors Publisher
    armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>(
        "/detector/armors", rclcpp::SensorDataQoS());

    //tf2
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

    // Debug
    debug_ = this->declare_parameter("debug", false);
    if (debug_) {
        createDebugPublishers();
    }
    debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    debug_cb_handle_ =
        debug_param_sub_->add_parameter_callback("debug", [this](const rclcpp::Parameter & p) {
            debug_ = p.as_bool();
            debug_ ? createDebugPublishers() : destroyDebugPublishers();
        });
}

// ==================== 初始化功能 ====================
std::unique_ptr<Detector> ArmorDetectorNode::initDetector()
{
    rcl_interfaces::msg::ParameterDescriptor param_desc;  //用于描述填充参数
    //设置二值化参数
    param_desc.integer_range.resize(1);
    param_desc.integer_range[0].step = 1;
    param_desc.integer_range[0].from_value = 0;
    param_desc.integer_range[0].to_value = 255;
    int binary_thres = declare_parameter("binary_thres", 80, param_desc);
    //填充light和armor类所需要的参数
    LightParams l_params = {

        .min_ratio = declare_parameter("light.min_ratio", 0.1),
        .max_ratio = declare_parameter("light.max_ratio", 0.4),
        .max_angle = declare_parameter("light.max_angle", 40.0)};

    ArmorParams a_params = {
        .min_light_ratio = declare_parameter("armor.min_light_ratio", 0.7),
        .min_small_center_distance = declare_parameter("armor.min_small_center_distance", 0.8),
        .max_small_center_distance = declare_parameter("armor.max_small_center_distance", 3.2),
        .min_large_center_distance = declare_parameter("armor.min_large_center_distance", 3.2),
        .max_large_center_distance = declare_parameter("armor.max_large_center_distance", 5.5),
        .max_angle = declare_parameter("armor.max_angle", 35.0)};

    //number classifier 参数
    auto pkg_path = ament_index_cpp::get_package_share_directory("armor_detector");
    auto model_path = pkg_path + "/model/mlp.onnx";
    auto label_path = pkg_path + "/model/label.txt";
    double threshold = this->declare_parameter("classifier_threshold", 0.7);
    std::vector<std::string> ignore_classes = this->declare_parameter(
        "ignore_classes", std::vector<std::string>{"negative"});  ////这里的this并非必须

    //初始化 detector
    auto detector = std::make_unique<Detector>(
        binary_thres, l_params, a_params, model_path, label_path, threshold, ignore_classes);

    RCLCPP_INFO(this->get_logger(), "Detector initialized");

    return detector;
}

std::unique_ptr<AIDetector> ArmorDetectorNode::initAIDetector()
{
    // 声明AI检测器相关参数
    auto model_path = ament_index_cpp::get_package_share_directory("armor_detector") +
                      this->declare_parameter("ai_model_path", "/model/0526.onnx");
    auto device = this->declare_parameter("ai_device", "CPU");
    auto conf_threshold = this->declare_parameter("ai_conf_threshold", 0.65);
    auto nms_threshold = this->declare_parameter("ai_nms_threshold", 0.45);

    // 创建AI检测器实例
    auto ai_detector = std::make_unique<AIDetector>(
        model_path, device, static_cast<float>(conf_threshold), static_cast<float>(nms_threshold));

    RCLCPP_INFO(this->get_logger(), "AI Detector initialized with model: %s", model_path.c_str());

    return ai_detector;
}

// ==================== 核心处理功能 ====================
void ArmorDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
    // 如果当前模式为打符模式，不进行装甲板检测
    if (!enable_) {
        return;
    }
    if (debug_) armors_msg_.image = *img_msg;

    // 检测装甲板
    cv::Mat img;
    std::vector<Armor> armors;
    if (use_ai_detector_) {
        armors = aiDetectArmors(img_msg, img);
    } else {
        armors = detectArmors(img_msg, img);
    }

    // 提取from odom to gimbal的坐标系变换
    if (!updateTransform(img_msg->header.frame_id, "odom", img_msg->header.stamp)) {
        return;
    }

    if (pnp_solver_ == nullptr) return;  //如果pnp解算未初始化

    //初始化消息，包括装甲板信息和调试信息
    armors_msg_.header = img_msg->header;
    armors_msg_.armors.clear();
    if (debug_) {
        armor_marker_.header = text_marker_.header = img_msg->header;
        marker_array_.markers.clear();
        armor_marker_.id = 0;
        text_marker_.id = 0;
    }

    // 进行位姿解算和降自由度优化

    //将装甲板分类存放，无效装甲板的装甲板数标为-1
    // std::string         int           std::vector<int>
    // 装甲板类别      这类装甲板的数目    这类装甲板的编号
    std::map<std::string, std::pair<int, std::vector<int>>> armor_num_map;
    std::vector<int> valid_armors;  //筛选出能解算，符合先验的有效装甲板，并储存编号
    for (size_t i = 0; i < armors.size(); i++) {
        cv::Mat rvec, tvec;
        bool success =
            pnp_solver_->solvePnP(armors[i], rvec, tvec);  //通过pnp解算获得两个装甲板位姿解
        if (success) {
            if (armor_num_map[armors[i].number].first == -1)
                continue;  // 如果发生了解算失败，这一帧的装甲板数据宁可放弃
            // 装甲板先验可知roll为0，pitch为15度
            // 通过装甲板的类型所决定的先验信息，在两个解中选择较好的那个
            // 同样通过这个先验信息，将原本三自由度的装甲板压到只有yaw一个自由度
            chooseBestPose(armors[i], rvec, tvec);
            armor_num_map[armors[i].number].first++;
            armor_num_map[armors[i].number].second.push_back(i);
            if (armor_num_map[armors[i].number].first > 2) {
                armor_num_map[armors[i].number].first = -1;
                RCLCPP_ERROR(this->get_logger(), "More than 2 armors detected!");
            }
        } else {
            armor_num_map[armors[i].number].first = -1;
            RCLCPP_ERROR(this->get_logger(), "PnP failed!");
        }
    }
    for (auto & armor_num : armor_num_map) {
        if (armor_num.second.first == 2) {
            //利用同一台车上的两块装甲板关系的先验提高解算正确程度
            bool success = ba_solver_->fixTwoArmors(
                armors[armor_num.second.second[0]], armors[armor_num.second.second[1]],
                r_odom_to_camera, t_odom_to_camera);
            if (!success) {
                RCLCPP_ERROR(this->get_logger(), "Fix two armors failed!");
                armor_num.second.first = -1;
                continue;
            }
            armors[armor_num.second.second[0]].setCameraArmor(r_odom_to_camera, t_odom_to_camera);
            armors[armor_num.second.second[1]].setCameraArmor(r_odom_to_camera, t_odom_to_camera);
        }
        if (armor_num.second.first != -1) {
            //视为有效装甲板
            valid_armors.insert(
                valid_armors.end(), armor_num.second.second.begin(), armor_num.second.second.end());
        }
    }
    //填充有效装甲板到消息中，发送给tracker或者调试
    auto_aim_interfaces::msg::Armor armor_msg;
    for (auto & index : valid_armors) {
        // Fill basic info
        armor_msg.type = ARMOR_TYPE_STR[static_cast<int>(armors[index].type)];
        armor_msg.number = armors[index].number;

        // Fill pose
        Eigen::Quaterniond eigen_quat(armors[index].r_camera_armor);
        tf2::Quaternion tf2_q(eigen_quat.x(), eigen_quat.y(), eigen_quat.z(), eigen_quat.w());
        armor_msg.pose.orientation = tf2::toMsg(tf2_q);
        armor_msg.pose.position.x = armors[index].t_camera_armor(0);
        armor_msg.pose.position.y = armors[index].t_camera_armor(1);
        armor_msg.pose.position.z = armors[index].t_camera_armor(2);

        // Fill the distance to image center
        armor_msg.distance_to_image_center =
            pnp_solver_->calculateDistanceToCenter(armors[index].center);

        // Fill the classification result
        armors_msg_.armors.emplace_back(armor_msg);

        // Fill the debug markers
        if (debug_) {
            armor_marker_.id++;
            armor_marker_.scale.y = armors[index].type == ArmorType::SMALL ? 0.135 : 0.23;
            armor_marker_.pose = armor_msg.pose;
            text_marker_.id++;
            text_marker_.pose.position = armor_msg.pose.position;
            text_marker_.pose.position.y -= 0.1;
            text_marker_.text = armors[index].classfication_result;
            marker_array_.markers.emplace_back(armor_marker_);
            marker_array_.markers.emplace_back(text_marker_);
        }
    }
    // Publishing detected armors
    armors_pub_->publish(armors_msg_);

    if (debug_) {
        // draw results
        drawResults(img_msg, img, armors);
        // Publishing marker
        publishMarkers();
    }
}

std::vector<Armor> ArmorDetectorNode::detectArmors(
    const sensor_msgs::msg::Image::ConstSharedPtr & img_msg, cv::Mat & img)
{
    // Convert ROS img to cv::Mat
    img = cv_bridge::toCvShare(img_msg, "rgb8")->image;

    // get color
    int detect_color = get_parameter("detect_color").as_int();

    auto armors = detector_->detect(img, detect_color);

    // Publish debug info
    if (debug_) {
        //计算延迟
        auto final_time = this->now();
        auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");

        binary_img_pub_.publish(
            cv_bridge::CvImage(img_msg->header, "mono8", detector_->getBinaryImage()).toImageMsg());

        if (!armors.empty()) {
            number_img_pub_.publish(
                *cv_bridge::CvImage(img_msg->header, "mono8", detector_->getAllNumbersImage())
                     .toImageMsg());
        }
    }
    return armors;
}

std::vector<Armor> ArmorDetectorNode::aiDetectArmors(
    const sensor_msgs::msg::Image::ConstSharedPtr & img_msg, cv::Mat & img)
{
    // Convert ROS img to cv::Mat
    img = cv_bridge::toCvShare(img_msg, "rgb8")->image;

    // 使用 AI 检测器
    int detect_color = get_parameter("detect_color").as_int();
    auto armors = ai_detector_->detect(img, detect_color);

    // Publish debug info
    if (debug_) {
        //计算延迟
        auto final_time = this->now();
        auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");
    }
    return armors;
}

// ==================== 坐标变换和位姿处理 ====================
bool ArmorDetectorNode::updateTransform(
    std::string target_frame, std::string source_frame, rclcpp::Time timestamp)
{
    try {
        auto latest_tf =
            tf2_buffer_->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
        const rclcpp::Time & target_time = timestamp;
        rclcpp::Time latest_time = latest_tf.header.stamp;
        // 比较时间戳
        geometry_msgs::msg::TransformStamped odom_to_camera_tf;
        if (target_time > latest_time) {
            // 使用最新变换
            odom_to_camera_tf = latest_tf;
        } else {
            // 查找指定时间的变换
            odom_to_camera_tf = tf2_buffer_->lookupTransform(
                target_frame, source_frame, target_time,
                rclcpp::Duration::from_nanoseconds(1000000));
        }
        auto msg_q = odom_to_camera_tf.transform.rotation;
        tf2::Quaternion tf_q;
        tf2::fromMsg(msg_q, tf_q);
        tf2::Matrix3x3 tf2_matrix = tf2::Matrix3x3(tf_q);
        r_odom_to_camera << tf2_matrix.getRow(0)[0], tf2_matrix.getRow(0)[1],
            tf2_matrix.getRow(0)[2], tf2_matrix.getRow(1)[0], tf2_matrix.getRow(1)[1],
            tf2_matrix.getRow(1)[2], tf2_matrix.getRow(2)[0], tf2_matrix.getRow(2)[1],
            tf2_matrix.getRow(2)[2];
        t_odom_to_camera = Eigen::Vector3d(
            odom_to_camera_tf.transform.translation.x, odom_to_camera_tf.transform.translation.y,
            odom_to_camera_tf.transform.translation.z);
        return 1;
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Something Wrong when lookUpTransform");
        return 0;
    }
}

void ArmorDetectorNode::chooseBestPose(Armor & armor, const cv::Mat & rvec, const cv::Mat & tvec)
{
    //提取云台系欧拉角
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    Eigen::Matrix3d rotation_matrix_eigen;
    cv::cv2eigen(rotation_matrix, rotation_matrix_eigen);
    Eigen::Matrix3d R_camera_to_gimble;
    R_camera_to_gimble << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    Eigen::Vector3d rpy = (R_camera_to_gimble * rotation_matrix_eigen).eulerAngles(0, 1, 2);

    //对于云台系来说，将yaw归一到-pi/2 到 pi/2中后：左侧装甲板yaw角为负，右侧装甲板yaw角为正
    rpy(1) = std::atan2(std::sin(rpy(2)), std::cos(rpy(2)));
    if (abs(rpy(2)) > M_PI / 2) {
        rpy(0) = std::atan2(std::sin(M_PI + rpy(0)), std::cos(M_PI + rpy(0)));  // 旋转roll 180度
        rpy(1) = std::atan2(std::sin(M_PI - rpy(1)), std::cos(M_PI - rpy(1)));  // pitch, 使用补角
        rpy(2) = std::atan2(std::sin(M_PI + rpy(2)), std::cos(M_PI + rpy(2)));  // 旋转yaw 180度
    }
    //前哨站装甲板负倾角
    if (armor.number == "outpost") armor.sign = !armor.sign;
    // armor.sign 为0则为右侧装甲板，为1则为左侧装甲板
    if (!armor.sign) {
        rpy = Eigen::Vector3d(rpy(0), rpy(1), abs(rpy(2)));  //右侧
    } else {
        rpy = Eigen::Vector3d(rpy(0), rpy(1), -abs(rpy(2)));  //左侧
    }

    //构造装甲板的旋转平移矩阵
    armor.r_odom_armor = r_odom_to_camera.inverse() * R_camera_to_gimble.inverse() *
                         (Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ()))
                             .toRotationMatrix();
    armor.t_odom_armor =
        r_odom_to_camera.inverse() *
        (Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)) -
         t_odom_to_camera);
    armor.setCameraArmor(r_odom_to_camera, t_odom_to_camera);
    if (abs(rpy(0)) < 0.26) {
        ba_solver_->solveBa(armor, r_odom_to_camera, t_odom_to_camera);
        armor.setCameraArmor(r_odom_to_camera, t_odom_to_camera);
    } else {
        RCLCPP_WARN(this->get_logger(), "The car is on the slope");
    }
}

// ==================== 可视化和调试功能 ====================
void ArmorDetectorNode::drawResults(
    const sensor_msgs::msg::Image::ConstSharedPtr & img_msg, cv::Mat & img,
    const std::vector<Armor> & armors)
{
    //计算延迟
    auto final_time = this->now();
    auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");
    if (!debug_) {
        return;
    }
    if (!use_ai_detector_) {
        detector_->drawResults(img);
    } else {
        ai_detector_->drawResults(img);
    }
    // Show yaw, pitch, roll
    for (const auto & armor : armors) {
        Eigen::Vector3d rpy = armor.r_odom_armor.eulerAngles(0, 1, 2);  //提取欧拉角
        // 归一化
        if (abs(rpy(1)) > M_PI / 2) {
            rpy(0) =
                std::atan2(std::sin(M_PI + rpy(0)), std::cos(M_PI + rpy(0)));  // 旋转roll 180度
            rpy(1) =
                std::atan2(std::sin(M_PI - rpy(1)), std::cos(M_PI - rpy(1)));  // pitch, 使用补角
            rpy(2) = std::atan2(std::sin(M_PI + rpy(2)), std::cos(M_PI + rpy(2)));  // 旋转yaw 180度
        }
        double distance = armor.t_camera_armor.norm();
        cv::putText(
            img, "y: " + std::to_string(int(rpy(2) / CV_PI * 180)) + " deg",
            cv::Point(armor.left_light.bottom.x, armor.left_light.bottom.y + 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        cv::putText(
            img, "d: " + std::to_string(distance) + "m",
            cv::Point(armor.left_light.bottom.x, armor.left_light.bottom.y + 60),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    }
    // Draw camera center
    cv::circle(img, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
    // Draw latency
    std::stringstream latency_ss;
    latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
    auto latency_s = latency_ss.str();
    cv::putText(
        img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    result_img_pub_.publish(cv_bridge::CvImage(img_msg->header, "rgb8", img).toImageMsg());
}

void ArmorDetectorNode::createDebugPublishers()
{
    marker_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>("/detector/marker", 10);

    binary_img_pub_ = image_transport::create_publisher(this, "/detector/binary_img");
    number_img_pub_ = image_transport::create_publisher(this, "/detector/number_img");
    result_img_pub_ = image_transport::create_publisher(this, "/detector/result_img");
}

void ArmorDetectorNode::destroyDebugPublishers()
{
    marker_pub_.reset();

    binary_img_pub_.shutdown();
    number_img_pub_.shutdown();
    result_img_pub_.shutdown();
}

void ArmorDetectorNode::publishMarkers()
{
    using Marker = visualization_msgs::msg::Marker;
    armor_marker_.action = armors_msg_.armors.empty() ? Marker::DELETE : Marker::ADD;
    marker_array_.markers.emplace_back(armor_marker_);
    marker_pub_->publish(marker_array_);
}

// ==================== 服务回调 ====================
void ArmorDetectorNode::setModeCallback(
    const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
    std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response)
{
    response->success = true;

    VisionMode mode = static_cast<VisionMode>(request->mode);
    std::string mode_name = visionModeToString(mode);
    if (mode_name == "UNKNOWN") {
        RCLCPP_ERROR(this->get_logger(), "Invalid mode: %d", request->mode);
        return;
    }

    switch (mode) {
        case VisionMode::RUNE: {
            enable_ = false;
            break;
        }
        default: {
            enable_ = true;
            break;
        }
    }

    RCLCPP_INFO(this->get_logger(), "Set Car Mode: %s", visionModeToString(mode).c_str());
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorNode)
