// Copyright 2022 Chen Jun
#include "armor_tracker/tracker_node.hpp"

// STD
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <vector>

namespace rm_auto_aim
{
ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions & options)
: Node("armor_tracker", options)
{
    RCLCPP_INFO(this->get_logger(), "Starting TrackerNode!");

    // Maximum allowable armor distance in the XOY plane
    max_armor_distance_ = this->declare_parameter("max_armor_distance", 10.0);
    debug_ = this->declare_parameter("debug", false);
    // Tracker
    double max_match_distance = this->declare_parameter("tracker.max_match_distance", 0.15);
    double max_match_yaw_diff = this->declare_parameter("tracker.max_match_yaw_diff", 1.0);
    

    // 初始化追踪器管理器替代单一追踪器
    tracker_manager_ = std::make_unique<TrackerManager>(
        max_match_distance,
        max_match_yaw_diff,
        this->declare_parameter("tracker.tracking_thres", 5),  // 传递tracking_thres
        this->declare_parameter("tracker.lost_time_thres", 0.3), 
        this->declare_parameter("tracker.switch_cooldown", 1.0));
        
    // 设置评分权重参数
    tracker_manager_->setWeights(
        this->declare_parameter("score.w_distance", 0.5),
        this->declare_parameter("score.w_tracking", 0.5));

    // Initialize EKF
    initializeEKF();
    // Reset tracker service
    reset_tracker_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/tracker/reset", [this](
                            const std_srvs::srv::Trigger::Request::SharedPtr,
                            std_srvs::srv::Trigger::Response::SharedPtr response) {
            tracker_manager_->reset();
            response->success = true;
            RCLCPP_INFO(this->get_logger(), "Tracker reset!");
            return;
        });

      // set_mode
    set_mode_srv_ = this->create_service<auto_aim_interfaces::srv::SetMode>(
        "armor_tracker/set_mode",
        std::bind(
        &ArmorTrackerNode::setModeCallback, this, std::placeholders::_1, std::placeholders::_2));
    // Camera info subscription
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
            cam_info_ = *camera_info;
            cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
            cam_info_sub_.reset();
        });
    // TF2 setup
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
    armors_sub_.subscribe(this, "/detector/armors", rmw_qos_profile_sensor_data);
    target_frame_ = this->declare_parameter("target_frame", "odom");

    tf2_filter_ = std::make_shared<tf2_ros::MessageFilter<auto_aim_interfaces::msg::Armors>>(
        armors_sub_, *tf2_buffer_, target_frame_, 10, this->get_node_logging_interface(),
        this->get_node_clock_interface(), std::chrono::duration<int>(1));
    tf2_filter_->registerCallback(&ArmorTrackerNode::armorsCallback, this);

    // Publishers
    info_pub_ = this->create_publisher<auto_aim_interfaces::msg::TrackerInfo>("/tracker/info", 10);
    target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
        "/tracker/target", rclcpp::SensorDataQoS());
    tracker_img_pub_ = image_transport::create_publisher(this, "/tracker/result_img");
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/tracker/marker", 10);

    // Visualization Marker setup
    if(debug_){
        position_marker_.ns = "position";
        position_marker_.type = visualization_msgs::msg::Marker::SPHERE;
        position_marker_.scale.x = position_marker_.scale.y = position_marker_.scale.z = 0.1;
        position_marker_.color.a = 1.0;
        position_marker_.color.g = 1.0;
        linear_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
        linear_v_marker_.ns = "linear_v";
        linear_v_marker_.scale.x = 0.03;
        linear_v_marker_.scale.y = 0.05;
        linear_v_marker_.color.a = 1.0;
        linear_v_marker_.color.r = 1.0;
        linear_v_marker_.color.g = 1.0;
        angular_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
        angular_v_marker_.ns = "angular_v";
        angular_v_marker_.scale.x = 0.03;
        angular_v_marker_.scale.y = 0.05;
        angular_v_marker_.color.a = 1.0;
        angular_v_marker_.color.b = 1.0;
        angular_v_marker_.color.g = 1.0;
        armor_marker_.ns = "armors";
        armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
        armor_marker_.scale.x = 0.03;
        armor_marker_.scale.z = 0.125;
        armor_marker_.color.a = 1.0;
        armor_marker_.color.r = 1.0;
    }
}

void ArmorTrackerNode::initializeEKF()
{
    // EKF
    // xa = x_armor, xc = x_robot_center
    // state: xc, v_xc, yc, v_yc, zc1, zc2, v_zc, v_yaw, r1, r2, yaw1, yaw2
    // measurement: xa, ya, za, yaw
    // f - Process function
    auto f = [this](const Eigen::VectorXd & x) {
        Eigen::VectorXd x_new = x;
        x_new(XC) += x(VXC) * dt_;
        x_new(YC) += x(VYC) * dt_;
        x_new(ZC1) += x(VZC) * dt_;
        x_new(ZC2) += x(VZC) * dt_;
        x_new(YAW1) += x(VYAW) * dt_;
        x_new(YAW2) += x(VYAW) * dt_;
        return x_new;
    };
    // J_f - Jacobian of process function
    auto j_f = [this](const Eigen::VectorXd &) {
        Eigen::MatrixXd f(12, 12);
        // clang-format off
        f <<1,   dt_, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // xc = xc + v_xc * dt
            0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, // v_xc = v_xc
            0,   0,   1,   dt_, 0,   0,   0,   0,   0,   0,   0,   0, // yc = yc + v_yc * dt
            0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0, // v_yc = v_yc
            0,   0,   0,   0,   1,   0, dt_,   0,   0,   0,   0,   0, // zc1 = zc1 + v_zc * dt
            0,   0,   0,   0,   0,   1, dt_,   0,   0,   0,   0,   0, // zc2 = zc2 + v_zc * dt
            0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0, // v_zc = v_zc
            0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0, // v_yaw = v_yaw
            0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0, // r1 = r1
            0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0, // r2 = r2
            0,   0,   0,   0,   0,   0,   0, dt_,   0,   0,   1,   0, // yaw1 = yaw1 + v_yaw * dt
            0,   0,   0,   0,   0,   0,   0, dt_,   0,   0,   0,   1; // yaw2 = yaw2 + v_yaw * dt
        // clang-format on
        return f;
    };
    // h1 - Observation function for armor 1
    auto h1 = [](const Eigen::VectorXd & x) {
        Eigen::VectorXd z(4);
        double xc = x(XC), yc = x(YC), yaw = x(YAW1), r = x(R1);
        z(0) = xc - r * cos(yaw);  // xa
        z(1) = yc - r * sin(yaw);  // ya
        z(2) = x(ZC1);                  // za
        z(3) = x(YAW1);                 // yaw
        return z;
    };
    // J_h1 - Jacobian of observation function for armor 1
    auto j_h1 = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(4, 12);
        double yaw = x(YAW1), r = x(R1);
        // clang-format off
        //    xc   v_xc yc   v_yc zc1  zc2  v_zc v_yaw r1  r2  yaw1 yaw2
        h <<  1,   0,   0,   0,   0,   0,   0,   0,    -cos(yaw), 0,  r*sin(yaw), 0, // xa = xc - r1 * cos(yaw1)
              0,   0,   1,   0,   0,   0,   0,   0,    -sin(yaw), 0, -r*cos(yaw), 0,  // ya = yc - r1 * sin(yaw1)
              0,   0,   0,   0,   1,   0,   0,   0,          0,   0,           0, 0, // za = zc1
              0,   0,   0,   0,   0,   0,   0,   0,          0,   0,           1, 0; // yaw = yaw1
        // clang-format on
        return h;
    };
    // h2 - Observation function for armor 2
    auto h2 = [](const Eigen::VectorXd & x) {
        Eigen::VectorXd z(4);
        double xc = x(XC), yc = x(YC), yaw = x(YAW2), r = x(R2);
        z(0) = xc - r * cos(yaw);  // xa
        z(1) = yc - r * sin(yaw);  // ya
        z(2) = x(ZC2);                  // za
        z(3) = x(YAW2);                 // yaw
        return z;
    };
    // J_h2 - Jacobian of observation function for armor 2
    auto j_h2 = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(4, 12);
        double yaw = x(YAW2), r = x(R2);
        // clang-format off
        //    xc   v_xc yc   v_yc zc1  zc2  v_zc v_yaw r1  r2  yaw1 yaw2
        h <<  1,   0,   0,   0,   0,   0,   0,   0,  0,    -cos(yaw), 0,  r*sin(yaw), // xa = xc - r2 * cos(yaw2)
              0,   0,   1,   0,   0,   0,   0,   0,  0,    -sin(yaw), 0, -r*cos(yaw), // ya = yc - r2 * sin(yaw2)
              0,   0,   0,   0,   0,   1,   0,   0,          0,   0,          0, 0, // za = zc2
              0,   0,   0,   0,   0,   0,   0,   0,          0,   0,          0, 1; // yaw = yaw2
        // clang-format on
        return h;
    }; 
    // h_two - Observation function for 2 armors
    auto h_two = [](const Eigen::VectorXd & x) {
        Eigen::VectorXd z(10);
        double xc = x(XC), yc = x(YC), yaw1 = x(YAW1), yaw2 = x(YAW2), r1 = x(R1), r2 = x(R2);
        z(0) = xc - r1 * cos(yaw1);  // xa1
        z(1) = yc - r1 * sin(yaw1);  // ya1
        z(2) = x(ZC1);                // za1
        z(3) = yaw1;                  // yaw1
        z(4) = xc - r2 * cos(yaw2);  // xa2
        z(5) = yc - r2 * sin(yaw2);  // ya2
        z(6) = x(ZC2);                // za2
        z(7) = yaw2;                  // yaw2
        z(8) = r1;                    // r1
        z(9) = r2;                    // r2
        return z;
    };
    // J_h_two - Jacobian of observation function for 2 armors
    auto j_h_two = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(10, 12);
        double yaw1 = x(YAW1), yaw2 = x(YAW2), r1 = x(R1), r2 = x(R2);
        // clang-format off
        //    xc   v_xc yc   v_yc zc1  zc2  v_zc v_yaw r1  r2  yaw1 yaw2
        h <<  1,   0,   0,   0,   0,   0,   0,   0,   -cos(yaw1), 0, r1*sin(yaw1), 0,   // xa1 = xc - r1 * cos(yaw1)
              0,   0,   1,   0,   0,   0,   0,   0,   -sin(yaw1), 0, -r1*cos(yaw1), 0,   // ya1 = yc - r1 * sin(yaw1)
              0,   0,   0,   0,   1,   0,   0,   0,          0,   0,          0, 0, // za1 = zc1
              0,   0,   0,   0,   0,   0,   0,   0,          0,   0,          1, 0, // yaw1 = yaw1
              1,   0,   0,   0,   0,   0,   0,   0,          0,   -cos(yaw2), 0, r2*sin(yaw2), // xa2 = xc - r2 * cos(yaw2)
              0,   0,   1,   0,   0,   0,   0,   0,          0,   -sin(yaw2), 0, -r2*cos(yaw2), // ya2 = yc - r2 * sin(yaw2)
              0,   0,   0,   0,   0,   1,   0,   0,          0,   0,          0, 0, // za2 = zc2
              0,   0,   0,   0,   0,   0,   0,   0,          0,   0,          0, 1, // yaw2 = yaw2
              0,   0,   0,   0,   0,   0,   0,   0,          1,   0,          0, 0, // r1 = r1 
              0,   0,   0,   0,   0,   0,   0,   0,          0,   1,          0, 0; // r2 = r2
        // clang-format on
        return h;
    };
    // update_Q - process noise covariance matrix
    s2qxy_ = declare_parameter("ekf.sigma2_q_xy", 20.0);
    s2qz_ = declare_parameter("ekf.sigma2_q_z", 20.0);
    s2qyaw_ = declare_parameter("ekf.sigma2_q_yaw", 100.0);
    s2qr_ = declare_parameter("ekf.sigma2_q_r", 800.0);
    auto u_q = [this]() {
        Eigen::MatrixXd q(12, 12);
        double t = dt_, x = s2qxy_, z = s2qz_, y = s2qyaw_, r = s2qr_;
        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        double q_y_y = pow(t, 4) / 4 * x, q_y_vy = pow(t, 3) / 2 * x, q_vy_vy = pow(t, 2) * x;
        double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 * z, q_vz_vz = pow(t, 2) * z;
        double q_yaw_yaw = pow(t, 4) / 4 * y, q_yaw_vyaw = pow(t, 3) / 2 * y, q_vyaw_vyaw = pow(t, 2) * y;
        double q_r = pow(t, 4) / 4 * r;
        // clang-format off
        //    xc      v_xc    yc      v_yc    zc1     zc2     v_zc    v_yaw   r1      r2      yaw1    yaw2
        q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,          0,      0,      0,         0,
              q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,          0,      0,      0,         0,
              0,      0,      q_y_y,  q_y_vy, 0,      0,      0,      0,          0,      0,      0,         0,
              0,      0,      q_y_vy, q_vy_vy,0,      0,      0,      0,          0,      0,      0,         0,
              0,      0,      0,      0,      q_z_z,  0,      q_z_vz, 0,          0,      0,      0,         0,
              0,      0,      0,      0,      0,      q_z_z,  q_z_vz, 0,          0,      0,      0,         0,
              0,      0,      0,      0,      q_z_vz, q_z_vz, q_vz_vz,0,          0,      0,      0,         0,
              0,      0,      0,      0,      0,      0,      0,      q_vyaw_vyaw,0,      0,      q_yaw_vyaw,q_yaw_vyaw,
              0,      0,      0,      0,      0,      0,      0,      0,          q_r,    0,      0,         0,
              0,      0,      0,      0,      0,      0,      0,      0,          0,      q_r,    0,         0,
              0,      0,      0,      0,      0,      0,      0,      q_yaw_vyaw, 0,      0,      q_yaw_yaw, 0,
              0,      0,      0,      0,      0,      0,      0,      q_yaw_vyaw, 0,      0,      0,         q_yaw_yaw;
        // clang-format on
        return q;
    };
    // update_R - measurement noise covariance matrix
    r_xyz_factor = declare_parameter("ekf.r_xyz_factor", 0.05);
    r_yaw = declare_parameter("ekf.r_yaw", 0.02);
    r_radius = declare_parameter("ekf.r_radius", 0.02);
    auto u_r = [this](const Eigen::VectorXd & z) {
        Eigen::DiagonalMatrix<double, 4> r;
        double x = r_xyz_factor;
        r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw;
        return r;
    };
    auto u_r_two = [this](const Eigen::VectorXd & z){
        Eigen::DiagonalMatrix<double, 10> r; 
        double x = r_xyz_factor;
        r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw, 
                        abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw, r_radius, r_radius;
        return r; 
    }; 
    // P - error estimate covariance matrix
    Eigen::DiagonalMatrix<double, 12> p0;
    p0.setIdentity();
    // 创建 EKF 并设置到 TrackerManager 中
    ExtendedKalmanFilter ekf{f, h1, h2, h_two, j_f, j_h1, j_h2, j_h_two, u_q, u_r, u_r_two, p0};
    tracker_manager_->setEKFTemplate(ekf);
}

void ArmorTrackerNode::armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
//它的主要作用是将装甲板的位置从图像帧坐标转换到世界坐标系，过滤掉异常的装甲板，更新追踪器的状态，
//并根据追踪结果发布相关信息和可视化标记
{
    auto start_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(start_time.time_since_epoch());
    current_sec = duration.count() / 1000;
    if(last_sec != current_sec){
        last_sec = current_sec;
        if(frame_count < 100) RCLCPP_INFO(get_logger(), "fps: %d", frame_count);
        frame_count = 0;
    }
    frame_count++;
    
    // Tranform armor position from image frame to world coordinate
    for (auto & armor : armors_msg->armors) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = armors_msg->header;
        ps.pose = armor.pose;
        try {
            armor.pose = tf2_buffer_->transform(ps, target_frame_).pose;
        } catch (const tf2::ExtrapolationException & ex) {
            RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
            return;
        }
    }

    // Filter abnormal armors
    

    armors_msg->armors.erase(
        std::remove_if(
            armors_msg->armors.begin(), armors_msg->armors.end(),
            [this](const auto_aim_interfaces::msg::Armor & armor) {
                return Eigen::Vector2d(armor.pose.position.x, armor.pose.position.y).norm() >
                            max_armor_distance_;
            }),
        armors_msg->armors.end());

    // 计算时间差
    rclcpp::Time time = armors_msg->header.stamp;
    try {
        dt_ = (time - last_time_).seconds();
    }
    catch (const std::exception & e) {
        RCLCPP_WARN(this->get_logger(), "last_time_ has not been initialized");
    }
    last_time_ = time;

    // 使用 TrackerManager 更新所有追踪器
    tracker_manager_->update(armors_msg);
    // 清理不活跃的追踪器
    tracker_manager_->cleanInactiveTrackers( this->now());
    // 获取当前追踪目标
    auto target_msg = tracker_manager_->getCurrentTarget();
    target_msg.header.stamp = time;
    target_msg.header.frame_id = target_frame_;
    // 如果跟踪状态有效，发布 TrackerInfo 消息
    if (target_msg.tracking) {
        // 获取当前活跃的追踪器
        auto current_tracker = tracker_manager_->getTracker(target_msg.id);
        if (current_tracker) {
            // 发布 TrackerInfo
            auto_aim_interfaces::msg::TrackerInfo info_msg;
            info_msg.position_diff = current_tracker->info_position_diff;
            info_msg.yaw_diff = current_tracker->info_yaw_diff;
            info_msg.position.x = current_tracker->measurement(0);
            info_msg.position.y = current_tracker->measurement(1);
            info_msg.position.z = current_tracker->measurement(2);
            info_msg.yaw = current_tracker->measurement(3);
            info_pub_->publish(info_msg);
        }
    }


    
    
    

    target_pub_->publish(target_msg);//发布target信息
    if(debug_){
        if(!armors_msg->image.data.empty() && armors_msg->image.header.stamp != last_img_time_){
            publishMarkers(target_msg);//发布可视化信息
            
            // 获取所有活跃的跟踪器ID
            std::vector<std::string> active_ids = tracker_manager_->getActiveTrackerIDs();
            
            // 创建一个副本用于绘制
            cv::Mat combined_image = cv_bridge::toCvCopy(armors_msg->image, "bgr8")->image;
            
            // 首先绘制当前活跃的主要目标
            if (target_msg.tracking) {
                publishImgAll(target_msg, combined_image, true); // true表示是主要目标，可以用不同颜色标识
            }
            
            // 然后绘制其他活跃的目标
            for (const auto& id : active_ids) {
                if (id != target_msg.id && id != "") {  // 排除当前已处理的ID和空ID
                    // 获取该ID的目标信息
                    auto id_target_msg = tracker_manager_->getIDTarget(id);
                    
                    // 在相同图像上绘制此ID的装甲板
                    publishImgAll(id_target_msg, combined_image, false); // false表示不是主要目标
                    
                }
            }
            
            // 添加通用信息（如相机中心、延迟等）
            cv::circle(combined_image, cam_center_, 5, cv::Scalar(0, 0, 255), 2);
            auto latency = (this->now() - rclcpp::Time(armors_msg->image.header.stamp)).seconds() * 1000; 
            std::stringstream text; 
            text << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
            cv::putText(combined_image, text.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            
            // 发布最终的组合图像
            auto processed_image_msg = cv_bridge::CvImage(armors_msg->image.header, "bgr8", combined_image).toImageMsg();
            tracker_img_pub_.publish(*processed_image_msg);
            
            last_img_time_ = armors_msg->image.header.stamp;
            
        }
    } 
}

void ArmorTrackerNode::publishImgAll(
    const auto_aim_interfaces::msg::Target & target_msg,
    cv::Mat & image,
    bool is_primary_target)
{
    if (!target_msg.tracking) {
        return;
    }
    
    // 获取相机内参矩阵
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 
        cam_info_.k[0], cam_info_.k[1], cam_info_.k[2],
        cam_info_.k[3], cam_info_.k[4], cam_info_.k[5],
        cam_info_.k[6], cam_info_.k[7], cam_info_.k[8]);

    // 获取相机畸变系数
    cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << 
        cam_info_.d[0], cam_info_.d[1], cam_info_.d[2], cam_info_.d[3], cam_info_.d[4]);
    
    // 选择颜色：主要目标使用绿色，其他目标使用不同颜色
    cv::Scalar color;
    if (is_primary_target) {
        color = cv::Scalar(0, 255, 0); // 绿色
    } else {
        // 基于ID生成不同的颜色
        int id_hash = 0;
        for (char c : target_msg.id) {
            id_hash += c;
        }
        // 使用简单的哈希生成不同颜色
        color = cv::Scalar(
            (id_hash * 90) % 255,  // B
            (id_hash * 50) % 255,  // G
            (id_hash * 140) % 255  // R
        );
    }
    
    auto current_tracker = tracker_manager_->getTracker(target_msg.id);
    if (!current_tracker) {
        return;
    }
    
    // 计算装甲板的位姿
    double yaw = target_msg.yaw, r1 = target_msg.radius_1, r2 = target_msg.radius_2;
    double xc = target_msg.position.x, yc = target_msg.position.y, za = target_msg.position.z;
    double dz = target_msg.dz;
    double pitch = target_msg.id == "outpost" ? -0.26 : 0.26;

    // 在图像右上角添加目标ID
    cv::putText(image, "ID: " + target_msg.id, 
                cv::Point(image.cols - 150, is_primary_target ? 30 : 60 + (std::hash<std::string>{}(target_msg.id) % 5) * 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

    bool is_current_pair = true;
    size_t a_n = target_msg.armors_num;
    double r = 0;
    for (size_t i = 0; i < a_n; i++) {
        double tmp_yaw = yaw + i * (2 * M_PI / a_n);
        // Only 4 armors has 2 radius and height
        double armor_z = za + (is_current_pair ? 0 : dz);
        if (a_n == 4) {
            r = is_current_pair ? r1 : r2;
            is_current_pair = !is_current_pair;
        } else {
            r = r1;
        }
        double armor_x = xc - r * cos(tmp_yaw);
        double armor_y = yc - r * sin(tmp_yaw);

        // 计算装甲板的四个角点
        std::vector<cv::Point3d> corners_world = {};
        double half_width = 0.5 * (current_tracker->tracked_armor.type == "small" ? 0.135 : 0.23);
        double half_height = 0.5 * 0.125;

        // 计算四个角点的世界坐标，考虑pitch
        double cos_pitch = cos(pitch);
        double sin_pitch = sin(pitch);

        corners_world.emplace_back(
            armor_x - half_width * sin(tmp_yaw) + half_height * cos(tmp_yaw) * sin_pitch,
            armor_y + half_width * cos(tmp_yaw) + half_height * sin(tmp_yaw) * sin_pitch,
            armor_z + half_height * cos_pitch);

        corners_world.emplace_back(
            armor_x + half_width * sin(tmp_yaw) + half_height * cos(tmp_yaw) * sin_pitch,
            armor_y - half_width * cos(tmp_yaw) + half_height * sin(tmp_yaw) * sin_pitch,
            armor_z + half_height * cos_pitch);

        corners_world.emplace_back(
            armor_x + half_width * sin(tmp_yaw) - half_height * cos(tmp_yaw) * sin_pitch,
            armor_y - half_width * cos(tmp_yaw) - half_height * sin(tmp_yaw) * sin_pitch,
            armor_z - half_height * cos_pitch);

        corners_world.emplace_back(
            armor_x - half_width * sin(tmp_yaw) - half_height * cos(tmp_yaw) * sin_pitch,
            armor_y + half_width * cos(tmp_yaw) - half_height * sin(tmp_yaw) * sin_pitch,
            armor_z - half_height * cos_pitch);

        // 将装甲板的角点从世界坐标系转换到相机坐标系
        std::vector<cv::Point2d> corners_image;
        cv::Mat rvec, tvec;
        // ROS 坐标系到 OpenCV 坐标系的变换矩阵
        cv::Mat ros_to_cv = (cv::Mat_<double>(3,3) <<
            0,-1, 0,
            0, 0,-1,
            1, 0, 0);
        try {
            geometry_msgs::msg::TransformStamped transform_stamped = tf2_buffer_->lookupTransform("camera_link", "odom", tf2::TimePointZero);
            tf2::Quaternion quat(
                transform_stamped.transform.rotation.x,
                transform_stamped.transform.rotation.y,
                transform_stamped.transform.rotation.z,
                transform_stamped.transform.rotation.w);

            tf2::Matrix3x3 mat(quat);
            // 原始旋转矩阵
            cv::Mat rotation_matrix = (cv::Mat_<double>(3, 3) <<
                mat[0][0], mat[0][1], mat[0][2],
                mat[1][0], mat[1][1], mat[1][2],
                mat[2][0], mat[2][1], mat[2][2]);
            // 调整后的旋转矩阵
            rotation_matrix = ros_to_cv * rotation_matrix;

            cv::Rodrigues(rotation_matrix, rvec);

            // 原始平移向量
            tvec = (cv::Mat_<double>(3, 1) <<
                transform_stamped.transform.translation.x,
                transform_stamped.transform.translation.y,
                transform_stamped.transform.translation.z);

            // 调整后的平移向量
            tvec = ros_to_cv * tvec;
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(rclcpp::get_logger("tracker_node"), "Could NOT transform: %s", ex.what());
            continue;
        }
        
        cv::projectPoints(corners_world, rvec, tvec, camera_matrix, dist_coeffs, corners_image);
        // 在图像上绘制四边形，使用不同颜色区分不同目标
        for (size_t j = 0; j < corners_image.size(); ++j) {
            cv::line(
                image, 
                corners_image[j], 
                corners_image[(j + 1) % corners_image.size()], 
                color, 
                is_primary_target ? 2 : 2  // 主要目标线条更粗
            ); 
        }
    }
}

void ArmorTrackerNode::publishMarkers(const auto_aim_interfaces::msg::Target & target_msg)
{
    position_marker_.header = target_msg.header;
    linear_v_marker_.header = target_msg.header;
    angular_v_marker_.header = target_msg.header;
    armor_marker_.header = target_msg.header;

    visualization_msgs::msg::MarkerArray marker_array;
    if (target_msg.tracking) {
        // 获取当前跟踪器
        auto current_tracker = tracker_manager_->getTracker(target_msg.id);
        if (!current_tracker) {
            // 如果找不到当前跟踪器，则清空所有标记
            position_marker_.action = visualization_msgs::msg::Marker::DELETE;
            linear_v_marker_.action = visualization_msgs::msg::Marker::DELETE;
            angular_v_marker_.action = visualization_msgs::msg::Marker::DELETE;
            armor_marker_.action = visualization_msgs::msg::Marker::DELETE;
            marker_array.markers.emplace_back(position_marker_);
            marker_array.markers.emplace_back(linear_v_marker_);
            marker_array.markers.emplace_back(angular_v_marker_);
            marker_array.markers.emplace_back(armor_marker_);
            marker_pub_->publish(marker_array);
            return;
        }

        double yaw = target_msg.yaw, r1 = target_msg.radius_1, r2 = target_msg.radius_2;
        double xc = target_msg.position.x, yc = target_msg.position.y, za = target_msg.position.z;
        double vx = target_msg.velocity.x, vy = target_msg.velocity.y, vz = target_msg.velocity.z;
        double dz = target_msg.dz;

        position_marker_.action = visualization_msgs::msg::Marker::ADD;
        position_marker_.pose.position.x = xc;
        position_marker_.pose.position.y = yc;
        position_marker_.pose.position.z = za + dz / 2;

        linear_v_marker_.action = visualization_msgs::msg::Marker::ADD;
        linear_v_marker_.points.clear();
        linear_v_marker_.points.emplace_back(position_marker_.pose.position);
        geometry_msgs::msg::Point arrow_end = position_marker_.pose.position;
        arrow_end.x += vx;
        arrow_end.y += vy;
        arrow_end.z += vz;
        linear_v_marker_.points.emplace_back(arrow_end);

        angular_v_marker_.action = visualization_msgs::msg::Marker::ADD;
        angular_v_marker_.points.clear();
        angular_v_marker_.points.emplace_back(position_marker_.pose.position);
        arrow_end = position_marker_.pose.position;
        arrow_end.z += target_msg.v_yaw / M_PI;
        angular_v_marker_.points.emplace_back(arrow_end);

        armor_marker_.action = visualization_msgs::msg::Marker::ADD;
        armor_marker_.scale.y = current_tracker->tracked_armor.type == "small" ? 0.135 : 0.23;
        bool is_current_pair = true;
        size_t a_n = target_msg.armors_num;
        geometry_msgs::msg::Point p_a;
        double r = 0;
        for (size_t i = 0; i < a_n; i++) {
            double tmp_yaw = yaw + i * (2 * M_PI / a_n);
            // Only 4 armors has 2 radius and height
            if (a_n == 4) {
                r = is_current_pair ? r1 : r2;
                p_a.z = za + (is_current_pair ? 0 : dz);
                is_current_pair = !is_current_pair;
            } else {
                r = r1;
                p_a.z = za;
            }
            p_a.x = xc - r * cos(tmp_yaw);
            p_a.y = yc - r * sin(tmp_yaw);

            armor_marker_.id = i;
            armor_marker_.pose.position = p_a;
            tf2::Quaternion q;
            q.setRPY(0, target_msg.id == "outpost" ? -0.26 : 0.26, tmp_yaw);
            armor_marker_.pose.orientation = tf2::toMsg(q);
            marker_array.markers.emplace_back(armor_marker_);
        }
    } else {
        position_marker_.action = visualization_msgs::msg::Marker::DELETE;
        linear_v_marker_.action = visualization_msgs::msg::Marker::DELETE;
        angular_v_marker_.action = visualization_msgs::msg::Marker::DELETE;

        armor_marker_.action = visualization_msgs::msg::Marker::DELETE;
        marker_array.markers.emplace_back(armor_marker_);
    }

    marker_array.markers.emplace_back(position_marker_);
    marker_array.markers.emplace_back(linear_v_marker_);
    marker_array.markers.emplace_back(angular_v_marker_);
    marker_pub_->publish(marker_array);
}
void ArmorTrackerNode::setModeCallback(
    const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
    std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response) {
    response->success = true;
  
    VisionMode mode = static_cast<VisionMode>(request->mode);
    std::string mode_name = visionModeToString(mode);
    if (mode_name == "UNKNOWN") {
        RCLCPP_ERROR(this->get_logger(), "Invalid mode: %d", request->mode);
        return;
    }
  
    mode_ = mode; 
    bool success = tracker_manager_->setMode(mode_);
  
    if(success) RCLCPP_INFO(this->get_logger(), "Set Car tracking Mode: %s", visionModeToString(mode).c_str());
    else RCLCPP_ERROR(this->get_logger(), "Failed to set Car tracking Mode: %s", visionModeToString(mode).c_str());
}
}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)