// Copyright 2022 Chen Jun
#include "armor_tracker/tracker_node.hpp"

// STD
#include <auto_aim_interfaces/msg/detail/target__struct.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> 
#include <iostream>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <sstream>
#include <vector>

#include "armor_tracker/types.hpp"

namespace rm_auto_aim
{
ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions & options)
: Node("armor_tracker", options)
{
    RCLCPP_INFO(this->get_logger(), "Starting TrackerNode!");

    // 回调组：主相机独占、广角独立、发布独立
    main_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    wide_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    publish_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    service_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // Maximum allowable armor distance in the XOY plane
    max_armor_distance_ = this->declare_parameter("max_armor_distance", 10.0);
    debug_ = this->declare_parameter("debug", false);

    // 初始化高频发布定时器 (例如 100Hz = 10ms)
    publish_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(&ArmorTrackerNode::publishCallback, this), publish_cb_group_);

    // 初始化tracker管理器
    tracker_manager_ = std::make_unique<TrackerManager>(
        this->declare_parameter("tracker.max_match_distance", 0.15),
        this->declare_parameter("tracker.max_match_yaw_diff", 1.0),
        this->declare_parameter("tracker.max_translation_speed", 5.0),
        this->declare_parameter("tracker.tracking_thres", 5),  // 传递tracking_thres
        this->declare_parameter("tracker.outpost_match_count_threshold", 2),
        this->declare_parameter("tracker.outpost_z_lost_threshold", 0.1),
        this->declare_parameter("tracker.outpost_z_mismatch_reinit_rounds", 3),
        this->declare_parameter("tracker.lost_time_thres", 0.3),
        this->declare_parameter("tracker.miss_match_time_thres", 0.4), // 新增错匹配时间阈值
        this->declare_parameter("tracker.switch_cooldown", 1.0));

    // 设置评分权重参数
    tracker_manager_->setWeights(
        this->declare_parameter("score.w_distance", 0.5),
        this->declare_parameter("score.w_tracking", 0.5));

    // 初始化EKF的各个矩阵
    initializeEKF();

    // set_mode（单独回调组，防止被长耗时回调饿死）
    const std::string set_mode_name = "/armor_tracker/set_mode";  // 使用绝对名称，便于发现
    set_mode_srv_ = this->create_service<auto_aim_interfaces::srv::SetMode>(
        set_mode_name,
        std::bind(&ArmorTrackerNode::setModeCallback, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default,
        service_cb_group_);
    RCLCPP_INFO(this->get_logger(), "SetMode service advertised at %s", set_mode_name.c_str());
    // Camera info subscription
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
            cam_info_ = *camera_info;
            cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
            cam_info_sub_.reset();
        });
    cam_info_sub_wide = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/wide_cam/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_wide) {
            cam_info_wide = *camera_info_wide;
            cam_center_wide = cv::Point2f(camera_info_wide->k[2], camera_info_wide->k[5]);
            cam_info_sub_wide.reset();
        });

    // TF2 setup
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

    // Subscriptions
    target_frame_ = this->declare_parameter("target_frame", "odom");
    rclcpp::SubscriptionOptions main_options;
    main_options.callback_group = main_cb_group_;
    main_armors_sub_ = this->create_subscription<ArmorsMsg>(
        "/detector/armors", rclcpp::SensorDataQoS(),
        std::bind(&ArmorTrackerNode::mainArmorsCallback, this, std::placeholders::_1),
        main_options);

    rclcpp::SubscriptionOptions wide_options;
    wide_options.callback_group = wide_cb_group_;
    wide_armors_sub_ = this->create_subscription<ArmorsMsg>(
        "/detector_wide/armors", rclcpp::SensorDataQoS(),
        std::bind(&ArmorTrackerNode::wideArmorsCallback, this, std::placeholders::_1),
        wide_options);

    // Publishers
    info_pub_ = this->create_publisher<auto_aim_interfaces::msg::TrackerInfo>("/tracker/info", 10);
    info_pub_graph = this->create_publisher<std_msgs::msg::Float32MultiArray>("/tracker/info_graph", 10);
    target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
        "/tracker/target", rclcpp::SensorDataQoS());
    tracker_img_pub_ = image_transport::create_publisher(this, "/tracker/result_img");
    marker_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>("/tracker/marker", 10);

    // debug mode
    if (debug_) {
        // Visualization Marker setup
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
    // 状态向量为16维：
    // state: xc, v_xc, yc, v_yc, zc1, zc2, zc3, v_zc, v_yaw, r1, r2, r3, yaw1, yaw2, yaw3, r_outpost
    // measurement: xa, ya, za, yaw
    // f - Process function
    auto f = [this](const Eigen::VectorXd & x) {
        Eigen::VectorXd x_new = x;
        x_new(XC) += x(VXC) * dt_;
        x_new(YC) += x(VYC) * dt_;
        x_new(ZC1) += x(VZC) * dt_;
        x_new(ZC2) += x(VZC) * dt_;
        x_new(ZC3) += x(VZC) * dt_;
        x_new(YAW1) += x(VYAW) * dt_;
        x_new(YAW2) += x(VYAW) * dt_;
        x_new(YAW3) += x(VYAW) * dt_;
        x_new(R_OUTPOST) = 0.275;
        return x_new;
    };
    // J_f - Jacobian of process function
    auto j_f = [this](const Eigen::VectorXd &) {
        Eigen::MatrixXd f = Eigen::MatrixXd::Identity(16, 16);
        f(XC, VXC) = dt_;
        f(YC, VYC) = dt_;
        f(ZC1, VZC) = dt_;
        f(ZC2, VZC) = dt_;
        f(ZC3, VZC) = dt_;
        f(YAW1, VYAW) = dt_;
        f(YAW2, VYAW) = dt_;
        f(YAW3, VYAW) = dt_;
        return f;
    };
    // h1 - Observation function for armor 1
    auto h1 = [](const Eigen::VectorXd & x) {
        Eigen::VectorXd z(4);
        double xc = x(XC), yc = x(YC), yaw = x(YAW1), r = x(R1);
        z(0) = xc - r * cos(yaw);  // xa
        z(1) = yc - r * sin(yaw);  // ya
        z(2) = x(ZC1);             // za
        z(3) = x(YAW1);            // yaw
        return z;
    };
    // J_h1 - Jacobian of observation function for armor 1
    auto j_h1 = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(4, 16);
        h.setZero();
        double yaw = x(YAW1), r = x(R1);
        h(0, XC) = 1;
        h(0, R1) = -cos(yaw);
        h(0, YAW1) = r * sin(yaw);
        h(1, YC) = 1;
        h(1, R1) = -sin(yaw);
        h(1, YAW1) = -r * cos(yaw);
        h(2, ZC1) = 1;
        h(3, YAW1) = 1;
        return h;
    };
    // h2 - Observation function for armor 2
    auto h2 = [](const Eigen::VectorXd & x) {
        Eigen::VectorXd z(4);
        double xc = x(XC), yc = x(YC), yaw = x(YAW2), r = x(R2);
        z(0) = xc - r * cos(yaw);  // xa
        z(1) = yc - r * sin(yaw);  // ya
        z(2) = x(ZC2);             // za
        z(3) = x(YAW2);            // yaw
        return z;
    };
    // J_h2 - Jacobian of observation function for armor 2
    auto j_h2 = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(4, 16);
        h.setZero();
        double yaw = x(YAW2), r = x(R2);
        h(0, XC) = 1;
        h(0, R2) = -cos(yaw);
        h(0, YAW2) = r * sin(yaw);
        h(1, YC) = 1;
        h(1, R2) = -sin(yaw);
        h(1, YAW2) = -r * cos(yaw);
        h(2, ZC2) = 1;
        h(3, YAW2) = 1;
        return h;
    };
    // h3 - Observation function for armor 3
    auto h3 = [](const Eigen::VectorXd & x) {
        Eigen::VectorXd z(4);
        double xc = x(XC), yc = x(YC), yaw = x(YAW3), r = x(R3);
        z(0) = xc - r * cos(yaw);  // xa
        z(1) = yc - r * sin(yaw);  // ya
        z(2) = x(ZC3);             // za
        z(3) = x(YAW3);            // yaw
        return z;
    };
    // J_h3 - Jacobian of observation function for armor 3
    auto j_h3 = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(4, 16);
        h.setZero();
        double yaw = x(YAW3), r = x(R3);
        h(0, XC) = 1;
        h(0, R3) = -cos(yaw);
        h(0, YAW3) = r * sin(yaw);
        h(1, YC) = 1;
        h(1, R3) = -sin(yaw);
        h(1, YAW3) = -r * cos(yaw);
        h(2, ZC3) = 1;
        h(3, YAW3) = 1;
        return h;
    };
    // h_two - Observation function for 2 armors
    auto h_two = [](const Eigen::VectorXd & x) {
        Eigen::VectorXd z(10);  // 2装甲板(8维) + R1/R2(2维)
        double xc = x(XC), yc = x(YC), yaw1 = x(YAW1), yaw2 = x(YAW2), r1 = x(R1), r2 = x(R2);
        z(0) = xc - r1 * cos(yaw1);  // xa1
        z(1) = yc - r1 * sin(yaw1);  // ya1
        z(2) = x(ZC1);               // za1
        z(3) = yaw1;                 // yaw1
        z(4) = xc - r2 * cos(yaw2);  // xa2
        z(5) = yc - r2 * sin(yaw2);  // ya2
        z(6) = x(ZC2);               // za2
        z(7) = yaw2;                 // yaw2
        z(8) = r1;                   // r1
        z(9) = r2;                   // r2
        return z;
    };
    // J_h_two - Jacobian of observation function for 2 armors
    auto j_h_two = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(10, 16);
        h.setZero();
        double yaw1 = x(YAW1), yaw2 = x(YAW2);
        double r1 = x(R1), r2 = x(R2);
        h(0, XC) = 1;
        h(0, R1) = -cos(yaw1);
        h(0, YAW1) = r1 * sin(yaw1);
        h(1, YC) = 1;
        h(1, R1) = -sin(yaw1);
        h(1, YAW1) = -r1 * cos(yaw1);
        h(2, ZC1) = 1;
        h(3, YAW1) = 1;

        h(4, XC) = 1;
        h(4, R2) = -cos(yaw2);
        h(4, YAW2) = r2 * sin(yaw2);
        h(5, YC) = 1;
        h(5, R2) = -sin(yaw2);
        h(5, YAW2) = -r2 * cos(yaw2);
        h(6, ZC2) = 1;
        h(7, YAW2) = 1;

        h(8, R1) = 1;
        h(9, R2) = 1;
        return h;
    };
    // update_Q - process noise covariance matrix
    s2qxy_ = declare_parameter("ekf.sigma2_q_xy", 20.0);
    s2qz_ = declare_parameter("ekf.sigma2_q_z", 20.0);
    s2qyaw_ = declare_parameter("ekf.sigma2_q_yaw", 100.0);
    s2qr_ = declare_parameter("ekf.sigma2_q_r", 800.0);
    auto u_q = [this]() {
        Eigen::MatrixXd q(16, 16);
        q.setZero();
        double t = dt_, x = s2qxy_, z = s2qz_, y = s2qyaw_, r = s2qr_;
        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        double q_y_y = pow(t, 4) / 4 * x, q_y_vy = pow(t, 3) / 2 * x, q_vy_vy = pow(t, 2) * x;
        double q_z_z = pow(t, 4) / 4 * z, q_z_vz = pow(t, 3) / 2 * z, q_vz_vz = pow(t, 2) * z;
        double q_yaw_yaw = pow(t, 4) / 4 * y, q_yaw_vyaw = pow(t, 3) / 2 * y,
               q_vyaw_vyaw = pow(t, 2) * y;
        double q_r = pow(t, 4) / 4 * r;
        q(XC, XC) = q_x_x;
        q(XC, VXC) = q_x_vx;
        q(VXC, XC) = q_x_vx;
        q(VXC, VXC) = q_vx_vx;

        q(YC, YC) = q_y_y;
        q(YC, VYC) = q_y_vy;
        q(VYC, YC) = q_y_vy;
        q(VYC, VYC) = q_vy_vy;

        q(ZC1, ZC1) = q_z_z;
        q(ZC2, ZC2) = q_z_z;
        q(ZC3, ZC3) = q_z_z;
        q(ZC1, VZC) = q_z_vz;
        q(ZC2, VZC) = q_z_vz;
        q(ZC3, VZC) = q_z_vz;
        q(VZC, ZC1) = q_z_vz;
        q(VZC, ZC2) = q_z_vz;
        q(VZC, ZC3) = q_z_vz;
        q(VZC, VZC) = q_vz_vz;

        q(VYAW, VYAW) = q_vyaw_vyaw;
        q(VYAW, YAW1) = q_yaw_vyaw;
        q(VYAW, YAW2) = q_yaw_vyaw;
        q(VYAW, YAW3) = q_yaw_vyaw;
        q(YAW1, VYAW) = q_yaw_vyaw;
        q(YAW2, VYAW) = q_yaw_vyaw;
        q(YAW3, VYAW) = q_yaw_vyaw;
        q(YAW1, YAW1) = q_yaw_yaw;
        q(YAW2, YAW2) = q_yaw_yaw;
        q(YAW3, YAW3) = q_yaw_yaw;

        q(R1, R1) = q_r;
        q(R2, R2) = q_r;
        q(R3, R3) = q_r;
        q(R_OUTPOST, R_OUTPOST) = 1e-6;
        return q;
    };
    // update_R - measurement noise covariance matrix
    r_xyz_factor = declare_parameter("ekf.r_xyz_factor", 0.05);
    r_yaw = declare_parameter("ekf.r_yaw", 0.02);
    r_radius = declare_parameter("ekf.r_radius", 0.02);
    auto u_r = [this](const Eigen::VectorXd & z) {
        Eigen::DiagonalMatrix<double, 4> r;
        
        // 借鉴同济：计算装甲板在相机坐标系下的真实偏角 (当前观测Yaw - 装甲板相对相机的方位角)
        double center_yaw = std::atan2(z[1], z[0]); // atan2(y, x)
        double delta_angle = std::abs(z[3] - center_yaw);
        // 归一化到 [-pi, pi] 以求得最小视差
        while(delta_angle > M_PI) delta_angle -= 2.0 * M_PI;
        while(delta_angle < -M_PI) delta_angle += 2.0 * M_PI;
        delta_angle = std::abs(delta_angle);

        // 视差越大（越侧面），深度方向 (Z, x,y同受影响但主要放大深度不信任度) 噪声越高
        double depth_noise = std::abs(r_xyz_factor * z[0]) * (log(delta_angle + 1.0) + 1.0);

        r.diagonal() << depth_noise, depth_noise, abs(r_xyz_factor * z[2]), r_yaw;
        return r;
    };
    auto u_r_two = [this](const Eigen::VectorXd & z) {
        Eigen::DiagonalMatrix<double, 10> r;
        
        double center_yaw1 = std::atan2(z[1], z[0]); 
        double delta_angle1 = std::abs(z[3] - center_yaw1);
        while(delta_angle1 > M_PI) delta_angle1 -= 2.0 * M_PI;
        while(delta_angle1 < -M_PI) delta_angle1 += 2.0 * M_PI;
        delta_angle1 = std::abs(delta_angle1);
        double depth_noise1 = std::abs(r_xyz_factor * z[0]) * (log(delta_angle1 + 1.0) + 1.0);

        double center_yaw2 = std::atan2(z[5], z[4]); 
        double delta_angle2 = std::abs(z[7] - center_yaw2);
        while(delta_angle2 > M_PI) delta_angle2 -= 2.0 * M_PI;
        while(delta_angle2 < -M_PI) delta_angle2 += 2.0 * M_PI;
        delta_angle2 = std::abs(delta_angle2);
        double depth_noise2 = std::abs(r_xyz_factor * z[4]) * (log(delta_angle2 + 1.0) + 1.0);

        // 装甲板1/2的XYZ噪声 + YAW噪声 + R1/R2噪声
        r.diagonal() << depth_noise1, depth_noise1, abs(r_xyz_factor * z[2]), r_yaw, depth_noise2,
            depth_noise2, abs(r_xyz_factor * z[6]), r_yaw, r_radius, r_radius;
        return r;
    };
    // P - error estimate covariance matrix
    Eigen::DiagonalMatrix<double, 16> p0;
    p0.setIdentity();
    // 创建 EKF 并设置到 TrackerManager 中
    ExtendedKalmanFilter ekf{f, h1, h2, h3, h_two, j_f, j_h1, j_h2, j_h3, j_h_two, u_q, u_r,
                             u_r_two, p0};
    tracker_manager_->setEKFTemplate(ekf);
}



void ArmorTrackerNode::mainArmorsCallback(const ArmorsMsg::SharedPtr armors_msg)
{
    // Check if camera info is ready
    if (cam_info_.k[0] == 0.0) {
        return;
    }

    struct FlagGuard {
        explicit FlagGuard(std::atomic_bool & flag) : flag_(flag) { flag_.store(true, std::memory_order_release); }
        ~FlagGuard() { flag_.store(false, std::memory_order_release); }
        std::atomic_bool & flag_;
    } guard(main_processing_);

    main_seq_.fetch_add(1, std::memory_order_acq_rel);
    
    if (debug_) {
        // 计算帧率
        auto start_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(start_time.time_since_epoch());
        current_sec = duration.count() / 1000;
        if (last_sec != current_sec) {
            last_sec = current_sec;
            if (frame_count < 100) RCLCPP_INFO(get_logger(), "fps: %d", frame_count);
            frame_count = 0;
        }
        frame_count++;
    }

    processArmors(armors_msg, cam_info_, cam_center_, "camera_main_link", true);
}

void ArmorTrackerNode::wideArmorsCallback(const ArmorsMsg::SharedPtr armors_msg)
{
    if (main_processing_.load(std::memory_order_acquire)) {
        RCLCPP_DEBUG(this->get_logger(), "Skip wide frame: main processing busy");
        return;
    }
    // 确保任意两次 wide 之间至少有一次 main 回调
    uint64_t current_main_seq = main_seq_.load(std::memory_order_acquire);
    uint64_t last_seen = wide_seen_main_seq_.load(std::memory_order_acquire);
    if (current_main_seq == last_seen) {
        RCLCPP_DEBUG(this->get_logger(), "Skip wide frame: no new main frame since last wide");
        return;
    }
    wide_seen_main_seq_.store(current_main_seq, std::memory_order_release);
    // Check if camera info is ready
    if (cam_info_wide.k[0] == 0.0) {
        return;
    }
    processArmors(armors_msg, cam_info_wide, cam_center_wide, "camera_wide_link", false);
}

void ArmorTrackerNode::processArmors(
    const ArmorsMsg::SharedPtr & armors_msg, const sensor_msgs::msg::CameraInfo & cam_info,
    const cv::Point2f & cam_center, const std::string & frame_id, bool is_main_camera)
{

     // 手动执行坐标变换 (替代 tf2_filter)
    for (auto & armor : armors_msg->armors) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = armors_msg->header;
        ps.pose = armor.pose;
        try {
            // 首先尝试获取与图像时间戳严格匹配的 TF (不阻塞等待)
            geometry_msgs::msg::TransformStamped transform = tf2_buffer_->lookupTransform(
                target_frame_, 
                ps.header.frame_id, 
                ps.header.stamp);
            
            tf2::doTransform(ps, ps, transform);
            armor.pose = ps.pose;
        } catch (const tf2::ExtrapolationException & ex) {
            // 如果 TF 没赶上来（请求的时间在未来），不等待，直接取最新可用的 TF 
            try {
                // tf2::TimePointZero 表示直接抓取 TF 树里当前缓冲的最新的那个变换
                geometry_msgs::msg::TransformStamped fallback_transform = tf2_buffer_->lookupTransform(
                    target_frame_, 
                    ps.header.frame_id, 
                    tf2::TimePointZero);
                
                tf2::doTransform(ps, ps, fallback_transform);
                armor.pose = ps.pose;
                //RCLCPP_WARN(get_logger(), "can't use target_frame for TF, use newest instead: %s", ex.what());
            } catch (const tf2::TransformException & fallback_ex) {
                RCLCPP_ERROR(get_logger(), "Fallback transform failed: %s", fallback_ex.what());
                return;
            }
        } catch (const tf2::TransformException & ex) {
            // 捕获除了外推异常之外的其他严重 TF 错误
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
    // 更新/清理/选目标，主相机独占锁，广角非阻塞尝试
    {
        std::unique_lock<std::shared_mutex> lock(tracker_mutex_, std::defer_lock);
        if (is_main_camera) {
            lock.lock();
        } else {
            if (!lock.try_lock()) {
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *this->get_clock(), 2000,
                    "Skip wide armors: tracker busy with main stream");
                return;
            }
        }

        tracker_manager_->update(armors_msg, is_main_camera);
        tracker_manager_->cleanInactiveTrackers();
        tracker_manager_->selectBestTarget();
    }

    if (!is_main_camera) {
        RCLCPP_DEBUG(get_logger(), "Wide TrackerManager updated with %zu armors.", armors_msg->armors.size());
    } else {
        RCLCPP_DEBUG(get_logger(), "Main TrackerManager updated with %zu armors.", armors_msg->armors.size());
    }
    
    // 在主相机传来的图像上画图
    if (debug_ && is_main_camera) {
        // 如果跟踪状态有效，发布 TrackerInfo 消息
        // 注意：这里获取 target 信息仅用于绘图，不用于发布
        std::shared_lock<std::shared_mutex> read_lock(tracker_mutex_);
        auto current_target_id = tracker_manager_->getCurrentTargetID();

        if (!armors_msg->image.data.empty() && armors_msg->image.header.stamp != last_img_time_) {
            // ... (获取 active_ids 和 marker_array 的逻辑保持不变) ...
            // 获取所有活跃的跟踪器ID
            std::vector<std::string> active_ids = tracker_manager_->getActiveTrackerIDs();

            // 创建一个副本用于绘制
            cv::Mat combined_image = cv_bridge::toCvCopy(armors_msg->image, "bgr8")->image;
            
            auto_aim_interfaces::msg::Target target_msg;
            bool success = tracker_manager_->getIDTarget(current_target_id, target_msg);
            if (!success) {
                target_msg.tracking = false;
            }

            // 首先绘制当前活跃的主要目标
            if (target_msg.tracking) {
                drawImgAll(target_msg, combined_image, true, cam_info, frame_id);  // 传递相机参数
            }

            // ... (绘制其他目标, 传递相机参数) ...
            for (const auto & id : active_ids) {
                if (id != target_msg.id && !id.empty()) {
                    auto_aim_interfaces::msg::Target id_target_msg;
                    if (tracker_manager_->getIDTarget(id, id_target_msg)) {
                        drawImgAll(id_target_msg, combined_image, false, cam_info, frame_id);
                    }
                }
            }

            // 添加通用信息
            cv::circle(combined_image, cam_center, 5, cv::Scalar(0, 0, 255), 2);
            // ... (绘制延迟文本的逻辑保持不变) ...
            auto latency =
                (this->now() - rclcpp::Time(armors_msg->image.header.stamp)).seconds() * 1000;
            
            std::stringstream text;
            text << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
            cv::putText(
                combined_image, text.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 0), 2);

            // 发布最终的组合图像
            auto processed_image_msg =
                cv_bridge::CvImage(armors_msg->image.header, "bgr8", combined_image).toImageMsg();
            tracker_img_pub_.publish(*processed_image_msg);

                    last_img_time_ = armors_msg->image.header.stamp;
        }
    }
}

void ArmorTrackerNode::publishCallback()
{
    std::shared_lock<std::shared_mutex> lock(tracker_mutex_);

    // 获取并发布目标
    auto current_target_id = tracker_manager_->getCurrentTargetID();
    auto_aim_interfaces::msg::Target target_msg;
        target_msg.header.frame_id = target_frame_;
    bool success = tracker_manager_->getIDTarget(current_target_id, target_msg);

    if (!success) {
        target_msg.tracking = false;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to get target with ID: %s", current_target_id.c_str());
    }
    else{
        target_pub_->publish(target_msg);
    }
    


    // 发布 TrackerInfo (调试用)
    if (debug_ && target_msg.tracking) {
        auto current_tracker = tracker_manager_->getTracker(target_msg.id);
        if (current_tracker) {
            auto_aim_interfaces::msg::TrackerInfo info_msg;
            info_msg.position_diff = current_tracker->info_position_diff;
            info_msg.yaw_diff = current_tracker->info_yaw_diff;
            info_msg.position.x = current_tracker->measurement(0);
            info_msg.position.y = current_tracker->measurement(1);
            info_msg.position.z = current_tracker->measurement(2);
            info_msg.yaw = current_tracker->measurement(3);
            info_pub_->publish(info_msg);

            std_msgs::msg::Float32MultiArray graph_msg;
            graph_msg.data.resize(6);
            graph_msg.data[0] = static_cast<float>(info_msg.position_diff);
            graph_msg.data[1] = static_cast<float>(info_msg.yaw_diff);
            graph_msg.data[2] = static_cast<float>(info_msg.position.x);
            graph_msg.data[3] = static_cast<float>(info_msg.position.y);
            graph_msg.data[4] = static_cast<float>(info_msg.position.z);
            graph_msg.data[5] = static_cast<float>(info_msg.yaw);

            info_pub_graph->publish(graph_msg);
        }
    }

    if (debug_) {
        visualization_msgs::msg::MarkerArray marker_array;
        drawMarkers(target_msg, marker_array);
        const auto active_ids = tracker_manager_->getActiveTrackerIDs();
        for (const auto & id : active_ids) {
            if (id != target_msg.id && !id.empty()) {
                auto_aim_interfaces::msg::Target id_target_msg;
                if (tracker_manager_->getIDTarget(id, id_target_msg)) {
                    id_target_msg.header.stamp = this->now();
                    id_target_msg.header.frame_id = target_frame_;
                    drawMarkers(id_target_msg, marker_array);
                }
            }
        }
        marker_pub_->publish(marker_array);
    }
}

void ArmorTrackerNode::drawImgAll(
    const auto_aim_interfaces::msg::Target & target_msg, cv::Mat & image, bool is_primary_target,
    const sensor_msgs::msg::CameraInfo & cam_info, const std::string & img_frame_id)
{
    if (!target_msg.tracking) {
        return;
    }

    // 获取相机内参矩阵
    cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << cam_info.k[0], cam_info.k[1], cam_info.k[2], cam_info.k[3],
         cam_info.k[4], cam_info.k[5], cam_info.k[6], cam_info.k[7], cam_info.k[8]);

    // 获取相机畸变系数
    cv::Mat dist_coeffs =
        (cv::Mat_<double>(1, 5) << cam_info.d[0], cam_info.d[1], cam_info.d[2], cam_info.d[3],
         cam_info.d[4]);

    // 选择颜色：主要目标使用绿色，其他目标使用不同颜色
    cv::Scalar color;
    if (is_primary_target) {
        color = cv::Scalar(0, 255, 0);  // 绿色
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
    cv::putText(
        image, "ID: " + target_msg.id,
        cv::Point(
            image.cols - 150,
            is_primary_target ? 30 : 60 + (std::hash<std::string>{}(target_msg.id) % 5) * 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

    bool is_current_pair = true;
    size_t a_n = target_msg.armors_num;
    double r = 0;
    // 扩展：支持3块装甲板绘制
    for (size_t i = 0; i < a_n; i++) {
        double tmp_yaw = yaw + i * (2 * M_PI / a_n);
        // 前哨站3块板Z坐标适配
        double armor_z = za;
        if (a_n == 4) {
            // Only 4 armors has 2 radius and height
            armor_z = za + (is_current_pair ? 0 : dz);
            r = is_current_pair ? r1 : r2;
            is_current_pair = !is_current_pair;
        } else if (target_msg.id == "outpost") {
            // 前哨站3块板直接使用状态中的 ZC1/ZC2/ZC3
            armor_z =
                (i == 0)
                    ? current_tracker->target_state(ZC1)
                    : ((i == 1) ? current_tracker->target_state(ZC2)
                                : current_tracker->target_state(ZC3));
            r = r1;  // 前哨站半径固定
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
        cv::Mat ros_to_cv = (cv::Mat_<double>(3, 3) << 0, -1, 0, 0, 0, -1, 1, 0, 0);
        try {
            geometry_msgs::msg::TransformStamped transform_stamped =
                tf2_buffer_->lookupTransform(img_frame_id, "odom", tf2::TimePointZero);
            
            tf2::Quaternion quat(
                transform_stamped.transform.rotation.x, transform_stamped.transform.rotation.y,
                transform_stamped.transform.rotation.z, transform_stamped.transform.rotation.w);

            tf2::Matrix3x3 mat(quat);
            // 原始旋转矩阵
            cv::Mat rotation_matrix =
                (cv::Mat_<double>(3, 3) << mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1],
                 mat[1][2], mat[2][0], mat[2][1], mat[2][2]);
            // 调整后的旋转矩阵
            rotation_matrix = ros_to_cv * rotation_matrix;

            cv::Rodrigues(rotation_matrix, rvec);

            // 原始平移向量
            tvec =
                (cv::Mat_<double>(3, 1) << transform_stamped.transform.translation.x,
                 transform_stamped.transform.translation.y,
                 transform_stamped.transform.translation.z);
                
            // 调整后的平移向量
            tvec = ros_to_cv * tvec;
        } catch (tf2::TransformException & ex) {
            RCLCPP_WARN(rclcpp::get_logger("tracker_node"), "Could NOT transform: %s", ex.what());
            continue;
        }

        cv::projectPoints(corners_world, rvec, tvec, camera_matrix, dist_coeffs, corners_image);
        // 在图像上绘制四边形，使用不同颜色区分不同目标
        bool is_wide_result = (img_frame_id == "camera_wide_link");
        for (size_t j = 0; j < corners_image.size(); ++j) {
            cv::line(
                image, corners_image[j], corners_image[(j + 1) % corners_image.size()], color,
                is_primary_target ? 2 : 2  // 主要目标线条更粗
            );
            if(!is_wide_result){
                cv::circle(image, corners_image[j], 5, color, -1);
            }
        }
    }
}

void ArmorTrackerNode::drawMarkers(
    const auto_aim_interfaces::msg::Target & target_msg,
    visualization_msgs::msg::MarkerArray & marker_array)
{
    position_marker_.header = target_msg.header;
    linear_v_marker_.header = target_msg.header;
    angular_v_marker_.header = target_msg.header;
    armor_marker_.header = target_msg.header;
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
            } else if (target_msg.id == "outpost") {
                r = r1;
                p_a.z =
                    (i == 0)
                        ? current_tracker->target_state(ZC1)
                        : ((i == 1) ? current_tracker->target_state(ZC2)
                                    : current_tracker->target_state(ZC3));
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
}
void ArmorTrackerNode::setModeCallback(
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

    mode_ = mode;
    bool success = tracker_manager_->setMode(mode_);

    if (success)
        RCLCPP_INFO(
            this->get_logger(), "Set Car tracking Mode: %s", visionModeToString(mode).c_str());
    else
        RCLCPP_ERROR(
            this->get_logger(), "Failed to set Car tracking Mode: %s",
            visionModeToString(mode).c_str());
}
}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)
