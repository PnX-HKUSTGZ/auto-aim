// Copyright 2022 Chen Jun
#include "armor_tracker/tracker_node.hpp"

// STD
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
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

    // Tracker
    double max_match_distance = this->declare_parameter("tracker.max_match_distance", 0.15);
    double max_match_yaw_diff = this->declare_parameter("tracker.max_match_yaw_diff", 1.0);
    tracker_ = std::make_unique<Tracker>(max_match_distance, max_match_yaw_diff);
    tracker_->tracking_thres = this->declare_parameter("tracker.tracking_thres", 5);
    lost_time_thres_ = this->declare_parameter("tracker.lost_time_thres", 0.3);

    // EKF
    // xa = x_armor, xc = x_robot_center
    // state: xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r
    // measurement: xa, ya, za, yaw
    // f - Process function状态转移函数预测值
    auto f = [this](const Eigen::VectorXd & x) {
        Eigen::VectorXd x_new = x;
        x_new(0) += x(1) * dt_;
        x_new(2) += x(3) * dt_;
        x_new(4) += x(5) * dt_;
        x_new(6) += x(7) * dt_;
        return x_new;
    };
    // J_f - Jacobian of process function
    auto j_f = [this](const Eigen::VectorXd &) {
        Eigen::MatrixXd f(9, 9);
        // clang-format off
        f <<  1,   dt_, 0,   0,   0,   0,   0,   0,   0,
                0,   1,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   1,   dt_, 0,   0,   0,   0,   0, 
                0,   0,   0,   1,   0,   0,   0,   0,   0,
                0,   0,   0,   0,   1,   dt_, 0,   0,   0,
                0,   0,   0,   0,   0,   1,   0,   0,   0,
                0,   0,   0,   0,   0,   0,   1,   dt_, 0,
                0,   0,   0,   0,   0,   0,   0,   1,   0,
                0,   0,   0,   0,   0,   0,   0,   0,   1;
        // clang-format on
        return f;
    };
    // h - Observation function 通过机器人中心位置计算装甲板的位置
    auto h = [](const Eigen::VectorXd & x) {
        Eigen::VectorXd z(4);
        double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
        z(0) = xc - r * cos(yaw);  // xa
        z(1) = yc - r * sin(yaw);  // ya
        z(2) = x(4);               // za
        z(3) = x(6);               // yaw
        return z;
    };
    // J_h - Jacobian of observation function
    auto j_h = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(4, 9);
        double yaw = x(6), r = x(8);
        // clang-format off
        //    xc   v_xc yc   v_yc za   v_za yaw         v_yaw r
        h <<  1,   0,   0,   0,   0,   0,   r*sin(yaw), 0,   -cos(yaw),
                0,   0,   1,   0,   0,   0,   -r*cos(yaw),0,   -sin(yaw),
                0,   0,   0,   0,   1,   0,   0,          0,   0,
                0,   0,   0,   0,   0,   0,   1,          0,   0;
        // clang-format on
        return h;
    };
    // update_Q - process noise covariance matrix
    s2qxyz_ = declare_parameter("ekf.sigma2_q_xyz", 20.0);
    s2qyaw_ = declare_parameter("ekf.sigma2_q_yaw", 100.0);
    s2qr_ = declare_parameter("ekf.sigma2_q_r", 800.0);
    auto u_q = [this]() {
        Eigen::MatrixXd q(9, 9);
        double t = dt_, x = s2qxyz_, y = s2qyaw_, r = s2qr_;
        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * x, q_vy_vy = pow(t, 2) * y;
        double q_r = pow(t, 4) / 4 * r;
        // clang-format off
        //    xc      v_xc    yc      v_yc    za      v_za    yaw     v_yaw   r
        q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,
                q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,
                0,      0,      q_x_x,  q_x_vx, 0,      0,      0,      0,      0,
                0,      0,      q_x_vx, q_vx_vx,0,      0,      0,      0,      0,
                0,      0,      0,      0,      q_x_x,  q_x_vx, 0,      0,      0,
                0,      0,      0,      0,      q_x_vx, q_vx_vx,0,      0,      0,
                0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy, 0,
                0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy,0,
                0,      0,      0,      0,      0,      0,      0,      0,      q_r;
        // clang-format on
        return q;
    };
    // update_R - measurement noise covariance matrix
    r_xyz_factor = declare_parameter("ekf.r_xyz_factor", 0.05);
    r_yaw = declare_parameter("ekf.r_yaw", 0.02);
    auto u_r = [this](const Eigen::VectorXd & z) {
        Eigen::DiagonalMatrix<double, 4> r;
        double x = r_xyz_factor;
        r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw;
        return r;
    };
    // P - error estimate covariance matrix
    Eigen::DiagonalMatrix<double, 9> p0;
    p0.setIdentity();
    tracker_->ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, p0};//初始化扩展kf滤波器

    // Reset tracker service，允许操作员重置追踪器
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    reset_tracker_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/tracker/reset", [this](
                            const std_srvs::srv::Trigger::Request::SharedPtr,
                            std_srvs::srv::Trigger::Response::SharedPtr response) {
            tracker_->tracker_state = Tracker::LOST;
            response->success = true;
            RCLCPP_INFO(this->get_logger(), "Tracker reset!");
            return;
        });

    //从相机的消息中提取内参和畸变参数
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
            cam_info_ = *camera_info;
            cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
            cam_info_sub_.reset();//取消订阅
        });

    // Subscriber with tf2 message_filter
    // tf2 relevant   
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    // Create the timer interface before call to waitForTransform,
    // to avoid a tf2_ros::CreateTimerInterfaceException exception
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface());

    tf2_buffer_->setCreateTimerInterface(timer_interface);//让tf2能够在需要时创建定时器
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);//接收坐标变换信息并更新buffer
    // subscriber and filter
    armors_sub_.subscribe(this, "/detector/armors", rmw_qos_profile_sensor_data);//订阅装甲板检测结果的主题
    target_frame_ = this->declare_parameter("target_frame", "odom");

    tf2_filter_ = std::make_shared<tf2_filter>(
        armors_sub_, *tf2_buffer_, target_frame_, 10, this->get_node_logging_interface(),
        this->get_node_clock_interface(), std::chrono::duration<int>(1));

    // Register a callback with tf2_ros::MessageFilter to be called when transforms are available
    tf2_filter_->registerCallback(&ArmorTrackerNode::armorsCallback, this);

    // Measurement publisher (for debug usage)
    info_pub_ = this->create_publisher<auto_aim_interfaces::msg::TrackerInfo>("/tracker/info", 10);

    // Publisher
    target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
        "/tracker/target", rclcpp::SensorDataQoS());
        
    // debug image publisher
    tracker_img_pub_ = image_transport::create_publisher(this, "/tracker/result_img");

    // Visualization Marker Publisher
    // See http://wiki.ros.org/rviz/DisplayTypes/Marker
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
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/tracker/marker", 10);
}

void ArmorTrackerNode::armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
//它的主要作用是将装甲板的位置从图像帧坐标转换到世界坐标系，过滤掉异常的装甲板，更新追踪器的状态，
//并根据追踪结果发布相关信息和可视化标记
{
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
                return abs(armor.pose.position.z) > 1.2 ||
                        Eigen::Vector2d(armor.pose.position.x, armor.pose.position.y).norm() >
                            max_armor_distance_;
            }),
        armors_msg->armors.end());

    // Init message
    auto_aim_interfaces::msg::TrackerInfo info_msg;
    auto_aim_interfaces::msg::Target target_msg;
    rclcpp::Time time = armors_msg->header.stamp;
    target_msg.header.stamp = time;
    target_msg.header.frame_id = target_frame_;

    // Update tracker
    if (tracker_->tracker_state == Tracker::LOST) {
        tracker_->init(armors_msg);
        target_msg.tracking = false;
    } else {
        dt_ = (time - last_time_).seconds();
        tracker_->lost_thres = static_cast<int>(lost_time_thres_ / dt_);
        tracker_->update(armors_msg);

        // Publish Info
        info_msg.position_diff = tracker_->info_position_diff;
        info_msg.yaw_diff = tracker_->info_yaw_diff;
        info_msg.position.x = tracker_->measurement(0);
        info_msg.position.y = tracker_->measurement(1);
        info_msg.position.z = tracker_->measurement(2);
        info_msg.yaw = tracker_->measurement(3);
        info_pub_->publish(info_msg);

        if (tracker_->tracker_state == Tracker::DETECTING) {
            target_msg.tracking = false;
        } else if (
            tracker_->tracker_state == Tracker::TRACKING ||
            tracker_->tracker_state == Tracker::TEMP_LOST) {
            target_msg.tracking = true;
            // Fill target message
            const auto & state = tracker_->target_state;
            target_msg.id = tracker_->tracked_id;
            target_msg.armors_num = static_cast<int>(tracker_->tracked_armors_num);
            target_msg.position.x = state(0);
            target_msg.velocity.x = state(1);
            target_msg.position.y = state(2);
            target_msg.velocity.y = state(3);
            target_msg.position.z = state(4);
            target_msg.velocity.z = state(5);
            target_msg.yaw = state(6);
            target_msg.v_yaw = state(7);
            target_msg.radius_1 = state(8);
            target_msg.radius_2 = tracker_->another_r;
            target_msg.dz = tracker_->dz;
        }
    }

    last_time_ = time;

    target_pub_->publish(target_msg);//发布target信息
    if(!armors_msg->image.data.empty() && armors_msg->image.header.stamp != last_img_time_){
        publishMarkers(target_msg);//发布可视化信息
        publishImg(target_msg, armors_msg->image); //发布图像信息
        last_img_time_ = armors_msg->image.header.stamp;
    } 
}
void ArmorTrackerNode::publishImg(
    const auto_aim_interfaces::msg::Target & target_msg,
    const sensor_msgs::msg::Image & image_msg)
{
    // 将 ROS 图像消息转换为 OpenCV 图像
    cv::Mat image = cv_bridge::toCvCopy(image_msg, "bgr8")->image;

    // 获取相机内参矩阵
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 
        cam_info_.k[0], cam_info_.k[1], cam_info_.k[2],
        cam_info_.k[3], cam_info_.k[4], cam_info_.k[5],
        cam_info_.k[6], cam_info_.k[7], cam_info_.k[8]);

    // 获取相机畸变系数
    cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << 
        cam_info_.d[0], cam_info_.d[1], cam_info_.d[2], cam_info_.d[3], cam_info_.d[4]);

    if(target_msg.tracking){
        // 计算装甲板的位姿
        double yaw = target_msg.yaw, r1 = target_msg.radius_1, r2 = target_msg.radius_2;
        double xc = target_msg.position.x, yc = target_msg.position.y, za = target_msg.position.z;
        double dz = target_msg.dz;
        double pitch = target_msg.id == "outpost" ? -0.26 : 0.26;

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
            double half_width = 0.5 * (tracker_->tracked_armor.type == "small" ? 0.135 : 0.23);
            double half_height = 0.5 * 0.125;

            // 计算四个角点的世界坐标，考虑pitch
            double cos_pitch = cos(pitch);
            double sin_pitch = sin(pitch);

            corners_world.emplace_back(
                armor_x - half_width * sin(tmp_yaw) + half_height * cos(tmp_yaw) * sin_pitch,
                armor_y - half_width * cos(tmp_yaw) - half_height * sin(tmp_yaw) * sin_pitch,
                armor_z + half_height * cos_pitch);

            corners_world.emplace_back(
                armor_x + half_width * sin(tmp_yaw) + half_height * cos(tmp_yaw) * sin_pitch,
                armor_y + half_width * cos(tmp_yaw) - half_height * sin(tmp_yaw) * sin_pitch,
                armor_z + half_height * cos_pitch);

            corners_world.emplace_back(
                armor_x + half_width * sin(tmp_yaw) - half_height * cos(tmp_yaw) * sin_pitch,
                armor_y + half_width * cos(tmp_yaw) + half_height * sin(tmp_yaw) * sin_pitch,
                armor_z - half_height * cos_pitch);

            corners_world.emplace_back(
                armor_x - half_width * sin(tmp_yaw) - half_height * cos(tmp_yaw) * sin_pitch,
                armor_y - half_width * cos(tmp_yaw) + half_height * sin(tmp_yaw) * sin_pitch,
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
                RCLCPP_WARN(this->get_logger(), "Could NOT transform: %s", ex.what());
                continue;
            }
            
            cv::projectPoints(corners_world, rvec, tvec, camera_matrix, dist_coeffs, corners_image);
            // 在图像上绘制四边形
            for (size_t j = 0; j < corners_image.size(); ++j) {
                cv::line(image, corners_image[j], corners_image[(j + 1) % corners_image.size()], cv::Scalar(0, 255, 0), 2); 
            }
        }
        // Draw camera center
        cv::circle(image, cam_center_, 5, cv::Scalar(0, 0, 255), 2);
        auto latency = (this->now() - rclcpp::Time(image_msg.header.stamp)).seconds() * 1000; 
        std::stringstream text; 
        text  << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
        cv::putText(image, text.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        // 将处理后的 OpenCV 图像转换回 ROS 图像消息
        auto processed_image_msg = cv_bridge::CvImage(image_msg.header, "bgr8", image).toImageMsg();

        // 发布处理后的图像
        tracker_img_pub_.publish(*processed_image_msg);
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
        armor_marker_.scale.y = tracker_->tracked_armor.type == "small" ? 0.135 : 0.23;
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

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)