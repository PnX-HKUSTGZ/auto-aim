// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <tf2_ros/create_timer_ros.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/core.hpp>
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

#include "armor_detector/armor.hpp"
#include "armor_detector/detector_node.hpp"

namespace rm_auto_aim
{
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions & options)
: Node("armor_detector", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting DetectorNode!");

  // Detector
  detector_ = initDetector();

  // Armors Publisher
  armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>(
    "/detector/armors", rclcpp::SensorDataQoS());

  // Visualization Marker Publisher
  // See http://wiki.ros.org/rviz/DisplayTypes/Marker
  armor_marker_.ns = "armors";
  armor_marker_.action = visualization_msgs::msg::Marker::ADD;
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.05;
  armor_marker_.scale.z = 0.125;
  armor_marker_.color.a = 1.0;
  armor_marker_.color.g = 0.5;
  armor_marker_.color.b = 1.0;
  armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  text_marker_.ns = "classification";
  text_marker_.action = visualization_msgs::msg::Marker::ADD;
  text_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text_marker_.scale.z = 0.1;
  text_marker_.color.a = 1.0;
  text_marker_.color.r = 1.0;
  text_marker_.color.g = 1.0;
  text_marker_.color.b = 1.0;
  text_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  marker_pub_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("/detector/marker", 10);

  // Debug Publishers
  debug_ = this->declare_parameter("debug", false);
  if (debug_) {
    createDebugPublishers();
  }

  // Debug param change moniter
  debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
  debug_cb_handle_ =
    debug_param_sub_->add_parameter_callback("debug", [this](const rclcpp::Parameter & p) {
      debug_ = p.as_bool();
      debug_ ? createDebugPublishers() : destroyDebugPublishers();
    });
  //从相机的消息中进一步提取信息
  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera_info", rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
      cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
      cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
      pnp_solver_ = std::make_unique<PnPSolver>(camera_info->k, camera_info->d);
      ba_solver_ = std::make_unique<BaSolver>(camera_info->k, camera_info->d);
      cam_info_sub_.reset();//取消订阅
    });
  //收到图像信息后回调imageCallback函数
  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/image_raw", rclcpp::SensorDataQoS(),
    std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1));
  // set_mode
  set_mode_srv_ = this->create_service<auto_aim_interfaces::srv::SetMode>(
    "armor_detector/set_mode",
    std::bind(
      &ArmorDetectorNode::setModeCallback, this, std::placeholders::_1, std::placeholders::_2));

  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
}

void ArmorDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
  if(!enable_){
    return;
  }
  if (debug_)armors_msg_.image = *img_msg;
  cv::Mat img; 
  auto armors = detectArmors(img_msg, img);

  // Get the transform from odom to gimbal
  try {
    auto latest_tf = tf2_buffer_->lookupTransform(img_msg->header.frame_id, "odom", tf2::TimePointZero);
    rclcpp::Time target_time = img_msg->header.stamp;
    rclcpp::Time latest_time = latest_tf.header.stamp;
    // 比较时间戳
    geometry_msgs::msg::TransformStamped odom_to_camera_tf;
    if (target_time > latest_time) {
        // 使用最新变换
        odom_to_camera_tf = latest_tf;
    } else {
        // 查找指定时间的变换
        odom_to_camera_tf = tf2_buffer_->lookupTransform(
            img_msg->header.frame_id, "odom", target_time,
            rclcpp::Duration::from_nanoseconds(1000000));
    }
    auto msg_q = odom_to_camera_tf.transform.rotation;
    tf2::Quaternion tf_q;
    tf2::fromMsg(msg_q, tf_q);
    tf2::Matrix3x3 tf2_matrix = tf2::Matrix3x3(tf_q);
    r_odom_to_camera << tf2_matrix.getRow(0)[0], tf2_matrix.getRow(0)[1],
        tf2_matrix.getRow(0)[2], tf2_matrix.getRow(1)[0],
        tf2_matrix.getRow(1)[1], tf2_matrix.getRow(1)[2],
        tf2_matrix.getRow(2)[0], tf2_matrix.getRow(2)[1],
        tf2_matrix.getRow(2)[2];
    t_odom_to_camera = Eigen::Vector3d(
        odom_to_camera_tf.transform.translation.x,
        odom_to_camera_tf.transform.translation.y,
        odom_to_camera_tf.transform.translation.z);
  } catch (...) {
    RCLCPP_ERROR(this->get_logger(), "Something Wrong when lookUpTransform");
    return;
  }

  if (pnp_solver_ != nullptr) {
    armors_msg_.header = armor_marker_.header = text_marker_.header = img_msg->header;
    armors_msg_.armors.clear();
    marker_array_.markers.clear();
    armor_marker_.id = 0;
    text_marker_.id = 0;

    std::map<std::string, std::pair<int, std::vector<int>>> armor_num_map;
    for(size_t i = 0; i < armors.size(); i++){
      std::vector<cv::Mat> rvecs, tvecs;
      bool success = pnp_solver_->solvePnP(armors[i], rvecs, tvecs);//获得两个矩阵
      if (success) {
        // choose the best result
        chooseBestPose(armors[i], rvecs, tvecs);
        armor_num_map[armors[i].number].first++;
        armor_num_map[armors[i].number].second.push_back(i);
      } else {
        RCLCPP_WARN(this->get_logger(), "PnP failed!");
        armors.erase(armors.begin() + i);
        i--;
      }
    }
    std::vector<int> erase_index;
    for(auto & armor_num : armor_num_map){
      if(armor_num.second.first > 2){
        RCLCPP_ERROR(this->get_logger(), "More than 2 armors detected!");
        erase_index = armor_num.second.second;
      }
      else if (armor_num.second.first == 2){
        bool success = ba_solver_->fixTwoArmors(armors[armor_num.second.second[0]], armors[armor_num.second.second[1]], r_odom_to_camera);
        if(!success){
          RCLCPP_ERROR(this->get_logger(), "Fix two armors failed!");
          erase_index.push_back(armor_num.second.second[0]);
          erase_index.push_back(armor_num.second.second[1]);
        }
        else {
          armors[armor_num.second.second[0]].setCameraArmor(r_odom_to_camera, t_odom_to_camera); 
          armors[armor_num.second.second[1]].setCameraArmor(r_odom_to_camera, t_odom_to_camera);
        }
      }
    }
    std::sort(erase_index.begin(), erase_index.end(), std::greater<int>());
    for(auto & index : erase_index){
      armors.erase(armors.begin() + index);
    }
    auto_aim_interfaces::msg::Armor armor_msg;
    for (auto & armor : armors) {
      // Fill basic info
      armor_msg.type = ARMOR_TYPE_STR[static_cast<int>(armor.type)];
      armor_msg.number = armor.number;

      Eigen::Quaterniond eigen_quat(armor.r_camera_armor);
      tf2::Quaternion tf2_q(
        eigen_quat.x(), eigen_quat.y(), eigen_quat.z(), eigen_quat.w());
      armor_msg.pose.orientation = tf2::toMsg(tf2_q);

      // Fill pose
      armor_msg.pose.position.x = armor.t_camera_armor(0);
      armor_msg.pose.position.y = armor.t_camera_armor(1);
      armor_msg.pose.position.z = armor.t_camera_armor(2);

      // Fill the distance to image center
      armor_msg.distance_to_image_center = pnp_solver_->calculateDistanceToCenter(armor.center);
      // Fill the classification result
      armors_msg_.armors.emplace_back(armor_msg);

      // Fill the markers
      if(debug_){
        armor_marker_.id++;
        armor_marker_.scale.y = armor.type == ArmorType::SMALL ? 0.135 : 0.23;
        armor_marker_.pose = armor_msg.pose;
        text_marker_.id++;
        text_marker_.pose.position = armor_msg.pose.position;
        text_marker_.pose.position.y -= 0.1;
        text_marker_.text = armor.classfication_result;
        marker_array_.markers.emplace_back(armor_marker_);
        marker_array_.markers.emplace_back(text_marker_);
      }
    }
    // Publishing detected armors
    armors_pub_->publish(armors_msg_);

    if(debug_){
      // draw results
      drawResults(img_msg, img, armors);
      // Publishing marker
      publishMarkers();
    }
  }
}
void ArmorDetectorNode::chooseBestPose(Armor & armor, const std::vector<cv::Mat> & rvecs, const std::vector<cv::Mat> & tvecs){
  // choose the best result
  // rvec to 3x3 rotation matrix
  cv::Mat rotation_matrix;
  cv::Rodrigues(rvecs[0], rotation_matrix);//将旋转向量转换为旋转矩阵

  // rotation matrix to quaternion
  Eigen::Matrix3d rotation_matrix_eigen;
  cv::cv2eigen(rotation_matrix, rotation_matrix_eigen);
  
  Eigen::Quaterniond q_gimbal_camera(
      Eigen::AngleAxisd(-CV_PI / 2, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(-CV_PI / 2, Eigen::Vector3d::UnitX())
  );
  Eigen::Quaterniond q_rotation(rotation_matrix_eigen);
  q_rotation = q_gimbal_camera * q_rotation;
  // get yaw
  Eigen::Vector3d rpy = q_rotation.toRotationMatrix().eulerAngles(0, 1, 2);
  //限制在-pi到pi之间
  rpy(0) = std::fmod(rpy(0) + M_PI, M_PI) > M_PI / 2 ? std::fmod(rpy(0) + M_PI, M_PI) - M_PI : std::fmod(rpy(0) + M_PI, M_PI);
  rpy(1) = std::fmod(rpy(1) + M_PI, M_PI) > M_PI / 2 ? std::fmod(rpy(1) + M_PI, M_PI) - M_PI : std::fmod(rpy(1) + M_PI, M_PI);
  rpy(2) = std::fmod(rpy(2) + M_PI, M_PI) > M_PI / 2 ? std::fmod(rpy(2) + M_PI, M_PI) - M_PI : std::fmod(rpy(2) + M_PI, M_PI);
  
  if(armor.number == "outpost") armor.sign = -armor.sign;
  // armor.sign 为0则为右侧装甲板，为1则为左侧装甲板
  if(!armor.sign && rpy(2) < 0){
    rpy = Eigen::Vector3d(rpy(0), rpy(1), -rpy(2));
  }
  else if(armor.sign && rpy(2) > 0){
    rpy = Eigen::Vector3d(rpy(0), rpy(1), -rpy(2));
  }
  q_rotation = Eigen::Quaterniond(Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX())) *
               Eigen::Quaterniond(Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY())) *
               Eigen::Quaterniond(Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ()));
  q_rotation = q_gimbal_camera.conjugate() * q_rotation;
  Eigen::Matrix3d eigen_mat = q_rotation.toRotationMatrix();
  armor.t_odom_armor = r_odom_to_camera.inverse() * 
                 (Eigen::Vector3d(tvecs[0].at<double>(0), 
                                 tvecs[0].at<double>(1), 
                                 tvecs[0].at<double>(2)) - t_odom_to_camera);
  armor.r_odom_armor = r_odom_to_camera.inverse() * eigen_mat;
  armor.setCameraArmor(r_odom_to_camera, t_odom_to_camera); 
  if(rpy(0) < 0.26){
    ba_solver_->solveBa(armor, r_odom_to_camera, t_odom_to_camera);
  }
  armor.setCameraArmor(r_odom_to_camera, t_odom_to_camera); 
}
std::unique_ptr<Detector> ArmorDetectorNode::initDetector()
{

  rcl_interfaces::msg::ParameterDescriptor param_desc;//用于描述填充参数
  //设置二值化参数
  param_desc.integer_range.resize(1);
  param_desc.integer_range[0].step = 1;
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 255;
  int binary_thres = declare_parameter("binary_thres", 80, param_desc);
  //设置需要探测的颜色
  param_desc.description = "0-RED, 1-BLUE";
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 1;
  auto detect_color = declare_parameter("detect_color", RED, param_desc);
  //填充light和armor类所需要的参数
  Detector::LightParams l_params = {
    
    .min_ratio = declare_parameter("light.min_ratio", 0.1),
    .max_ratio = declare_parameter("light.max_ratio", 0.4),
    .max_angle = declare_parameter("light.max_angle", 40.0)};
  
  Detector::ArmorParams a_params = {
    .min_light_ratio = declare_parameter("armor.min_light_ratio", 0.7),
    .min_small_center_distance = declare_parameter("armor.min_small_center_distance", 0.8),
    .max_small_center_distance = declare_parameter("armor.max_small_center_distance", 3.2),
    .min_large_center_distance = declare_parameter("armor.min_large_center_distance", 3.2),
    .max_large_center_distance = declare_parameter("armor.max_large_center_distance", 5.5),
    .max_angle = declare_parameter("armor.max_angle", 35.0)};
  //
  auto detector = std::make_unique<Detector>(binary_thres, detect_color, l_params, a_params);

  // Init classifier
  auto pkg_path = ament_index_cpp::get_package_share_directory("armor_detector");
  auto model_path = pkg_path + "/model/mlp.onnx";
  auto label_path = pkg_path + "/model/label.txt";
  double threshold = this->declare_parameter("classifier_threshold", 0.7);
  std::vector<std::string> ignore_classes =
    this->declare_parameter("ignore_classes", std::vector<std::string>{"negative"});////这里的this并非必须
    
  detector->classifier =
    std::make_unique<NumberClassifier>(model_path, label_path, threshold, ignore_classes);
  


  return detector;
}

std::vector<Armor> ArmorDetectorNode::detectArmors(
  const sensor_msgs::msg::Image::ConstSharedPtr & img_msg, cv::Mat & img)
{
  // Convert ROS img to cv::Mat
  img = cv_bridge::toCvShare(img_msg, "rgb8")->image;
  // Update params
  detector_->binary_thres = get_parameter("binary_thres").as_int();
  detector_->detect_color = get_parameter("detect_color").as_int();
  detector_->classifier->threshold = get_parameter("classifier_threshold").as_double();
  
  //   //动态调参
  // param_callback_handle_ = this->add_on_set_parameters_callback(
  //   std::bind(&ArmorDetectorNode::onParameterChanged, this, std::placeholders::_1)
  // );

  auto armors = detector_->detect(img);
  for(auto & armor : armors){
    lcc.correctCorners(armor, detector_->gray_img);
  }
  //计算延迟
  auto final_time = this->now();
  auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
  RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");

  // Publish debug info
  if (debug_) {
    binary_img_pub_.publish(
      cv_bridge::CvImage(img_msg->header, "mono8", detector_->binary_img).toImageMsg());

    // Sort lights and armors data by x coordinate
    std::sort(
      detector_->debug_lights.data.begin(), detector_->debug_lights.data.end(),
      [](const auto & l1, const auto & l2) { return l1.center_x < l2.center_x; });
    std::sort(
      detector_->debug_armors.data.begin(), detector_->debug_armors.data.end(),
      [](const auto & a1, const auto & a2) { return a1.center_x < a2.center_x; });

    lights_data_pub_->publish(detector_->debug_lights);
    armors_data_pub_->publish(detector_->debug_armors);

    if (!armors.empty()) {
      auto all_num_img = detector_->getAllNumbersImage();
      number_img_pub_.publish(
        *cv_bridge::CvImage(img_msg->header, "mono8", all_num_img).toImageMsg());
    }
  }
  return armors;
}
void ArmorDetectorNode::drawResults(
  const sensor_msgs::msg::Image::ConstSharedPtr & img_msg, cv::Mat & img, const std::vector<Armor> & armors){
  //计算延迟
  auto final_time = this->now();
  auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
  RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");
  if(!debug_){
    return;
  }
  detector_->drawResults(img);
  // Show yaw, pitch, roll
  for (const auto & armor : armors) {
    Eigen::Vector3d rpy = armor.r_odom_armor.eulerAngles(0, 1, 2); //提取欧拉角
    double distance = armor.t_camera_armor.norm();
    cv::putText(
      img, "y: " + std::to_string(int(rpy(2) / CV_PI * 180)) + " deg", cv::Point(armor.left_light.bottom.x, armor.left_light.bottom.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(0, 255, 255), 2);
    cv::putText(
      img, "d: " + std::to_string(distance) + "m" , cv::Point(armor.left_light.bottom.x, armor.left_light.bottom.y + 60), cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(0, 255, 255), 2);
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
// //动态调参
//  rcl_interfaces::msg::SetParametersResult ArmorDetectorNode::onParameterChanged(const std::vector<rclcpp::Parameter> &parameters)
//  {
//   rcl_interfaces::msg::SetParametersResult result;
//   result.successful = true;

//   for(const auto & param : parameters)
//   {
//     if(param.get_name() == "binary_thres")
//     {
//       detector_->binary_thres = param.as_int();
//     }
//     else if(param.get_name() == "light.min_ratio")
//     {
//       detector_->l.min_ratio = param.as_double();
//     }
//     else if(param.get_name() == "light.max_ratio")
//     {
//       detector_->l.max_ratio = param.as_double();
//     }
//     else if(param.get_name() == "light.max_angle")
//     {
//       detector_->l.max_angle = param.as_double();
//     }

//     else if(param.get_name() == "armor.min_light_ratio")
//     {
//       detector_->a.min_light_ratio = param.as_double();
//     }
//     else if(param.get_name() == "armor.min_small_center_distance")
//     {
//       detector_->a.min_small_center_distance = param.as_double();
//     }
//     else if(param.get_name() == "armor.max_small_center_distance")
//     {
//       detector_->a.max_small_center_distance = param.as_double();
//     }
//     else if(param.get_name() == "armor.min_large_center_distance")
//     {
//       detector_->a.min_large_center_distance = param.as_double();
//     }
//     else if(param.get_name() == "armor.max_large_center_distance")
//     {
//       detector_->a.max_large_center_distance = param.as_double();
//     }
//     else if(param.get_name() == "armor.max_angle")
//     {
//       detector_->a.max_angle = param.as_double();
//     }
    
//     else if(param.get_name() == "classifier_threshold")
//     {
//       detector_->classifier->threshold = param.as_double();
//     }
//     else
//     {
//       result.successful = false;
//       result.reason = "Unknown parameter: " + param.get_name();
//     }
//   }
//   return result;
//  }

void ArmorDetectorNode::createDebugPublishers()
{
  lights_data_pub_ =
    this->create_publisher<auto_aim_interfaces::msg::DebugLights>("/detector/debug_lights", 10);
  armors_data_pub_ =
    this->create_publisher<auto_aim_interfaces::msg::DebugArmors>("/detector/debug_armors", 10);

  binary_img_pub_ = image_transport::create_publisher(this, "/detector/binary_img");
  number_img_pub_ = image_transport::create_publisher(this, "/detector/number_img");
  result_img_pub_ = image_transport::create_publisher(this, "/detector/result_img");
}

void ArmorDetectorNode::destroyDebugPublishers()
{
  lights_data_pub_.reset();
  armors_data_pub_.reset();

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

void ArmorDetectorNode::setModeCallback(
  const std::shared_ptr<auto_aim_interfaces::srv::SetMode::Request> request,
  std::shared_ptr<auto_aim_interfaces::srv::SetMode::Response> response) {
  response->success = true;

  VisionMode mode = static_cast<VisionMode>(request->mode);
  std::string mode_name = visionModeToString(mode);
  if (mode_name == "UNKNOWN") {
    RCLCPP_ERROR(this->get_logger(), "Invalid mode: %d", request->mode);
    return;
  }

  switch (mode) {
    case VisionMode::AUTO_AIM_FLAT:{
      is_flat_mode_ = true;
      enable_ = true;
      break; 
    }
    case VisionMode::AUTO_AIM_SLOPE:{
      is_flat_mode_ = false;
      enable_ = true;
      break; 
    }
    default: {
      enable_ = false;
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
