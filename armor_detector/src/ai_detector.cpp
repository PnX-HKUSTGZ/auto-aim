// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#include "armor_detector/ai_detector.hpp"

#include <algorithm>
#include <vector>

namespace rm_auto_aim
{

AIDetector::AIDetector(
    const std::string & model_path, 
    const std::string & device,
    float conf_threshold,
    float nms_threshold)
: conf_threshold_(conf_threshold), nms_threshold_(nms_threshold)
{
    // 设置输入形状
    input_shape = {1, static_cast<size_t>(IMAGE_HEIGHT), static_cast<size_t>(IMAGE_WIDTH), 3};
    
    // 读取模型
    model = core.read_model(model_path);
    
    // 初始化预处理器
    ppp = std::make_unique<ov::preprocess::PrePostProcessor>(model);
    
    // 指定输入图像格式
    ppp->input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    
    // 指定预处理管道
    ppp->input().preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({255.0f, 255.0f, 255.0f});
    
    // 指定模型输入布局
    ppp->input().model().set_layout("NCHW");
    
    // 指定输出结果格式
    ppp->output().tensor().set_element_type(ov::element::f32);
    
    // 构建模型
    model = ppp->build();
    
    // 编译模型
    compiled_model = core.compile_model(model, device);
}

AIDetector::~AIDetector() = default;

std::vector<Armor> AIDetector::detect(const cv::Mat & input, int detect_color)
{
    // 清空之前的结果
    objects_.clear();
    tmp_objects_.clear();
    ious_.clear();
    debug_armors.data.clear();
    
    // 执行推理
    infer(input, detect_color);
    
    // 将检测对象转换为装甲板
    std::vector<Armor> armors;
    armors.reserve(tmp_objects_.size());
    
    for (const auto & obj : tmp_objects_) {
        Armor armor = objectToArmor(obj);
        armors.push_back(armor);
        
        // 添加调试信息
        auto_aim_interfaces::msg::DebugArmor debug_armor;
        debug_armor.center_x = static_cast<int32_t>(armor.center.x);
        debug_armor.type = armor.type == ArmorType::SMALL ? "small" : "large";
        debug_armor.light_ratio = obj.ratio;
        debug_armor.center_distance = cv::norm(armor.left_light.center - armor.right_light.center);
        debug_armor.angle = 0.0; // AI检测器暂时不计算角度
        debug_armors.data.push_back(debug_armor);
    }
    
    return armors;
}

void AIDetector::infer(const cv::Mat & img, int detect_color)
{
    // 创建输入张量
    uchar* input_data = (uchar*)img.data;
    ov::Tensor input_tensor = ov::Tensor(
        compiled_model.input().get_element_type(), 
        compiled_model.input().get_shape(), 
        input_data);
    
    // 创建推理请求
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    
    // 执行推理
    infer_request.infer();
    
    // 获取输出张量
    auto output = infer_request.get_output_tensor(0);
    ov::Shape output_shape = output.get_shape();
    
    // 创建输出矩阵 (25200 x 22)
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());
    
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<float> confidences;
    
    // 解析输出结果
    for (int i = 0; i < output_buffer.rows; i++) {
        // 获取置信度 (第8列)
        float confidence = output_buffer.at<float>(i, 8);
        confidence = sigmoid(confidence);
        
        if (confidence < conf_threshold_) {
            continue;
        }
        
        // 获取颜色和类别得分
        cv::Mat color_scores = output_buffer.row(i).colRange(9, 13);  // 颜色 (4列)
        cv::Mat classes_scores = output_buffer.row(i).colRange(13, 22); // 数字类别 (9列)
        
        cv::Point class_id, color_id;
        double score_color, score_num;
        cv::minMaxLoc(classes_scores, nullptr, &score_num, nullptr, &class_id);
        cv::minMaxLoc(color_scores, nullptr, &score_color, nullptr, &color_id);
        
        // 过滤不需要的颜色
        // color_id.x: 0=red, 1=blue, 2=none, 3=purple
        if (color_id.x == 2 || color_id.x == 3) {  // None 或 Purple
            continue;
        }
        
        if (detect_color == 0 && color_id.x == 1) {  // 检测蓝色但发现红色
            continue;
        }
        
        if (detect_color == 1 && color_id.x == 0) {  // 检测红色但发现蓝色
            continue;
        }
        
        // 创建检测对象
        Object obj;
        obj.prob = confidence;
        obj.color = color_id.x;
        obj.label = class_id.x;
        
        // 获取关键点坐标 (前8列)
        for (int j = 0; j < 8; j++) {
            obj.landmarks[j] = output_buffer.at<float>(i, j);
        }
        
        // 计算长度和宽度
        obj.length = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[6], 
                                         obj.landmarks[1] - obj.landmarks[7]));
        obj.width = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[2], 
                                        obj.landmarks[1] - obj.landmarks[3]));
        obj.ratio = obj.length / obj.width;
        
        // 构建四个角点 (左上逆时针 -> 左上顺时针)
        std::vector<cv::Point2f> points;
        points.reserve(4);
        points.push_back(cv::Point2f(obj.landmarks[0], obj.landmarks[1])); // 左上
        points.push_back(cv::Point2f(obj.landmarks[6], obj.landmarks[7])); // 右上
        points.push_back(cv::Point2f(obj.landmarks[4], obj.landmarks[5])); // 右下
        points.push_back(cv::Point2f(obj.landmarks[2], obj.landmarks[3])); // 左下
        
        // 计算边界框
        float min_x = points[0].x;
        float max_x = points[0].x;
        float min_y = points[0].y;
        float max_y = points[0].y;
        
        for (size_t j = 1; j < points.size(); j++) {
            min_x = std::min(min_x, points[j].x);
            max_x = std::max(max_x, points[j].x);
            min_y = std::min(min_y, points[j].y);
            max_y = std::max(max_y, points[j].y);
        }
        
        obj.rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
        
        objects_.push_back(obj);
        boxes.push_back(obj.rect);
        confidences.push_back(static_cast<float>(score_num));
    }
    
    // 非极大值抑制 (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);
    
    for (int valid_index : indices) {
        if (valid_index < static_cast<int>(objects_.size())) {
            tmp_objects_.push_back(objects_[valid_index]);
        }
    }
}

Armor AIDetector::objectToArmor(const Object & obj)
{
    Armor armor;
    
    // 设置装甲板类型 (根据长宽比判断)
    armor.type = (obj.ratio > 2.0) ? ArmorType::LARGE : ArmorType::SMALL;
    
    // 创建虚拟的左右灯条
    Light left_light, right_light;
    
    // 使用关键点信息构建灯条
    cv::Point2f left_top(obj.landmarks[0], obj.landmarks[1]);
    cv::Point2f left_bottom(obj.landmarks[2], obj.landmarks[3]);
    cv::Point2f right_top(obj.landmarks[6], obj.landmarks[7]);
    cv::Point2f right_bottom(obj.landmarks[4], obj.landmarks[5]);
    
    // 左灯条
    left_light.top = left_top;
    left_light.bottom = left_bottom;
    left_light.center = (left_top + left_bottom) / 2;
    left_light.length = cv::norm(left_top - left_bottom);
    left_light.width = 5.0; // 估计值
    left_light.color = obj.color;
    left_light.tilt_angle = 0.0; // 初始化角度
    
    // 右灯条
    right_light.top = right_top;
    right_light.bottom = right_bottom;
    right_light.center = (right_top + right_bottom) / 2;
    right_light.length = cv::norm(right_top - right_bottom);
    right_light.width = 5.0; // 估计值
    right_light.color = obj.color;
    right_light.tilt_angle = 0.0; // 初始化角度
    
    armor.left_light = left_light;
    armor.right_light = right_light;
    armor.center = (left_light.center + right_light.center) / 2;
    
    // 设置数字识别结果
    armor.number = std::to_string(obj.label);
    armor.confidence = obj.prob;
    armor.classfication_result = std::to_string(obj.label);
    
    return armor;
}

void AIDetector::drawResults(cv::Mat & img)
{
    for (const auto & obj : tmp_objects_) {
        // 绘制边界框
        cv::rectangle(img, obj.rect, cv::Scalar(0, 255, 0), 2);
        
        // 绘制关键点
        std::vector<cv::Point2f> points;
        points.reserve(4);
        for (int i = 0; i < 4; i++) {
            points.push_back(cv::Point2f(obj.landmarks[i*2], obj.landmarks[i*2+1]));
        }
        
        // 绘制装甲板四个角点
        for (size_t i = 0; i < points.size(); i++) {
            cv::circle(img, points[i], 3, cv::Scalar(0, 0, 255), -1);
            cv::line(img, points[i], points[(i+1) % points.size()], cv::Scalar(255, 0, 0), 2);
        }
        
        // 显示类别和置信度
        std::string label = "Class: " + std::to_string(obj.label) + 
                           " Color: " + (obj.color == 0 ? "Red" : "Blue") + 
                           " Conf: " + std::to_string(obj.prob);
        
        cv::putText(img, label, cv::Point(obj.rect.x, obj.rect.y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }
}

}  // namespace rm_auto_aim