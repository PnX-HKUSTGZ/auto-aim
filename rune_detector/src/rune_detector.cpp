#include "rune_detector/rune_detector.hpp"
#include <opencv2/highgui.hpp>
#include "rune_detector/types.hpp"
namespace rm_auto_aim {
RuneDetector::RuneDetector(int max_iterations, double distance_threshold, double prob_threshold, EnemyColor detect_color): 
    max_iterations(max_iterations), distance_threshold(distance_threshold), prob_threshold(prob_threshold), detect_color(detect_color)
{
}
std::vector<RuneObject> RuneDetector::detectRune(const cv::Mat &img){
    frame = img.clone(); 
    cv::Mat gray;

    // 转换为灰度图像
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 二值化
    cv::threshold(gray, aim_img, 80, 255, cv::THRESH_BINARY);

    // 定义结构元素
    cv::Mat element_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat element_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // 流水灯，先膨胀后侵蚀
    cv::dilate(aim_img, flow_img, element_dilate);
    cv::erode(flow_img, flow_img, element_erode);

    hitting_light_mask = cv::Mat::ones(aim_img.size(), aim_img.type()); // 击打灯条的蒙版初始化

    std::vector<RuneObject> rune_objects; 
    RuneObject rune_object;
    std::vector<cv::Point2f> signal_points_hitting = processHittingLights();
    if(signal_points_hitting.size() != 6) return {};
    else{
        rune_object.pts.arm_bottom = signal_points_hitting[0];
        rune_object.pts.arm_top = signal_points_hitting[1];
        rune_object.pts.hit_bottom = signal_points_hitting[2];
        rune_object.pts.hit_left = signal_points_hitting[3];
        rune_object.pts.hit_top = signal_points_hitting[4];
        rune_object.pts.hit_right = signal_points_hitting[5];
        rune_object.type = RuneType::ACTIVATED;
        rune_objects.push_back(rune_object);
    }
    
    center = signal_points_hitting[0] + (signal_points_hitting[1] - signal_points_hitting[0]) * 0.5;
    aim_img = aim_img.mul(hitting_light_mask);
    cv::erode(aim_img, arm_img, element_dilate);
    cv::dilate(aim_img, hit_img, element_dilate);

    std::vector<std::vector<cv::Point2f>> signal_points_hit = processhitLights();
    for(auto & signal_point_hit : signal_points_hit){
        if(signal_point_hit.size() == 6){
            rune_object.pts.arm_bottom = signal_point_hit[0];
            rune_object.pts.arm_top = signal_point_hit[1];
            rune_object.pts.hit_bottom = signal_point_hit[2];
            rune_object.pts.hit_left = signal_point_hit[3];
            rune_object.pts.hit_top = signal_point_hit[4];
            rune_object.pts.hit_right = signal_point_hit[5];
            rune_object.type = RuneType::INACTIVATED;
            rune_objects.push_back(rune_object);
        }
    }

    return rune_objects;
}
std::vector<cv::Point2f> RuneDetector::processHittingLights()// 用于绘制的原图或当前帧
{
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(flow_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 筛选并保存长宽比在 2.5 到 7 之间的旋转矩形
    std::vector<cv::RotatedRect> flow_lights;
    for (const auto& contour : contours) {
        cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
        if(!isRightColor(rotated_rect)) continue;
        double aspect_ratio = static_cast<double>(rotated_rect.size.width) / rotated_rect.size.height;
        if (aspect_ratio < 1.0) {
            aspect_ratio = 1.0 / aspect_ratio;
        }
        if (aspect_ratio >= 2.5 && aspect_ratio <= 7.0 && rotated_rect.size.area() >= 20) {
            flow_lights.push_back(rotated_rect);
        }
    }

    // 储存含有两个以上子轮廓的父轮廓并用椭圆拟合
    std::vector<cv::RotatedRect> aim_lights;
    for (size_t i = 0; i < contours.size(); i++) {
        int child_count = 0;
        for (int j = hierarchy[i][2]; j != -1; j = hierarchy[j][0]) {
            child_count++;
        }
        // 检查是否有两个以上子轮廓并且轮廓点数大于等于 5
        if (child_count >= 2 && contours[i].size() >= 5) {
            cv::RotatedRect ellipse = cv::fitEllipse(contours[i]);
            if(!isRightColor(ellipse)) continue;
            aim_lights.push_back(ellipse);
        }
    }

    // 进行匹配
    std::pair<cv::RotatedRect, cv::RotatedRect> matched_light; 
    bool matched = false;
    double min_score = 1e9;

    for (const auto& aim_light : aim_lights) {
        for(const auto& flow_light : flow_lights) {
            double score = calculateMatchScoreHitting(flow_light, aim_light);
            if (score < min_score && score != -1) {
                min_score = score;
                matched = true;
                matched_light.first = flow_light;
                matched_light.second = aim_light;
            }
        }
    }

    std::vector<cv::Point2f> signal_points_hitting;
    if (matched) {
        cv::RotatedRect flow_light = matched_light.first;
        cv::RotatedRect aim_light = matched_light.second;

        // 绘制匹配的灯条
        cv::Point2f flow_light_vertices[4];
        flow_light.points(flow_light_vertices);
        for (int i = 0; i < 4; i++) {
            cv::line(frame, flow_light_vertices[i], flow_light_vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        // 绘制匹配的椭圆
        cv::ellipse(frame, aim_light, cv::Scalar(0, 255, 0), 2);

        // 获取关键点
        signal_points_hitting = getSignalPoints(aim_light, flow_light);

        // 将检测到的矩形和椭圆的内部的mask设置为0
        std::vector<cv::Point> int_flow_light_vertices;
        int_flow_light_vertices.reserve(4);
        for (auto & flow_light_vertice : flow_light_vertices) {
            int_flow_light_vertices.push_back(cv::Point(static_cast<int>(flow_light_vertice.x), static_cast<int>(flow_light_vertice.y)));
        }
        cv::fillConvexPoly(hitting_light_mask, int_flow_light_vertices, cv::Scalar(0));
        cv::ellipse(hitting_light_mask, aim_light, cv::Scalar(0), -1);
    }

    return signal_points_hitting;
}
std::vector<std::vector<cv::Point2f>> RuneDetector::processhitLights()
{
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // 处理 arm_img
    cv::findContours(arm_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::RotatedRect> arm_lights;
    for (const auto& contour : contours) {
        cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
        if(!isRightColor(rotated_rect)) continue;
        double aspect_ratio = static_cast<double>(rotated_rect.size.width) / rotated_rect.size.height;
        if (aspect_ratio < 1.0) {
            aspect_ratio = 1.0 / aspect_ratio;
        }
        if (aspect_ratio >= 2.5 && aspect_ratio <= 7.0 && rotated_rect.size.area() >= 20) {
            arm_lights.push_back(rotated_rect);
        }
    }

    // 清空并处理 hit_img
    contours.clear();
    cv::findContours(hit_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::RotatedRect> hit_lights;
    for (size_t i = 0; i < contours.size(); i++) {
        int child_count = 0;
        for (int j = hierarchy[i][2]; j != -1; j = hierarchy[j][0]) {
            child_count++;
        }
        // 检查是否有一个子轮廓并且轮廓点数 >=5
        if (child_count == 1 && contours[i].size() >= 5 && contours[hierarchy[i][2]].size() >= 5) {
            cv::RotatedRect ellipse_father = cv::fitEllipse(contours[i]);
            if(!isRightColor(ellipse_father)) continue;
            cv::RotatedRect ellipse_child = cv::fitEllipse(contours[hierarchy[i][2]]);
            cv::RotatedRect ellipse;
            ellipse.size = (ellipse_father.size + ellipse_child.size) / 2.0f;
            ellipse.center = (ellipse_father.center + ellipse_child.center) / 2.0f;
            ellipse.angle = (ellipse_father.angle + ellipse_child.angle) / 2.0f;
            hit_lights.push_back(ellipse);
        }
    }

    // 进行匹配
    std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> matched_arm_lights;
    for (const auto& arm_light : arm_lights) {
        double local_min_score = 1e9;
        cv::RotatedRect matched_hit_light;
        for(const auto& hit_light : hit_lights){
            double score = calculateMatchScorehit(arm_light, hit_light);
            if (score < local_min_score && score != -1) {
                local_min_score = score;
                matched_hit_light = hit_light;
            }
        }
        if(local_min_score != 1e9) {
            matched_arm_lights.push_back(std::make_pair(arm_light, matched_hit_light));
        }
        else if(arm_light.size.area() > 100){
            cv::Rect roi = calculateROI(arm_light);
            roi &= cv::Rect(0, 0, hit_img.cols, hit_img.rows);
            if(roi.area() == 0) continue;
            cv::Mat roi_img = hit_img(roi);
            cv::RotatedRect ellipse = detectBestEllipse(roi_img);
            ellipse.center.x += roi.x;
            ellipse.center.y += roi.y;
            if(ellipse.size.area() == 0 || !isRightColor(ellipse)) continue;
            if(calculateMatchScorehit(arm_light, ellipse) != -1){
                matched_arm_lights.push_back(std::make_pair(arm_light, ellipse));
            }
        }
    }

    // 绘制结果并保存关键点
    std::vector<std::vector<cv::Point2f>> signal_points_hit;
    for (const auto& matched_arm_light : matched_arm_lights) {
        cv::RotatedRect arm_light = matched_arm_light.first;
        cv::RotatedRect hit_light = matched_arm_light.second;

        // 绘制匹配的灯条
        cv::Point2f arm_light_vertices[4];
        arm_light.points(arm_light_vertices);
        for (int i = 0; i < 4; i++) {
            cv::line(frame, arm_light_vertices[i], arm_light_vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
        }
        // 绘制匹配的椭圆
        cv::ellipse(frame, hit_light, cv::Scalar(0, 0, 255), 2);

        // 获取关键点
        signal_points_hit.push_back(getSignalPoints(hit_light, arm_light));
    }

    return signal_points_hit;
}
std::tuple<cv::Point2f, cv::Mat> RuneDetector::detectRTag(const cv::Mat &img, const cv::Point2f &prior) {
    if (prior.x < 0 || prior.x > img.cols || prior.y < 0 || prior.y > img.rows) {
        return {prior, cv::Mat::zeros(cv::Size(200, 200), CV_8UC3)};
    }

    // Create ROI
    cv::Rect roi = cv::Rect(prior.x - 100, prior.y - 100, 200, 200) &
                                 cv::Rect(0, 0, img.cols, img.rows);
    const cv::Point2f prior_in_roi = prior - cv::Point2f(roi.tl());

    cv::Mat img_roi = img(roi);

    // Gray -> Binary -> Dilate
    cv::Mat gray_img;
    cv::cvtColor(img_roi, gray_img, cv::COLOR_BGR2GRAY);
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, 0, 255,
                cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binary_img, binary_img, kernel);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL,
                cv::CHAIN_APPROX_NONE);

    auto it = std::find_if(
            contours.begin(), contours.end(),
            [p = prior_in_roi](const std::vector<cv::Point> &contour) -> bool {
                return cv::boundingRect(contour).contains(p);
            });

    // For visualization
    cv::cvtColor(binary_img, binary_img, cv::COLOR_GRAY2BGR);

    if (it == contours.end()) {
        return {prior, binary_img};
    }

    cv::drawContours(binary_img, contours, it - contours.begin(),
        cv::Scalar(0, 255, 0), 2);

    cv::Point2f center =
            std::accumulate(it->begin(), it->end(), cv::Point(0, 0));
    center /= static_cast<float>(it->size());
    center += cv::Point2f(roi.tl());

    return {center, binary_img};
}
// 计算角度差
double RuneDetector::calculateAngleDifference(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse) {
    cv::Point2f rect_center = rect.center;
    cv::Point2f ellipse_center = ellipse.center;
    cv::Point2f rect_vertices[4];
    rect.points(rect_vertices);

    // 计算矩形长边方向向量
    cv::Point2f rect_long_edge = rect_vertices[1] - rect_vertices[0];
    if (cv::norm(rect_vertices[2] - rect_vertices[1]) > cv::norm(rect_long_edge)) {
        rect_long_edge = rect_vertices[2] - rect_vertices[1];
    }

    // 计算中心连线向量
    cv::Point2f center_line = ellipse_center - rect_center;

    // 计算角度
    double rect_angle = std::atan2(rect_long_edge.y, rect_long_edge.x);
    double center_line_angle = std::atan2(center_line.y, center_line.x);
    double angle_diff = std::abs(rect_angle - center_line_angle);
    angle_diff = std::min(angle_diff, CV_PI * 2 - angle_diff);
    return std::min(angle_diff, CV_PI - angle_diff);
}
// 计算椭圆沿特定方向的轴的长度
double RuneDetector::calculateAxisLength(const cv::RotatedRect& ellipse, const cv::Point2f& direction) {
    // 椭圆的长轴和短轴长度
    double a = ellipse.size.width / 2.0;  // 长轴的一半
    double b = ellipse.size.height / 2.0; // 短轴的一半

    // 椭圆的旋转角度（以弧度表示）
    double theta = ellipse.angle * CV_PI / 180.0;

    // 方向向量的单位化
    cv::Point2f unit_direction = direction / cv::norm(direction);

    // 计算旋转后的方向向量
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double x_prime = unit_direction.x * cos_theta + unit_direction.y * sin_theta;
    double y_prime = -unit_direction.x * sin_theta + unit_direction.y * cos_theta;

    // 计算沿特定方向的弦的长度
    double chord_length = 2 * a * b / std::sqrt(b * b * x_prime * x_prime + a * a * y_prime * y_prime);

    return chord_length;
}
// 计算比例差
double RuneDetector::calculateRatioDifferenceHitting(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse) {
    double center_distance = cv::norm(rect.center - ellipse.center);
    // 计算椭圆沿特定方向的轴的长度
    cv::Point2f rect_vertices[4];
    rect.points(rect_vertices);
    cv::Point2f rect_long_edge = rect_vertices[1] - rect_vertices[0];
    if (cv::norm(rect_vertices[2] - rect_vertices[1]) > cv::norm(rect_long_edge)) {
        rect_long_edge = rect_vertices[2] - rect_vertices[1];
    }
    double ellipse_axis = calculateAxisLength(ellipse, rect_long_edge);

    double ratio1 = ellipse_axis / cv::norm(rect_long_edge);
    double ratio2 = cv::norm(rect_long_edge) / center_distance;

    double target_ratio1 = 310.0 / 330.0;
    double target_ratio2 = 330.0 / 355.0;

    double ratio_diff1 = std::abs(ratio1 - target_ratio1);
    double ratio_diff2 = std::abs(ratio2 - target_ratio2);

    return ratio_diff1 + ratio_diff2;
}

// 计算匹配程度
double RuneDetector::calculateMatchScoreHitting(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse) {
    double angle_diff = calculateAngleDifference(rect, ellipse);
    double ratio_diff = calculateRatioDifferenceHitting(rect, ellipse);

    // 归一化并计算总分
    double angle_score = angle_diff / CV_PI;
    double ratio_score = ratio_diff / 2.0; // 假设最大比例差为2

    if(angle_diff < CV_PI / 12 && ratio_score < 0.2) return angle_score + ratio_score;
    else return -1; // 不匹配
}
// 计算比例差
double RuneDetector::calculateRatioDifferencehit(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse) {
    double rect_long_edge = std::max(rect.size.width, rect.size.height);
    double center_distance = cv::norm(rect.center - ellipse.center);
    double ratio_score = rect_long_edge / center_distance;
    double target_ratio = 330.0 / 355.0;
    return std::abs(ratio_score - target_ratio);
}
// 计算匹配程度
double RuneDetector::calculateMatchScorehit(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse) {
    double angle_diff = calculateAngleDifference(rect, ellipse);
    double ratio_diff = calculateRatioDifferencehit(rect, ellipse);

    // 归一化并计算总分
    double angle_score = angle_diff / CV_PI;
    double ratio_score = ratio_diff; // 假设最大比例差为1
    if(angle_diff < CV_PI / 12 && ratio_score < 0.2) return angle_score + ratio_score;
    else return -1; // 不匹配
}
// 计算指定方向与椭圆的两个交点
std::vector<cv::Point2f> RuneDetector::ellipseIntersections(const cv::RotatedRect& ellipse, const cv::Point2f& dir) {
    // 单位化方向
    cv::Point2f unit_dir = dir / cv::norm(dir);

    // 椭圆长短轴半径
    double a = ellipse.size.width * 0.5;
    double b = ellipse.size.height * 0.5;

    // 旋转角（弧度）
    double theta = ellipse.angle * CV_PI / 180.0;
    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);

    // 方向向量旋转到椭圆坐标系
    double x =  unit_dir.x * cos_t + unit_dir.y * sin_t;
    double y = -unit_dir.x * sin_t + unit_dir.y * cos_t;

    // 计算半弦长
    double half_len = (a * b) / std::sqrt(b * b * x * x + a * a * y * y);

    // 椭圆中心
    cv::Point2f c = ellipse.center;

    // 原方向向量在图像坐标系下的分量（逆旋转）
    // 为了得到在图像坐标系下 ±half_len 的坐标，需要将 (±x_, ±y_) 再旋转回来
    // 可直接在单位化方向上乘以 half_len，分别正负即可
    cv::Point2f dir_n = unit_dir * static_cast<float>(half_len);

    // 交点1、交点2 = 椭圆中心 ± dir_n
    std::vector<cv::Point2f> pts(2);
    pts[0] = c + dir_n; 
    pts[1] = c - dir_n;
    return pts;
}
// 提取 6 个 signal points
std::vector<cv::Point2f> RuneDetector::getSignalPoints(const cv::RotatedRect& ellipse, const cv::RotatedRect& rect){
    // 最终返回的 76 个点
    // [0]、[1] = 矩形两条短边的中心；[2] ~ [5] = 椭圆交点；(共 6 个)
    std::vector<cv::Point2f> result(6, cv::Point2f(0,0));

    // 1. 找矩形的两条短边中心
    cv::Point2f pts[4];
    rect.points(pts);
    // 计算四条边长度
    std::vector<std::pair<float,int>> edges; // (边长度, 起点索引)
    for(int i=0; i<4; i++){
        float len = cv::norm(pts[(i+1)%4] - pts[i]);
        edges.push_back(std::make_pair(len, i));
    }
    // 按边长排序
    std::sort(edges.begin(), edges.end(),
              [](auto &a, auto &b){return a.first < b.first;});

    // edges[0], edges[1] 即为两条短边
    auto idx0 = edges[0].second; 
    auto idx1 = edges[1].second; 
    cv::Point2f mid0 = 0.5f * (pts[idx0] + pts[(idx0+1)%4]);
    cv::Point2f mid1 = 0.5f * (pts[idx1] + pts[(idx1+1)%4]);

    // 根据离椭圆中心距离判断谁放 0 号位
    float d0 = cv::norm(mid0 - ellipse.center);
    float d1 = cv::norm(mid1 - ellipse.center);
    if(d0 > d1){
        result[0] = mid0; 
        result[1] = mid1;
    } else {
        result[0] = mid1; 
        result[1] = mid0;
    }

    // 2. 计算椭圆中心与矩形中心之间的连线 dir
    cv::Point2f dir = rect.center - ellipse.center;
    // 垂直方向 dir_perp
    cv::Point2f dir_perp(-dir.y, dir.x);

    // 3. 分别计算这两条方向与椭圆的交点 (各自 2 个)
    std::vector<cv::Point2f> pts_dir  = ellipseIntersections(ellipse, dir);
    std::vector<cv::Point2f> pts_perp = ellipseIntersections(ellipse, dir_perp);

    // 合并成 4 个点
    std::vector<cv::Point2f> four_pts;
    four_pts.insert(four_pts.end(), pts_dir.begin(),  pts_dir.end());
    four_pts.insert(four_pts.end(), pts_perp.begin(), pts_perp.end());

    // 4. 找到距离矩形中心最近的点放在位置 [2]，其余按顺时针顺序放 [3]、[4]、[5]
    // 先找距离 rect.center 最近的点
    float min_dist = 1e9f;
    for(int i=0; i<4; i++){
        float dist = cv::norm(four_pts[i] - rect.center);
        if(dist < min_dist){
            min_dist = dist;
        }
    }
    
    // 剩下 3 个点，按顺时针顺序放 [3],[4],[5]
    cv::Point2f base = ellipse.center;
    // 以矩形中心为参考，按顺时针(atan2)排序
    std::sort(four_pts.begin(), four_pts.end(), 
              [base](const cv::Point2f &p1, const cv::Point2f &p2){
                  double a1 = std::atan2(p1.y - base.y, p1.x - base.x);
                  double a2 = std::atan2(p2.y - base.y, p2.x - base.x);
                  return a1 < a2;
              });
    for(int i = 0; i < 4; i++){
        float dist = cv::norm(four_pts[i] - rect.center);
        if(dist == min_dist){
            for(int j = i; j < 4 + i; j++){
                result[j - i + 2] = four_pts[j % 4];
            }
            break;
        }
    }

    return result;
}
// 计算线段AB的两个可能的点B，并以B为中心计算正方形ROI
cv::Rect RuneDetector::calculateROI(const cv::RotatedRect& rect) {
    std::vector<cv::Rect> rois;

    // 获取矩形的四个顶点
    cv::Point2f rect_vertices[4];
    rect.points(rect_vertices);

    // 计算矩形长边方向向量
    cv::Point2f rect_long_edge = rect_vertices[1] - rect_vertices[0];
    if (cv::norm(rect_vertices[2] - rect_vertices[1]) > cv::norm(rect_long_edge)) {
        rect_long_edge = rect_vertices[2] - rect_vertices[1];
    }

    // 计算长边长度
    double long_edge_length = cv::norm(rect_long_edge);

    // 计算线段AB的两个可能的点B
    cv::Point2f rect_center = rect.center;
    cv::Point2f b1 = rect_center + rect_long_edge * (400.0 / 330.0);
    cv::Point2f b2 = rect_center - rect_long_edge * (400.0 / 330.0);
    // 选择距离中心更远的点
    cv::Point2f b = cv::norm(b1 - center) > cv::norm(b2 - center) ? b1 : b2;

    // 计算正方形ROI的边长
    double roi_side_length = long_edge_length * (400.0 / 330.0);

    // 计算 ROI 并检查边界
    cv::Rect roi(
        b.x - roi_side_length / 2, 
        b.y - roi_side_length / 2, 
        roi_side_length, 
        roi_side_length
    );
    
    // 确保 ROI 在图像范围内
    roi &= cv::Rect(0, 0, hitting_light_mask.cols, hitting_light_mask.rows);
    return roi;
}
// 计算点到椭圆的距离
double RuneDetector::pointToEllipseDistance(const cv::Point2f& point, const cv::RotatedRect& ellipse) {
    // 将点从图像坐标系转换到椭圆坐标系
    cv::Point2f centered = point - ellipse.center;
    double angle = ellipse.angle * CV_PI / 180.0;
    double cos_angle = std::cos(angle);
    double sin_angle = std::sin(angle);
    
    // 旋转点到椭圆的主轴方向
    double x = centered.x * cos_angle + centered.y * sin_angle;
    double y = -centered.x * sin_angle + centered.y * cos_angle;
    
    // 计算椭圆的半长轴和半短轴
    double a = ellipse.size.width * 0.5;
    double b = ellipse.size.height * 0.5;
    
    // 计算点到椭圆的距离
    double px = std::abs(x);
    double py = std::abs(y);
    
    // 迭代求解最近点
    double t = std::atan2(py * a, px * b);
    double dx = a * std::cos(t);
    double dy = b * std::sin(t);
    
    return std::sqrt((px - dx) * (px - dx) + (py - dy) * (py - dy));
}
// 修改后的RANSAC拟合椭圆函数
cv::RotatedRect RuneDetector::fitEllipseRANSAC(const std::vector<cv::Point>& points) {
    cv::RotatedRect best_ellipse;
    int best_inliers = 0;
    const int total_points = points.size();
    if(total_points < 5) return cv::RotatedRect();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points.size() - 1);
    std::vector<cv::Point> best_points, inlier_points;

    for (int iter = 0; iter < max_iterations; ++iter) {
        inlier_points.clear();
        // 随机选择5个点
        std::vector<cv::Point> sample_points;
        while (sample_points.size() < 5) {
            int idx = dis(gen);
            sample_points.push_back(points[idx]);
        }

        // 拟合椭圆
        if (sample_points.size() >= 5) {
            cv::RotatedRect ellipse = cv::fitEllipse(sample_points);

            // 计算内点数量
            int inliers = 0;
            for (const auto& point : points) {
                if (pointToEllipseDistance(point, ellipse) < distance_threshold) {
                    inliers++;
                    inlier_points.push_back(point);
                }
            }
            // 更新最佳椭圆
            if (inliers > best_inliers) {
                best_inliers = inliers;
                best_ellipse = ellipse;
                best_points = inlier_points;
                if(best_inliers > total_points * 0.8) break;
            }
        }
    }
    // 重新拟合最佳椭圆
    if(best_inliers > total_points * prob_threshold && best_points.size() >= 50){
        best_ellipse = cv::fitEllipse(best_points);
        if (best_ellipse.size.width > 0 && best_ellipse.size.height > 0 &&
            !std::isnan(best_ellipse.center.x) && !std::isnan(best_ellipse.center.y) &&
            !std::isnan(best_ellipse.angle)) {
            return best_ellipse;
        }
    }
    return cv::RotatedRect();
}

cv::RotatedRect RuneDetector::detectBestEllipse(const cv::Mat& src) {
    cv::RotatedRect best_ellipse;
    const int STANDARD_SIZE = 25; // 标准分辨率大小
    
    // 先进行形态学操作
    cv::Mat processed;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(src, processed, element);
    
    // 计算缩放比例
    float scale_x = static_cast<float>(STANDARD_SIZE) / src.cols;
    float scale_y = static_cast<float>(STANDARD_SIZE) / src.rows;
    
    // 缩放处理后的图像到标准大小
    cv::Mat resized;
    cv::resize(processed, resized, cv::Size(STANDARD_SIZE, STANDARD_SIZE));
    
    // 收集点
    std::vector<cv::Point> points;
    for(int i = 0; i < STANDARD_SIZE; i++) {
        for(int j = 0; j < STANDARD_SIZE; j++) {
            if(resized.at<uchar>(i, j) > 0 && 
               cv::norm(cv::Point(j, i) - cv::Point(STANDARD_SIZE/2, STANDARD_SIZE/2)) * 2 < STANDARD_SIZE) {
                points.push_back(cv::Point(j, i));
            }
        }
    }

    // 使用 RANSAC 算法拟合椭圆
    cv::RotatedRect standard_ellipse = fitEllipseRANSAC(points);
    
    // 将椭圆参数映射回原始尺寸
    if(standard_ellipse.size.width > 0 && standard_ellipse.size.height > 0) {
        best_ellipse = cv::RotatedRect(
            cv::Point2f(standard_ellipse.center.x / scale_x, standard_ellipse.center.y / scale_y),
            cv::Size2f(standard_ellipse.size.width / scale_x, standard_ellipse.size.height / scale_y),
            standard_ellipse.angle
        );
    }

    return best_ellipse;
}

bool RuneDetector::isRightColor(const cv::RotatedRect& rect){
    // 计算 ROI
    cv::Rect roi_rect(
        rect.center.x - rect.size.width / 2, 
        rect.center.y - rect.size.height / 2, 
        rect.size.width, 
        rect.size.height
    );
    
    // 边界检查
    roi_rect &= cv::Rect(0, 0, frame.cols, frame.rows);
    if(roi_rect.area() == 0) return false;
    
    cv::Mat roi = frame(roi_rect);
    cv::Scalar mean_color = cv::mean(roi);
    if(detect_color == EnemyColor::RED){
        return mean_color[2] > mean_color[0] && mean_color[2] > mean_color[1];
    } 
    else if(detect_color == EnemyColor::BLUE){
        return mean_color[0] > mean_color[1] && mean_color[0] > mean_color[2];
    }
    else if(detect_color == EnemyColor::WHITE){
        return true;
    }
    return false;
}

}  // namespace rm_auto_aim