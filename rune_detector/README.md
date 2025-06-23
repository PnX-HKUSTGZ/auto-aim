# rune_detector

符文（Rune）检测器节点，订阅相机图像流进行符文识别与定位，输出符文目标的二维像素坐标及状态信息，支持动态参数调整与模式切换

## rune_detector_node.cpp

符文检测与定位节点，识别符文目标并发布其位置信息

### 订阅话题 

* `image_raw` (`sensor_msgs/msg/Image`) - 相机图像信息，用于符文检测与识别

### 发布话题 

* `rune_detector/rune` (`auto_aim_interfaces/msg/Rune`) - 输出检测到的符文目标信息，包括位置与状态
* `rune_detector/result_img` (`sensor_msgs/msg/Image`) - 调试模式下输出绘制检测结果的图像

### 服务

* `rune_detector/set_mode` (`auto_aim_interfaces/srv/SetMode`) - 设置符文检测模式（启用/禁用）

### 参数 

* `debug` (`bool`, default: false) - 是否开启调试模式（开启后发布调试图像）
* `binary_thresh` (`int`, default: 100) - 灯条检测的二值化阈值
* `detect_color` (`int`, default: 1) - 检测颜色，0为红，1为蓝
* `detect_r_tag` (`bool`, default: true) - 是否检测R标识（符文中心标记）
* `frame_id` (`string`, default: "camera_optical_frame") - 输出消息的坐标系ID
* `max_iterations` (`int`, default: 99) - 检测算法最大迭代次数
* `distance_threshold` (`double`, default: 2.0) - 目标距离阈值（像素）
* `prob_threshold` (`double`, default: 0.6) - 目标检测概率阈值

## rune_detector.cpp

RuneDetector类实现了对符文目标的检测与定位功能，可识别符文的灯条特征，计算其关键点位置并判断激活状态

### 核心功能

1. 图像预处理
* 灰度转换与二值化：将输入RGB图像转为灰度图，并通过阈值分割提取目标区域
* 形态学操作：使用膨胀、腐蚀操作处理二值图像，增强灯条特征
2. 灯条与椭圆检测
* 轮廓提取：使用findContours获取图像中的灯条轮廓
* 旋转矩形拟合：对灯条轮廓拟合最小面积矩形，筛选符合长宽比（2.5-7.0）的灯条
* 椭圆拟合：对包含子轮廓的父轮廓拟合椭圆，使用RANSAC算法增强椭圆拟合鲁棒性，通过迭代优化内点数量; 支持在ROI区域内检测最佳椭圆
3. 目标匹配与关键点提取
* 匹配评分系统：计算角度差(矩形与椭圆方向一致性)、比例差（尺寸比例与目标比例的偏差），综合评分筛选最佳匹配对
* 关键点计算：提取矩形短边中心作为装甲板关键点，计算椭圆与方向向量的交点作为击打区域关键点，按顺时针顺序排列6个关键点（装甲底部、顶部、击打区域四点）
4. R标识检测
* ROI创建：以先验位置为中心创建正方形感兴趣区域
* 自适应阈值分割：使用OTSU算法自动计算二值化阈值
* 轮廓筛选：查找包含先验位置的轮廓并计算中心
* 可视化反馈：返回R标签中心坐标与二值化ROI图像