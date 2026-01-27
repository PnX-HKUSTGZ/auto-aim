# rune_solver

符文解算节点，订阅符文检测结果并结合相机参数解算符文三维位置，通过扩展卡尔曼滤波器预测目标运动，输出可供射击的瞄准参数

## rune_solver_node.cpp

符文目标解算与跟踪节点，主要用于将二维图像检测结果转换为三维世界坐标，并进行运动预测

### 订阅话题 

* `rune_detector/rune` (`auto_aim_interfaces/msg/Rune`) - 符文检测结果，包含二维像素坐标
* `camera_info` (`sensor_msgs/msg/CameraInfo`) - 相机内参信息，用于PnP解算三维坐标

### 发布话题 

* `rune_solver/rune_target` (`auto_aim_interfaces/msg/RuneTarget`) - 解算后的符文三维目标信息
* `rune_solver/observed_angle` (`auto_aim_interfaces/msg/debugRuneAngle`) - 调试模式下输出观测角度
* `rune_solver/fitting_info` (`std_msgs/msg/String`) - 调试模式下输出拟合状态信息
* `rune_solver/marker` (`visualization_msgs/msg/MarkerArray`) - 可视化符文位置与预测轨迹

### 服务

* `rune_solver/set_mode` (`auto_aim_interfaces/srv/SetMode`) - 设置符文解算器工作模式

### 参数 

* `debug` (`bool`, default: false) - 是否开启调试模式
* `predict_time` (`double`, default: 0.1) - 预测时间偏移（秒）
* `gravity` (`double`, default: 9.8) - 重力加速度（m/s²）
* `bullet_speed` (`double`, default: 26.0) - 弹丸速度（m/s）
* `angle_offset_thres` (`double`, default: 0.78) - 角度偏移阈值（弧度）
* `lost_time_thres` (`double`, default: 0.5) - 目标丢失时间阈值（秒）
* `ekf.q` (`vector<double>`, default: [0.001, 0.001, 0.001, 0.001]) - 过程噪声协方差
* `ekf.r` (`vector<double>`, default: [0.1, 0.1, 0.1, 0.1]) - 测量噪声协方差

## rune_solver.cpp

RuneSolver 类实现了从二维图像检测结果到三维世界坐标的转换，并通过扩展卡尔曼滤波器（EKF）进行目标状态估计与运动预测

### 核心功能

1. 三维坐标解算
* PnP求解：利用相机内参与符文特征点进行透视投影变换解算
* 坐标变换：通过TF2将相机坐标系下的坐标转换为世界坐标系（odom帧）
* 异常值过滤：根据距离阈值筛选有效三维坐标（MIN_RUNE_DISTANCE ~ MAX_RUNE_DISTANCE）
2. 状态估计与跟踪
* 扩展卡尔曼滤波
* 目标状态管理：DETECTING：检测中; TRACKING：正常跟踪; LOST：目标丢失
3. 运动预测与曲线拟合
* 多项式曲线拟合：自动判断运动类型 (BIG/SMALL)，支持二次/三次多项式拟合，根据历史数据预测未来角度
* 运动状态验证：通过拟合误差判断跟踪可靠性
4. 射击参数计算
* 弹道模型：考虑重力影响的抛物线轨迹，计算弹丸飞行时间与提前量
* 角度补偿：基于符文姿态计算瞄准偏移角，处理角度跳变与目标切换情况

## curve_fitter.cpp

CurveFitter类实现了对符文目标运动轨迹的多项式曲线拟合，通过历史角度数据拟合出运动模型，支持同时拟合大符文和小符文两种运动模式，并自动选择最优模型。该类主要用于预测符文目标的未来角度，为射击提前量计算提供依据

### 核心功能

1. 数据管理与状态判断
* 历史数据队列：存储时间-角度数据对，支持动态维护队列长度
* 静态目标检测：通过角度变化量判断目标是否静止
* 运动方向确定：根据角度变化趋势确定顺时针/逆时针旋转方向
2. 曲线拟合算法
* 双曲线并行拟合：同时拟合大符文（BIG）和小符文（SMALL）模型，使用Ceres Solver进行非线性最小二乘优化，基于拟合代价自动选择最优模型
* 指定类型拟合：按预设类型（BIG/SMALL）进行曲线拟合。为不同类型设置合理的参数初始值与约束范围
3. 拟合模型
* 大符文模型：五次多项式模型
* 小符文模型：三次多项式模型
4. 并行计算优化
* 异步拟合：使用 std::async 实现并行计算
* 自适应并行策略：根据数据量决定是否启用并行拟合
* 资源管理：避免同时启动多个拟合任务导致资源浪费

## pnp_solver.cpp

核心思想与armor_detector中的pnp_solver.cpp一致