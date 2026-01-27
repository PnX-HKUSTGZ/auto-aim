# Ballistic Calculation

## 概述

`ballistic_calculation` 是 RoboMaster 自瞄系统中的弹道计算模块，负责预测目标位置并计算最优射击角度。该包实现了考虑空气阻力的弹道计算、装甲板选择策略以及多种目标类型的处理。

## 主要功能

- **弹道计算**: 基于空气阻力模型的精确弹道计算
- **目标预测**: 支持装甲车辆和能量机关的运动预测
- **装甲板选择**: 三级火控策略的智能装甲板选择
- **坐标变换**: 从odom坐标系到枪口坐标系的精确变换
- **开火判断**: 基于云台位姿的开火条件判断

## 架构设计

### 核心组件

1. **BallisticCalculateNode**: 主节点，负责协调各个组件
2. **Ballistic**: 弹道计算器，实现迭代优化算法
3. **AimInfo**: 目标信息基类，定义目标预测接口
   - `CarInfo`: 装甲车辆信息处理
   - `ArmorInfo`: 装甲板信息处理
   - `RuneInfo`: 能量机关信息处理
4. **ArmorSelector**: 装甲板选择器，实现三级火控策略
5. **数学工具**: 坐标变换和优化算法支持

### 类图关系

```
BallisticCalculateNode
├── Ballistic (弹道计算器)
├── AimInfo (目标信息基类)
│   ├── CarInfo (车辆信息)
│   ├── ArmorInfo (装甲板信息)
│   └── RuneInfo (能量机关信息)
└── ArmorSelector (装甲板选择器)
```

## 算法原理

### 弹道计算

使用考虑空气阻力的弹道方程：

```
x(t) = (1/k) * ln(k*v₀*cos(θ)*t + 1)
y(t) = (v₀*sin(θ) + g/k) * (1 - e^(-kt))/k - g*t/k
```

其中：
- `k`: 空气阻力系数
- `v₀`: 初始速度
- `θ`: 发射角度
- `g`: 重力加速度

### 三级火控策略

1. **一级策略**: 通过点乘计算最优装甲板
2. **二级策略**: 引入放弃角度，选择更容易击中的装甲板
3. **三级策略**: 高速目标时瞄准中心点

### 迭代优化

采用双重迭代优化：
1. 使用Ceres优化器优化飞行时间
2. 使用固定迭代法修正俯仰角
3. 迭代直到收敛

## 节点接口

### 订阅话题

#### `/tracker/target` 
- **消息类型**: `auto_aim_interfaces/msg/Target`
- **QoS**: `rclcpp::SensorDataQoS()`
- **描述**: 接收装甲车辆目标跟踪信息
- **主要字段**:
  ```cpp
  std_msgs/Header header           # 时间戳和坐标系信息
  string id                        # 目标ID
  bool tracking                    # 是否正在跟踪
  geometry_msgs/Point position     # 目标中心位置 (odom坐标系)
  geometry_msgs/Vector3 velocity   # 目标线速度
  float64 yaw                      # 目标偏航角
  float64 v_yaw                    # 目标偏航角速度
  int32 armors_num                 # 装甲板数量 (3或4)
  float64 radius_1                 # 第一装甲板半径
  float64 radius_2                 # 第二装甲板半径  
  float64 dz                       # 装甲板高度差
  ```

#### `/tracker/rune_target`
- **消息类型**: `auto_aim_interfaces/msg/RuneTarget`  
- **QoS**: `rclcpp::SensorDataQoS()`
- **描述**: 接收能量机关目标信息
- **主要字段**:
  ```cpp
  std_msgs/Header header           # 时间戳和坐标系信息
  bool tracking                    # 是否正在跟踪
  geometry_msgs/Point center       # 能量机关中心位置
  float64 yaw                      # 偏航角
  float64 roll                     # 滚转角
  bool is_big                      # 是否为大能量机关
  int32 direction                  # 旋转方向 (1或-1)
  float64[] fitting_curve          # 拟合曲线参数 [a, ω, b, c, d]
  ```

#### `/camera_info`
- **消息类型**: `sensor_msgs/msg/CameraInfo`
- **QoS**: `rclcpp::SensorDataQoS()`  
- **描述**: 接收相机内参信息，用于3D点投影
- **用途**: 计算目标在图像平面的投影位置

### 发布话题

#### `/firecontrol`
- **消息类型**: `auto_aim_interfaces/msg/Firecontrol`
- **QoS**: 默认QoS (队列深度10)
- **描述**: 发布火控指令给底盘控制节点
- **主要字段**:
  ```cpp
  std_msgs/Header header           # 时间戳信息
  float64 pitch                    # 目标俯仰角 (弧度)
  float64 yaw                      # 目标偏航角 (弧度)
  bool tracking                    # 是否在跟踪目标
  string id                        # 目标ID
  bool iffire                      # 是否满足开火条件
  float64 projected_x              # 目标在图像中的X坐标
  float64 projected_y              # 目标在图像中的Y坐标
  ```

### TF变换

#### 监听的TF变换
- **`odom` → `gimbal_link`**: 
  - 用于获取当前云台位姿
  - 用于判断是否满足开火条件
  - 查找时机: `tf2::TimePointZero` (最新可用变换)

- **`odom` → `camera_link`**: 
  - 用于3D点到图像平面的投影
  - 结合相机内参计算目标在图像中的位置

### 服务接口

该节点当前不提供ROS服务接口。

### 动作接口

该节点当前不使用ROS动作接口。

### 参数接口

节点通过ROS参数服务器接收配置参数，支持运行时动态调整：

```bash
# 查看所有参数
ros2 param list /ballistic_calculation

# 获取参数值
ros2 param get /ballistic_calculation bullet_speed

# 设置参数值  
ros2 param set /ballistic_calculation bullet_speed 25.0
```

## 配置参数

### 弹道参数

```yaml
# 空气阻力系数
air_resistence: 0.1

# 子弹速度 (m/s)
bullet_speed: 23.0
```

### 迭代参数

```yaml
# 第一次迭代系数
iteration_coeffcient_first: 0.1

# 第二次迭代系数  
iteration_coeffcient_second: 0.05
```

### 火控策略参数

```yaml
# 开火角度阈值 (rad)
ifFireK: 0.05

# 一级策略切换阈值 (°/s)
switch_stategy_1: 5.0

# 二级策略切换阈值 (°/s)
switch_stategy_2: 30.0

# 云台最大角速度 (rad/s)
max_v_yaw_gimble: 0.8

# 停止开火时间 (s)
stop_fire_time: 0.1
```

### 坐标系参数

```yaml
# 枪口相对odom的位置 [x, y, z]
xyz: [0.0, 0.0, 0.0]

# 枪口相对odom的姿态 [roll, pitch, yaw]
rpy: [0.0, 0.0, 0.0]
```

## 编译与运行

### 依赖项

- ROS2 Humble/Foxy
- Eigen3
- OpenCV
- Ceres Solver
- angles
- tf2

### 编译

```bash
cd /path/to/your/ros2_ws
colcon build --packages-select ballistic_calculation
```

### 运行

```bash
# 启动弹道计算节点
ros2 run ballistic_calculation ballistic_calculation_node

# 或通过launch文件启动
ros2 launch ballistic_calculation ballistic_calculation.launch.py
```

## 调试与监控

### 查看话题

```bash
# 查看火控指令输出
ros2 topic echo /firecontrol

# 查看装甲车辆目标信息
ros2 topic echo /tracker/target
```

### 参数调整

```bash
# 动态调整参数
ros2 param set /ballistic_calculation bullet_speed 25.0
ros2 param set /ballistic_calculation air_resistence 0.12
```

### 性能监控

```bash
# 查看节点性能
ros2 run rqt_graph rqt_graph
ros2 run rqt_plot rqt_plot
```

## 维护与开发

### 代码结构

```
ballistic_calculation/
├── include/ballistic_calculation/
│   ├── aim_info.hpp           # 目标信息基类
│   ├── armor_selector.hpp     # 装甲板选择器
│   ├── ballistic_calculator.hpp # 弹道计算器
│   ├── ballistic_node.hpp     # 主节点
│   └── math_uitl.hpp          # 数学工具
├── src/
│   └── ballistic_node.cpp     # 主节点实现
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 许可证

Apache-2.0