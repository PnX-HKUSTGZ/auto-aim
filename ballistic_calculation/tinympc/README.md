## TinyMPC

### 核心求解器模块（ADMM实现）
- admm.cpp ADMM核心算法实现
- admm.hpp
    - 求解器核心函数：solve、backward_pass_grad、forward_pass 等；
    - 投影工具函数：project_soc、project_hyperplane；
    - 约束处理函数：update_slack、update_dual、update_linear_cost

### 数据类型与常量定义
- types.hpp 核心数据结构定义

| 结构体名        | 功能描述       | 核心成员                                                     |
| --------------- | -------------- | ------------------------------------------------------------ |
| `TinySolution`  | 求解结果存储   | `iter`（迭代次数）、`solved`（收敛标志）、`x`（状态轨迹）、`u`（控制输入）。 |
| `TinyCache`     | 预计算矩阵缓存 | 存储 LQR 相关预计算矩阵（`Kinf`、`Pinf`、`Quu_inv`）、`rho`（惩罚参数），以及自适应 `rho` 所需的灵敏度矩阵（`dKinf_drho`、`dPinf_drho`）。 |
| `TinySettings`  | 求解器配置     | 收敛阈值（`abs_pri_tol`/`abs_dua_tol`）、最大迭代次数（`max_iter`）、约束使能开关（`en_state_bound`/`en_state_soc`）、自适应 `rho` 配置（`adaptive_rho`、`rho_min`/`rho_max`）。 |
| `TinyWorkspace` | 求解器工作空间 | 存储实时计算的变量：- 状态 / 输入（`x`/`u`）、协态 / 前馈项（`p`/`d`）；- 松弛变量（`v`/`vnew`/`zc`/`zcnew`）、对偶变量（`g`/`y`/`gc`/`yc`）；- 系统参数（`Adyn`/`Bdyn`/`fdyn`）、成本矩阵（`Q`/`R`）、参考轨迹（`Xref`/`Uref`）；- 约束相关参数（`x_min`/`x_max`、`numStateCones`、`Alin_x`/`blin_x`）。 |
| `TinySolver`    | 求解器总入口   | 聚合 `TinySolution`（结果）、`TinySettings`（配置）、`TinyCache`（缓存）、`TinyWorkspace`（工作空间），是外部调用的唯一入口。 |

tinytype：默认 double，确保数值精度，可按需改为 float（嵌入式场景）；

tinyMatrix/tinyVector：基于 Eigen 的动态尺寸矩阵 / 向量，适配任意状态 / 输入维度

- tiny_api_constants.hpp 默认配置常量

| 常量名                        | 默认值 | 功能                   |
| ----------------------------- | ------ | ---------------------- |
| `TINY_DEFAULT_ABS_PRI_TOL`    | `1e-3` | 原始残差收敛阈值       |
| `TINY_DEFAULT_MAX_ITER`       | `1000` | 最大迭代次数           |
| `TINY_DEFAULT_EN_STATE_BOUND` | `1`    | 默认启用状态边界约束   |
| `TINY_DEFAULT_EN_STATE_SOC`   | `0`    | 默认禁用状态二阶锥约束 |

### API 接口模块（用户交互层）
- tiny_api.cpp/hpp 用户友好型API实现
封装核心求解逻辑，提供简洁的用户接口，包括求解器初始化、约束配置、参数更新等，隐藏底层实现细节

### 自适应 Rho 模块（动态惩罚参数）
- rho_benchmark.cpp/hpp 自适应 Rho 实现

### 代码生成模块（嵌入式部署）
- codegen.cpp/hpp
将 TinyMPC 求解器生成为可直接编译的嵌入式代码（C/C++），避免在嵌入式系统中动态分配内存，适配资源受限环境

### 辅助模块
- error.hpp 错误处理
