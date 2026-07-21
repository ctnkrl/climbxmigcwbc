# xMIGCS

xMIGCS (X-humanoid Motion Intelligence Group Control System) 是一个用于机器人控制的软件系统，专注于通过有限状态机（FSM）和策略模块实现对机器人的灵活控制。项目服务于运动智能领域的研究与开发，支持多种控制模式和外部输入方式（如键盘、手柄等）。

## 功能特性

- **状态机管理**: 基于 FSM 模块实现机器人行为的状态流转控制
- **多策略支持**: 提供多种控制策略（如 dh, mlp, pbhc, zero 等）
- **人机交互控制**: 支持键盘、手柄等外设进行机器人实时操控
- **配置驱动**: 使用 YAML 文件进行参数配置，支持不同场景下的快速部署
- **模块化设计**: 各策略独立封装，便于扩展与维护

## 安装与运行

### 环境要求

- Python 3.12
- Git (用于版本控制)
- bodyctrl_msgs

### 安装步骤

#### 方法1：使用uv构建环境和编译、安装

```bash
cd your_project_folder
git clone https://git.x-humanoid-cloud.com/motion-intelligence-group/xmigcs.git .
cd xmigcs
```

##### 执行一步安装命令

```bash
./install.sh
```

##### 如遇报错，执行下面的分步安装

```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# 如遇网络问题，使用下面方式安装,注意不要下载到xmigcs目录下
uname -m # 查看系统架构,如x86_64，则执行
wget https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
#解压
tar -xzf uv-*.tar.gz
cd uv-*/
# 将 uv 可执行文件移动到系统目录
sudo mv uv /usr/local/bin/
#确保目标目录在 PATH 中
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc  # 如果用bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc  # 如果用zsh
# 创建环境
uv venv && uv sync --no-install-project
# 开发环境方式部署
source .venv/bin/activate
export XMIGCS_DEV=1
uv pip install -e .

# 安装脚踝串并联转换库
# ros jazzy
uv pip install downloads/sptlib_python-0.1.0-cp312-cp312-linux_x86_64.whl
# ros humble
uv pip install downloads/sptlib_python-0.1.0-cp310-cp310-linux_x86_64.whl
```

#### 安装 xlog

`install.sh` 会安装 xlog 预编译包。如果一键安装失败，请查看 [XLOG_USAGE.md](XLOG_USAGE.md) 来完成安装。

# 导航消息安装(xos < 5.1)
```bash
# 实机
tar -C ~ -xf downloads/robot_motion_control.tar.gz
cd ~/robot_motion_control
source /opt/ros/jazzy/setup.bash
colcon build
# 仿真
cp downloads/hric_msgs ~/robot_motion_control -r
cd ~/robot_motion_control
source /opt/ros/jazzy/setup.bash
colcon build
```

### 代码打包(ubuntu 24环境下执行)

```bash
# 生产环境，编译wheel包(仅需要构建wheel包才运行下面的代码)
cd xmigcs
source .venv/bin/activate
unset XMIGCS_DEV
rm -r build
rm -r dist
uv build --wheel
# 安装xMIGCS wheel包
#仿真
uv pip install dist/xmigcs-*.whl
#真机
pip show xmigcs
#如果上面有输出，就手动卸载
pip uninstall xmigcs --break-system-packages
# 将运控包拷贝到Ubuntu下,运行下面的指令安装包
pip install xmigcs-xx.whl(替换为拷贝的运控包) --break-system-packages
# 确认安装成功，查看版本
pip show xmigcs
```

### 运行项目

真机配置分 **两层**，各自独立选择，可自由组合：

#### ① 底层通信（本体 + xmigcs）— 二选一

两套起法**等价**，仅底层电机通信方式不同，**与 HRIC / 云卓桥接无关**。


|       | 本体（CPU 6–9，`sudo`）                                                          | xmigcs（CPU 0–5）                                                        |
| ----- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **A** | `chrt -r 99 taskset -c 6-9 ros2 launch body_control body_control.launch.py` | `chrt -r 1 taskset -c 0-5 xmigcs`                                      |
| **B** | `chrt -r 99 taskset -c 6-9 bash …/robot_control_xos.sh`（支持遥操）               | `chrt -r 1 taskset -c 0-5 xmigcs --ros-args -p comm_mode:=ros2_bridge` |


#### ② 高层收令 — 二选一

与上表 A/B **无关**，只决定状态/速度指令从哪来。


|                 | xmigcs 额外参数                         | 还需启动                                     |
| --------------- | ----------------------------------- | ---------------------------------------- |
| **云卓桥接**        | 无（`hric_server_enabled` 默认 `false`） | `joystick` + `xmigcs_joystick`           |
| **HRIC Server** | `-p hric_server_enabled:=true`      | HRIC 客户端发令；**不要**启桥接节点 `xmigcs_joystick` |


云卓桥接发布 `/hric/robot/cmd_vel` 等话题，主控订阅；HRIC 走 RPC（`dex_config.yaml` 中 `hric_server.comm_mode` 须与 `robot_control` 端 `--hric-comm-mode` 一致）。

### 真机运行

真机系统为 **Ubuntu 24**

#### 安装包方式启动运控

##### 方式一：自启

直接开启自启，并替换系统上的 xmigcs whl 安装包。适用于状态机、命令与系统默认遥控器节点**完全一致**的情况。

```bash
sudo systemctl enable proc_manager.service
sudo systemctl restart proc_manager.service
```

##### 方式二：手启 + 自定义桥接

替换 whl 后，若新增了系统遥控器未定义的状态，须关闭自启，再分终端手动启动（含运控自带的 `xmigcs_joystick`）。

```bash
# 启动本体
sudo su && source /home/ubuntu/xos/setup.bash
chrt -r 99 taskset -c 6-9 bash /home/ubuntu/xos/robot_control/share/scripts/robot_control_xos.sh

# 启动遥控器
sudo su && source /home/ubuntu/xos/setup.bash
ros2 launch joystick joystick.launch.py

# 启动运控
sudo su && source /home/ubuntu/xos/setup.bash
chrt -r 1 taskset -c 0-5 bash /home/ubuntu/xos/robot_control/share/scripts/run_xmigcs.sh

# 启动运控自带的遥控器桥接
source /home/ubuntu/xos/setup.bash
xmigcs_joystick
```

#### 源码启动运控

开发调试时逐终端手动启动，各进程需自行绑核。

**手起前须先关闭自启**：

```bash
sudo systemctl disable proc_manager.service
sudo systemctl stop proc_manager.service
```

恢复自启：`sudo systemctl enable proc_manager.service` && `sudo systemctl start proc_manager.service`。

**完整手起示例**（先选底层 A 或 B，再起高层云卓）：

```bash
# ── ① 底层（二选一）──

# A：body_control + 默认 xmigcs
sudo su && source /home/ubuntu/xos/setup.bash
chrt -r 99 taskset -c 6-9 ros2 launch body_control body_control.launch.py
# 另开终端：
source /home/ubuntu/xos/setup.bash
cd xmigcs && source .venv/bin/activate
chrt -r 1 taskset -c 0-5 xmigcs

# B：robot_control_xos + ros2_bridge xmigcs(支持遥操作的通信方式)
sudo su && source /home/ubuntu/xos/setup.bash
chrt -r 99 taskset -c 6-9 bash /home/ubuntu/xos/robot_control/share/scripts/robot_control_xos.sh
# 另开终端：
source /home/ubuntu/xos/setup.bash
cd xmigcs && source .venv/bin/activate
chrt -r 1 taskset -c 0-5 xmigcs --ros-args -p comm_mode:=ros2_bridge

# ── ② 高层：云卓桥接（与 A/B 无关，另开终端）──
sudo su && source /home/ubuntu/xos/setup.bash
ros2 launch joystick joystick.launch.py
# 再另开终端：
source /home/ubuntu/xos/setup.bash
cd xmigcs && source .venv/bin/activate
xmigcs_joystick

```

### 仿真运行

仿真环境可为 **Ubuntu 22 或 24**，**均为源码 venv 开发**（`cd xmigcs && source .venv/bin/activate`）。

#### 云卓手柄 + 桥接节点

```bash
# 1. 仿真主控（另开终端；不要加 hric_server_enabled:=true）
export ROS_DOMAIN_ID=YOUR_DOMAIN_ID
source /opt/ros/humble/setup.bash   # Ubuntu 22；24 请 source 本机 ROS 环境
cd xmigcs && source .venv/bin/activate
xmigcs_sim

# 2. SBUS → /sbus_data（若仿真环境未自带 Joy 源）
export ROS_DOMAIN_ID=YOUR_DOMAIN_ID
source ros2ws/install/setup.bash
ros2 launch usb_sbus usb_sbus.launch.py

# 3. 云卓键位映射桥接（再另开终端）
export ROS_DOMAIN_ID=YOUR_DOMAIN_ID
# ubuntu22
source /opt/ros/humble/setup.bash   
cd xmigcs && source .venv/bin/activate
python3 src/xmigcs/joystick_bridge_node.py
# ubuntu24
source /opt/ros/jazzy/setup.bash   # Ubuntu 22；24 请 source 本机 ROS 环境
cd xmigcs && source .venv/bin/activate
xmigcs_joystick
```

## 导航信息安装

```bash
# 编译前先退出 conda，避免 colcon 误用 conda 的 python3
conda deactivate  # 如果提示 CommandNotFound，可忽略
source /opt/ros/humble/setup.bash

# 创建导航消息文件夹
mkdir -p ~/ros2ws_nav/src

# 将hric_msgs放入到src中

#然后编译
cd ~/ros2ws_nav
colcon build

source ~/ros2ws_nav/install/setup.bash
```

## 配置文件说明

xMIGCS 将**运控核心参数**与**人机交互映射**拆分为两个 YAML 文件，修改后需重启对应节点生效。


| 文件                  | 路径                                    | 用途                                                       |
| ------------------- | ------------------------------------- | -------------------------------------------------------- |
| `dex_config.yaml`   | `src/xmigcs/config/dex_config.yaml`   | 主控节点：FSM 状态/转移、关节限位、通讯方式、速度限制、`moe_state_commands_map` 等 |
| `control_tool.yaml` | `src/xmigcs/config/control_tool.yaml` | 外设控制：云卓手柄 / 键盘 / Xbox 的按键映射、摇杆速度、长按阈值等                   |


### dex_config.yaml（主控）

常用段落：

- `**states` / `transitions`**：FSM 状态编号与状态转移表
- `**robot_interface.cmd_limits`**：各状态下的行走速度上限（与手柄 `state_speed_limits` 对应）
- `**robot_interface.moe_state_commands_map`**：FSM 命令写入 `robot_data` 的映射，例如 STAND/NAVIGATE MOE 子模式：

```yaml
moe_state_commands_map:
  gotoSTAND:
    nav_stand_mode: 0
  gotoNAVIGATE:
    nav_stand_mode: 1
```

### control_tool.yaml（外设）

顶层 `control_tool` 字段选择当前外设类型：`joystick` / `xbox` / `keyboard`。

云卓手柄（`joystick` 段）核心配置分为三块：


| 配置项                   | 作用                                      |
| --------------------- | --------------------------------------- |
| `fsm_command_rules`   | **按键 → 切 FSM 命令**（只负责改 `command`）       |
| `fsm_command_actions` | **命令 → 摇杆副作用**（每帧按当前 `command` 更新速度/姿态） |
| `fsm_stop_rule`       | 全局急停，匹配后覆盖 `command`                    |


`fsm_command_rules` 规则字段说明：

- `when.switch_count`：e/f/g/h 中非零拨杆的数量
- `when.buttons`：按键/拨杆期望值（`1.0` 按下，`−1.0` 松开/拨动，`0.0` 中位）
- `when.press`：短按 `short` / 长按 `long`（阈值见 `long_press_thresholds`）
- `when.press_count`：有效连按次数（见 `press_counters`）
- `command`：发出的 FSM 命令，如 `gotoNAVIGATE`
- `reset_press_count`：触发后清零的按键计数

`fsm_command_actions` 可用 action：


| action              | 说明                                      |
| ------------------- | --------------------------------------- |
| `x_y_yaw_speed`     | 普通行走速度（HBWALK / NAVIGATE）               |
| `swr_x_y_yaw_speed` | SWR 走跑渐变速度（`gotoSTART`）                 |
| `rpyz`              | 站立姿态 roll/pitch/yaw/height（`gotoSTAND`） |


新增状态时，在 `fsm_command_rules` 增加切状态规则，并在 `fsm_command_actions` 中挂上对应 action 即可，无需改 Python 代码。

## 控制器使用说明

### 云卓手柄桥接节点

订阅 `/sbus_data`，发布：


| 话题                           | 类型                           | 说明                            |
| ---------------------------- | ---------------------------- | ----------------------------- |
| `/hric/robot/fsm_state_cmd`  | `std_msgs/String`            | FSM 状态切换命令                    |
| `/hric/robot/cmd_vel`        | `geometry_msgs/TwistStamped` | 行走速度（HBWALK / NAVIGATE / SWR） |
| `/hric/robot/stand_cmd`      | `geometry_msgs/TwistStamped` | 站立姿态指令                        |
| `/hric/robot/fsm_resume_cmd` | `std_msgs/String`            | 单次恢复命令                        |


数据流（桥接模式）：

```
/sbus_data → joystick_bridge → /hric/robot/* → xmigcs（订阅话题，hric_server_enabled=false）
```

### 云卓 T12 手柄键位映射

使用前请将所有拨杆回中、按键松开。手柄按键在代码中对应 `YUNZHUOMap`：`a/b/c/d` 为按键，`e/f/g/h` 为三档拨杆，`x1/y1/x2/y2` 为左右摇杆轴。

#### 使能开关


| 拨杆  | 位置    | 效果                                    |
| --- | ----- | ------------------------------------- |
| e   | 上拨    | 禁用行走/站立控制（`enable=false`）；c 长按/双击仍可急停 |
| e   | 中位/下拨 | 正常控制                                  |


#### 状态切换（`fsm_command_rules`）

**三档均回中（`switch_count: 0`）**


| 操作   | FSM 命令      | 说明           |
| ---- | ----------- | ------------ |
| 按 d  | `gotoZERO`  | 回零           |
| 长按 a | `gotoSTART` | 进入 SWR（走跑模式） |


**一个拨杆离中（`switch_count: 1`）**
**两个拨杆离中（`switch_count: 2`）**
**任意时刻**


| 操作                 | FSM 命令     | 说明  |
| ------------------ | ---------- | --- |
| c 长按，或 c 有效连按 ≥2 次 | `gotoSTOP` | 急停  |


#### 摇杆控制（`fsm_command_actions`）

进入对应状态后，每帧根据当前 `command` 更新摇杆指令（无需持续按住切状态键）：


| 当前命令             | 摇杆功能                                    |
| ---------------- | --------------------------------------- |
| `gotoSTART`（SWR） | 左摇杆 Y1 前进/后退（走跑渐变），X1 横移，右摇杆 X2 旋转      |
| `gotoHBWALK`     | 左摇杆 Y1 前后，X1 横移，右摇杆 X2 旋转               |
| `gotoNAVIGATE`   | 同上                                      |
| `gotoSTAND`      | 右摇杆 X1/Y1 控制 roll/pitch，左摇杆 X2 偏航、Y2 高度 |


各状态速度上限在 `control_tool.yaml` 的 `state_speed_limits` 中配置（如 `SWR`、`HBWALK`、`NAVIGATE`）。

#### 自定义按键映射

编辑 `src/xmigcs/config/control_tool.yaml`：

```yaml
# 1. 增加切状态规则
fsm_command_rules:
  - when: {switch_count: 1, buttons: {h: 1.0, a: 1.0}, press_count: {b: 5}}
    command: gotoMYNEWSTATE
    reset_press_count: [b]

# 2. 若新状态需要摇杆控制，挂上 action
fsm_command_actions:
  gotoMYNEWSTATE: [x_y_yaw_speed]
```

同时需在 `dex_config.yaml` 的 `transitions` 中注册 `gotoMYNEWSTATE` 转移边。

### XBOX 手柄键位映射

```bash
# 启动XBOX手柄数据节点
export ROS_DOMAIN_ID=YOUR_DOMAIN_ID
source /opt/ros/jazzy/setup.bash
ros2 run joy joy_node --ros-args --remap joy:=xbox_data
```

xMIGCS支持标准XBOX手柄控制，以下是详细键位映射关系：

#### 状态映射关系

##### 单按钮状态切换


| 按钮  | 对应状态     | 功能说明   |
| --- | -------- | ------ |
| X   | gotoZERO | 回到零位状态 |
| Y   | gotoSTOP | 停止状态   |


##### 基础运动控制


| 控制方式  | 功能            |
| ----- | ------------- |
| 左摇杆Y轴 | 前后移动控制（正向为前进） |
| 左摇杆X轴 | 左右移动控制        |
| 右摇杆X轴 | 机身旋转控制        |


> Xbox / 键盘参数见 `control_tool.yaml` 中 `xbox`、`keyboard` 段；映射逻辑在 `common/xbox_control.py`、`common/stdin_keyboard_control.py`。

## 项目结构

```
.
├── src/xmigcs/                     # 核心源码
│   ├── FSM/                        # 状态机框架
│   │   ├── fsm_base.py             # FSM 基类与状态名定义
│   │   └── robot_fsm.py            # 状态注册、转移与调度
│   ├── common/                     # 通用模块
│   │   ├── robot_data.py           # 机器人运行时数据
│   │   ├── robot_interface.py      # 与底层通信、FSM 命令解析
│   │   ├── joystick.py             # 云卓 T12 键位映射（读 control_tool.yaml）
│   │   ├── hric_server_adapter.py  # HRIC Server RPC 适配
│   │   ├── xbox_control.py         # Xbox 手柄控制
│   │   └── stdin_keyboard_control.py
│   ├── config/                     # 配置文件
│   │   ├── dex_config.yaml         # FSM 状态/转移、通讯、限位、moe_state_commands_map
│   │   ├── control_tool.yaml       # 云卓/Xbox/键盘按键与速度映射
│   │   └── tiangong3.urdf          # 机器人 URDF
│   ├── policy/                     # 各策略 FSM 实现（每目录一种行为）
│   │   ├── navigate/               # 例：fsm_navigate.py + config/ + agent/ + model/
│   │   ├── hbwalk/
│   │   ├── stand_nav_moe/
│   │   └── ...                     # stop、damping、swr、kneeldown 等
│   ├── utils/                      # 滤波、日志、对称矩阵等工具
│   ├── rl_control_node.py          # 真机主控入口（console: xmigcs）
│   ├── rl_control_node_sim.py      # 仿真主控入口（console: xmigcs_sim）
│   └── joystick_bridge_node.py     # 云卓桥接入口（console: xmigcs_joystick）
├── scripts/                        # 辅助脚本
│   ├── nav_target_pub.py           # 导航目标发布
│   └── detect_xbox_buttons.py      # Xbox 按键探测
├── test/                           # 测试与对比脚本
├── docs/
│   └── dex_fsm_transitions.svg     # FSM 转移图
├── logs/                           # 运行时日志（gitignore）
├── install.sh                      # 开发环境一键安装（uv + venv）
├── install_wheel_real.sh           # 真机 whl 安装
├── publish_release.sh              # 打包发布 whl
├── startup_nav_real.sh             # 真机 tmux 手启示例
├── startup_nav_sim.sh              # 仿真 tmux 手启示例
├── pyproject.toml                  # 项目配置与 console 入口定义
├── setup.py
├── requirements.txt
├── uv.lock
└── README.md
```

**入口命令**（`pyproject.toml` → `[project.scripts]`）：

| 命令 | 模块 | 用途 |
| --- | --- | --- |
| `xmigcs` | `rl_control_node` | 真机主控 |
| `xmigcs_sim` | `rl_control_node_sim` | 仿真主控 |
| `xmigcs_joystick` | `joystick_bridge_node` | 云卓手柄桥接 |

## 如何添加新的控制策略

1. 在 policy 目录下创建新的策略文件夹，例如 dance
2. 在新文件夹中创建以下文件：
  - fsm_dance.py - 实现具体的FSM状态类
  - config/dance.yaml - 策略配置文件（可选）
3. 在 fsm_dance.py 中实现 FSMState 类：

```python
    from xmigcs.FSM.fsm_base import FSMState, FSMStateName, ControlFlag
    from xmigcs.common.robot_data import RobotData

    class FSMStateDANCE(FSMState):
        def __init__(self, robot_data: RobotData):
            super().__init__(robot_data)
            # 初始化策略特定变量

        def on_enter(self):
            # 进入状态时的初始化操作
            pass

        def run(self, flag: ControlFlag):
            # 策略的主要运行逻辑
            pass

        def on_exit(self):
            # 退出状态时的清理操作
            pass

        def check_transition(self, *args, **kwargs):
            # 检查是否需要转换到其他状态
            pass
```

1. 在 `config/dex_config.yaml` 中注册新状态与转移：

```yaml
states:
  DANCE: <新编号>

transitions:
  gotoDANCE:
    - {from: DANCE, to: DANCE}
    - {from: SWR, to: DANCE}
```

1. 若需手柄触发，在 `config/control_tool.yaml` 的 `fsm_command_rules` / `fsm_command_actions` 中添加映射（参见「云卓 T12 手柄键位映射 → 自定义按键映射」）

## 开发与贡献

欢迎对项目进行贡献，开发前请确保：

1. 遵循项目代码规范
2. 添加适当的测试用例
3. 提交前运行所有测试确保无误

## 许可证

本项目仅供内部使用。

## 项目状态

项目正在积极开发中。