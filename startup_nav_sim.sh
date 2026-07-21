#!/usr/bin/env bash

# 设置bash严格模式：遇到错误立即退出，使用未定义变量报错，管道中任一命令失败则整个管道失败
set -euo pipefail

# NOTE: 使用前修改此处设置
ROS_SETUP="/opt/ros/humble/setup.bash"
NAV_WS="/home/eai/allCode/ros2ws"
SIM_DIR="/home/eai/allCode/xsim_mujoco"
CTRL_DIR="/home/eai/allCode/xmigcs"
open_nav_pub=false  # 是否启动 nav_pub

# tmux name
SESSION_NAME="rl_sim"
WINDOW_NAME="rl_run"
TARGET_WINDOW="${SESSION_NAME}:${WINDOW_NAME}"

export ROS_LOCALHOST_ONLY=1  # 只收发本地消息

# 清理已有的tmux会话
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

# Pane 1: MuJoCo Simulator
tmux new-session -d -s "${SESSION_NAME}" -n "${WINDOW_NAME}" bash -lc "
  source ${ROS_SETUP}
  source ${NAV_WS}/install/setup.bash
  cd ${SIM_DIR} || exit 1
  # /usr/bin/python3 scripts/simulator_view_asyn.py -m dex_evt_hand
  /usr/bin/python3 scripts/simulator_view_asyn.py -m tg3
  exec bash # 保持窗格打开
"

# Pane 2: 云卓手柄
tmux split-window -t "${TARGET_WINDOW}" -h bash -lc "
  source ${NAV_WS}/install/setup.bash
  ros2 launch usb_sbus usb_sbus.launch.py
  exec bash # 保持窗格打开
"

# Pane 3: RL Control Node
tmux split-window -t "${TARGET_WINDOW}" -v bash -lc "
  source ${ROS_SETUP}
  source ${NAV_WS}/install/setup.bash
  export XMIGCS_LOG_LEVEL=DEBUG
  sleep 3
  cd ${CTRL_DIR} || exit 1
  source .venv/bin/activate
  xmigcs_sim
  exec bash # 保持窗格打开
"

# Pane 4: nav_pub
if [ "${open_nav_pub}" = true ]; then
  tmux split-window -t "${TARGET_WINDOW}" -v bash -lc "
    source ${ROS_SETUP}
    source ${NAV_WS}/install/setup.bash
    cd ${CTRL_DIR}
    source .venv/bin/activate
    python scripts/nav_target_pub.py --target=8.0,0.0,0.0 --ref_walk_speed 0.8 --ref_rotate_speed 1.0
    exec bash # 保持窗格打开
  "
fi

# Pane 5: xmigcs_joystick_bridge
tmux split-window -t "${TARGET_WINDOW}" -v bash -lc "
  source ${ROS_SETUP}
  source .venv/bin/activate
  python src/xmigcs/joystick_bridge_node.py
  exec bash # 保持窗格打开
"

# 调整窗格布局为平铺模式（自动调整大小）
tmux select-layout -t "${TARGET_WINDOW}" tiled
# 附加到tmux会话（用户可以看到所有窗格）
tmux attach-session -t "${SESSION_NAME}"
