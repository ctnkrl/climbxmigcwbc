
#!/usr/bin/env bash


set -euo pipefail

TARGET_DOMAIN_ID=11
# PASSWD=123
PASSWD=dionamuhx

SESSION_NAME="rl_sim"
WINDOW_NAME="rl_run"
TARGET_WINDOW="${SESSION_NAME}:${WINDOW_NAME}"


if ! command -v tmux >/dev/null 2>&1; then
    echo "错误: 系统未安装 tmux。"
    exit 1
fi


tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

echo "正在启动项目 (ROS_DOMAIN_ID=11)..."


# Pane 1: Body Control Launch
# tmux new-session -d -s "${SESSION_NAME}" -n "${WINDOW_NAME}" bash -lc "
#   echo 123 | sudo -S su -c '
#     export ROS_DOMAIN_ID=11
#     source /opt/ros/humble/setup.bash
#     source /home/ubuntu/ros2ws_nav/install/setup.bash
#     ros2 launch body_control body.launch.py
#   '
#   exec bash
# "
tmux new-session -d -s "${SESSION_NAME}" -n "${WINDOW_NAME}" bash -lc "
  # 1. 先在普通用户环境导出关键变量，确保 sudo 能继承
  #export ROS_DOMAIN_ID=${TARGET_DOMAIN_ID}
  export HOME=/home/ubuntu  # 强制指定普通用户 HOME，避免 root 环境路径错乱
  # 2. 使用 sudo -E 保留环境变量，移除多余的 su 嵌套
  #    密码通过管道传递仍保留，但解决环境变量丢失问题
  echo ${PASSWD} | sudo -S -E bash -c '
    source /home/ubuntu/xos/setup.bash
    # 可选：调试用，验证环境变量是否正确加载
    # echo \"COLCON路径: \$COLCON_PREFIX_PATH\"
    # echo \"ROS域ID: \$ROS_DOMAIN_ID\"
    # chrt -r 99 taskset -c 6-9 ros2 launch body_control body_control.launch.py
    chrt -r 99 taskset -c 6-9 bash /home/ubuntu/xos/robot_control/share/scripts/robot_control_xos.sh
  '
  exec bash
"

# Pane 2: 云卓手柄
tmux split-window -t "${TARGET_WINDOW}" -h bash -lc "
  export HOME=/home/ubuntu  # 强制指定普通用户 HOME，避免 root 环境路径错乱
  # 2. 使用 sudo -E 保留环境变量，移除多余的 su 嵌套
  #    密码通过管道传递仍保留，但解决环境变量丢失问题
  echo ${PASSWD} | sudo -S -E bash -c '
    source /home/ubuntu/xos/setup.bash
    ros2 launch joystick joystick.launch.py
	'
  exec bash # 保持窗格打开
"

# Pane 3: RL Control Node
tmux split-window -t "${TARGET_WINDOW}" -h bash -lc "
  #export ROS_DOMAIN_ID=${TARGET_DOMAIN_ID}
	# 1. 先在普通用户环境导出关键变量，确保 sudo 能继承
  #export ROS_DOMAIN_ID=${TARGET_DOMAIN_ID}
  export HOME=/home/ubuntu  # 强制指定普通用户 HOME，避免 root 环境路径错乱
  # 2. 使用 sudo -E 保留环境变量，移除多余的 su 嵌套
  #    密码通过管道传递仍保留，但解决环境变量丢失问题
  echo ${PASSWD} | sudo -S -E bash -c '
  	source /home/ubuntu/xos/setup.bash
    source /home/ubuntu/robot_motion_control/install/setup.bash
  	sleep 3
  	cd /home/ubuntu/XMIGCS_DEPLOY/xmigcs || exit 1
  	source .venv/bin/activate
    export XMIGCS_LOG_LEVEL=DEBUG
  	chrt -r 1 taskset -c 0-5 xmigcs --ros-args -p comm_mode:=ros2_bridge
	'
  exec bash
"


tmux select-layout -t "${TARGET_WINDOW}" tiled
tmux attach-session -t "${SESSION_NAME}"
