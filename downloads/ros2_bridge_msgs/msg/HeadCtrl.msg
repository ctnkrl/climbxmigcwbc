# HeadCtrl.msg
# 头部关节控制消息

std_msgs/Header header
uint8 mode  # 控制模式: 0=位置模式, 1=力位混合模式, 2=速度模式, 3=标零模式, 4=距离模式, 5=电流模式
uint8 label
ros2_bridge_msgs/MotorCtrl[] ctrl

