# MotorCtrl.msg
# 电机控制消息

uint16 name
float64 kp      # 比例增益（力位混合模式使用）
float64 kd      # 微分增益（力位混合模式使用）
float64 pos     # 期望位置 (rad)（位置模式和力位混合模式使用）
float64 spd     # 期望速度 (rad/s)（位置模式、速度模式和力位混合模式使用）
float64 tor     # 前馈力矩（力位混合模式使用）
float64 cur     # 最大电流 (A)（位置模式和速度模式使用）
string joint_ids  # 关节ID字符串，逗号分隔，如 "1,2,3"（标零模式使用）

