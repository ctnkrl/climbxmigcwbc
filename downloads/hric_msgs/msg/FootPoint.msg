# 落脚点信息消息
# 用于发布机器人距离目标点1米左右时的落脚点信息

# 时间戳
builtin_interfaces/Time stamp

# 开关标志：距离目标点小于1米时为true，否则为false
bool footflag

# 相对坐标信息（机器人坐标系下）
# x坐标：机器人到目标点的相对x坐标
float64 relative_x
# y坐标：机器人到目标点的相对y坐标  
float64 relative_y
# 相对航向角：机器人到目标点的相对航向角（弧度）
float64 relative_yaw

# 距离信息
# 机器人与目标点的直线距离
float64 distance

# 坐标系信息
string frame_id
