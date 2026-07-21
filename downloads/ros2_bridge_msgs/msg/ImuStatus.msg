# ImuStatus.msg
# IMU 状态消息

std_msgs/Header header
uint8 label

# 四元数
float64 qx
float64 qy
float64 qz
float64 qw

# 欧拉角（度）
float64 roll
float64 pitch
float64 yaw

# 角速度 (rad/s)
float64 wx
float64 wy
float64 wz

# 线性加速度 (m/s^2)
float64 ax
float64 ay
float64 az

# 协方差矩阵（按行展开）
float64[9] orientation_covariance
float64[9] angular_velocity_covariance
float64[9] linear_acceleration_covariance

# 错误码
uint32 error

