import sys
import argparse
import math
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from hric_msgs.msg import FootPoint
from xmigcs.utils.tool_functions import wrap_to_pi


class Navigate_Global_Target_Publisher(Node):
    """
    全局导航目标发布节点

    订阅 robot/abs_position (TwistStamped) 获取机器人在世界坐标系下的实时位姿 (x, y, heading) \\
    根据启动参数 --target x,y,heading 指定的全局目标点, \\
    计算出机器人坐标系下的相对坐标 (relative_x, relative_y, relative_yaw), \\
    并通过 /hric/robot/footpoint 话题发布 FootPoint 消息。
    """
    def __init__(
        self,
        target_x: float,
        target_y: float,
        target_heading: float,
        ref_walk_speed: float = 0.3,
        ref_rotate_speed: float = 0.5
    ):
        super().__init__("nav_gcs_target_publisher")

        self.target_x_gcs = target_x
        self.target_y_gcs = target_y
        self.target_heading_gcs = target_heading
        self.ref_walk_speed = ref_walk_speed
        self.ref_rotate_speed = ref_rotate_speed

        # 缓存：heading 从 waist_pitch_link 获取，x/y 从 robot/abs_position 获取
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.current_heading: float = 0.0
        self.position_received = False
        self.heading_received = False

        # 订阅机器人(pelvis)绝对位姿
        self.sub_pelvis = self.create_subscription(
            TwistStamped,
            "robot/abs_position",
            self._abs_position_callback,
            10,
        )

        # 订阅trunk绝对位姿
        self.sub_trunk = self.create_subscription(
            TwistStamped,
            "waist_pitch_link/abs_position",
            self._heading_callback,
            10,
        )

        self.pub_footpoint = self.create_publisher(FootPoint, "/hric/robot/footpoint", 10)
        self.pub_cmd_vel = self.create_publisher(TwistStamped, "/hric/robot/cmd_vel", 10)

        # 用于计算实际速度的滑动窗口
        self.pose_history: deque[tuple[float, float, float, int]] = deque(maxlen=100)

        # 定时器：周期性执行发布逻辑
        self.publish_timer = self.create_timer(1.0 / 50.0, self.publish)

        self.get_logger().info(
            f"NavTargetPublisher 启动: "
            f"target=({target_x:.3f}, {target_y:.3f}, {target_heading:.3f} rad), "
            f"walk_speed={ref_walk_speed:.2f}, rotate_speed={ref_rotate_speed:.2f}"
        )

    def calculate_velocity(
        self, current_x: float, current_y: float,
        current_heading: float, current_time_ns: int
    ) -> tuple[float, float, float]:
        """
        计算实际速度（机器人坐标系）。
        返回 (vx_body, vy_body, v_yaw)。
        """
        self.pose_history.append((current_x, current_y, current_heading, current_time_ns))

        if len(self.pose_history) < 2:
            return 0.0, 0.0, 0.0

        first_x, first_y, first_heading, first_time = self.pose_history[0]

        dt = (current_time_ns - first_time) * 1e-9
        if dt <= 0.0:
            return 0.0, 0.0, 0.0

        # 全局系速度
        vx_global = (current_x - first_x) / dt
        vy_global = (current_y - first_y) / dt
        v_yaw = wrap_to_pi(current_heading - first_heading) / dt

        # 转换到机器人坐标系
        cos_h = math.cos(current_heading)
        sin_h = math.sin(current_heading)
        vx_body = vx_global * cos_h + vy_global * sin_h
        vy_body = -vx_global * sin_h + vy_global * cos_h

        return vx_body, vy_body, v_yaw

    def _heading_callback(self, msg: TwistStamped) -> None:
        """heading从trunk获取"""
        # self.current_x = msg.twist.linear.x
        # self.current_y = msg.twist.linear.y
        self.current_heading = msg.twist.angular.z
        self.heading_received = True

    def _abs_position_callback(self, msg: TwistStamped) -> None:
        """xy座标从pelvis获取"""
        self.current_x = msg.twist.linear.x
        self.current_y = msg.twist.linear.y
        # self.current_heading = msg.twist.angular.z
        self.position_received = True

    def publish(self) -> None:
        if not (self.position_received and self.heading_received):
            return

        current_x_gcs = self.current_x
        current_y_gcs = self.current_y
        current_heading_gcs = self.current_heading

        # position
        dx = self.target_x_gcs - current_x_gcs
        dy = self.target_y_gcs - current_y_gcs
        cos_h = math.cos(current_heading_gcs)
        sin_h = math.sin(current_heading_gcs)
        x_bcs = dx * cos_h + dy * sin_h
        y_bcs = -dx * sin_h + dy * cos_h

        # heading
        heading_bcs = wrap_to_pi(self.target_heading_gcs - current_heading_gcs)
        heading_bcs_deg = heading_bcs * 180 / math.pi

        # 计算实际速度
        now_time = self.get_clock().now()
        current_time_ns = now_time.nanoseconds
        vx, vy, vyaw = self.calculate_velocity(current_x_gcs, current_y_gcs, current_heading_gcs, current_time_ns)

        print(
            f"rel_pos=({x_bcs:+.3f}, {y_bcs:+.3f}, {heading_bcs_deg:+.1f}deg) | "
            f"vel=(vx={vx:+.3f}, vy={vy:+.3f}, vyaw={vyaw:+.3f})"
        )

        now = self.get_clock().now().to_msg()

        # 发布 FootPoint (相对目标坐标)
        fp = FootPoint()
        fp.footflag = True
        fp.relative_x = x_bcs
        fp.relative_y = y_bcs
        fp.relative_yaw = heading_bcs
        fp.stamp = now
        self.pub_footpoint.publish(fp)

        # 发布参考速度 (cmd_vel: x=walk_speed, y=0, yaw=rotate_speed)
        vel = TwistStamped()
        vel.header.stamp = now
        vel.twist.linear.x = float(self.ref_walk_speed)
        vel.twist.linear.y = 0.0
        vel.twist.angular.z = float(self.ref_rotate_speed)
        self.pub_cmd_vel.publish(vel)


def main():
    parser = argparse.ArgumentParser(
        description="全局导航目标发布节点：将全局目标转换为机器人系相对坐标并发布 FootPoint。"
    )
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--ref_walk_speed", type=float, required=True)
    parser.add_argument("--ref_rotate_speed", type=float, required=True)

    args = parser.parse_args()

    # 解析 target 参数
    parts = args.target.split(",")
    assert len(parts) == 3

    target_x = float(parts[0].strip())
    target_y = float(parts[1].strip())
    target_heading = float(parts[2].strip())

    rclpy.init()

    node = Navigate_Global_Target_Publisher(
        target_x,
        target_y,
        target_heading,
        args.ref_walk_speed,
        args.ref_rotate_speed,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("收到中断信号，正在退出...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0

if __name__ == "__main__":
    sys.exit(main())

    # source /opt/ros/humble/setup.bash
    # source /home/eai/allCode/ros2ws/install/setup.bash
    # python scripts/nav_target_pub.py --target 1.0,2.0,0.5
