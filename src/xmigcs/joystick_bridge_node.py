"""
独立 ROS2 节点：订阅手柄 Joy（默认 /sbus_data），复用 JoystickHumanoid 按键与平滑逻辑，
将指令发布为与 robot_interface._init_ros_highlevel_interfaces 一致的 /hric/robot/* 话题。
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import String

from xmigcs.common.joystick import JoystickHumanoid
from xmigcs.utils.xlog_utils import configure_xlog, xlog


class JoystickBridgeNode(Node):
    """手柄桥接：Joy -> cmd_vel / stand_cmd / fsm_state_cmd / fsm_resume_cmd"""

    def __init__(self) -> None:
        super().__init__('joystick_bridge_node')

        self.declare_parameter('joy_topic', '/sbus_data')
        self.declare_parameter('frame_id', '')

        joy_topic = self.get_parameter('joy_topic').get_parameter_value().string_value
        self._frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        qos = QoSProfile(depth=10)

        self._pub_cmd_vel = self.create_publisher(
            TwistStamped, '/hric/robot/cmd_vel', qos)
        self._pub_stand_cmd = self.create_publisher(
            TwistStamped, '/hric/robot/stand_cmd', qos)
        self._pub_fsm_state = self.create_publisher(
            String, '/hric/robot/fsm_state_cmd', qos)
        self._pub_fsm_resume = self.create_publisher(
            String, '/hric/robot/fsm_resume_cmd', qos)

        self._joystick = JoystickHumanoid()
        self._joystick.init()

        self.create_subscription(Joy, joy_topic, self._joy_callback, qos)

        self.get_logger().info(
            f'Joystick bridge: subscribe Joy on "{joy_topic}", '
            'publish /hric/robot/cmd_vel, stand_cmd, fsm_state_cmd, fsm_resume_cmd'
        )

    def _stamp_twist(self, msg: TwistStamped) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        if self._frame_id:
            msg.header.frame_id = self._frame_id

    def _joy_callback(self, msg: Joy) -> None:
        # 与原先 robot_interface.process_controller_data 中一致：读入映射再更新标志
        self._joystick.joy_map_read(msg)
        self._joystick.joy_flag_update()

        with self._joystick.data_mutex:
            flag = self._joystick.joy_flag

        # 未 enable：不发 cmd_vel / stand / resume；仅 gotoSTOP 时发 fsm_state_cmd
        if not flag.enable:
            if flag.fsm_state_command == "gotoSTOP":
                self._pub_fsm_state.publish(
                    String(data=str(flag.fsm_state_command)))
            return
        if flag.fsm_state_command in ['gotoHBWALK', 'gotoNAVIGATE',  'gotoSTART']:
            cmd_vel = TwistStamped()
            cmd_vel.twist.linear.x = float(flag.x_speed_command)
            cmd_vel.twist.linear.y = float(flag.y_speed_command)
            cmd_vel.twist.angular.z = float(flag.yaw_speed_command)
            self._stamp_twist(cmd_vel)
            self._pub_cmd_vel.publish(cmd_vel)
        if flag.fsm_state_command == "gotoSTAND":
            stand = TwistStamped()
            stand.twist.angular.x = float(flag.waist_roll_command)
            stand.twist.angular.y = float(flag.waist_pitch_command)
            stand.twist.angular.z = float(flag.waist_yaw_command)
            stand.twist.linear.z = float(flag.waist_height_command)
            self._stamp_twist(stand)
            self._pub_stand_cmd.publish(stand)

        self._pub_fsm_state.publish(String(data=str(flag.fsm_state_command)))
        self._pub_fsm_resume.publish(String(data=str(flag.fsm_resume_command)))


def main(args=None) -> None:
    rclpy.init(args=args)
    configure_xlog(logger_id="xmigcs_joystick_bridge")

    node = None
    try:
        node = JoystickBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
