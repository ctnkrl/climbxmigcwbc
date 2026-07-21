"""
Robot Interface
Python equivalent of the C++ RobotInterface class
"""
from __future__ import annotations
import queue
from xmigcs.common.peekqueue import PeekableQueue
import yaml
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import numpy as np
# ROS messages
# from bodyctrl_msgs.msg import MotorStatusMsg, CmdMotorCtrl, MotorCtrl, Imu, MotorStatus
import rclpy
from std_msgs.msg import String
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# TF2 imports for transformations
from tf2_ros import TransformListener, Buffer
import tf2_ros
import transforms3d as t3d
from xmigcs.common.body_id_map import BodyServoIdMap
from xmigcs.common.robot_data import RobotData
import functools
import time
import math
# from common.func_sp_trans_dex_opt import OptimizedFuncSPTrans as FuncSPTrans
# from xmigcs.common.func_sp_trans_evt_opt import OptimizedFuncSPTrans as FuncSPTrans
# from sptlib_python import funcSPTrans as FuncSPTrans
from xmigcs.common.func_sp_trans_evt_analytic import AnalyticFuncSPTrans as FuncSPTrans

from xmigcs.common.control_flag import FSMControlFlag
from geometry_msgs.msg import TwistStamped
from xmigcs.FSM.fsm_base import FSMStateName, FSMState
# from hric_msgs.msg import FootPoint
from xmigcs.common.dynamic_model import DynamicModel
from diagnostic_msgs.msg import KeyValue
from diagnostic_msgs.msg import DiagnosticStatus
from xmigcs.utils.xlog_utils import xlog
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

try:
    from sensor_msgs.msg import PointCloud2, PointField
    _point_cloud2_available = True
except ImportError:
    PointCloud2 = None
    PointField = None
    _point_cloud2_available = False

# 避免在没有导航消息的环境中报错
class FootPoint: pass
try:
    from hric_msgs.msg import FootPoint
    has_navigate_msg = True
except ImportError:
    has_navigate_msg = False

import threading
# ros2_bridge_msgs 可选导入（与 bodyctrl_msgs 二选一用于 ROS2 底层）
try:
    from ros2_bridge_msgs.msg import (
        RobotState, ArmCtrl, LegCtrl, WaistCtrl, HeadCtrl,
        LegStatus, MotorStatus as BridgeMotorStatus, MotorCtrl as BridgeMotorCtrl,
        ArmStatus, WaistStatus, ImuStatus,
    )
    _ros2_bridge_msgs_available = True
except ImportError:
    _ros2_bridge_msgs_available = False

# 全局变量，用于存储动态导入的模块
_imported_modules = {
    'bodyctrl_msgs': None,
    'ros2_bridge_msgs': None,  # 用于 comm_mode == 'ros2_bridge'
    'robot_control_lpc_python': None,      # LPC 模块名（LPC - Local Procedure Call）
    'robot_control_shm_rpc_python': None,  # SHM RPC 模块名（Shared Memory RPC - ICEORYX2）
    'robot_control_tcp_rpc_python': None   # TCP RPC 模块名（TCP RPC - YLT）
}

def import_bodyctrl_msgs():
    """
    动态导入bodyctrl_msgs模块
    """
    global _imported_modules
    if _imported_modules['bodyctrl_msgs'] is not None:
        return _imported_modules['bodyctrl_msgs']

    # 如果 Python bridge 已经导入，则不能导入bodyctrl_msgs
    if (_imported_modules['robot_control_lpc_python'] is not None or
        _imported_modules['robot_control_shm_rpc_python'] is not None or
        _imported_modules['robot_control_tcp_rpc_python'] is not None):
        raise RuntimeError("Cannot import bodyctrl_msgs when Python bridge is already imported")

    try:
        from bodyctrl_msgs.msg import MotorStatusMsg, CmdMotorCtrl, MotorCtrl, Imu, MotorStatus
        _imported_modules['bodyctrl_msgs'] = {
            'MotorStatusMsg': MotorStatusMsg,
            'CmdMotorCtrl': CmdMotorCtrl,
            'MotorCtrl': MotorCtrl,
            'Imu': Imu,
            'MotorStatus': MotorStatus
        }
        return _imported_modules['bodyctrl_msgs']
    except ImportError as e:
        _imported_modules['bodyctrl_msgs'] = None
        return None

def import_control_sdk_lpc_python_bridge():
    """
    动态导入 robot_control_lpc_python 模块（LPC - Local Procedure Call）
    """
    global _imported_modules

    # 如果已经导入，直接返回
    if _imported_modules['robot_control_lpc_python'] is not None:
        return _imported_modules['robot_control_lpc_python']

    # 如果bodyctrl_msgs已经导入，则不能导入 Python bridge
    if _imported_modules['bodyctrl_msgs'] is not None:
        raise RuntimeError("Cannot import robot_control_lpc_python when bodyctrl_msgs is already imported")

    # 如果其他 bridge 已经导入，则不能导入 LPC bridge
    if (_imported_modules['robot_control_shm_rpc_python'] is not None or
        _imported_modules['robot_control_tcp_rpc_python'] is not None):
        raise RuntimeError("Cannot import robot_control_lpc_python when other RPC bridges are already imported")

    # 导入 robot_control_lpc_python 模块
    try:
        import robot_control_lpc_python
        _imported_modules['robot_control_lpc_python'] = robot_control_lpc_python
        return _imported_modules['robot_control_lpc_python']
    except ImportError as e:
        _imported_modules['robot_control_lpc_python'] = None
        return None

def import_control_sdk_shm_rpc_python_bridge():
    """
    动态导入 robot_control_shm_rpc_python 模块（SHM RPC - Shared Memory RPC，使用 ICEORYX2）
    """
    global _imported_modules

    # 如果已经导入，直接返回
    if _imported_modules['robot_control_shm_rpc_python'] is not None:
        return _imported_modules['robot_control_shm_rpc_python']

    # 如果bodyctrl_msgs已经导入，则不能导入 Python bridge
    if _imported_modules['bodyctrl_msgs'] is not None:
        raise RuntimeError("Cannot import robot_control_shm_rpc_python when bodyctrl_msgs is already imported")

    # 如果其他 bridge 已经导入，则不能导入 SHM RPC bridge
    if (_imported_modules['robot_control_lpc_python'] is not None or
        _imported_modules['robot_control_tcp_rpc_python'] is not None):
        raise RuntimeError("Cannot import robot_control_shm_rpc_python when other bridges are already imported")

    # 导入 robot_control_shm_rpc_python 模块
    try:
        import robot_control_shm_rpc_python
        _imported_modules['robot_control_shm_rpc_python'] = robot_control_shm_rpc_python
        return _imported_modules['robot_control_shm_rpc_python']
    except ImportError as e:
        _imported_modules['robot_control_shm_rpc_python'] = None
        return None

def import_control_sdk_tcp_rpc_python_bridge():
    """
    动态导入 robot_control_tcp_rpc_python 模块（TCP RPC - TCP Socket RPC，使用 YLT）
    """
    global _imported_modules

    # 如果已经导入，直接返回
    if _imported_modules['robot_control_tcp_rpc_python'] is not None:
        return _imported_modules['robot_control_tcp_rpc_python']

    # 如果bodyctrl_msgs已经导入，则不能导入 Python bridge
    if _imported_modules['bodyctrl_msgs'] is not None:
        raise RuntimeError("Cannot import robot_control_tcp_rpc_python when bodyctrl_msgs is already imported")

    # 如果其他 bridge 已经导入，则不能导入 TCP RPC bridge
    if (_imported_modules['robot_control_lpc_python'] is not None or
        _imported_modules['robot_control_shm_rpc_python'] is not None):
        raise RuntimeError("Cannot import robot_control_tcp_rpc_python when other bridges are already imported")

    # 导入 robot_control_tcp_rpc_python 模块
    try:
        import robot_control_tcp_rpc_python
        _imported_modules['robot_control_tcp_rpc_python'] = robot_control_tcp_rpc_python
        return _imported_modules['robot_control_tcp_rpc_python']
    except ImportError as e:
        _imported_modules['robot_control_tcp_rpc_python'] = None
        return None

def get_imported_modules():
    """
    获取当前导入的模块信息
    """
    if _imported_modules['robot_control_lpc_python']:
        return 'robot_control_lpc_python'
    elif _imported_modules['robot_control_shm_rpc_python']:
        return 'robot_control_shm_rpc_python'
    elif _imported_modules['robot_control_tcp_rpc_python']:
        return 'robot_control_tcp_rpc_python'
    elif _imported_modules['ros2_bridge_msgs']:
        return 'ros2_bridge_msgs'
    elif _imported_modules['bodyctrl_msgs']:
        return 'bodyctrl_msgs'
    else:
        return None


def timing_decorator(func):
    """
    装饰器：记录函数执行时间
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result
    return wrapper


def decay_if_ros_stamp_stale(
    node: Node,
    last_stamp,
    out: np.ndarray,
    *,
    indices: Sequence[int],
    stale_ms: float = 100.0,
    decay_alpha: float = 0.35,
    eps: float = 1e-4,
) -> None:
    """若 ``node`` 时钟比 ``last_stamp`` 晚超过 ``stale_ms``，对 ``out`` 在 ``indices``
    指定下标上做指数衰减（维数任意，由 ``indices`` 决定）。

    ``last_stamp`` 为 ``None`` 或与 ROS ``Time`` 一样具有 ``sec`` / ``nanosec`` 字段。
    ``out`` 为一维可写数组，下标为扁平下标。
    """
    if last_stamp is None:
        return
    current_time = node.get_clock().now().to_msg()
    time_diff_ns = (
        (current_time.sec - last_stamp.sec) * 1000000000
        + (current_time.nanosec - last_stamp.nanosec)
    )
    time_diff_ms = max(0.0, time_diff_ns / 1000000.0)
    if time_diff_ms <= stale_ms:
        return

    factor = 1.0 - decay_alpha
    for i in indices:
        v = float(out[i]) * factor
        out[i] = 0.0 if abs(v) < eps else v


class RobotInterface(ABC):
    """机器人接口抽象基类"""

    def __init__(self, robot_data: RobotData):
        self.robot_data_ = robot_data

    @abstractmethod
    def init(self, node: Node):
        """初始化接口"""
        pass

    @abstractmethod
    def update_robot_data(self, time_passed: float, control_step: int = 0):
        """更新机器人状态"""
        pass


    @abstractmethod
    def send_motor_commands(self):
        """发布电机控制命令"""
        pass



class RobotInterfaceImpl(RobotInterface):
    """机器人接口具体实现"""

    def __init__(self, robot_data: RobotData, config_path: str = ''):
        super().__init__(robot_data)
        self.initialized = False
        self.node = None
        self.config_path = config_path

        # ID映射
        self.id_map = BodyServoIdMap()
        self.id_map.body_can_id_map_init()

        # 消息队列
        self.queue_leg_motor_state = PeekableQueue(maxsize=1)
        self.queue_arm_motor_state = PeekableQueue(maxsize=1)
        self.queue_waist_motor_state = PeekableQueue(maxsize=1)
        self.queue_head_motor_state = PeekableQueue(maxsize=1)

        self.queue_imu_xsens = PeekableQueue(maxsize=1)
        self.queue_walk_cmd = PeekableQueue(maxsize=1)
        self.queue_walk_height_cmd = PeekableQueue(maxsize=1)
        self.queue_stand_cmd = PeekableQueue(maxsize=1)
        self.queue_footpoint_cmd = PeekableQueue(maxsize=1)
        self.queue_fsm_state_cmd = PeekableQueue(maxsize=1)
        self.queue_fsm_resume_cmd = PeekableQueue(maxsize=1)

        # joysticks 消息队列
        self.queue_joy_cmd = queue.Queue(maxsize=1)
        self.queue_xbox_cmd = queue.Queue(maxsize=1)
        self.fsm_control_flag = FSMControlFlag()

        # 关节维度
        self.floating_base_dof = 6
        self.whole_joint_nums = self.id_map.whole_motor_nums + self.floating_base_dof

        # 临时变量用于优化计算
        self.temp_q = np.empty(self.id_map.whole_motor_nums)
        # 预分配另一个用于存储中间计算的临时数组
        self._temp_zero_cnt = np.empty(self.id_map.whole_motor_nums)

        # 电机控制参数
        self.motor_dir = np.ones(self.id_map.whole_motor_nums)
        self.zero_offset = np.zeros(self.id_map.whole_motor_nums)

        # 添加标志位，用于跟踪是否是首次接收数据
        self.first_leg_data_received = False
        self.first_arm_data_received = False
        self.first_waist_data_received = False

        # 关节限位
        self.joint_limits = {}
        self.joint_pos_threshold = math.pi
        self.joint_vel_threshold = 30  # rad/s
        # 串并联转换器
        self.fun_s2p = FuncSPTrans()

        # 串并联转换相关变量
        self.left_ankle_indices = np.array([4, 5]) + self.floating_base_dof
        self.right_ankle_indices = np.array([10, 11]) + self.floating_base_dof
        self.q_a_p = np.zeros(4)  # 并联关节位置
        self.qdot_a_p = np.zeros(4)  # 并联关节速度
        self.tor_a_p = np.zeros(4)  # 并联关节力矩
        self.ankle_kp_p = np.zeros(4)  # 并联关节刚度
        self.ankle_kd_p = np.zeros(4)  # 并联关节阻尼

        # TF相关属性
        self.tf_buffer = None
        self.tf_listener = None

        # ROS publishers and subscribers
        self.pub_leg_motor_cmd = None
        self.pub_arm_motor_cmd = None
        self.pub_waist_motor_cmd = None
        self.pub_head_motor_cmd = None   # 头部控制 publisher
        self.sub_leg_state = None
        self.sub_arm_state = None
        self.sub_waist_state = None
        self.sub_terrain_scan = None
        self.terrain_scan_enabled = False
        self.terrain_scan_topic = ""
        self.terrain_scan_dim = (33, 21, 3)
        self._terrain_scan_error_logged = False
        self._terrain_scan_debug_printed = False
        self._terrain_scan_last_debug_time = 0.0

        # 当前机器人所处状态
        self.current_state: FSMStateName = FSMStateName.STOP
        self.current_state_class: FSMState = None
        self.last_state: FSMStateName = FSMStateName.STOP

        # 控制消息变量
        self.walk_cmd_ = None
        self.stand_cmd_ = None
        self.footpoint_cmd_ = None
        self.trans_flag_ = False

        # 动力学模型
        self.dynamic_model = DynamicModel(robot_data=self.robot_data_)
        self._cached_transition_start_time = None
        self._cached_last_state_serial_qd = None
        self._cached_last_state_serial_kp = None
        self._cached_last_state_serial_kd = None
        self.motion_control_topic_publish_hz = 0.0
        self._motion_control_topic_publish_period = 0.0
        self._last_motion_control_topic_publish_time = -math.inf

    def update_fsm_state(self, current_state: FSMStateName = None, current_state_class: FSMState = None):
        """更新机器人接口"""
        if current_state is not None:
            if self.current_state != current_state:
                self.last_state = self.current_state
                self.current_state = current_state
                self.current_state_class = current_state_class

    def load_config(self):
        """从配置文件加载关键参数"""
        config_path = self.config_path
        if not os.path.exists(config_path):
            self.node.get_logger().error(
                f"Joint limits config file not found: {config_path}")
            raise FileNotFoundError(
                f"Joint limits config file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.node.get_logger().error(
                f"Failed to load joint limits config from {config_path}: {e}")
            raise RuntimeError(
                f"Failed to load joint limits config from {config_path}: {e}")
        # 读取运行模式
        self.sim = config.get('sim')
        self.debug = config.get('debug')
        # 机器人接口配置
        interface_config = config.get('robot_interface')

        # 优先使用节点参数中的 comm_mode 覆盖 YAML 中的配置
        param = self.node.get_parameter('comm_mode')
        comm_mode_from_param = str(param.value)
        self.comm_mode = comm_mode_from_param

        self.hric_server_enabled = self.node.get_parameter('hric_server_enabled').value

        # 是否限位
        self.clip_actions = interface_config.get('clip_actions')
        # 加载关节限位值
        self._load_joint_limits(interface_config)
        # 加载控制状态
        self._load_control_status(interface_config)
        # 零位
        self.zero_pos = np.array(interface_config.get('zero_pos'))
        # 电流转换比例
        self.ct_scale_sim = np.array(interface_config.get('ct_scale_sim'))
        self.ct_scale_real = np.array(interface_config.get('ct_scale_real'))
        if self.sim:
            self.ct_scale = self.ct_scale_sim
        else:
            self.ct_scale = self.ct_scale_real

        # IMU 数据偏移（度 → 弧度）
        self.imu_ypr_offset = (
            np.array(interface_config.get('imu_ypr_offset')) * np.pi / 180.0
        )
        # 禁用电机
        self.disable_joints_ = interface_config.get('disable_joints', False)
        # 脚踝Kp,Kd
        self.ankle_kp_p = np.array(interface_config.get('ankle_kp_p'))
        self.ankle_kd_p = np.array(interface_config.get('ankle_kd_p'))

        # 状态速度配置
        self._load_state_speed_limits(interface_config.get('cmd_limits'))
        # 过渡时间配置
        self._load_trans_time(interface_config)
        # 运控话题发布频率配置
        self._load_motion_control_topic_publish_hz(interface_config)
        # MOE状态命令映射配置
        self.moe_state_commands_map = interface_config.get('moe_state_commands_map', {})
        # 地形点云输入配置；默认复用 stairs 策略配置，避免 topic 配置分散。
        self._load_terrain_scan_config(interface_config)
    
    def rewrite_config(self, config: Dict[str, Any]):
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key == 'sim':
                    if self.sim:
                        self.ct_scale = self.ct_scale_sim
                    else:
                        self.ct_scale = self.ct_scale_real
                    xlog.info(f"Rewrite ct_scale: {self.ct_scale}")
            else:
                xlog.warning(f"Key {key} not found in robot interface")


    def _load_trans_time(self, config: Dict[str, Any]):
        self.trans_time = config['trans_time']

    def _load_motion_control_topic_publish_hz(self, config: Dict[str, Any]):
        """加载运控话题发布频率，Hz <= 0 表示跟随控制循环每周期发布。"""
        self.motion_control_topic_publish_hz = float(
            config.get('motion_control_topic_publish_hz', 0.0))
        self._motion_control_topic_publish_period = (
            1.0 / self.motion_control_topic_publish_hz
            if self.motion_control_topic_publish_hz > 0.0 else 0.0)
        self._last_motion_control_topic_publish_time = -math.inf

    def _load_terrain_scan_config(self, config: Dict[str, Any]):
        """加载 terrain scan 配置，优先使用 dex_config，默认读取 stairs.yaml。"""
        terrain_cfg = config.get('terrain_scan') or {}
        if not terrain_cfg:
            terrain_cfg = self._load_policy_terrain_scan_config()

        self.terrain_scan_enabled = bool(terrain_cfg.get('enabled', False))
        self.terrain_scan_topic = str(terrain_cfg.get('topic', '') or '')
        scan_dim = terrain_cfg.get('scan_dim', self.terrain_scan_dim)
        if len(scan_dim) == 2:
            scan_dim = [*scan_dim, 3]
        self.terrain_scan_dim = tuple(int(v) for v in scan_dim)
        if self.terrain_scan_enabled:
            self.robot_data_.configure_terrain_scan(self.terrain_scan_dim)

    def _load_policy_terrain_scan_config(self) -> Dict[str, Any]:
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stairs_config_path = os.path.join(
            package_dir, 'policy', 'stairs', 'config', 'stairs.yaml')
        if not os.path.exists(stairs_config_path):
            return {}
        try:
            with open(stairs_config_path, 'r') as f:
                stairs_config = yaml.safe_load(f) or {}
            return stairs_config.get('terrain_scan') or {}
        except Exception as exc:
            xlog.warning(f"[TerrainScan] failed to load stairs terrain_scan config: {exc}")
            return {}

    def _should_publish_motion_control_topic(self, now: float) -> bool:
        if self._motion_control_topic_publish_period <= 0.0:
            return True
        if now - self._last_motion_control_topic_publish_time < (
                self._motion_control_topic_publish_period):
            return False
        self._last_motion_control_topic_publish_time = now
        return True

    def _get_current_transition_time(self) -> float:
        """获取上一状态到当前状态的过渡时间"""
        current_state_name = self.current_state.name
        last_state_name = self.last_state.name
        state_trans_time = self.trans_time['state_trans_time']

        if current_state_name in state_trans_time:
            current_state_config = state_trans_time[current_state_name]
            last_state_trans_time = current_state_config.get('last_state', {})
            if last_state_name in last_state_trans_time:
                return last_state_trans_time[last_state_name]
            if 'default_trans_time' in current_state_config:
                return current_state_config['default_trans_time']

        return self.trans_time['default_time']

    def _load_state_speed_limits(self, config: Dict[str, Any]):
        self.x_command_offset = config.get("x_command_offset")
        self.y_command_offset = config.get("y_command_offset")
        self.yaw_command_offset = config.get("yaw_command_offset")
        self.max_x_plus_speed = config.get("max_x_plus_speed")
        self.max_x_minus_speed = config.get("max_x_minus_speed")
        self.max_y_speed = config.get("max_y_speed")
        self.max_yaw_speed = config.get("max_yaw_speed")
        state_limits = config.get('state_speed_limits')
        self.state_speed_limits = {}
        for state_name, limits in state_limits.items():
            self.state_speed_limits[state_name] = {
                "max_x_plus":
                limits.get("max_x_plus", self.max_x_plus_speed),
                "max_x_minus":
                limits.get("max_x_minus", self.max_x_minus_speed),
                "max_y":
                limits.get("max_y", self.max_y_speed),
                "max_yaw":
                limits.get("max_yaw", self.max_yaw_speed),
                "x_command_offset":
                limits.get("x_command_offset", self.x_command_offset),
                "y_command_offset":
                limits.get("y_command_offset", self.y_command_offset),
                "yaw_command_offset":
                limits.get("yaw_command_offset", self.yaw_command_offset)
            }

    def safe_get_states(self, state_names):
        """安全地获取状态枚举，跳过无效的状态名称"""
        valid_states = []
        for name in state_names:
            try:
                valid_states.append(FSMStateName[name])
            except (KeyError, AttributeError):
                pass
        return valid_states

    def _load_control_status(self, config: Dict[str, Any]):
        # 字符串命令到枚举值的映射
        all_states = list(FSMStateName.__members__.values())
        self.head_control_status = self.safe_get_states(
            config.get('head_control_status', []))
        self.waist_control_status = self.safe_get_states(
            config.get('waist_control_status', []))
        self.legs_control_status = self.safe_get_states(
            config.get('legs_control_status', []))
        self.head_control_status = self.safe_get_states(
            config.get('head_control_status', []))
        if self.legs_control_status == []:
            # 如果未配置腿部控制状态，则默认添加所有状态
            self.legs_control_status = all_states
        self.arms_control_status = self.safe_get_states(
            config.get('arms_control_status', []))
        self.left_arm_only_status = self.safe_get_states(
            config.get('left_arm_only_status', []))
        self.right_arm_only_status = self.safe_get_states(
            config.get('right_arm_only_status', []))
        #配置需要双臂动力学插值的状态列表
        self.dynamic_inter_status = self.safe_get_states(
            config.get('dynamic_inter_status', []))
        #配置MOE状态列表
        self.moe_states = self.safe_get_states(
            config.get('moe_states', []))

        # 为每一个状态创建控制的电机id映射，如stop=[0,1,2...,28], MLP=[0,1,...,11]
        self.state_motor_id_map = {}
        for state_enum in all_states:
            motor_ids = []
            # 如果状态在腿部控制状态中，添加腿部电机ID
            if state_enum in self.legs_control_status:
                motor_ids.extend(list(range(self.id_map.leg_motor_nums)))

            # 如果状态在腰部控制状态中，添加腰部电机ID
            if state_enum in self.waist_control_status:
                start_idx = self.id_map.leg_motor_nums
                end_idx = start_idx + self.id_map.waist_motor_nums
                motor_ids.extend(list(range(start_idx, end_idx)))

            # 如果状态在手臂控制状态中，添加手臂电机ID
            if state_enum in self.arms_control_status:
                if state_enum in self.left_arm_only_status:
                    start_idx = self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                    end_idx = start_idx + self.id_map.arm_motor_nums // 2
                elif state_enum in self.right_arm_only_status:
                    start_idx = self.id_map.leg_motor_nums + self.id_map.waist_motor_nums + self.id_map.arm_motor_nums // 2
                    end_idx = start_idx + self.id_map.arm_motor_nums // 2
                else:
                    start_idx = self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                    end_idx = start_idx + self.id_map.arm_motor_nums
                motor_ids.extend(list(range(start_idx, end_idx)))

            # 去重并排序
            motor_ids = sorted(list(set(motor_ids)))

            self.state_motor_id_map[state_enum] = motor_ids

        # 打印映射关系用于调试
        for state_enum, motor_ids in self.state_motor_id_map.items():
            pass

    def _load_joint_limits(self, config: Dict[str, Any]):
        """从配置文件加载关节限位值"""
        # 从配置中获取关节限位信息
        joint_limits_config = config.get('joint_limits', None)

        if joint_limits_config is None:
            error_msg = "No joint_limits section in config"
            self.node.get_logger().error(error_msg)
            raise ValueError(error_msg)
        else:
            # 从配置中加载限位值
            for joint_name, limits in joint_limits_config.items():
                if 'min' in limits and 'max' in limits:
                    self.joint_limits[joint_name] = {
                        'min': float(limits['min']),
                        'max': float(limits['max'])
                    }
                else:
                    error_msg = f"Invalid limits for joint {joint_name}"
                    self.node.get_logger().error(error_msg)
                    raise ValueError(error_msg)

            xlog.info(f"Loaded joint limits from {config}")
        # 预计算ID到限位的映射
        self.id_to_limits = {}
        for joint_name, limits in self.joint_limits.items():
            index = self.id_map.get_index_by_name(joint_name)
            if index >= 0:
                motor_id = self.id_map.get_id_by_index(index)
                self.id_to_limits[motor_id] = limits

        # 记录加载的限位值
        for joint_name, limits in self.joint_limits.items():
            self.node.get_logger().debug(
                f"Joint {joint_name}: [{limits['min']}, {limits['max']}]")

    def init(self, node: Node):
        """初始化接口"""
        self.node = node
        self.initialized = True
        # 加载配置文件
        self.load_config()

        if self.comm_mode == 'ros2':
            # 初始化 ROS 底层接口（bodyctrl_msgs：/leg/cmd_ctrl, /leg/status 等）
            self._init_ros_lowlevel_interfaces()
        elif self.comm_mode == 'ros2_bridge':
            # 初始化 ROS 底层接口（ros2_bridge_msgs：/robot_state, /leg/cmd 等）
            self._init_ros_lowlevel_new_interfaces()
        elif self.comm_mode == 'lpc':
            # 初始化LPC SDK接口（本地过程调用）
            self._init_sdk_lpc_interface()
        elif self.comm_mode == 'shm_rpc':
            # 初始化SHM RPC SDK接口（共享内存 RPC，使用 ICEORYX2）
            self._init_sdk_shm_rpc_interface()
        elif self.comm_mode == 'tcp_rpc':
            # 初始化TCP RPC SDK接口（TCP Socket RPC，使用 YLT）
            self._init_sdk_tcp_rpc_interface()
        else:
            raise ValueError(f"Invalid communication mode: {self.comm_mode}")
        self._init_ros_highlevel_interfaces()
        self._init_terrain_scan_interface()

        # # 初始化控制系统
        # self._init_control_system()
        # 初始化TF缓冲区和监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

        xlog.info("Robot interface initialized")

    def _init_sdk_lpc_interface(self):
        """初始化 LPC SDK 接口（本地过程调用）"""
        # 尝试动态导入 robot_control_lpc_python 模块
        robot_control_lpc_python = import_control_sdk_lpc_python_bridge()
        if robot_control_lpc_python is None:
            return

        # 检查 Python Bridge 是否可用（单例已初始化且 Python Bridge 已启用）
        if not robot_control_lpc_python.is_available():
            return

        # 保存模块引用，后续通过模块函数调用 SDK
        self.sdk_lpc_interface = robot_control_lpc_python

        # 启动 SDK 状态轮询循环
        self._start_sdk_polling_loop()

    def _init_sdk_shm_rpc_interface(self):
        """初始化 SHM RPC SDK 接口（共享内存 RPC，使用 ICEORYX2）"""
        # 尝试动态导入 robot_control_shm_rpc_python 模块
        robot_control_shm_rpc_python = import_control_sdk_shm_rpc_python_bridge(
        )
        if robot_control_shm_rpc_python is None:
            return

        # 连接到 RPC 服务器（进程内通信，使用共享内存）
        try:
            # 无参 connect() 使用进程内通信（SHM/ICEORYX2）
            if not robot_control_shm_rpc_python.connect():
                return
        except Exception as e:
            return

        # 验证连接状态
        if not robot_control_shm_rpc_python.is_connected():
            return

        # 保存模块引用，后续通过模块函数调用 SDK
        self.sdk_rpc_interface = robot_control_shm_rpc_python

        # 启动 SDK 状态轮询循环
        self._start_sdk_polling_loop()

    def _init_sdk_tcp_rpc_interface(self):
        """初始化 TCP RPC SDK 接口（TCP Socket RPC，使用 YLT）"""
        # 尝试动态导入 robot_control_tcp_rpc_python 模块
        robot_control_tcp_rpc_python = import_control_sdk_tcp_rpc_python_bridge(
        )
        if robot_control_tcp_rpc_python is None:
            return

        # 连接到 RPC 服务器（跨机器通信，使用 TCP）
        try:
            # 默认连接到本地 RPC 服务器（端口 50051）
            if not robot_control_tcp_rpc_python.connect("127.0.0.1", "50051"):
                return
        except Exception as e:
            return

        # 验证连接状态
        if not robot_control_tcp_rpc_python.is_connected():
            return

        # 保存模块引用，后续通过模块函数调用 SDK
        self.sdk_rpc_interface = robot_control_tcp_rpc_python

        # 启动 SDK 状态轮询循环
        self._start_sdk_polling_loop()

    def _init_ros_lowlevel_interfaces(self):
        """初始化ROS底层接口"""
        # 动态导入bodyctrl_msgs模块
        bodyctrl_msgs = import_bodyctrl_msgs()
        if bodyctrl_msgs is None:
            return

        MotorStatusMsg = bodyctrl_msgs['MotorStatusMsg']
        CmdMotorCtrl = bodyctrl_msgs['CmdMotorCtrl']
        Imu = bodyctrl_msgs['Imu']

        qos_profile = QoSProfile(
            # reliability=ReliabilityPolicy.RELIABLE,
            # history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # 发布者
        self.pub_leg_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/leg/cmd_ctrl', qos_profile)
        self.pub_arm_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/arm/cmd_ctrl', qos_profile)
        self.pub_waist_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/waist/cmd_ctrl', qos_profile)

        # 订阅者
        self.sub_leg_state = self.node.create_subscription(
            MotorStatusMsg, '/leg/status', self._leg_motor_status_callback,
            qos_profile)
        self.sub_arm_state = self.node.create_subscription(
            MotorStatusMsg, '/arm/status', self._arm_motor_status_callback,
            qos_profile)
        self.sub_waist_state = self.node.create_subscription(
            MotorStatusMsg, '/waist/status', self._waist_motor_status_callback,
            qos_profile)

        #（非电机相关）
        self.sub_imu_xsens = self.node.create_subscription(
            Imu, '/imu/status', self._imu_status_callback, qos_profile)

    def _init_terrain_scan_interface(self):
        """订阅策略需要的 PointCloud2 地形扫描输入。"""
        if not self.terrain_scan_enabled:
            return
        if not self.terrain_scan_topic:
            xlog.warning("[TerrainScan] enabled but topic is empty")
            return
        if not _point_cloud2_available:
            xlog.warning("[TerrainScan] sensor_msgs PointCloud2 support is unavailable")
            return

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        self.sub_terrain_scan = self.node.create_subscription(
            PointCloud2,
            self.terrain_scan_topic,
            self._terrain_scan_callback,
            qos_profile)
        self.robot_data_.configure_terrain_scan(self.terrain_scan_dim)
        xlog.info(
            f"[TerrainScan] subscribed topic={self.terrain_scan_topic} dim={self.terrain_scan_dim}")

    def _terrain_scan_callback(self, msg: PointCloud2):
        """缓存 terrain policy 使用的局部 xyz scan。"""
        try:
            points = self._pointcloud2_to_xyz(msg)
            self.robot_data_.set_terrain_scan(
                points.reshape(-1),
                scan_dim=self.terrain_scan_dim,
                valid=True,
                stamp=time.perf_counter(),
            )
            self._log_terrain_scan_stats(points)
        except Exception as exc:
            if not self._terrain_scan_error_logged:
                self._terrain_log_warning(f"Failed to parse terrain scan PointCloud2: {exc}")
                self._terrain_scan_error_logged = True

    def _log_terrain_scan_stats(self, points: np.ndarray):
        now = time.perf_counter()
        if (
            self._terrain_scan_debug_printed
            and now - self._terrain_scan_last_debug_time < 5.0
        ):
            return

        finite = np.isfinite(points).all(axis=1)
        finite_count = int(np.count_nonzero(finite))
        if finite_count == 0:
            self._terrain_log_warning("[terrain_scan] received finite=0")
        else:
            finite_points = points[finite]
            width = int(self.terrain_scan_dim[0])
            height = int(self.terrain_scan_dim[1])
            center_idx = (height // 2) * width + (width // 2)
            center_idx = min(center_idx, points.shape[0] - 1)
            center = points[center_idx]
            near_zero = int(np.count_nonzero(
                np.linalg.norm(finite_points, axis=1) < 1.0e-3))
            max_range = int(np.count_nonzero(
                np.isclose(finite_points[:, 0], 5.0, atol=0.05)))
            mins = finite_points.min(axis=0)
            maxs = finite_points.max(axis=0)
            means = finite_points.mean(axis=0)
            self._terrain_log_info(
                "[terrain_scan] received "
                f"finite={finite_count}/{points.shape[0]}, zero={near_zero}, "
                f"max_x={max_range}, "
                f"x=[{mins[0]:.3f},{maxs[0]:.3f},{means[0]:.3f}], "
                f"y=[{mins[1]:.3f},{maxs[1]:.3f},{means[1]:.3f}], "
                f"z=[{mins[2]:.3f},{maxs[2]:.3f},{means[2]:.3f}], "
                f"center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})"
            )

        self._terrain_scan_debug_printed = True
        self._terrain_scan_last_debug_time = now

    def _terrain_log_info(self, message: str):
        if self.node is not None:
            self.node.get_logger().info(message)
        else:
            logger.info(message)

    def _terrain_log_warning(self, message: str):
        if self.node is not None:
            self.node.get_logger().warn(message)
        else:
            logger.warning(message)

    def _pointcloud2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        fields = {field.name: field for field in msg.fields}
        missing = [name for name in ("x", "y", "z") if name not in fields]
        if missing:
            raise ValueError(f"missing fields: {missing}")
        for name in ("x", "y", "z"):
            if fields[name].datatype != PointField.FLOAT32:
                raise ValueError(f"field {name} must be FLOAT32")

        endian = ">" if msg.is_bigendian else "<"
        dtype = np.dtype({
            "names": ["x", "y", "z"],
            "formats": [endian + "f4", endian + "f4", endian + "f4"],
            "offsets": [fields["x"].offset, fields["y"].offset, fields["z"].offset],
            "itemsize": msg.point_step,
        })
        row_data_width = msg.point_step * msg.width
        raw = bytes(msg.data)
        if msg.height > 1 and msg.row_step != row_data_width:
            rows = [
                raw[row * msg.row_step:row * msg.row_step + row_data_width]
                for row in range(msg.height)
            ]
            raw = b"".join(rows)
        count = int(msg.width * msg.height)
        cloud = np.frombuffer(raw, dtype=dtype, count=count)
        points = np.empty((count, 3), dtype=np.float32)
        points[:, 0] = cloud["x"]
        points[:, 1] = cloud["y"]
        points[:, 2] = cloud["z"]
        np.nan_to_num(points, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return points

    def _init_ros_lowlevel_new_interfaces(self):
        """初始化 ROS 底层接口（ros2_bridge_msgs：订阅 /robot_state，发布 /leg/cmd、/arm/cmd、/waist/cmd）"""
        if not _ros2_bridge_msgs_available:
            return
        _imported_modules['ros2_bridge_msgs'] = True

        qos_profile = QoSProfile(depth=10)

        # 发布者：ros2_bridge 使用 /leg/cmd, /arm/cmd, /waist/cmd
        self.pub_leg_motor_cmd = self.node.create_publisher(
            LegCtrl, '/leg/cmd', qos_profile)
        self.pub_arm_motor_cmd = self.node.create_publisher(
            ArmCtrl, '/arm/cmd', qos_profile)
        self.pub_waist_motor_cmd = self.node.create_publisher(
            WaistCtrl, '/waist/cmd', qos_profile)
        self.pub_head_motor_cmd = self.node.create_publisher(
            HeadCtrl, '/head/cmd', qos_profile)

        # 订阅者：单一话题 /robot_state 汇总腿/臂/腰/IMU
        self.sub_robot_state = self.node.create_subscription(
            RobotState, '/robot_state', self._robot_state_callback_ros2_bridge,
            qos_profile)

        # 无独立 leg/arm/waist/imu 订阅，由 _robot_state_callback_ros2_bridge 解包
        self.sub_leg_state = None
        self.sub_arm_state = None
        self.sub_waist_state = None
        self.sub_imu_xsens = None

        xlog.info("[RobotInterface] ROS2 bridge (ros2_bridge_msgs) low-level interface initialized")

    def _init_ros_highlevel_interfaces(self):
        """初始化高层抽象控制接口"""
        qos_profile = QoSProfile(
            # reliability=ReliabilityPolicy.RELIABLE,
            # history=HistoryPolicy.KEEP_LAST,
            depth=10)
        # roy 添加你的发布话题
        self.pub_walk_cmd = self.node.create_publisher(
            TwistStamped, '/hric/robot/cmd_vel_status', qos_profile)
        self.pub_stand_cmd = self.node.create_publisher(
            TwistStamped, '/hric/robot/stand_cmd_status', qos_profile)
        
        self.pub_walk_speed = self.node.create_publisher(
            TwistStamped, '/hric/robot/walk_speed', qos_profile)
        self.pub_stand_status = self.node.create_publisher(
            TwistStamped, '/hric/robot/stand_status', qos_profile)
        self.pub_robot_state = self.node.create_publisher(
            DiagnosticStatus, '/hric/robot/rl_state', qos_profile)
        
        if not self.hric_server_enabled:
            self.sub_walk_cmd = self.node.create_subscription(
                TwistStamped, '/hric/robot/cmd_vel', self._walk_cmd_callback,
                qos_profile)
            #TODO:合并高度和移动速度话题
            self.sub_stand_cmd = self.node.create_subscription(
                TwistStamped, '/hric/robot/stand_cmd', self._stand_cmd_callback,
                qos_profile)
            if has_navigate_msg:
                self.sub_footpoint_cmd = self.node.create_subscription(
                    FootPoint, '/hric/robot/footpoint',
                    self._footpoint_cmd_callback, qos_profile)
            self.sub_fsm_state_cmd = self.node.create_subscription(
                String, '/hric/robot/fsm_state_cmd', self._fsm_state_cmd_callback,
                qos_profile)
            
            self.sub_fsm_resume_cmd = self.node.create_subscription(
                String, '/hric/robot/fsm_resume_cmd', self._fsm_resume_cmd_callback,
                qos_profile)
        #TODO:添加机器人状态信息订阅：如电量、电压等信息

    # @timing_decorator
    def get_imu_data(self):
        """处理传感器数据（非电机），兼容 bodyctrl_msgs 与 ros2_bridge_msgs(ImuStatus) 两种格式"""
        while True:
            try:
                msg = self.queue_imu_xsens.peek()
                if hasattr(msg, 'euler') and hasattr(
                        msg, 'angular_velocity') and hasattr(
                            msg, 'linear_acceleration'):
                    # bodyctrl_msgs 格式
                    self.robot_data_.imu_data_[0] = msg.euler.yaw
                    self.robot_data_.imu_data_[1] = msg.euler.pitch
                    self.robot_data_.imu_data_[2] = msg.euler.roll
                    self.robot_data_.imu_data_[3] = msg.angular_velocity.x
                    self.robot_data_.imu_data_[4] = msg.angular_velocity.y
                    self.robot_data_.imu_data_[5] = msg.angular_velocity.z
                    self.robot_data_.imu_data_[6] = msg.linear_acceleration.x
                    self.robot_data_.imu_data_[7] = msg.linear_acceleration.y
                    self.robot_data_.imu_data_[8] = msg.linear_acceleration.z
                else:
                    # ros2_bridge_msgs ImuStatus 等扁平格式：yaw/pitch/roll, wx/wy/wz, ax/ay/az
                    self.robot_data_.imu_data_[0] = getattr(msg, 'yaw', 0.0)
                    self.robot_data_.imu_data_[1] = getattr(msg, 'pitch', 0.0)
                    self.robot_data_.imu_data_[2] = getattr(msg, 'roll', 0.0)
                    av = getattr(msg, 'angular_velocity', None)
                    self.robot_data_.imu_data_[
                        3] = av.x if av is not None else getattr(
                            msg, 'wx', 0.0)
                    self.robot_data_.imu_data_[
                        4] = av.y if av is not None else getattr(
                            msg, 'wy', 0.0)
                    self.robot_data_.imu_data_[
                        5] = av.z if av is not None else getattr(
                            msg, 'wz', 0.0)
                    la = getattr(msg, 'linear_acceleration', None)
                    self.robot_data_.imu_data_[
                        6] = la.x if la is not None else getattr(
                            msg, 'ax', 0.0)
                    self.robot_data_.imu_data_[
                        7] = la.y if la is not None else getattr(
                            msg, 'ay', 0.0)
                    self.robot_data_.imu_data_[
                        8] = la.z if la is not None else getattr(
                            msg, 'az', 0.0)
                break
            except queue.Empty:
                time.sleep(0.0001)

    def update_robot_state(self):
        """读取电机状态数据更新为机器人状态数据"""
        # 处理腿部电机状态
        while True:
            try:
                msg = self.queue_leg_motor_state.peek()
                if self.debug:
                    current_time = self.node.get_clock().now().to_msg()
                    msg_time = msg.header.stamp
                    time_diff = (current_time.sec -
                                 msg_time.sec) * 1000000000 + (
                                     current_time.nanosec - msg_time.nanosec)
                    time_diff_ms = time_diff / 1000000.0
                for status in msg.status:
                    self.motor_state_to_robot_state(
                        status, self.first_leg_data_received)
                break
            except queue.Empty:
                time.sleep(0.0001)

        # 处理手臂电机状态
        while True:
            try:
                msg = self.queue_arm_motor_state.peek()
                if self.debug:
                    current_time = self.node.get_clock().now().to_msg()
                    msg_time = msg.header.stamp
                    time_diff = (current_time.sec -
                                 msg_time.sec) * 1000000000 + (
                                     current_time.nanosec - msg_time.nanosec)
                    time_diff_ms = time_diff / 1000000.0
                for status in msg.status:
                    self.motor_state_to_robot_state(
                        status, self.first_arm_data_received)

                break
            except queue.Empty:
                time.sleep(0.0001)

        # 处理腰部电机状态
        while True:
            try:
                msg = self.queue_waist_motor_state.peek()
                if self.debug:
                    current_time = self.node.get_clock().now().to_msg()
                    msg_time = msg.header.stamp
                    time_diff = (current_time.sec -
                                 msg_time.sec) * 1000000000 + (
                                     current_time.nanosec - msg_time.nanosec)
                    time_diff_ms = time_diff / 1000000.0
                for status in msg.status:
                    self.motor_state_to_robot_state(
                        status, self.first_waist_data_received)
                break
            except queue.Empty:
                time.sleep(0.0001)

        # 处理头部电机状态, 不阻塞
        """从 queue_head_motor_state 消费最新头部状态，更新 robot_data_.q_a_"""
        try:
            msg = self.queue_head_motor_state.peek()
            for motor_status in msg.status:
                index = self.id_map.get_index_by_id(motor_status.name)
                robotdata_index = index + self.floating_base_dof  # 偏移到完整关节数组中
                self.robot_data_.q_a_[robotdata_index] = float(motor_status.pos) #只更新关节位置，不更新速度和力矩
        except queue.Empty:
            pass

    def get_walk_cmd(self):
        """处理运动控制命令"""
        try:
            #TODO:添加平滑处理
            msg: TwistStamped = self.queue_walk_cmd.get_nowait()
            self.walk_cmd_ = msg
            xyyaw_speed_limits = self.get_state_xyyaw_speed_limits(self.current_state.name)
            self.robot_data_.set_walk_cmd(np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z]), xyyaw_speed_limits, self.trans_flag_)
        except queue.Empty:
            last_stamp = (
                self.walk_cmd_.header.stamp
                if self.walk_cmd_ is not None
                else None
            )
            walk_cmd = self.robot_data_.get_walk_cmd().copy()
            decay_if_ros_stamp_stale(
                self.node,
                last_stamp,
                walk_cmd,
                indices=(0, 1, 2),
                stale_ms=200.0,
                decay_alpha=0.35,
                eps=1e-4,
            )
            xyyaw_speed_limits = self.get_state_xyyaw_speed_limits(
                self.current_state.name)
            xyyaw_speed_limits["x_command_offset"] = 0.0
            xyyaw_speed_limits["y_command_offset"] = 0.0
            xyyaw_speed_limits["yaw_command_offset"] = 0.0
            self.robot_data_.set_walk_cmd(walk_cmd, xyyaw_speed_limits, self.trans_flag_)

    def get_stand_cmd(self):
        """Parse stand command from TwistStamped topic"""
        try:
            msg: TwistStamped = self.queue_stand_cmd.get_nowait()
            self.stand_cmd_ = msg
            self.robot_data_.set_floating_base_cmd(np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z, msg.twist.linear.z]), self.trans_flag_)            
        except queue.Empty:
            pass  # 如果队列为空，保持上次命令不变

    def get_footpoint_cmd(self):
        """获取足点命令"""
        if not has_navigate_msg:
            return
        try:
            msg: FootPoint = self.queue_footpoint_cmd.get_nowait()
            self.robot_data_.set_footpoint_cmd(
                np.array([msg.footflag, msg.relative_x, msg.relative_y, msg.relative_yaw], dtype=object),
                self.trans_flag_,
            )
            self.footpoint_cmd_ = msg
        except queue.Empty:
            # 从未收到过 footpoint 命令: footpoint_cmd_ 仍为 None, 没有 last_stamp 可比较;
            # robot_data_.footpoint_cmd_ 默认即 0/False, 无须再做"超时清零", 直接 return.
            if self.footpoint_cmd_ is None:
                return
            last_stamp = self.footpoint_cmd_.stamp
            current_time = self.node.get_clock().now().to_msg()
            time_diff = (current_time.sec - last_stamp.sec) * 1000000000 + (
                current_time.nanosec - last_stamp.nanosec)
            time_diff_ms = time_diff / 1000000.0
            if time_diff_ms > 500.0:
                self.robot_data_.set_footpoint_cmd(np.array([False, 0.0, 0.0, 0.0]), self.trans_flag_)

    def get_fsm_state_cmd(self):
        """获取FSM状态命令.

        语义 (v3.5.3 与参考实现 robot_control_xmigcs 对齐):
            队列空时 **保持 flag.fsm_state_command 上次值不变**, 不再每 tick 清空.
            理由: 参考实现下 flag 持久保留, FSM trigger 是幂等的
            (transition 表里大多带 self-loop 边或 _active_reject_key 去重),
            重复 trigger 不会有副作用. 这样:
              1. HRIC adapter 入队一次后, FSM 自然下一 tick 消费, 无需轮询重入队;
              2. trigger 被拒绝时, 不会被 "空 tick" 立刻擦掉, 留给后续 safe_guard / 上层
                 业务一次"再来"的机会;
              3. 与 robot_control_xmigcs (参考实现) 保持完全一致的命令生命周期语义.
            旧行为 (每 tick 清空) 导致 HRIC adapter 必须每 1s 重入队才能让 trigger 多次
            生效, 已在 _set_command_fsm_state_confirmed 同步移除.
        """
        try:
            msg: String = self.queue_fsm_state_cmd.get_nowait()
            self.fsm_control_flag.fsm_state_command = msg.data
        except queue.Empty:
            pass

    def get_fsm_resume_cmd(self):
        """获取FSM RESUME状态命令. 与 get_fsm_state_cmd 同语义, 队列空时保持 flag."""
        try:
            msg: String = self.queue_fsm_resume_cmd.get_nowait()
            self.fsm_control_flag.fsm_resume_command = msg.data
        except queue.Empty:
            pass  # 保持上次命令不变, 与参考实现 robot_control_xmigcs 一致

    def motor_state_to_robot_state(self, status, received_flag: bool):
        index = self.id_map.get_index_by_id(status.name)
        if index >= 0:
            self.robot_data_.diagnostic_info['motor_status'][status.name] = status.error
            robotdata_index = index + self.floating_base_dof  # 偏移到完整关节数组中
            # 直接赋值
            self.robot_data_.q_a_[robotdata_index] = status.pos
            last_qdot_a = self.robot_data_.q_dot_a_[robotdata_index]
            # 计算速度差值
            speed_gap = status.speed - last_qdot_a
            if abs(speed_gap) < self.joint_vel_threshold:
                self.robot_data_.q_dot_a_[robotdata_index] = status.speed
            self.robot_data_.tau_a_[
                robotdata_index] = status.current * self.ct_scale[min(
                    index,
                    len(self.ct_scale) - 1)]
            self.robot_data_.temperature_a[
                robotdata_index - self.floating_base_dof] = status.temperature

            self.robot_data_.q_a_[robotdata_index] = (
                status.pos - self.zero_pos[index]
            ) * self.motor_dir[index] + self.zero_offset[index]
            # if self.robot_data_.q_a_[robotdata_index] > math.pi:
            #     self.zero_cnt[index] = -1.0
            # elif self.robot_data_.q_a_[robotdata_index] < -math.pi:
            #     self.zero_cnt[index] = 1.0

            # self.robot_data_.q_a_[
            #     robotdata_index] += self.zero_cnt[index] * 2.0 * math.pi
            self.robot_data_.q_dot_a_[robotdata_index] *= self.motor_dir[index]
            self.robot_data_.tau_a_[robotdata_index] *= self.motor_dir[index]

            if not received_flag or abs(
                    self.robot_data_.q_a_[robotdata_index] -
                    self.robot_data_.q_a_last[robotdata_index]
            ) > self.joint_pos_threshold:
                if received_flag:
                    self.node.get_logger().warn(
                        f"Joint {index} error detected")
                    self.robot_data_.q_a_[
                        robotdata_index] = self.robot_data_.q_a_last[
                            robotdata_index]
                    self.robot_data_.q_dot_a_[
                        robotdata_index] = self.robot_data_.qdot_a_last[
                            robotdata_index]
                    self.robot_data_.tau_a_[
                        robotdata_index] = self.robot_data_.tor_a_last[
                            robotdata_index]
                else:
                    # 首次接收数据，更新标志位
                    received_flag = True
            self.robot_data_.q_a_last[robotdata_index] = self.robot_data_.q_a_[
                robotdata_index]
            self.robot_data_.qdot_a_last[
                robotdata_index] = self.robot_data_.q_dot_a_[robotdata_index]
            self.robot_data_.tor_a_last[
                robotdata_index] = self.robot_data_.tau_a_[robotdata_index]

    def update_sensor_states(self):
        # 获取Imu数据
        self.get_imu_data()
        # 添加IMU偏移
        self.robot_data_.imu_data_[0:3] += self.imu_ypr_offset

    def get_pelvis_ankle_transform(self):
        """
        获取pelvis_link相对于ankle_roll_l_link的坐标变换
        将四元数转换为xyz顺序的欧拉角，并记录z轴高度
        """
        try:
            # 获取pelvis_link相对于ankle_roll_l_link的变换
            trans = self.tf_buffer.lookup_transform(
                'ankle_roll_l_link',  # 目标坐标系
                'pelvis',  # 源坐标系
                rclpy.time.Time(),  # 最新可用的变换
                timeout=rclpy.duration.Duration(seconds=0.0)  # 超时时间
            )

            # 提取平移部分（xyz位置）
            z_offset = 0.056
            self.robot_data_.q_a_[0] = trans.transform.translation.x
            self.robot_data_.q_a_[1] = trans.transform.translation.y
            self.robot_data_.q_a_[2] = trans.transform.translation.z + z_offset

            # 提取旋转部分（四元数）
            quat = [
                trans.transform.rotation.w,
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
            ]

            # 将四元数转换为xyz顺序的欧拉角
            euler = t3d.euler.quat2euler(quat, axes="rxyz")
            # 存储欧拉角
            self.robot_data_.q_a_[3] = euler[0]  # x轴旋转
            self.robot_data_.q_a_[4] = euler[1]  # y轴旋转
            self.robot_data_.q_a_[5] = euler[2]  # z轴旋转

            return True

        except tf2_ros.TransformException as ex:
            # logger.warning(f'Could not get transform: {ex}')
            return False

    def update_robot_data(self, time_passed: float, control_step: int = 0):
        # 更新传感器数据
        self.update_sensor_states()
        # 更新电机状态数据
        self.update_robot_state()
        # 更新机器人控制命令
        self.update_robot_cmd()
        # 脚踝并联转串联
        if not self.sim:
            #真机
            self.ankle_parallel_to_serial()
        # 获取stand的状态反馈
        self.dynamic_model.compute_stand_status()
        # 更新robot_data 时间戳
        self.robot_data_.time_now_ = time_passed
        self.robot_data_.control_step_ = control_step

    def backup_serial_cmd(self):
        self.robot_data_.q_d_s_ = self.robot_data_.q_d_.copy()
        self.robot_data_.joint_kp_s_ = self.robot_data_.joint_kp_p_.copy()
        self.robot_data_.joint_kd_s_ = self.robot_data_.joint_kd_p_.copy()

    def _check_and_clip_joint_limits_fast(
            self, cmd_name: int, position: float) -> tuple[bool, float]:
        """
        快速检查并修正关节位置限位（避免重复查询）
        """
        if not self.sim and self.clip_actions:
            limit = self.id_to_limits[cmd_name]
            clipped_pos = np.clip(position, limit["min"], limit["max"])
            return clipped_pos == position, clipped_pos
        else:
            return True, position

    # @timing_decorator
    def convert_to_motor_commands(self):
        """将机器人控制命令转换为电机控制命令"""
        # 使用切片操作一次性复制数据，避免逐个元素赋值
        q_d_reordered = self.robot_data_.q_d_[self.floating_base_dof:]
        qdot_d_reordered = self.robot_data_.q_dot_d_[self.floating_base_dof:]
        tor_d_reordered = self.robot_data_.tau_d_[self.floating_base_dof:]

        # 计算 q_d_reordered - self.zero_offset
        np.subtract(q_d_reordered, self.zero_offset, out=self.temp_q)
        # 计算 (q_d_reordered - self.zero_offset) * self.motor_dir
        np.multiply(self.temp_q, self.motor_dir, out=self.temp_q)
        # 计算最终结果 (q_d_reordered - self.zero_offset) * self.motor_dir + self.zero_pos
        np.add(self.temp_q,
               self.zero_pos,
               out=self.robot_data_.q_d_[self.floating_base_dof:])

        # qdot_d和tor_d的计算也可以向量化
        np.multiply(qdot_d_reordered,
                    self.motor_dir,
                    out=self.robot_data_.q_dot_d_[self.floating_base_dof:])
        np.multiply(tor_d_reordered,
                    self.motor_dir,
                    out=self.robot_data_.tau_d_[self.floating_base_dof:])
        # 如果关节被禁用
        #TODO: 添加通过flag的急停
        if self.disable_joints_:
            self.robot_data_.joint_kp_p_.fill(0.0)
            self.robot_data_.joint_kd_p_.fill(0.0)
            self.robot_data_.tau_d_.fill(0.0)
            self.node.get_logger().warn("Joints disabled!")

    # @timing_decorator
    def publish_motor_commands(self):
        """发布电机控制命令，支持 bodyctrl_msgs 与 ros2_bridge_msgs 两种接口"""
        current_state = self.current_state

        if self.comm_mode == 'ros2_bridge' and _imported_modules.get(
                'ros2_bridge_msgs'):
            self._publish_motor_commands_ros2_bridge()
            return

        if _imported_modules['bodyctrl_msgs'] is None:
            return
        CmdMotorCtrl = _imported_modules['bodyctrl_msgs']['CmdMotorCtrl']
        MotorCtrl = _imported_modules['bodyctrl_msgs']['MotorCtrl']
        # 发布腿部控制命令 (legs_control_status==[] 视为允许所有状态, 与 SDK 路径 _send_motor_commands 对齐)
        leg_gate_ok = (self.legs_control_status == [] or current_state in self.legs_control_status)
        if leg_gate_ok:
            leg_msg = CmdMotorCtrl()
            leg_msg.header.stamp = self.node.get_clock().now().to_msg()
            for i in range(self.id_map.leg_motor_nums):
                cmd = MotorCtrl()
                cmd.name = self.id_map.get_id_by_index(i)
                cmd.kp = float(self.robot_data_.joint_kp_p_[i])
                cmd.kd = float(self.robot_data_.joint_kd_p_[i])
                cmd.pos = float(self.robot_data_.q_d_[i +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[i + self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[i + self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                leg_msg.cmds.append(cmd)
            self.pub_leg_motor_cmd.publish(leg_msg)

        # 只在特定模式下控制腰部
        if current_state in self.waist_control_status:
            # 腰部控制命令
            waist_msg = CmdMotorCtrl()
            waist_msg.header.stamp = self.node.get_clock().now().to_msg()
            for i in range(self.id_map.waist_motor_nums):
                cmd = MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums
                cmd.name = self.id_map.get_id_by_index(
                    motor_idx)  # 12 -> 33(yaw)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                waist_msg.cmds.append(cmd)
            # logger.debug(f'waist_msg {waist_msg}')
            self.pub_waist_motor_cmd.publish(waist_msg)

        # 只在特定模式下控制手臂
        if current_state in self.arms_control_status:
            # 手臂控制命令
            arm_msg = CmdMotorCtrl()
            arm_msg.header.stamp = self.node.get_clock().now().to_msg()
            if current_state in self.left_arm_only_status:
                control_index = np.arange(0, 7)
            elif current_state in self.right_arm_only_status:
                control_index = np.arange(self.id_map.arm_motor_nums - 7,
                                          self.id_map.arm_motor_nums)
            else:
                control_index = np.arange(0, self.id_map.arm_motor_nums)
            for i in control_index:
                cmd = MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                arm_msg.cmds.append(cmd)
            # logger.debug(f'arm_msg {arm_msg}')
            self.pub_arm_motor_cmd.publish(arm_msg)

    def _publish_motor_commands_ros2_bridge(self):
        """ros2_bridge_msgs：发布 LegCtrl/ArmCtrl/WaistCtrl 到 /leg/cmd、/arm/cmd、/waist/cmd"""
        current_state = self.current_state
        if self.pub_leg_motor_cmd is None:
            return
        # 腿部 (legs_control_status==[] 视为允许所有状态, 与 SDK 路径 _send_motor_commands 对齐)
        leg_gate_ok = (self.legs_control_status == [] or current_state in self.legs_control_status)
        if leg_gate_ok:
            leg_msg = LegCtrl()
            leg_msg.header.stamp = self.node.get_clock().now().to_msg()
            leg_msg.header.frame_id = "leg_ctrl"
            leg_msg.mode = 1
            leg_msg.label = 0
            leg_msg.ctrl = []
            for i in range(self.id_map.leg_motor_nums):
                cmd = BridgeMotorCtrl()
                cmd.name = self.id_map.get_id_by_index(i)
                cmd.kp = float(self.robot_data_.joint_kp_p_[i])
                cmd.kd = float(self.robot_data_.joint_kd_p_[i])
                cmd.pos = float(self.robot_data_.q_d_[i +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[i + self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[i + self.floating_base_dof])
                cmd.cur = 0.0
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                leg_msg.ctrl.append(cmd)
            self.pub_leg_motor_cmd.publish(leg_msg)
        # 腰部
        if current_state in self.waist_control_status:
            waist_msg = WaistCtrl()
            waist_msg.header.stamp = self.node.get_clock().now().to_msg()
            waist_msg.header.frame_id = "waist_ctrl"
            waist_msg.mode = 1
            waist_msg.label = 0
            waist_msg.ctrl = []
            for i in range(self.id_map.waist_motor_nums):
                motor_idx = i + self.id_map.leg_motor_nums
                cmd = BridgeMotorCtrl()
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[motor_idx +
                                            self.floating_base_dof])
                cmd.cur = 0.0
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                waist_msg.ctrl.append(cmd)
            self.pub_waist_motor_cmd.publish(waist_msg)
        # 手臂
        if current_state in self.arms_control_status:
            arm_msg = ArmCtrl()
            arm_msg.header.stamp = self.node.get_clock().now().to_msg()
            arm_msg.header.frame_id = "arm_ctrl"
            arm_msg.mode = 1
            arm_msg.label = 0
            arm_msg.ctrl = []
            if current_state in self.left_arm_only_status:
                control_index = np.arange(0, 7)
            elif current_state in self.right_arm_only_status:
                control_index = np.arange(self.id_map.arm_motor_nums - 7,
                                          self.id_map.arm_motor_nums)
            else:
                control_index = np.arange(0, self.id_map.arm_motor_nums)
            for i in control_index:
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                cmd = BridgeMotorCtrl()
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[motor_idx +
                                            self.floating_base_dof])
                cmd.cur = 0.0
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                arm_msg.ctrl.append(cmd)
            self.pub_arm_motor_cmd.publish(arm_msg)

        # 头部
        if current_state in self.head_control_status:
            if self.pub_head_motor_cmd is None:
                return
            head_msg = HeadCtrl()
            head_msg.header.stamp = self.node.get_clock().now().to_msg()
            head_msg.header.frame_id = "head_ctrl"
            head_msg.mode = 1   # ControlMode::CTRL（力位混合）
            head_msg.label = 0
            head_msg.ctrl = []
            for i in range(self.id_map.head_motor_nums):
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums + self.id_map.arm_motor_nums
                cmd = BridgeMotorCtrl()
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[motor_idx +
                                            self.floating_base_dof])
                cmd.cur = 0.0
                head_msg.ctrl.append(cmd)
            self.pub_head_motor_cmd.publish(head_msg)

    # @timing_decorator
    def publish_robot_cmd(self):
        """发布机器人控制命令"""
        now = time.perf_counter()
        if not self._should_publish_motion_control_topic(now):
            return
        # roy 在这里实现你的话题发布代码
        timestamp = self.node.get_clock().now().to_msg()
        # 发布行走控制命令
        walk_msg = TwistStamped()
        walk_msg.header.stamp = timestamp
        walk_msg.twist.linear.x = float(self.robot_data_.walk_cmd_[0])
        walk_msg.twist.linear.y = float(self.robot_data_.walk_cmd_[1])
        walk_msg.twist.angular.z = float(self.robot_data_.walk_cmd_[2])
        self.pub_walk_cmd.publish(walk_msg)

        # 发布腰部控制命令
        stand_msg = TwistStamped()
        stand_msg.header.stamp = timestamp
        stand_msg.twist.angular.x = float(
            self.robot_data_.floating_base_cmd_[0])
        stand_msg.twist.angular.y = float(
            self.robot_data_.floating_base_cmd_[1])
        stand_msg.twist.angular.z = float(
            self.robot_data_.floating_base_cmd_[2])
        stand_msg.twist.linear.z = float(
            self.robot_data_.floating_base_cmd_[3])
        self.pub_stand_cmd.publish(stand_msg)

        # 发布基座线速度
        timestamp = self.node.get_clock().now().to_msg()
        walk_speed_msg = TwistStamped()
        walk_speed_msg.header.stamp = timestamp
        walk_speed_msg.twist.linear.x = self.robot_data_.q_dot_a_[0]
        walk_speed_msg.twist.linear.y = self.robot_data_.q_dot_a_[1]
        walk_speed_msg.twist.linear.z = self.robot_data_.q_dot_a_[2]
        self.pub_walk_speed.publish(walk_speed_msg)

        # 发布站立状态
        stand_status_msg = TwistStamped()
        stand_status_msg.header.stamp = timestamp
        stand_status_msg.twist.angular.x = float(self.robot_data_.q_a_[3])
        stand_status_msg.twist.angular.y = float(self.robot_data_.q_a_[4])
        stand_status_msg.twist.angular.z = float(self.robot_data_.q_a_[5])
        stand_status_msg.twist.linear.x = float(self.robot_data_.q_a_[0])
        stand_status_msg.twist.linear.y = float(self.robot_data_.q_a_[1])
        stand_status_msg.twist.linear.z = float(self.robot_data_.q_a_[2])
        self.pub_stand_status.publish(stand_status_msg)

        # 发布机器人的其余状态
        rl_state_msg = DiagnosticStatus()
        rl_state_msg.name = "rl_state"
        # 添加多个键值对
        #从current_state_class中获取状态信息
        state_info = self.get_state_info()
        # 获取子状态
        child_state = self.get_child_state()
        pairs = [("current_state", state_info["current_state"]),
                 ("child_state", child_state),
                 ("status", state_info["status"]),
                 ]

        for key, value in pairs:
            kv = KeyValue()
            kv.key = key
            kv.value = value
            rl_state_msg.values.append(kv)
        self.pub_robot_state.publish(rl_state_msg)

    @staticmethod
    def _make_sdk_ctrl_body_cmd(sdk_module, frame_id: str, motor_ctrls):
        """将 MotorCtrl 列表封装为新版 BodyCmd（CTRL 模式）。"""
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1e9)
        ctrl_msg = sdk_module.CtrlMsg()
        ctrl_msg.header.stamp.sec = sec
        ctrl_msg.header.stamp.nanosec = nanosec
        ctrl_msg.header.frame_id = frame_id
        ctrl_msg.ctrl = motor_ctrls
        return sdk_module.make_body_cmd_ctrl(ctrl_msg)

    def _send_motor_commands(self):
        """发送电机控制命令，根据 comm_mode 使用对应的 SDK 接口"""
        current_state = self.current_state

        # 根据 comm_mode 选择对应的 SDK 接口和模块
        if self.comm_mode == 'lpc' and hasattr(self, 'sdk_lpc_interface'):
            sdk_interface = self.sdk_lpc_interface
            sdk_module = _imported_modules['robot_control_lpc_python']
        elif self.comm_mode == 'shm_rpc' and hasattr(self,
                                                     'sdk_rpc_interface'):
            sdk_interface = self.sdk_rpc_interface
            sdk_module = _imported_modules['robot_control_shm_rpc_python']
        elif self.comm_mode == 'tcp_rpc' and hasattr(self,
                                                     'sdk_rpc_interface'):
            sdk_interface = self.sdk_rpc_interface
            sdk_module = _imported_modules['robot_control_tcp_rpc_python']
        else:
            return

        if sdk_module is None:
            return

        # [HRIC-INFO] leg gate: 是否允许下发腿部
        leg_gate_ok = (self.legs_control_status == [] or current_state in self.legs_control_status)

        # 发布腿部控制命令（新版接口：send_leg_cmd(BodyCmd)）
        if leg_gate_ok:
            leg_cmds = []
            for i in range(self.id_map.leg_motor_nums):
                cmd = sdk_module.MotorCtrl()
                cmd.name = self.id_map.get_id_by_index(i)
                cmd.kp = float(self.robot_data_.joint_kp_p_[i])
                cmd.kd = float(self.robot_data_.joint_kd_p_[i])
                cmd.pos = float(self.robot_data_.q_d_[i +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[i + self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[i + self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                leg_cmds.append(cmd)
            leg_cmd = self._make_sdk_ctrl_body_cmd(
                sdk_module, "leg_ctrl", leg_cmds)
            try:
                ok = sdk_interface.send_leg_cmd(leg_cmd)
                if not ok:
                    pass
            except Exception as e:
                pass

        # 只在特定模式下控制腰部
        if current_state in self.waist_control_status:
            # 腰部控制命令
            waist_cmds = []
            for i in range(self.id_map.waist_motor_nums):
                cmd = sdk_module.MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums
                cmd.name = self.id_map.get_id_by_index(
                    motor_idx)  # 12 -> 33(yaw)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                waist_cmds.append(cmd)
            waist_cmd = self._make_sdk_ctrl_body_cmd(
                sdk_module, "waist_ctrl", waist_cmds)
            try:
                ok = sdk_interface.send_waist_cmd(waist_cmd)
                if not ok:
                    pass
            except Exception as e:
                pass

        # 只在特定模式下控制手臂
        if current_state in self.arms_control_status:
            # 手臂控制命令
            arm_cmds = []
            if current_state in self.left_arm_only_status:
                control_index = np.arange(0, 7)
            elif current_state in self.right_arm_only_status:
                control_index = np.arange(self.id_map.arm_motor_nums - 7,
                                          self.id_map.arm_motor_nums)
            else:
                control_index = np.arange(0, self.id_map.arm_motor_nums)
            for i in control_index:
                cmd = sdk_module.MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                arm_cmds.append(cmd)
            arm_cmd = self._make_sdk_ctrl_body_cmd(
                sdk_module, "arm_ctrl", arm_cmds)
            try:
                ok = sdk_interface.send_arm_cmd(arm_cmd)
                if not ok:
                    pass
            except Exception as e:
                pass
        
        # 头部
        if current_state in self.head_control_status:
            head_cmds = []
            for i in range(self.id_map.head_motor_nums):
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums + self.id_map.arm_motor_nums
                cmd = sdk_module.MotorCtrl()
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[motor_idx +
                                            self.floating_base_dof])
                head_cmds.append(cmd)
            head_cmd = self._make_sdk_ctrl_body_cmd(
                sdk_module, "head_ctrl", head_cmds)
            try:
                ok = sdk_interface.send_head_cmd(head_cmd)
                if not ok:
                    pass
            except Exception as e:
                pass

    # @timing_decorator
    def send_motor_commands(self):
        """发布电机控制命令"""
        if not self.initialized:
            return
        self.interpolate_action()
        self.backup_serial_cmd()
        if not self.sim:
            #真机
            self.ankle_serial_to_parallel()
        self.convert_to_motor_commands()
        if self.comm_mode in ('ros2', 'ros2_bridge'):
            self.publish_motor_commands()
        elif self.comm_mode in ['lpc', 'shm_rpc', 'tcp_rpc']:
            self._send_motor_commands()
        self.publish_robot_cmd()

    # ROS回调函数
    def _leg_motor_status_callback(self, msg):
        """腿部电机状态回调"""
        try:
            self.queue_leg_motor_state.put_nowait(msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_leg_motor_state.get_nowait()  # 移除旧数据
                self.queue_leg_motor_state.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _arm_motor_status_callback(self, msg):
        """手臂电机状态回调"""
        try:
            self.queue_arm_motor_state.put_nowait(msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_arm_motor_state.get_nowait()  # 移除旧数据
                self.queue_arm_motor_state.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _waist_motor_status_callback(self, msg):
        """腰部电机状态回调"""
        try:
            self.queue_waist_motor_state.put_nowait(msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_waist_motor_state.get_nowait()  # 移除旧数据
                self.queue_waist_motor_state.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _imu_status_callback(self, msg):
        """IMU状态回调"""
        try:
            self.queue_imu_xsens.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_imu_xsens.get_nowait()  # 移除旧数据
                self.queue_imu_xsens.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _robot_state_callback_ros2_bridge(self, msg):
        """ros2_bridge_msgs：将 /robot_state 解包为 leg/arm/waist/imu 队列（与 bodyctrl 下游一致）"""
        try:
            if hasattr(msg, 'leg') and msg.leg.status:
                leg_msg = type('LegStatusMsg', (), {
                    'header': msg.leg.header,
                    'status': msg.leg.status
                })()
                try:
                    self.queue_leg_motor_state.put_nowait(leg_msg)
                except queue.Full:
                    try:
                        self.queue_leg_motor_state.get_nowait()
                        self.queue_leg_motor_state.put_nowait(leg_msg)
                    except Exception:
                        pass
            if hasattr(msg, 'arm') and msg.arm.status:
                arm_msg = type('ArmStatusMsg', (), {
                    'header': msg.arm.header,
                    'status': msg.arm.status
                })()
                try:
                    self.queue_arm_motor_state.put_nowait(arm_msg)
                except queue.Full:
                    try:
                        self.queue_arm_motor_state.get_nowait()
                        self.queue_arm_motor_state.put_nowait(arm_msg)
                    except Exception:
                        pass
            if hasattr(msg, 'waist') and msg.waist.status:
                waist_msg = type('WaistStatusMsg', (), {
                    'header': msg.waist.header,
                    'status': msg.waist.status
                })()
                try:
                    self.queue_waist_motor_state.put_nowait(waist_msg)
                except queue.Full:
                    try:
                        self.queue_waist_motor_state.get_nowait()
                        self.queue_waist_motor_state.put_nowait(waist_msg)
                    except Exception:
                        pass
            if hasattr(msg, 'imu') and msg.imu is not None:
                try:
                    self.queue_imu_xsens.put_nowait(msg.imu)
                except queue.Full:
                    try:
                        self.queue_imu_xsens.get_nowait()
                        self.queue_imu_xsens.put_nowait(msg.imu)
                    except Exception:
                        pass
            if hasattr(msg, 'head') and msg.head.status:
                head_msg = type('HeadStatusMsg', (), {
                    'header': msg.head.header,
                    'status': msg.head.status
                })()
                try:
                    self.queue_head_motor_state.put_nowait(head_msg)
                except queue.Full:
                    try:
                        self.queue_head_motor_state.get_nowait()
                        self.queue_head_motor_state.put_nowait(head_msg)
                    except Exception:
                        pass
        except Exception as e:
            self.node.get_logger().warn(
                f"Error in _robot_state_callback_ros2_bridge: {e}")

    def _walk_cmd_callback(self, msg: TwistStamped):
        """步态控制回调"""
        try:
            self.queue_walk_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_walk_cmd.get_nowait()  # 移除旧数据
                self.queue_walk_cmd.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _stand_cmd_callback(self, msg: TwistStamped):
        """Stand posture command callback (TwistStamped)"""
        try:
            self.queue_stand_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_stand_cmd.get_nowait()
                self.queue_stand_cmd.put_nowait(msg)
            except:
                pass

    def _footpoint_cmd_callback(self, msg: FootPoint):
        """Footpoint command callback"""
        try:
            self.queue_footpoint_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_footpoint_cmd.get_nowait()
                self.queue_footpoint_cmd.put_nowait(msg)
            except:
                pass

    def _fsm_state_cmd_callback(self, msg: String):
        """FSM状态命令回调"""
        try:
            self.queue_fsm_state_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_fsm_state_cmd.get_nowait()  # 移除旧数据
                self.queue_fsm_state_cmd.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略
    
    def _fsm_resume_cmd_callback(self, msg: String):
        """FSM RESUME状态命令回调"""
        try:
            self.queue_fsm_resume_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_fsm_resume_cmd.get_nowait()  # 移除旧数据
                self.queue_fsm_resume_cmd.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

# sdk接口获取数据

    def _start_sdk_polling_loop(self):
        self.sdk_polling_thread = threading.Thread(
            target=self._sdk_robot_status_polling_loop, daemon=True)
        self.sdk_polling_thread.start()

    def _sdk_robot_status_polling_loop(self):
        """SDK 状态轮询循环，根据 comm_mode 使用对应的接口"""
        while True:
            time_start = time.perf_counter()
            ideal_time_interval = 1.0 / 400
            # 根据 comm_mode 选择对应的 SDK 接口
            if self.comm_mode == 'lpc' and hasattr(self, 'sdk_lpc_interface'):
                sdk_interface = self.sdk_lpc_interface
            elif self.comm_mode == 'shm_rpc' and hasattr(
                    self, 'sdk_rpc_interface'):
                sdk_interface = self.sdk_rpc_interface
            elif self.comm_mode == 'tcp_rpc' and hasattr(
                    self, 'sdk_rpc_interface'):
                sdk_interface = self.sdk_rpc_interface
            else:
                time.sleep(0.1)
                continue
            # 手臂状态轮询
            try:
                success, arm_status = sdk_interface.get_arm_status()
            except Exception as e:
                success = False
            if success:
                try:
                    self.queue_arm_motor_state.put_nowait(arm_status)
                except queue.Full:
                    # 队列满时移除旧数据，加入新数据
                    try:
                        self.queue_arm_motor_state.get_nowait()  # 移除旧数据
                        self.queue_arm_motor_state.put_nowait(
                            arm_status)  # 加入新数据
                    except:
                        pass  # 如果仍然无法加入，忽略

            else:
                pass
            # 腿部状态轮询
            try:
                success, leg_status = sdk_interface.get_leg_status()
            except Exception as e:
                success = False
            if success:
                try:
                    self.queue_leg_motor_state.put_nowait(leg_status)
                except queue.Full:
                    # 队列满时移除旧数据，加入新数据
                    try:
                        self.queue_leg_motor_state.get_nowait()  # 移除旧数据
                        self.queue_leg_motor_state.put_nowait(
                            leg_status)  # 加入新数据
                    except:
                        pass
            else:
                pass
            # 腰部状态轮询
            try:
                success, waist_status = sdk_interface.get_waist_status()
            except Exception as e:
                success = False
            if success:
                try:
                    self.queue_waist_motor_state.put_nowait(waist_status)
                except queue.Full:
                    # 队列满时移除旧数据，加入新数据
                    try:
                        self.queue_waist_motor_state.get_nowait()  # 移除旧数据
                        self.queue_waist_motor_state.put_nowait(
                            waist_status)  # 加入新数据
                    except:
                        pass
            else:
                pass
            # 头部状态轮询
            try:
                success, head_status = sdk_interface.get_head_status()
            except Exception as e:
                success = False
            if success:
                try:
                    self.queue_head_motor_state.put_nowait(head_status)
                except queue.Full:
                    try:
                        self.queue_head_motor_state.get_nowait()
                        self.queue_head_motor_state.put_nowait(head_status)
                    except:
                        pass
            else:
                pass
            # Imu状态轮询
            try:
                success, imu_status = sdk_interface.get_imu_status()
            except Exception as e:
                success = False
            if success:
                try:
                    self.queue_imu_xsens.put_nowait(imu_status)
                except queue.Full:
                    # 队列满时移除旧数据，加入新数据
                    try:
                        self.queue_imu_xsens.get_nowait()  # 移除旧数据
                        self.queue_imu_xsens.put_nowait(imu_status)  # 加入新数据
                    except:
                        pass
            else:
                pass
            time_left = ideal_time_interval - (time.perf_counter() -
                                               time_start)
            if time_left < 0:
                pass
            elif time_left > 0.001:
                time.sleep(time_left)

    # @timing_decorator
    def ankle_parallel_to_serial(self):
        # 串并联转换：并转串 (类似C++版本中的处理)
        # 提取左右脚两个踝关节（并联关节）
        q_a_p = np.zeros(4)  # 并联关节角度（实际）
        qdot_a_p = np.zeros(4)  # 并联关节速度（实际）
        tor_a_p = np.zeros(4)  # 并联关节力矩（实际）
        q_a_s = np.zeros(4)  # 串联关节角度（实际）
        qdot_a_s = np.zeros(4)  # 串联关节速度（实际）
        tor_a_s = np.zeros(4)  # 串联关节力矩（实际）

        q_a_p[:2] = self.robot_data_.q_a_[
            self.left_ankle_indices]  # 左脚踝关节 (pitch, roll)
        q_a_p[2:] = self.robot_data_.q_a_[
            self.right_ankle_indices]  # 右脚踝关节 (pitch, roll)

        qdot_a_p[:2] = self.robot_data_.q_dot_a_[self.left_ankle_indices]
        qdot_a_p[2:] = self.robot_data_.q_dot_a_[self.right_ankle_indices]

        tor_a_p[:2] = self.robot_data_.tau_a_[self.left_ankle_indices]
        tor_a_p[2:] = self.robot_data_.tau_a_[self.right_ankle_indices]

        self.q_a_p = q_a_p.copy()
        self.qdot_a_p = qdot_a_p.copy()
        self.tor_a_p = tor_a_p.copy()
        if self.debug:
            pass

        # 计算并转串（正运动学）
        self.fun_s2p.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        self.fun_s2p.calcFK()
        self.fun_s2p.calcIK()

        self.fun_s2p.get_s_state(q_a_s, qdot_a_s, tor_a_s)

        if self.debug:
            pass

        # 用串联关节值替换原来的并联关节值
        self.robot_data_.q_a_[self.left_ankle_indices] = q_a_s[:2]  # 左脚踝关节串联值
        self.robot_data_.q_a_[self.right_ankle_indices] = q_a_s[2:]  # 右脚踝关节串联值

        self.robot_data_.q_dot_a_[self.left_ankle_indices] = qdot_a_s[:2]
        self.robot_data_.q_dot_a_[self.right_ankle_indices] = qdot_a_s[2:]

        self.robot_data_.tau_a_[self.left_ankle_indices] = tor_a_s[:2]
        self.robot_data_.tau_a_[self.right_ankle_indices] = tor_a_s[2:]

    # @timing_decorator
    def ankle_serial_to_parallel(self):
        # 串转并：将串联关节命令转换为并联关节命令（类似C++版本）
        # 提取踝关节两关节的串联命令
        q_d_p = np.zeros(4)  # 并联关节角度（期望）
        qdot_d_p = np.zeros(4)  # 并联关节速度（期望）
        tor_d_p = np.zeros(4)  # 并联关节力矩（期望）
        q_d_s = np.zeros(4)  # 串联关节角度（期望）
        qdot_d_s = np.zeros(4)  # 串联关节速度（期望）
        tor_d_s = np.zeros(4)  # 串联关节力矩（期望）

        q_d_s[:2] = self.robot_data_.q_d_[self.left_ankle_indices]  # 左脚踝关节串联命令
        q_d_s[2:] = self.robot_data_.q_d_[
            self.right_ankle_indices]  # 右脚踝关节串联命令

        qdot_d_s[:2] = self.robot_data_.q_dot_d_[self.left_ankle_indices]
        qdot_d_s[2:] = self.robot_data_.q_dot_d_[self.right_ankle_indices]

        tor_d_s[:2] = self.robot_data_.tau_d_[self.left_ankle_indices]
        tor_d_s[2:] = self.robot_data_.tau_d_[self.right_ankle_indices]

        q_a_s = np.zeros(4)  # 串联关节角度（实际）
        qdot_a_s = np.zeros(4)  # 串联关节速度（实际）
        q_a_s[:2] = self.robot_data_.q_a_[self.left_ankle_indices]  # 左脚踝关节串联值
        q_a_s[2:] = self.robot_data_.q_a_[self.right_ankle_indices]  # 右脚踝关节串联值
        qdot_a_s[:2] = self.robot_data_.q_dot_a_[
            self.left_ankle_indices]  # 左脚踝关节串联速度
        qdot_a_s[2:] = self.robot_data_.q_dot_a_[
            self.right_ankle_indices]  # 右脚踝关节串联速度

        kp = np.zeros(4)  # 串联关节刚度
        kd = np.zeros(4)  # 串联关节阻尼
        kp[:2] = self.robot_data_.joint_kp_p_[self.left_ankle_indices -
                                              self.floating_base_dof]
        kp[2:] = self.robot_data_.joint_kp_p_[self.right_ankle_indices -
                                              self.floating_base_dof]
        kd[:2] = self.robot_data_.joint_kd_p_[self.left_ankle_indices -
                                              self.floating_base_dof]
        kd[2:] = self.robot_data_.joint_kd_p_[self.right_ankle_indices -
                                              self.floating_base_dof]

        tor_d_s = kp * (q_d_s - q_a_s) + kd * (qdot_d_s - qdot_a_s)

        if self.debug:
            pass

        # 串转并计算
        self.fun_s2p.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        self.fun_s2p.calc_joint_pos_ref()
        self.fun_s2p.calc_joint_tor_des()
        self.fun_s2p.get_p_des(q_d_p, qdot_d_p, tor_d_p)

        q_d_p = (tor_d_p - self.ankle_kd_p *
                 (qdot_d_p - self.qdot_a_p)) / self.ankle_kp_p + self.q_a_p

        if self.debug:
            pass

        # 用并联关节命令覆盖原来的串联命令
        self.robot_data_.q_d_[self.left_ankle_indices] = q_d_p[:2]  # 左脚踝关节并联命令
        self.robot_data_.q_d_[self.right_ankle_indices] = q_d_p[
            2:]  # 右脚踝关节并联命令

        self.robot_data_.q_dot_d_[self.left_ankle_indices] = qdot_d_p[:2]
        self.robot_data_.q_dot_d_[self.right_ankle_indices] = qdot_d_p[2:]

        # self.robot_data_.tau_d_[self.left_ankle_indices] = tor_d_p[:2]
        # self.robot_data_.tau_d_[self.right_ankle_indices] = tor_d_p[2:]
        self.robot_data_.tau_d_[self.left_ankle_indices] = 0.0
        self.robot_data_.tau_d_[self.right_ankle_indices] = 0.0
        # 更新脚踝Kp,Kd
        self.robot_data_.joint_kp_p_[
            self.left_ankle_indices -
            self.floating_base_dof] = self.ankle_kp_p[:2]
        self.robot_data_.joint_kp_p_[
            self.right_ankle_indices -
            self.floating_base_dof] = self.ankle_kp_p[2:]
        self.robot_data_.joint_kd_p_[
            self.left_ankle_indices -
            self.floating_base_dof] = self.ankle_kd_p[:2]
        self.robot_data_.joint_kd_p_[
            self.right_ankle_indices -
            self.floating_base_dof] = self.ankle_kd_p[2:]

    def update_robot_cmd(self):
        """更新机器人控制命令"""
        # 在这里把topic 收到的指令传递给robot_data中的命令变量
        self.get_walk_cmd()
        # self.get_walk_height_cmd()
        self.get_stand_cmd()
        self.get_footpoint_cmd()
        #通过话题获取状态切换指令
        self.get_fsm_state_cmd()
        self.apply_moe_state_commands()
        #通过话题获取resume指令
        self.get_fsm_resume_cmd()

    def get_control_flag(self):
        return self.fsm_control_flag


    def interpolate_action(self):
        #根据self.current_state 和 self.last_state的状态决定要对哪些电机插值
        self.trans_flag_ = False
        if self.robot_data_.trans_start_time > 0.01:
            # 计算插值
            trans_time = self._get_current_transition_time()
            if trans_time == 0.0:
                current_time = self.robot_data_.time_now_
                default_protect_trans_time = 2.0
                if current_time - self.robot_data_.trans_start_time <= default_protect_trans_time:
                    # 由于状态未指定明确过渡时间，为保护切换稳定性，强制{default_protect_trans_time}秒内，设转换flag设为True
                    self.trans_flag_ = True
                return
            alpha = (self.robot_data_.time_now_ -
                     self.robot_data_.trans_start_time) / trans_time
            if alpha > 1.0:
                return
            self.trans_flag_ = True
            alpha = np.clip(alpha, 0.0, 1.0)

            transition_start_time = self.robot_data_.trans_start_time
            if self._cached_transition_start_time != transition_start_time:
                self._cached_transition_start_time = transition_start_time
                self._cached_last_state_serial_qd = self.robot_data_.get_last_state_serial_qd()
                self._cached_last_state_serial_kp = self.robot_data_.get_last_state_serial_kp()
                self._cached_last_state_serial_kd = self.robot_data_.get_last_state_serial_kd()
                if self.last_state in self.dynamic_inter_status:
                    q_d_s_arm, joint_index = self.dynamic_model.compute_desired_joint_pos(
                        self._cached_last_state_serial_kp, self._cached_last_state_serial_kd)
                    self._cached_last_state_serial_qd[joint_index] = q_d_s_arm

            q_d_s_last_state = self._cached_last_state_serial_qd
            joint_kp_s_last_state = self._cached_last_state_serial_kp
            joint_kd_s_last_state = self._cached_last_state_serial_kd


            # 获取需要插值的电机ID集合
            last_ids = self.state_motor_id_map.get(self.last_state, [])
            current_ids = self.state_motor_id_map.get(self.current_state, [])
            # intersection = set(last_ids) & set(current_ids)
            intersection = current_ids
            
            q_d_ = self.robot_data_.get_desired_joint_pos()
            joint_kp = self.robot_data_.get_joint_kp()
            joint_kd = self.robot_data_.get_joint_kd()
            for motor_id in intersection:
                idx = self.floating_base_dof + motor_id
                self.robot_data_.q_d_[idx] = (q_d_s_last_state[motor_id] *
                                              (1 - alpha) +
                                              q_d_[motor_id] * alpha)
                self.robot_data_.joint_kp_p_[motor_id] = (joint_kp_s_last_state[motor_id] *
                                                          (1 - alpha) +
                                                          joint_kp[motor_id] * alpha)
                self.robot_data_.joint_kd_p_[motor_id] = (joint_kd_s_last_state[motor_id] *
                                                          (1 - alpha) +
                                                          joint_kd[motor_id] * alpha)

    def get_state_xyyaw_speed_limits(self, state_name: str):
        xyyaw_speed_limits = {
            "max_x_plus": self.max_x_plus_speed,
            "max_x_minus": self.max_x_minus_speed,
            "max_y": self.max_y_speed,
            "max_yaw": self.max_yaw_speed,
            "x_command_offset": self.x_command_offset,
            "y_command_offset": self.y_command_offset,
            "yaw_command_offset": self.yaw_command_offset
        }
        if state_name in self.state_speed_limits:
            state_limits = self.state_speed_limits[state_name]
            xyyaw_speed_limits["max_x_plus"] = state_limits.get("max_x_plus", self.max_x_plus_speed)
            xyyaw_speed_limits["max_x_minus"] = state_limits.get("max_x_minus", self.max_x_minus_speed)
            xyyaw_speed_limits["max_y"] = state_limits.get("max_y", self.max_y_speed)
            xyyaw_speed_limits["max_yaw"] = state_limits.get("max_yaw", self.max_yaw_speed)
            xyyaw_speed_limits["x_command_offset"] = state_limits.get("x_command_offset", self.x_command_offset)
            xyyaw_speed_limits["y_command_offset"] = state_limits.get("y_command_offset", self.y_command_offset)
            xyyaw_speed_limits["yaw_command_offset"] = state_limits.get("yaw_command_offset", self.yaw_command_offset)
        return xyyaw_speed_limits

    
    def get_motion_end_flag(self):
        """获取运动结束标志
        当前状态类有is_motion_end属性则返回其值，否则根据trans_flag_判断：
        trans_flag_为True表示正在过渡状态，运动未结束
        trans_flag_为False表示不在过渡状态，运动已结束
        """
        if hasattr(self.current_state_class, 'is_motion_end'):
            return self.current_state_class.is_motion_end
        else:
            return False
    
    def get_state_info(self):
        """获取状态信息"""
        is_motion_end = self.get_motion_end_flag()
        status = "start" if self.trans_flag_ else "running"
        if is_motion_end:
            status = "finish"
        return {
            "current_state": self.current_state.name,
            "last_state": self.last_state.name,
            "trans_flag": self.trans_flag_,
            "status": status
        }

    def apply_moe_state_commands(self):
        """根据 FSM 命令，将配置映射写入 robot_data 对应属性"""
        cmd = self.fsm_control_flag.fsm_state_command
        moe_state_command = self.moe_state_commands_map.get(cmd)
        if not moe_state_command:
            return
        self.robot_data_.set_moe_state_command(moe_state_command)

    def get_child_state(self):
        """获取子状态"""
        if self.current_state in self.moe_states:
            return self.current_state_class.current_state
        return self.current_state.name

def get_robot_interface(robot_data: RobotData, config_path: str) -> RobotInterface:
    """工厂函数，返回机器人接口实例"""
    return RobotInterfaceImpl(robot_data, config_path)
