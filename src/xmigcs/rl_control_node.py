"""
RL Control Plugin (Python Version)
Main ROS2 node for humanoid robot RL control system
"""
import math
import os
import threading
import time
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# Local imports
from xmigcs.common.robot_data import RobotData
from xmigcs.FSM.robot_fsm import get_robot_fsm
from xmigcs.common.robot_interface import get_robot_interface
from xmigcs.common.hric_server_adapter import create_hric_server
from xmigcs.utils.logging_utils import get_logger
from xmigcs.utils.xlog_utils import configure_xlog, xlog
# from xmigcs.common.stdin_keyboard_control  import KeyboardController
import functools


logger = get_logger(__name__)

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
        logger.info(f"[TIMING] {func.__name__} executed in {execution_time:.6f} seconds")
        return result
    return wrapper


class XMIGCSControlNode(Node):
    """xMIGCS控制节点Python版本"""

    def __init__(self, ):
        super().__init__('xmigcs_control_node')

        # 配置和参数
        self.floating_base_dof = 6

        # 加载配置
        self._load_config()

        # 初始化数据结构
        self._init_data_structures()

        # 机器人FSM
        self.robot_fsm = get_robot_fsm(
            self.robot_data,
            self.config,
        )

        # 机器人接口
        self.robot_interface = get_robot_interface(self.robot_data,
                                                   self.config_file)
        self.robot_interface.init(self)  # 传入node实例

        # 启动 HRIC 反向 RPC Server (接收 robot_control 转发的 /hric/robot/* 命令)
        self._init_hric_server()

        # 启动控制线程
        self._start_control_thread()

    def _init_hric_server(self):
        """根据 dex_config.yaml 中 hric_server 段创建并启动 HRIC 反向 RPC Server."""
        self.hric_server = None
        self.hric_adapter = None  # 必须持有强引用, 否则 pybind11 trampoline 会失效

        hric_cfg = (self.config or {}).get('hric_server') or {}
        if not self.get_parameter('hric_server_enabled').value:
            xlog.info("[HRIC] hric_server.enabled=false, HRIC reverse RPC disabled")
            return

        # 单一通信模式, **无 fallback**, 必须与 robot_control 端 --hric-comm-mode 一致.
        # 非法值直接抛 ValueError, 早失败优于运行期诡异行为.
        comm_mode = str(hric_cfg.get('comm_mode', 'shm_rpc')).lower()
        tcp_port = int(hric_cfg.get('tcp_port', 50052))
        shm_service_name = str(hric_cfg.get('shm_service_name', 'XmigcsHricRPC'))

        # ROS 参数可覆盖 yaml (便于 mock / 联调, 不必改 dex_config)
        override = str(self.get_parameter('hric_comm_mode').value).strip().lower()
        if override:
            comm_mode = override
        port_override = int(self.get_parameter('hric_tcp_port').value)
        if port_override > 0:
            tcp_port = port_override
        shm_override = str(self.get_parameter('hric_shm_service').value).strip()
        if shm_override:
            shm_service_name = shm_override

        server, adapter = create_hric_server(self.robot_interface,
                                             comm_mode=comm_mode,
                                             tcp_port=tcp_port,
                                             shm_service_name=shm_service_name)
        if server is None:
            return

        self.hric_server = server
        self.hric_adapter = adapter
        if self.hric_server.start():
            # 只汇报当前选中的 backend 状态
            if comm_mode == 'tcp_rpc':
                endpoint_info = f"tcp={tcp_port} running={self.hric_server.is_tcp_running()}"
            elif comm_mode == 'shm_rpc':
                endpoint_info = f"shm_service={shm_service_name} running={self.hric_server.is_shm_running()}"
            else:  # lpc
                endpoint_info = f"lpc_registered={self.hric_server.is_lpc_registered()}"
            xlog.info(f"[HRIC] HricServer started (comm_mode={comm_mode} {endpoint_info})")
        else:
            xlog.warning(f"[HRIC] HricServer failed to start "
                         f"(comm_mode={comm_mode}, tcp_port={tcp_port}, "
                         f"shm_service_name={shm_service_name})")

    def _load_config(self):
        """加载配置文件"""
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(current_dir, 'config',
                                        'dex_config.yaml')

        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # 提取关键配置参数
        self.motor_num = self.config.get('motor_num')
        self.dt = self.config.get('dt')
        self.sim = self.config.get('sim')
        self.debug = self.config.get('debug')

        # 从配置中获取 robot_interface 配置，作为参数默认值来源
        robot_interface_cfg = self.config.get('robot_interface')
        default_comm_mode = robot_interface_cfg.get('comm_mode', 'ros2')

        # 声明 ROS2 参数，用于在运行时覆盖通讯方式
        # 例如：--ros-args -p comm_mode:=ros2_bridge / lpc / shm_rpc / tcp_rpc
        if not self.has_parameter('comm_mode'):
            self.declare_parameter('comm_mode', default_comm_mode)
        if not self.has_parameter('hric_comm_mode'):
            self.declare_parameter('hric_comm_mode', '')
        if not self.has_parameter('hric_tcp_port'):
            self.declare_parameter('hric_tcp_port', 0)
        if not self.has_parameter('hric_shm_service'):
            self.declare_parameter('hric_shm_service', '')
        if not self.has_parameter('hric_server_enabled'):
            self.declare_parameter('hric_server_enabled', False)

        # 检查当前用户名，如果是ubuntu则抛出异常
        import getpass
        self.user_name = getpass.getuser().lower()
        if self.sim and self.user_name == 'ubuntu':
            raise RuntimeError("On ubuntu user, sim must be set to false")

    def _init_data_structures(self):
        """初始化数据结构"""
        # 机器人数据
        self.robot_data = RobotData(self.motor_num, self.motor_num + self.floating_base_dof)
        self.robot_data.config_file_ = getattr(self, 'config_file', '')


    def _start_control_thread(self):
        """启动控制线程"""
        self.control_running = True
        self.control_thread = threading.Thread(target=self._rl_control_loop,
                                               daemon=True)
        self.control_thread.start()
        xlog.info("Control thread started")

    def _rl_control_loop(self):
        """主控制循环"""
        xlog.info("RL control loop starting...")

        # 初始化时间戳
        time_passed = 0.0
        control_step = 0
        # 更新机器人数据
        self._update_robot_data(time_passed, control_step)
        # 切换到状态机初始状态
        self.robot_fsm.first_enter()
        last_control_freq_log_time = time.perf_counter()

        while self.control_running and rclpy.ok():

            # 运行FSM
            loop_start = time.perf_counter()
            self.robot_fsm.run_fsm(self.robot_interface.get_control_flag())

            # 发布控制命令
            self.robot_interface.update_fsm_state(
                current_state=self.robot_fsm.get_current_state(),
                current_state_class=self.robot_fsm.get_current_state_class())
            self._send_control_commands()

            # 更新时间戳
            time_passed += self.dt
            control_step += 1

            # 更新机器人数据
            self._update_robot_data(time_passed, control_step)

            # 控制频率
            self._precise_sleep_until(loop_start + self.dt)
            control_freq = 1 / (time.perf_counter() - loop_start)
            if time.perf_counter() - last_control_freq_log_time >= 5.0:
                xlog.info(f"current control freq: {control_freq:.2f} Hz")
                last_control_freq_log_time = time.perf_counter()

        xlog.info("RL control loop ended")

    def _precise_sleep_until(self, target_time):
        """精确睡眠到目标时间"""
        current_time = time.perf_counter()
        sleep_time = target_time - current_time

        if sleep_time <= 0:
            return  # 已经超时，立即返回

        # 分级睡眠策略
        if sleep_time > 0.003:  # 3ms以上使用混合睡眠
            # 先睡眠大部分时间
            time.sleep(sleep_time * 0.9)
            # 剩余时间忙等待
            while time.perf_counter() < target_time:
                pass
        else:  # 3ms以内纯忙等待
            while time.perf_counter() < target_time:
                pass

    # @timing_decorator
    def _update_robot_data(self, time_passed: float, control_step: int):
        """更新机器人数据"""
        self.robot_interface.update_robot_data(time_passed, control_step)

    def _send_control_commands(self, ):
        """发布控制命令"""
        # 通过robot_interface发布控制命令
        self.robot_interface.send_motor_commands()

    def destroy_node(self):
        """节点销毁"""
        self.control_running = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)

        # 停止 HRIC server (必须在 adapter 被释放前停止 RPC 线程)
        try:
            if getattr(self, 'hric_server', None) is not None:
                self.hric_server.stop()
        except Exception as exc:
            xlog.warning(f"[HRIC] error while stopping HricServer: {exc}")
        finally:
            self.hric_server = None
            self.hric_adapter = None

        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

        super().destroy_node()


def main(args=None):
    """主函数"""
    configure_xlog(logger_id="xmigcs")
    rclpy.init(args=args)
    node = None
    try:
        node = XMIGCSControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals() and node is not None:
            node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
