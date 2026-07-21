"""
HRIC Server Adapter (v3.5.2)

实现 IHricService (来自 xmigcs_hric_python pybind11 模块):
- 整个类对外 **只有一个 set_command 接口**;
- 内部按 cmd.type 查描述表 _DISPATCH_TABLE, 直接调用 xmigcs 原生接口
  (RobotInterfaceImpl 的 5 个 PeekableQueue: queue_walk_cmd / queue_stand_cmd /
  queue_footpoint_cmd / queue_fsm_state_cmd / queue_fsm_resume_cmd);
- 无任何单独的 _on_xxx_cmd 方法 (v3.0 起去除).

返回值语义 (v3.5.1 修复 v3.5 字面匹配 bug):
- fire-and-forget (walk_cmd / stand_cmd / footpoint_cmd / fsm_resume_cmd):
  True 表示"已成功入队", 不保证 FSM 实际执行
- FSM_STATE_CMD 强制确认 (无降级开关):
  set_command 主动轮询 self._iface.current_state.name (由 RL 主循环 ≥50Hz
  通过 robot_interface.update_param 持续刷新). v3.5.2 起把"状态检查"和
  "重新入队"两个周期解耦:
    * 状态检查 50ms 一次 (FSM_STATE_CHECK_INTERVAL_S, 决定响应延迟)
    * 重入队 1s 一次 (fsm_state_confirm_poll_s, 避免刷 FSM 队列)
    * 总超时 5s (fsm_state_confirm_timeout_s)
  这样 FSM 实测 <20ms 切完, adapter ~50ms 就 return True, 不必等满 1s.
  - **成功判据**: current_state.name 从 prev 变化为任意非空的其他态.
    FSM 接受 trigger 并 transition 即视为成功, 不再要求字面匹配
    'goto<X>' 中的 X (实测 FSM 配置里 trigger 名 ≠ 目标态名是常见场景:
    gotoSTART -> SWR, gotoZERO -> STANDZERO).
  - **失败判据**: 整个 timeout 内 current_state.name 始终未变化, 返回 False.
  - **幂等快速路径**: 入队前若 current_state.name 字面 == 'goto<X>' 中的 X,
    立即返回 True 不入队 (joystick 50Hz 按住按键的优化, 仅适用 X 恰好等于
    某个状态名的场景, 如 gotoSTANDZERO/gotoHBWALK).
  返回 True  = FSM 发生 transition / 幂等快速路径
  返回 False = 总超时仍未切换 (期间已重试 N=timeout/poll 次)

构造期强校验 (v3.5 新增, 替代旧降级):
- robot_interface **必须** 暴露 current_state.name, 缺失 -> 抛 RuntimeError
- fsm_state_confirm_timeout_s **必须** > 0, 否则 -> 抛 ValueError
- fsm_state_confirm_poll_s **必须** > 0 且 <= timeout, 否则 -> 抛 ValueError
- 旧的"自动回退 fire-and-forget"行为已彻底移除, 调用方必须显式满足前置条件

零侵入:
- 不依赖 FSM transition listener, 不依赖任何 FSM 内部 API, 不要求 rl_control_node
  额外注入. adapter 仅访问已有的 self._iface.current_state.name (CPython GIL
  保证原子读), 主循环线程负责写, RPC 线程负责读, 无竞争.

重试语义:
- 第一次入队若被 FSM 拒绝 (condition_failed / no_transition 等), adapter 不放弃,
  会在每个 poll 周期 (默认 1s) 再次入队同一命令, 让 FSM 在后续 tick 仍有机会
  消费. 典型场景: 目标状态需要前置条件 (如电池电量恢复 / 控制器 ready), 重试
  期间条件满足后即可成功切换, 调用方无需自己写重试循环.

数据流:
    robot_control (rclcpp 订阅 5 个 topic)
        --> HricRpcClient.setXxxCmd (上层零改动, 内部都构造 envelope)
        --> lpc / shm_rpc / tcp_rpc
        --> HricServer (xmigcs)
        --> HricServerAdapter.set_command(envelope)
        --> 查 _DISPATCH_TABLE[envelope.type]
            -> 取 payload (twist / string_data / footpoint)
            -> POD->ROS 转换
            -> _push_latest(self._iface.queue_*, ros_msg)
            -> 若 FSM_STATE_CMD: 轮询 self._iface.current_state.name 直到目标或超时
        --> FSM (现有消费侧无感知)

扩展新命令仅需 3 步, 不改方法签名:
    1) C++ HricCommandType 末尾新增枚举值 + HricCommand envelope 增加 payload 字段
    2) 在 _DISPATCH_TABLE 末尾追加一行 _DispatchEntry
    3) (若是新 ROS 消息类型) 增加一个 POD->ROS 转换函数

线程安全:
  HricServer 在独立 RPC 线程调用 set_command. PeekableQueue 内部使用 deque + lock,
  天然线程安全; 本类内部的去重字段 _last_*_data 仅在 RPC 线程读写, 无并发风险.
  current_state 字段 CPython 下单赋值原子, 轮询读无须加锁.

时间戳:
  不依赖 rclpy 的 now(), 直接用 RPC envelope 携带的 stamp 字段重建 ROS msg,
  使 FSM 中 decay_if_ros_stamp_stale 仍能基于真实下发时间工作.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped

import sys as _sys  # noqa: E402  用于 _safe_log 的 stderr 输出
from xmigcs.utils.xlog_utils import xlog  # noqa: F401  保留以兼容外部 monkey-patch

# 关键: HRIC RPC 后端线程 (TCP/SHM/LPC) 经由 pybind11 trampoline 进入 set_command,
# 此时所在线程**不是** Python 主线程, 也没有合法的 Python frame.
# - xlog (g3log C++ binding) 在这种上下文取 frame info (filename/line) 时会 SEGV
# - Python 标准 logging.info 也会调 sys._getframe() 查 caller, 同样不安全
# 因此 adapter 内所有日志一律走 **print(file=sys.stderr, flush=True)** 直出.
# print 由 CPython 用 stdout/stderr lock 保护, 多线程下不交错, 不依赖任何
# Python frame inspection 机制. 这是实测唯一稳定的方案.


class _SafeLog:
    """thread-safe + frame-free logger, 接口 mimic logging.Logger.{info,warning}."""

    @staticmethod
    def info(msg: str) -> None:
        try:
            print(f"[INFO] [hric_adapter] {msg}", file=_sys.stderr, flush=True)
        except Exception:
            pass

    @staticmethod
    def warning(msg: str) -> None:
        try:
            print(f"[WARN] [hric_adapter] {msg}", file=_sys.stderr, flush=True)
        except Exception:
            pass


_safe_logger = _SafeLog()


# ============================================================
# [HRIC-INFO] 简易节流日志: 每个 key 按时间间隔输出一次
# (joystick 按住时 walk_cmd 50Hz 上报, 不节流会刷屏)
# 用模块级 dict + 单赋值: CPython GIL 保证原子, 无须加锁
# ============================================================
_DBG_LAST_T: Dict[str, float] = {}


def _dbg_throttled(key: str, interval_s: float = 1.0) -> bool:
    """对同 key 至多 interval_s 一次返回 True. 多线程下安全 (GIL 原子写)."""
    now = time.monotonic()
    last = _DBG_LAST_T.get(key, 0.0)
    if now - last >= interval_s:
        _DBG_LAST_T[key] = now
        return True
    return False

# hric_msgs/FootPoint 可选
try:
    from hric_msgs.msg import FootPoint
    HAS_FOOTPOINT_MSG = True
except ImportError:  # pragma: no cover
    HAS_FOOTPOINT_MSG = False

# pybind11 模块 (由 robot_control/src/bridge/hric/python 构建)
try:
    import xmigcs_hric_python as _hric_native
    IHricServiceBase = _hric_native.IHricService
    HricServer = _hric_native.HricDeviceServer  # libhric_rpc.so SDK (HricServer 为向后兼容别名)
    HricDeviceServer = _hric_native.HricDeviceServer
    HricRpcConfig = _hric_native.HricRpcConfig
    HricCommandType = _hric_native.HricCommandType
    HAS_HRIC_NATIVE = True
except ImportError:  # pragma: no cover
    _hric_native = None
    HAS_HRIC_NATIVE = False

    class IHricServiceBase:  # 占位, 仅当 native 模块不可用时使用
        pass

    HricServer = None  # type: ignore
    HricDeviceServer = None  # type: ignore
    HricRpcConfig = None  # type: ignore
    HricCommandType = None  # type: ignore


# ============================================================
# POD -> ROS msg 转换工具 (与 ros2 订阅路径行为等价)
# ============================================================

def _push_latest(q: "queue.Queue", msg) -> bool:
    """与 robot_interface._walk_cmd_callback 等回调一致的'满则丢旧入新'语义."""
    try:
        q.put_nowait(msg)
        return True
    except queue.Full:
        try:
            q.get_nowait()
            q.put_nowait(msg)
            return True
        except Exception:
            return False


def _twist_pod_to_ros(pod) -> TwistStamped:
    msg = TwistStamped()
    msg.header.stamp.sec = int(pod.header.stamp.sec)
    msg.header.stamp.nanosec = int(pod.header.stamp.nanosec)
    msg.header.frame_id = pod.header.frame_id
    msg.twist.linear.x = float(pod.linear_x)
    msg.twist.linear.y = float(pod.linear_y)
    msg.twist.linear.z = float(pod.linear_z)
    msg.twist.angular.x = float(pod.angular_x)
    msg.twist.angular.y = float(pod.angular_y)
    msg.twist.angular.z = float(pod.angular_z)
    return msg


def _string_pod_to_ros(pod) -> String:
    msg = String()
    msg.data = pod.data
    return msg


def _footpoint_pod_to_ros(pod):
    if not HAS_FOOTPOINT_MSG:
        return None
    msg = FootPoint()
    msg.stamp.sec = int(pod.stamp.sec)
    msg.stamp.nanosec = int(pod.stamp.nanosec)
    msg.footflag = bool(pod.footflag)
    msg.relative_x = float(pod.relative_x)
    msg.relative_y = float(pod.relative_y)
    msg.relative_yaw = float(pod.relative_yaw)
    msg.distance = float(pod.distance)
    msg.frame_id = pod.frame_id
    return msg


# ============================================================
# 命令描述表 (Dispatch Entry): set_command 内部完全数据驱动
# ============================================================

@dataclass(frozen=True)
class _DispatchEntry:
    """单条命令分发描述: 描述一个 HricCommandType 如何映射到 xmigcs 原生接口.

    字段:
      queue_attr   : robot_interface 上 PeekableQueue 的属性名
      pod_field    : HricCommand envelope 上要读取的 payload 字段名
                     (twist / string_data / footpoint)
      converter    : POD -> ROS msg 转换函数
      label        : 日志短标识
      dedup_attr   : 可选, 高频去重日志状态字段名 (如 '_last_fsm_state_data')
                     None 表示不去重, 不打 INFO 日志
      requires_footpoint_msg : True 表示该 type 依赖 hric_msgs/FootPoint 包
    """
    queue_attr: str
    pod_field: str
    converter: Callable
    label: str
    dedup_attr: Optional[str] = None
    requires_footpoint_msg: bool = False


def _build_dispatch_table() -> Dict[int, _DispatchEntry]:
    """构造 type -> _DispatchEntry 表. native 模块缺失时返回空表."""
    if not HAS_HRIC_NATIVE:
        return {}
    return {
        int(HricCommandType.WALK_CMD): _DispatchEntry(
            queue_attr="queue_walk_cmd",
            pod_field="twist",
            converter=_twist_pod_to_ros,
            label="walk_cmd",
        ),
        int(HricCommandType.STAND_CMD): _DispatchEntry(
            queue_attr="queue_stand_cmd",
            pod_field="twist",
            converter=_twist_pod_to_ros,
            label="stand_cmd",
        ),
        int(HricCommandType.FOOTPOINT_CMD): _DispatchEntry(
            queue_attr="queue_footpoint_cmd",
            pod_field="footpoint",
            converter=_footpoint_pod_to_ros,
            label="footpoint_cmd",
            requires_footpoint_msg=True,
        ),
        int(HricCommandType.FSM_STATE_CMD): _DispatchEntry(
            queue_attr="queue_fsm_state_cmd",
            pod_field="string_data",
            converter=_string_pod_to_ros,
            label="fsm_state_cmd",
            dedup_attr="_last_fsm_state_data",
        ),
        int(HricCommandType.FSM_RESUME_CMD): _DispatchEntry(
            queue_attr="queue_fsm_resume_cmd",
            pod_field="string_data",
            converter=_string_pod_to_ros,
            label="fsm_resume_cmd",
            dedup_attr="_last_fsm_resume_data",
        ),
    }


# ============================================================
# HricServerAdapter: 唯一公共接口 set_command, 内部纯数据驱动
# ============================================================

class HricServerAdapter(IHricServiceBase):
    """实现 IHricService, 只暴露 set_command 一个方法.

    与 robot_interface 解耦: 依赖其 5 个 queue_* 属性 + current_state 属性.
    (current_state 是 FSM_STATE_CMD "真切换确认" 的强依赖, v3.5 起为**必填**;
    构造时缺失直接抛 RuntimeError, 不再回退 fire-and-forget.)
    """

    # FSM_STATE_CMD 切换确认参数 (v3.5.2: 检查/重入队解耦):
    #   timeout : 总等待上限. FSM on_enter 极端情况可达数秒, 给 5s. 必须 > 0.
    #   poll    : 重新入队周期 (避免刷 FSM 队列, 1s 足够). 必须 > 0 且 ≤ timeout.
    #   check   : 状态检查周期 (细粒度, 决定响应延迟, 默认 50ms).
    #            FSM 实测切换 <20ms 完成, 50ms check 即可立即返回, 不必等满 poll.
    #            自动取 min(FSM_STATE_CHECK_INTERVAL_S, poll), 不暴露给调用方.
    DEFAULT_FSM_STATE_CONFIRM_TIMEOUT_S: float = 5.0
    DEFAULT_FSM_STATE_CONFIRM_POLL_S: float = 1.0
    FSM_STATE_CHECK_INTERVAL_S: float = 0.05   # 50ms 细粒度状态轮询, 决定确认响应延迟
    GOTO_PREFIX: str = "goto"  # 事件名约定: gotoZERO -> 期望状态 ZERO (字面 hint, 非强约束)

    def __init__(self,
                 robot_interface,
                 fsm_state_confirm_timeout_s: Optional[float] = None,
                 fsm_state_confirm_poll_s: Optional[float] = None):
        super().__init__()
        self._iface = robot_interface
        # 高频按键去重日志状态 (由 _DISPATCH_TABLE 中的 dedup_attr 指向)
        self._last_fsm_state_data: Optional[str] = None
        self._last_fsm_resume_data: Optional[str] = None
        self._dispatch_table: Dict[int, _DispatchEntry] = _build_dispatch_table()
        # FSM 确认在后台线程轮询, set_command 入队后立即返回, 避免阻塞 iceoryx2 RPC 线程.
        self._fsm_confirm_lock = threading.Lock()
        self._fsm_confirm_inflight_event: Optional[str] = None

        # ----- FSM_STATE_CMD 轮询确认参数 (v3.5: 强校验, 不接受 ≤ 0) -----
        timeout = (float(fsm_state_confirm_timeout_s)
                   if fsm_state_confirm_timeout_s is not None
                   else self.DEFAULT_FSM_STATE_CONFIRM_TIMEOUT_S)
        poll = (float(fsm_state_confirm_poll_s)
                if fsm_state_confirm_poll_s is not None
                else self.DEFAULT_FSM_STATE_CONFIRM_POLL_S)
        if timeout <= 0:
            raise ValueError(
                f"[HricServerAdapter] fsm_state_confirm_timeout_s must be > 0 "
                f"(got {timeout}). FSM_STATE_CMD confirmation is mandatory in v3.5+; "
                f"fire-and-forget downgrade has been removed."
            )
        if poll <= 0:
            raise ValueError(
                f"[HricServerAdapter] fsm_state_confirm_poll_s must be > 0 (got {poll})."
            )
        if poll > timeout:
            raise ValueError(
                f"[HricServerAdapter] fsm_state_confirm_poll_s ({poll}) must be <= "
                f"fsm_state_confirm_timeout_s ({timeout})."
            )
        self._fsm_confirm_timeout_s: float = timeout
        self._fsm_confirm_poll_s: float = poll

        # ----- current_state 是 FSM_STATE_CMD 确认的强依赖, 缺失直接拒绝启动 -----
        if not self._probe_iface_has_current_state(robot_interface):
            raise RuntimeError(
                "[HricServerAdapter] robot_interface must expose 'current_state' attribute "
                "(with a '.name: str' field) for FSM_STATE_CMD confirmation. "
                "Fire-and-forget downgrade has been removed in v3.5+. "
                "RobotInterfaceImpl already provides this field by default; "
                "if you are passing a custom/mock interface, please add: "
                "`self.current_state = <obj with .name attribute>`."
            )

        _safe_logger.info(
            f"[HricServerAdapter] FSM_STATE_CMD confirmation enabled (v3.5.3, single-push) "
            f"(timeout={self._fsm_confirm_timeout_s:.3f}s, "
            f"check={self.FSM_STATE_CHECK_INTERVAL_S:.3f}s, "
            f"poll_s={self._fsm_confirm_poll_s:.3f}s [legacy, no longer used for repush], "
            f"goto-convention='{self.GOTO_PREFIX}<STATE>')"
        )

    @staticmethod
    def _probe_iface_has_current_state(iface) -> bool:
        """检测 robot_interface 是否暴露 .current_state.name 路径."""
        try:
            state = getattr(iface, "current_state", None)
            if state is None:
                return False
            return hasattr(state, "name")
        except Exception:
            return False

    def _read_iface_state_name(self) -> Optional[str]:
        """安全读取 robot_interface.current_state.name. CPython 单赋值原子, 无须加锁."""
        try:
            state = getattr(self._iface, "current_state", None)
            if state is None:
                return None
            name = getattr(state, "name", None)
            return name if isinstance(name, str) else None
        except Exception:
            return None

    # ----------- 由 C++ HricServer 在 RPC 线程调用 (唯一公共接口) -----------

    def get_current_fsm_state(self) -> str:
        """供 robot_control HRIC client 查询当前 FSM 状态 (shm_rpc, 非 topic)."""
        return self._read_iface_state_name() or ""

    def set_command(self, cmd) -> bool:
        """统一命令入口 (唯一公共方法).

        步骤:
            1) 校验 cmd 与 cmd.type
            2) 查 _dispatch_table 取 _DispatchEntry
            3) 取 cmd 上对应 payload 字段 (entry.pod_field)
            4) 调 entry.converter 转 ROS msg
            5) (可选) 去重日志: 仅在数据变化时打 INFO, 避免高频按键刷屏
            6) _push_latest 写入 self._iface 上对应队列 (entry.queue_attr)
            7) 若 entry.label == "fsm_state_cmd":
               入队后 _set_command_fsm_state_confirmed 立即 return True;
               状态确认在后台线程轮询 (不阻塞 RPC, 避免饿死 walk_cmd).
               其他命令立即返回入队结果 (fire-and-forget).
        """
        if cmd is None:
            _safe_logger.warning("[HricServerAdapter] set_command got None")
            return False

        # cmd.type 在 pybind11 binding 中是 HricCommandType 枚举, int() 提取底层值
        try:
            cmd_type_int = int(cmd.type)
        except Exception as e:
            _safe_logger.warning(f"[HricServerAdapter] invalid cmd.type: {cmd.type!r} ({e})")
            return False

        entry = self._dispatch_table.get(cmd_type_int)
        if entry is None:
            _safe_logger.warning(f"[HricServerAdapter] no handler for cmd.type={cmd_type_int}")
            return False

        # FOOTPOINT 等依赖可选 ROS 包的命令: 编译环境无该包时安静拒绝
        if entry.requires_footpoint_msg and not HAS_FOOTPOINT_MSG:
            _safe_logger.warning(f"[HricServerAdapter] {entry.label}: hric_msgs/FootPoint not available, drop")
            return False

        pod = getattr(cmd, entry.pod_field, None)
        if pod is None:
            _safe_logger.warning(f"[HricServerAdapter] {entry.label}: envelope.{entry.pod_field} missing")
            return False

        ros_msg = entry.converter(pod)
        if ros_msg is None:
            _safe_logger.warning(f"[HricServerAdapter] {entry.label}: converter returned None")
            return False

        # 高频去重日志 (FSM_*_CMD 用; joystick 按住按键会以 50Hz 上报, 不希望刷屏)
        if entry.dedup_attr is not None:
            data = str(getattr(pod, "data", "") or "")
            last = getattr(self, entry.dedup_attr, None)
            if data != last:
                _safe_logger.info(f"[HricServerAdapter] {entry.label}: '{data}' (was '{last}')")
                setattr(self, entry.dedup_attr, data)

        q = getattr(self._iface, entry.queue_attr, None)
        if q is None:
            _safe_logger.warning(f"[HricServerAdapter] {entry.label}: robot_interface.{entry.queue_attr} missing")
            return False

        # FSM_STATE_CMD 强制走轮询确认 (v3.5+ 无降级路径; 构造期已校验前置条件)
        # 其余 4 个命令仍 fire-and-forget
        if entry.label == "fsm_state_cmd":
            # [HRIC-INFO] 每次进 adapter 都打一条 (无 dedup), 证明 RPC 到了 adapter
            try:
                _data = str(getattr(pod, "data", "") or "")
            except Exception:
                _data = "<?>"
            print(
                f"[HRIC-INFO] [adapter] fsm_state_cmd ENTER data='{_data}' "
                f"current_state={self._read_iface_state_name()}",
                file=_sys.stderr, flush=True,
            )
            return self._set_command_fsm_state_confirmed(q, ros_msg, str(getattr(pod, "data", "") or ""))

        ok = _push_latest(q, ros_msg)
        if not ok:
            _safe_logger.warning(f"[HricServerAdapter] {entry.label}: queue push failed")

        # [HRIC-INFO] walk_cmd / stand_cmd 节流日志: 每 1s 打一行, 看 RPC 是否真到 adapter
        # 若这条日志看得到 => 杆 -> robot_control -> RPC -> xmigcs 入队 链路通; 不动是后面 gate
        # 若看不到       => 链路在 robot_control 或 RPC 层就断了
        if entry.label in ("walk_cmd", "stand_cmd"):
            try:
                lx = float(ros_msg.twist.linear.x)
                ly = float(ros_msg.twist.linear.y)
                az = float(ros_msg.twist.angular.z)
            except Exception:
                lx = ly = az = float("nan")
            cur_state = self._read_iface_state_name() or "<unknown>"
            if _dbg_throttled(f"set_cmd_{entry.label}", 1.0):
                print(
                    f"[HRIC-INFO] [adapter] set_command {entry.label} "
                    f"lx={lx:+.3f} ly={ly:+.3f} az={az:+.3f} "
                    f"current_state={cur_state} push_ok={ok}",
                    file=_sys.stderr, flush=True,
                )
        return ok

    # ------------------------------------------------------------------
    # FSM_STATE_CMD 专用: 轮询 robot_interface.current_state.name + 失败重试入队
    # ------------------------------------------------------------------

    def _set_command_fsm_state_confirmed(self, q: "queue.Queue", ros_msg: String,
                                         event_name: str) -> bool:
        """入队后立即返回; 状态确认在后台线程完成, 不阻塞 HRIC RPC 服务线程.

        joystick_bridge 会以 ~50Hz 重复同一 fsm_state (如 gotoSTART). 若在 RPC 线程
        内轮询 5s, 会阻塞 walk_cmd 等 fire-and-forget 命令. 因此:
          - 入队成功 => 立即 return True (bridge 侧 async worker 可继续)
          - 相同 event 确认进行中 => 直接 return True (去重)
          - 确认结果仅写日志, 不再阻塞 RPC 返回值
        """
        event_name = (event_name or "").strip()
        if not event_name:
            _safe_logger.warning("[HricServerAdapter] fsm_state_cmd: empty event_name, drop")
            return False

        goto_literal: Optional[str] = None
        if event_name.startswith(self.GOTO_PREFIX) and len(event_name) > len(self.GOTO_PREFIX):
            goto_literal = event_name[len(self.GOTO_PREFIX):]

        prev_state_name = self._read_iface_state_name()

        if goto_literal is not None and prev_state_name == goto_literal:
            print(
                f"[HRIC-INFO] [adapter] fsm_state_cmd '{event_name}' IDEMPOTENT_FAST_PATH "
                f"(prev_state already == '{goto_literal}'), return True without enqueue",
                file=_sys.stderr, flush=True,
            )
            return True

        with self._fsm_confirm_lock:
            if self._fsm_confirm_inflight_event == event_name:
                return True

        if not _push_latest(q, ros_msg):
            print(
                f"[HRIC-INFO] [adapter] fsm_state_cmd '{event_name}' ENQUEUE_FAILED",
                file=_sys.stderr, flush=True,
            )
            _safe_logger.warning(
                f"[HricServerAdapter] fsm_state_cmd '{event_name}': initial enqueue failed"
            )
            return False

        with self._fsm_confirm_lock:
            self._fsm_confirm_inflight_event = event_name

        print(
            f"[HRIC-INFO] [adapter] fsm_state_cmd '{event_name}' ENQUEUED ok, "
            f"prev_state='{prev_state_name}' goto_literal='{goto_literal}', "
            f"confirm async (timeout={self._fsm_confirm_timeout_s}s)",
            file=_sys.stderr, flush=True,
        )

        threading.Thread(
            target=self._fsm_confirm_poll_worker,
            args=(event_name, goto_literal, prev_state_name),
            daemon=True,
            name=f"hric_fsm_confirm_{event_name}",
        ).start()
        return True

    def _fsm_confirm_poll_worker(self,
                                 event_name: str,
                                 goto_literal: Optional[str],
                                 prev_state_name: Optional[str]) -> None:
        """后台轮询 FSM 状态变化; 仅日志, 不影响 RPC 返回."""
        try:
            deadline = time.monotonic() + self._fsm_confirm_timeout_s
            check_interval = self.FSM_STATE_CHECK_INTERVAL_S
            polls = 0
            while True:
                now_state = self._read_iface_state_name()
                if now_state is not None and now_state != prev_state_name:
                    literal_hint = (f" (goto-literal='{goto_literal}' did not match, "
                                    f"FSM mapped event to '{now_state}')"
                                    if goto_literal is not None and now_state != goto_literal
                                    else "")
                    _safe_logger.info(
                        f"[HricServerAdapter] fsm_state_cmd '{event_name}' confirmed "
                        f"after {polls} polls ('{prev_state_name}' -> '{now_state}')"
                        f"{literal_hint}"
                    )
                    return

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._log_fsm_state_timeout(event_name, goto_literal, prev_state_name,
                                                self._read_iface_state_name(), polls)
                    return

                sleep_s = min(check_interval, remaining)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                polls += 1
        finally:
            with self._fsm_confirm_lock:
                if self._fsm_confirm_inflight_event == event_name:
                    self._fsm_confirm_inflight_event = None

    def _log_fsm_state_timeout(self,
                               event_name: str,
                               goto_literal: Optional[str],
                               prev: Optional[str],
                               now: Optional[str],
                               polls: int) -> None:
        """超时日志: 'state 始终未变化' = FSM 拒绝/忽略 trigger. goto 字面仅作 hint."""
        reason = (f"state did not change (current='{now}', prev='{prev}'); "
                  f"FSM appears to reject or ignore trigger '{event_name}'; "
                  f"possible cause: event not registered for prev, on_enter blocked, "
                  f"condition_failed, or safe_guard overrode")
        if goto_literal is not None and now != goto_literal:
            reason += (f" | note: goto-literal='{goto_literal}' is a HINT only; "
                       f"actual mapping is decided by FSM transition table")
        _safe_logger.warning(
            f"[HricServerAdapter] fsm_state_cmd '{event_name}' confirm timeout "
            f"(total={self._fsm_confirm_timeout_s:.3f}s, check={self.FSM_STATE_CHECK_INTERVAL_S:.3f}s, "
            f"polls={polls}): {reason}"
        )


_VALID_COMM_MODES = ("lpc", "shm_rpc", "tcp_rpc")


def create_hric_server(robot_interface,
                       comm_mode: str = "shm_rpc",
                       tcp_port: int = 50052,
                       shm_service_name: str = "XmigcsHricRPC",
                       fsm_state_confirm_timeout_s: Optional[float] = None,
                       fsm_state_confirm_poll_s: Optional[float] = None):
    """
    创建并返回 (HricDeviceServer, HricServerAdapter) 元组. 调用方需:
        1. 持有 adapter 强引用 (HricDeviceServer 内部仅保存裸指针)
        2. 在合适时机调用 server.start() / server.stop()

    底层使用 libhric_rpc.so (HricDeviceServer), 与 librobot_rpc.so 模式对齐.

    comm_mode: 三选一, **无 fallback**, 必须与 robot_control 端 --hric-comm-mode 一致:
        - "lpc"      HricLpcRegistry 全局单例, 同进程零开销直调
        - "shm_rpc"  iceoryx2 共享内存 (**默认**, 需 RPC_ENABLE_ICEORYX2, 启动顺序: 先 robot_control 后 xmigcs)
        - "tcp_rpc"  coro_rpc TCP (跨进程 fallback / 调试, client 内置重连)

    非法 comm_mode 抛 ValueError (早失败, 不静默退化).
    tcp_port / shm_service_name 仅在对应 mode 下生效.

    FSM_STATE_CMD 切换确认 (v3.5, 强制启用, 无降级路径):
        - 调用方无需注入 FSM 实例. adapter 在构造期强制校验 robot_interface 已暴露
          .current_state 属性 (xmigcs RobotInterfaceImpl 默认就有). 缺失 -> RuntimeError.
          运行期对每个 FSM_STATE_CMD 命令执行"基于轮询的确认":
            * 入队前快照 current_state.name
            * 入队后以 poll 周期轮询 current_state.name 直到目标或超时
            * 事件名遵循 'goto<STATE>' 约定时等待 current_state.name == '<STATE>'
            * 其他事件名等待 current_state.name 与快照不同即视为成功
        - fsm_state_confirm_timeout_s: 超时秒, None 用默认 5.0s, **必须 > 0**, 否则 ValueError
        - fsm_state_confirm_poll_s    : 轮询/重试入队间隔秒, None 用默认 1.0s,
                                         **必须 > 0 且 <= timeout**, 否则 ValueError
        - 该确认逻辑仅作用于 FSM_STATE_CMD, 其余 4 个命令仍 fire-and-forget.

    如 xmigcs_hric_python 未安装则返回 (None, None) 并打印 warning.
    构造 adapter 时若 robot_interface 缺 current_state 或参数非法, 异常向上传播 (不静默吞)
    """
    mode = (comm_mode or "").lower()
    if mode not in _VALID_COMM_MODES:
        raise ValueError(
            f"[create_hric_server] invalid comm_mode='{comm_mode}', "
            f"must be one of {_VALID_COMM_MODES}"
        )

    if not HAS_HRIC_NATIVE:
        _safe_logger.warning("[HricServerAdapter] xmigcs_hric_python not importable, HRIC server disabled. "
                     "Please build robot_control with bridge/hric/python target.")
        return None, None

    adapter = HricServerAdapter(
        robot_interface,
        fsm_state_confirm_timeout_s=fsm_state_confirm_timeout_s,
        fsm_state_confirm_poll_s=fsm_state_confirm_poll_s,
    )

    cfg = HricRpcConfig()
    cfg.comm_mode = mode
    cfg.tcp_port = int(tcp_port)
    cfg.shm_service_name = str(shm_service_name)
    server = HricDeviceServer(cfg)
    server.register_service(adapter)
    endpoint = (f"tcp_port={tcp_port}" if mode == "tcp_rpc"
                else f"shm_service='{shm_service_name}'" if mode == "shm_rpc"
                else "same-process direct call")
    _safe_logger.info(f"[HricServerAdapter] HRIC server created via libhric_rpc.so "
              f"(comm_mode={mode}, {endpoint}); "
              f"unified set_command interface, {len(adapter._dispatch_table)} dispatch entries; "
              f"fsm_state_cmd=confirmed("
              f"timeout={adapter._fsm_confirm_timeout_s:.3f}s,"
              f"poll={adapter._fsm_confirm_poll_s:.3f}s)")
    return server, adapter
