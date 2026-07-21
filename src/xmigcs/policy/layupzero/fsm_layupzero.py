"""
FSM State Implementations
Concrete implementations of different FSM states
"""
import numpy as np
from xmigcs.FSM.fsm_base import FSMState, FSMStateName
from xmigcs.common.control_flag import FSMControlFlag
from xmigcs.common.robot_data import RobotData
import os
import yaml
import time
from xmigcs.utils.xlog_utils import xlog
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)
try:
    import onnxruntime
except ImportError:  # pragma: no cover - runtime fallback when onnxruntime isn't installed
    onnxruntime = None


LAYUPZERO_MODEL_JOINT_ORDER = [
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
]

LAYUPZERO_LOCKED_JOINT_MAP = np.array([
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12, 13, 14,
    15, 16, 17, 18,
    22, 23, 24, 25,
], dtype=np.int32)


class FSMStateLAYUPZERO(FSMState):
    """零位状态实现（完整C++逻辑移植）"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.q_factor_ = 0.0
        self._last_far_from_zero_log_time = 0.0
        self._last_not_face_up_log_time = 0.0
        self._last_exit_blocked_log_time = 0.0
        self.is_motion_end = False
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "layupzero.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)

        self.motor_num_ = policy_config["motor_num"]
        self.num_obs_ = int(policy_config["num_obs"])
        self.zero_positions_ = np.array(policy_config["zero_positions"],
                                        dtype=float)
        self.kp_pos_ = np.array(policy_config["kp_pos"], dtype=float)
        self.kd_pos_ = np.array(policy_config["kd_pos"], dtype=float)
        self.interp_step_ = float(policy_config["interp_step"])
        self.interp_max_ = float(policy_config["interp_max"])
        self.zero_motion_onnx_path_ = policy_config.get("zero_motion_onnx_path")
        self.zero_motion_frame_ = int(policy_config.get("zero_motion_frame", 0))
        self.num_obs_ = policy_config.get("num_obs")
        self.close_threshold_ = float(policy_config.get("close_threshold"))
        self.zero_motion_onnx_path_ = self._resolve_zero_motion_onnx_path(
            current_dir
        )

        self._override_zero_positions_from_motion(current_dir)

    def on_enter(self):
        self.q_factor_ = 0.0
        self.is_motion_end = False

    def run(self, flag: FSMControlFlag):
        if self.robot_data_ is None:
            return
        # q_est = self.robot_data_.q_a_[-self.motor_num_:].copy()  # numpy数组切片    
        q_est = self.robot_data_.get_serial_joint_pos_desired()
        if self.q_factor_ < self.interp_max_:
            pos_cmd = (1.0 - self.q_factor_
                       ) * q_est + self.q_factor_ * self.zero_positions_
            self.q_factor_ = min(self.q_factor_ + self.interp_step_,
                                 self.interp_max_)
        else:
            pos_cmd = self.zero_positions_
        self.robot_data_.q_d_[-self.motor_num_:] = pos_cmd
        self.robot_data_.q_dot_d_[-self.motor_num_:] = 0
        self.robot_data_.tau_d_[-self.motor_num_:] = 0
        self.robot_data_.joint_kp_p_[:self.motor_num_] = self.kp_pos_
        self.robot_data_.joint_kd_p_[:self.motor_num_] = self.kd_pos_
        is_close, q_est, position_diff, max_diff, max_idx, max_diff_id = self.is_close_to_zero_positions(threshold=self.close_threshold_ / 180 * np.pi)
        if is_close:
            self.is_motion_end = True


    def on_exit(self):
        self.is_motion_end = False

    def _override_zero_positions_from_motion(self, current_dir: str):
        if not self.zero_motion_onnx_path_:
            return

        try:
            zero_from_motion = self._load_zero_positions_from_motion(
                current_dir,
                self.zero_motion_onnx_path_,
                self.zero_motion_frame_,
            )
            self.zero_positions_ = zero_from_motion
        except Exception as exc:
            pass

    def _resolve_zero_motion_onnx_path(self, current_dir: str) -> str | None:
        layup_config_path = os.path.normpath(
            os.path.join(current_dir, "..", "layup", "config", "layup.yaml")
        )
        if not os.path.exists(layup_config_path):
            logger.warning(f"[FSMStateZero] Layup config not found: {layup_config_path}")
            return None

        with open(layup_config_path, "r") as f:
            layup_config = yaml.safe_load(f) or {}

        layup_onnx_path = layup_config.get("onnx_path")
        if not layup_onnx_path:
            logger.warning(f"[FSMStateZero] Missing onnx_path in {layup_config_path}")
            return None

        if os.path.isabs(layup_onnx_path):
            return layup_onnx_path

        layup_dir = os.path.normpath(os.path.join(current_dir, "..", "layup"))
        return os.path.normpath(os.path.join(layup_dir, "model", layup_onnx_path))

    def _load_zero_positions_from_motion(
        self,
        current_dir: str,
        motion_onnx_path: str,
        motion_frame: int,
    ) -> np.ndarray:
        if onnxruntime is None:
            raise RuntimeError("onnxruntime is not available")

        resolved_onnx_path = motion_onnx_path
        if not os.path.isabs(resolved_onnx_path):
            resolved_onnx_path = os.path.normpath(
                os.path.join(current_dir, resolved_onnx_path)
            )
        if not os.path.exists(resolved_onnx_path):
            raise FileNotFoundError(f"ONNX file not found: {resolved_onnx_path}")

        session = onnxruntime.InferenceSession(
            resolved_onnx_path,
            providers=["CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        if len(inputs) < 2:
            raise RuntimeError(
                f"Expected at least 2 ONNX inputs (obs/time_step), got {len(inputs)}"
            )

        obs_input = inputs[0]
        time_input = inputs[1]
        if not isinstance(self.num_obs_ , int) or self.num_obs_ <= 0:
            raise RuntimeError(f"Invalid num_obs: {self.num_obs_}")
        frame_idx = max(0, int(motion_frame))
        metadata = getattr(session.get_modelmeta(), "custom_metadata_map", {}) or {}
        motion_length_raw = metadata.get("motion_length") or metadata.get("time_step_total")
        if motion_length_raw is not None:
            try:
                frame_idx = min(frame_idx, int(float(motion_length_raw)) - 1)
            except (TypeError, ValueError):
                pass

        outputs = session.run(
            ["joint_pos"],
            {
                obs_input.name: np.zeros((1, self.num_obs_), dtype=np.float32),
                time_input.name: np.array([[frame_idx]], dtype=np.float32),
            },
        )
        joint_pos = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

        ordered_joint_pos = self._reorder_motion_joint_pos(joint_pos, metadata)
        if ordered_joint_pos.shape[0] != LAYUPZERO_LOCKED_JOINT_MAP.shape[0]:
            raise RuntimeError(
                "Motion joint_pos length mismatch: "
                f"expected {LAYUPZERO_LOCKED_JOINT_MAP.shape[0]}, "
                f"got {ordered_joint_pos.shape[0]}"
            )

        zero_positions = np.array(self.zero_positions_, copy=True, dtype=float)
        zero_positions[LAYUPZERO_LOCKED_JOINT_MAP] = ordered_joint_pos.astype(float)
        return zero_positions

    def _reorder_motion_joint_pos(self, joint_pos: np.ndarray, metadata: dict) -> np.ndarray:
        joint_names_raw = metadata.get("joint_names")
        if not joint_names_raw:
            return joint_pos

        joint_names = [name.strip() for name in joint_names_raw.split(",") if name.strip()]
        if len(joint_names) != joint_pos.shape[0]:
            raise RuntimeError(
                f"joint_names length {len(joint_names)} does not match joint_pos length {joint_pos.shape[0]}"
            )

        joint_map = {name: joint_pos[idx] for idx, name in enumerate(joint_names)}
        missing = [name for name in LAYUPZERO_MODEL_JOINT_ORDER if name not in joint_map]
        if missing:
            raise RuntimeError(f"Missing joints in ONNX metadata: {missing}")

        return np.array(
            [joint_map[name] for name in LAYUPZERO_MODEL_JOINT_ORDER],
            dtype=np.float32,
        )

    def check_transition(self, *args, **kwargs):
        """检查是否允许切入/退出 layupzero。"""
        result = {}
        result['allow_transition'] = True
        action = kwargs.get("action")
        target_state = kwargs.get("target_state")
        white_list = kwargs.get("white_list")
        if action == "exit":
            #TODO: 判断是否允许退出 layupzero
            if target_state is not None and target_state.name in white_list.values():
                return result
            if not self.is_motion_end:
                current_time = time.perf_counter()
                if current_time - self._last_exit_blocked_log_time >= 5.0:
                    xlog.warning("[FSMStateLAYUPZERO] 当前机器人未执行完成 layupzero 动作，禁止退出 layupzero")
                    is_close, q_est, position_diff, max_diff, max_idx, max_diff_id = self.is_close_to_zero_positions(threshold=self.close_threshold_ / 180 * np.pi)
                    if not is_close:
                        xlog.warning("[FSMStateLAYUPZERO] 当前距离躺下零位位置过远")
                        xlog.warning(f"  各关节位置差值：{position_diff}")
                        xlog.warning(
                            f"  最大差值：{max_diff:.4f} (关节 ID: {max_diff_id}, 索引: {max_idx})"
                        )
                        xlog.warning(f"  当前估计位置：{q_est.tolist()}")
                        xlog.warning(f"  目标零位位置：{self.zero_positions_.tolist()}")
                    self._last_exit_blocked_log_time = current_time
                result['allow_transition'] = False
            return result
        else :
            projected_gravity = self.robot_data_.get_project_gravity()
            is_face_up_lay_down = projected_gravity[0] < -0.5 and abs(projected_gravity[2]) < 0.5
            if not is_face_up_lay_down:
                current_time = time.perf_counter()
                if current_time - self._last_not_face_up_log_time >= 5.0:
                    roll = float(self.robot_data_.imu_data_[2])
                    pitch = float(self.robot_data_.imu_data_[1])
                    xlog.warning("[FSMStateLAYUPZERO] 当前机器人未处于正面朝上躺下状态，禁止切入 layupzero")
                    xlog.warning(f"  当前 roll/pitch：{roll:.4f}, {pitch:.4f}")
                    xlog.warning(f"  当前重力投影：{projected_gravity.tolist()}")
                    self._last_not_face_up_log_time = current_time
                result['allow_transition'] = False
            is_close, q_est, position_diff, max_diff, max_idx, max_diff_id = self.is_close_to_zero_positions(threshold=120.0 / 180 * np.pi)
            if not is_close:
                current_time = time.perf_counter()
                if current_time - self._last_far_from_zero_log_time >= 5.0:
                    xlog.warning("[FSMStateLAYUPZERO] 当前距离躺下零位位置过远")
                    xlog.warning(f"  各关节位置差值：{position_diff}")
                    xlog.warning(
                        f"  最大差值：{max_diff:.4f} (关节 ID: {max_diff_id}, 索引: {max_idx})"
                    )
                    xlog.warning(f"  当前估计位置：{q_est.tolist()}")
                    xlog.warning(f"  目标零位位置：{self.zero_positions_.tolist()}")
                    self._last_far_from_zero_log_time = current_time
                result['allow_transition'] = False
            return result

    def is_close_to_zero_positions(self, threshold: float = 120.0 / 180 * np.pi) -> bool:
        q_est = self.robot_data_.q_a_[-self.motor_num_:].copy()  # numpy数组切片
        motor_ids = [51, 52, 53, 54, 55, 56,  # 左腿
                    61, 62, 63, 64, 65, 66, # 右腿
                    33, 32, 31, # 腰
                    11, 12, 13, 14, 15, 16, 17, # 左臂
                    21, 22, 23, 24, 25, 26, 27, # 右臂
                    1, 2] # 头部
        position_diff = np.abs(q_est - self.zero_positions_)
        max_diff = np.max(position_diff)
        max_idx = int(np.argmax(position_diff))
        ids_for_slice = motor_ids[: self.motor_num_]
        max_diff_id = ids_for_slice[max_idx] if max_idx < len(ids_for_slice) else max_idx
        return max_diff < threshold, q_est, position_diff, max_diff, max_idx, max_diff_id
