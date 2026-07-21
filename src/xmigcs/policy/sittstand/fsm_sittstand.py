"""SITTSTAND FSM state.

与 ``policy/new_stand`` 完全一致的 DEX_MOSAIC *student* 运动策略：
  * ONNX 为单输入/单输出学生策略：
        input  : obs   [1, 620]
        output : action[1, 23]
    不吐参考动作、无 time_step 输入。
  * 参考轨迹（command 关节位/速 + anchor 位姿）来自预录的 NPZ 动作文件（NPZ 回放，不涉及 GMR）。
  * 620 维观测为 per-term history（oldest -> newest），与 IsaacLab ``ObservationManager``
    group ``history_length = 5`` 一致：
        620 = cmd46*5 + ori6*5 + ang_vel3*5 + jp23*5 + jv23*5 + act23*5
    其中 cmd46 = ref_joint_pos(23) + ref_joint_vel(23)。

与 ``new_stand`` 唯一的差异：``check_transition`` 在动作未播放完成前禁止退出
（除非目标为 STOP），保留 sittstand 原有的安全门控行为。
"""
import inspect
import os
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import onnxruntime
import yaml
import xlog

from xmigcs.FSM.fsm_base import FSMState, FSMStateName
from xmigcs.common.control_flag import FSMControlFlag
from xmigcs.common.robot_data import RobotData

DEFAULT_BODY_NAMES_24 = [
    "pelvis",
    "hip_pitch_l_link",
    "hip_roll_l_link",
    "hip_yaw_l_link",
    "knee_pitch_l_link",
    "ankle_pitch_l_link",
    "ankle_roll_l_link",
    "hip_pitch_r_link",
    "hip_roll_r_link",
    "hip_yaw_r_link",
    "knee_pitch_r_link",
    "ankle_pitch_r_link",
    "ankle_roll_r_link",
    "waist_yaw_link",
    "waist_roll_link",
    "waist_pitch_link",
    "shoulder_pitch_l_link",
    "shoulder_roll_l_link",
    "shoulder_yaw_l_link",
    "elbow_pitch_l_link",
    "shoulder_pitch_r_link",
    "shoulder_roll_r_link",
    "shoulder_yaw_r_link",
    "elbow_pitch_r_link",
]

# 23-dof "lab" joint order (wrists + elbow_yaw locked out), identical to sitrise/new_stand.
JOINT_XML_23 = [
    "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
]

_HISTORY_LEN = 5


class FSMStateSITTSTAND(FSMState):
    default_config_name = "sittstand.yaml"

    def __init__(self, robot_data: RobotData, config_path: Optional[str] = None, variant_name: str = "default"):
        super().__init__(robot_data)
        self.counter_step = 0
        self.variant_name = variant_name
        self.is_motion_end = False
        self._last_exit_blocked_log_time = 0.0

        current_dir = self.get_state_dir()
        if config_path is None:
            config_path = os.path.join(current_dir, "config", self.default_config_name)
        self.config_path = os.path.abspath(config_path)
        with open(self.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.onnx_path = config["onnx_path"]
        self.onnx_path = self.resolve_onnx_path(self.onnx_path, current_dir)
        self.physical_dt = config["physical_dt"]
        self.decimation_ = config["decimation"]
        self.num_actions = config["num_actions"]
        self.motor_nums = config["motor_nums"]
        self.kps = config["kps"]
        self.kds = config["kds"]
        self.body_names = config.get("body_names", DEFAULT_BODY_NAMES_24)
        self.locked_joint_map = config["locked_joint_map"]
        self.anchor_body_name = config.get("anchor_body", "pelvis")
        if self.anchor_body_name not in self.body_names:
            raise ValueError(f"Anchor body {self.anchor_body_name} missing from body list.")
        self.anchor_body_index = self.body_names.index(self.anchor_body_name)
        self.num_bodies = len(self.body_names)

        # Per-tick scratch buffers.
        self.action = np.zeros((1, self.num_actions), dtype=np.float32)

        # ONNX session + metadata.
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        model_meta = self.ort_session.get_modelmeta()
        custom_map = getattr(model_meta, "custom_metadata_map", {})
        metadata_props = [SimpleNamespace(key=k, value=v) for k, v in custom_map.items()]
        self.input_name = [inpt.name for inpt in self.ort_session.get_inputs()]
        self.num_obs = config.get("num_obs")
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

        # Per-term history buffers (oldest -> newest along axis 0).
        # 620 = cmd46*5 + ori6*5 + ang_vel3*5 + jp23*5 + jv23*5 + act23*5
        self._cmd_dim = 2 * self.num_actions  # ref_joint_pos + ref_joint_vel
        self._buf_cmd = np.zeros((_HISTORY_LEN, self._cmd_dim), dtype=np.float32)
        self._buf_ori = np.zeros((_HISTORY_LEN, 6), dtype=np.float32)
        self._buf_ang_vel = np.zeros((_HISTORY_LEN, 3), dtype=np.float32)
        self._buf_joint_pos = np.zeros((_HISTORY_LEN, self.num_actions), dtype=np.float32)
        self._buf_joint_vel = np.zeros((_HISTORY_LEN, self.num_actions), dtype=np.float32)
        self._buf_actions = np.zeros((_HISTORY_LEN, self.num_actions), dtype=np.float32)
        self._history_initialized = False

        _expected_obs = _HISTORY_LEN * (
            self._cmd_dim + 6 + 3 + 3 * self.num_actions
        )
        if self.num_obs != _expected_obs:
            raise ValueError(
                f"num_obs ({self.num_obs}) != expected {_expected_obs} "
                f"(history={_HISTORY_LEN}, num_actions={self.num_actions})."
            )

        # Reference (filled from the NPZ each tick).
        self.ref_joint_pos = np.zeros((1, self.num_actions), dtype=np.float32)
        self.ref_joint_vel = np.zeros((1, self.num_actions), dtype=np.float32)
        self.ref_anchor_pos_w = np.zeros(3, dtype=np.float32)
        self.ref_anchor_quat_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # ONNX metadata: joint order + default pose + action scale.
        self.joint_seq = None
        self.joint_pos_array_seq = None
        self.action_scale = None
        for prop in metadata_props:
            if prop.key == "joint_names":
                self.joint_seq = prop.value.split(",")
            elif prop.key == "default_joint_pos":
                self.joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
            elif prop.key == "action_scale":
                self.action_scale = np.array([float(x) for x in prop.value.split(",")])
        if self.joint_seq is None or self.joint_pos_array_seq is None or self.action_scale is None:
            raise RuntimeError(
                "ONNX metadata missing joint_names/default_joint_pos/action_scale."
            )

        self.joint_xml = JOINT_XML_23
        self.mj2lab = np.array([self.joint_xml.index(j) for j in self.joint_seq])
        self.default_angles_lab = self.joint_pos_array_seq
        self.action_scale_lab = self.action_scale

        self.init_to_world = np.eye(3, dtype=np.float32)

        # ---- NPZ reference replay ----
        self.hold_final_reference = bool(config.get("hold_final_reference", False))
        self.motion_length = config.get("motion_length")
        self.npz_anchor_body_index = int(config.get("npz_anchor_body_index", 0))
        motion_file = config.get("motion_file")
        if not motion_file:
            raise ValueError(
                f"{self.__class__.__name__} requires 'motion_file' in {self.default_config_name}"
            )
        motion_path = (
            motion_file
            if os.path.isabs(motion_file)
            else os.path.join(current_dir, "model", motion_file)
        )
        if not os.path.exists(motion_path):
            raise FileNotFoundError(f"motion_file not found: {motion_path}")
        npz = np.load(motion_path)
        self._npz_joint_pos = npz["joint_pos"].astype(np.float32)
        self._npz_joint_vel = npz["joint_vel"].astype(np.float32)
        self._npz_anchor_pos_w = npz["body_pos_w"][:, self.npz_anchor_body_index, :].astype(np.float32)
        self._npz_anchor_quat_w = npz["body_quat_w"][:, self.npz_anchor_body_index, :].astype(np.float32)
        self._npz_num_frames = int(self._npz_joint_pos.shape[0])
        if self.motion_length is None:
            self.motion_length = self._npz_num_frames
        else:
            self.motion_length = int(self.motion_length)

    # ---------- FSM lifecycle ----------
    def on_enter(self):
        self.counter_step = 0
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros((1, self.num_actions), dtype=np.float32)
        for _buf in (
            self._buf_cmd, self._buf_ori, self._buf_ang_vel,
            self._buf_joint_pos, self._buf_joint_vel, self._buf_actions,
        ):
            _buf[:] = 0.0
        self._history_initialized = False
        # Seed references from frame 0.
        self.ref_joint_pos = self._npz_joint_pos[0:1]
        self.ref_joint_vel = self._npz_joint_vel[0:1]
        self.ref_anchor_pos_w = self._npz_anchor_pos_w[0]
        self.ref_anchor_quat_w = self._npz_anchor_quat_w[0]
        xlog.info(
            f"[{self.__class__.__name__}] entered: onnx={os.path.basename(self.onnx_path)}, "
            f"npz_frames={self._npz_num_frames}, motion_length={self.motion_length}, "
            f"hold_final={self.hold_final_reference}"
        )

    def on_exit(self):
        self.action = np.zeros((1, self.num_actions), dtype=np.float32)
        self.counter_step = 0
        self.is_motion_end = False
        self._history_initialized = False

    def check_transition(self, *args, **kwargs):
        """检查是否允许切入/退出 sittstand。

        动作未播放完成前禁止退出（目标为 STOP 时除外），保留原有安全门控。
        """
        result = {}
        result['allow_transition'] = True
        action = kwargs.get("action")
        target_state = kwargs.get("target_state")
        white_list = kwargs.get("white_list")
        if action == "enter":
            return result
        # exit
        if target_state is not None and target_state.name in white_list.values():
            return result
        if not self.is_motion_end:
            current_time = time.perf_counter()
            if current_time - self._last_exit_blocked_log_time >= 5.0:
                xlog.warning("[FSMStateSITTSTAND] 当前机器人未执行完成 sittstand 动作，禁止退出 sittstand")
                self._last_exit_blocked_log_time = current_time
            result['allow_transition'] = False
        return result

    def run(self, flag: FSMControlFlag):
        if self.robot_data_.control_step_ % self.decimation_ == 0:
            self.inner_run()
        self.set_kp_kd()

    def set_kp_kd(self):
        self.robot_data_.joint_kp_p_[: self.motor_nums] = self.kps
        self.robot_data_.joint_kd_p_[: self.motor_nums] = self.kds

    # ---------- Reference plumbing ----------
    def _advance_npz_reference(self):
        """Pick the NPZ frame for the current control tick (frame == counter_step)."""
        time_index = max(self.counter_step, 0)
        if (
            self.hold_final_reference
            and self.motion_length is not None
            and self.motion_length > 0
        ):
            time_index = min(time_index, self.motion_length - 1)
        motion_frame = min(time_index, self._npz_num_frames - 1)
        self.ref_joint_pos = self._npz_joint_pos[motion_frame:motion_frame + 1]
        self.ref_joint_vel = self._npz_joint_vel[motion_frame:motion_frame + 1]
        self.ref_anchor_pos_w = self._npz_anchor_pos_w[motion_frame]
        self.ref_anchor_quat_w = self._npz_anchor_quat_w[motion_frame]

    # ---------- Geometry helpers (identical to new_stand / teleop_yibo) ----------
    @staticmethod
    def matrix_from_quat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ])

    @staticmethod
    def yaw_quat(q):
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    # ---------- Main inference step ----------
    def inner_run(self):
        self._advance_npz_reference()

        robot_quat = self.robot_data_.get_robot_quat()
        qj = self.robot_data_.get_joint_pos()
        qj = qj[self.locked_joint_map]
        qj = qj[self.mj2lab]
        qj = qj - self.default_angles_lab

        ref_anchor_ori_w = self.ref_anchor_quat_w.copy()

        # First two ticks: lock-in yaw alignment between robot and reference anchor.
        if self.counter_step < 2:
            init_to_anchor = self.matrix_from_quat(self.yaw_quat(ref_anchor_ori_w))
            world_to_anchor = self.matrix_from_quat(self.yaw_quat(robot_quat))
            self.init_to_world = world_to_anchor @ init_to_anchor.T
            self.counter_step += 1
            return

        robot_rot_mat = self.matrix_from_quat(robot_quat)
        motion_anchor_ori_b = (
            robot_rot_mat.T @ self.init_to_world @ self.matrix_from_quat(ref_anchor_ori_w)
        )

        ang_vel = self.robot_data_.get_angular_velocity()
        dqj = self.robot_data_.get_joint_vel()
        dqj = dqj[self.locked_joint_map]

        command_vec = np.concatenate(
            (self.ref_joint_pos.squeeze(0), self.ref_joint_vel.squeeze(0)),
            dtype=np.float32,
        )

        term_cmd = command_vec.astype(np.float32)
        term_ori = motion_anchor_ori_b[:, :2].reshape(-1).astype(np.float32)
        term_ang_vel = ang_vel.astype(np.float32)
        term_jp = qj.astype(np.float32)
        term_jv = dqj[self.mj2lab].astype(np.float32)
        term_act = np.asarray(self.action, dtype=np.float32).reshape(-1)

        if not self._history_initialized:
            self._buf_cmd[:] = term_cmd
            self._buf_ori[:] = term_ori
            self._buf_ang_vel[:] = term_ang_vel
            self._buf_joint_pos[:] = term_jp
            self._buf_joint_vel[:] = term_jv
            self._buf_actions[:] = term_act
            self._history_initialized = True
        else:
            self._buf_cmd = np.roll(self._buf_cmd, -1, axis=0); self._buf_cmd[-1] = term_cmd
            self._buf_ori = np.roll(self._buf_ori, -1, axis=0); self._buf_ori[-1] = term_ori
            self._buf_ang_vel = np.roll(self._buf_ang_vel, -1, axis=0); self._buf_ang_vel[-1] = term_ang_vel
            self._buf_joint_pos = np.roll(self._buf_joint_pos, -1, axis=0); self._buf_joint_pos[-1] = term_jp
            self._buf_joint_vel = np.roll(self._buf_joint_vel, -1, axis=0); self._buf_joint_vel[-1] = term_jv
            self._buf_actions = np.roll(self._buf_actions, -1, axis=0); self._buf_actions[-1] = term_act

        mimic_obs_tensor = np.expand_dims(
            np.concatenate([
                self._buf_cmd.reshape(-1),
                self._buf_ori.reshape(-1),
                self._buf_ang_vel.reshape(-1),
                self._buf_joint_pos.reshape(-1),
                self._buf_joint_vel.reshape(-1),
                self._buf_actions.reshape(-1),
            ]),
            axis=0,
        ).astype(np.float32)

        if mimic_obs_tensor.shape[1] != self.num_obs:
            raise RuntimeError(
                f"Observation length mismatch. Expected {self.num_obs}, got {mimic_obs_tensor.shape[1]}."
            )

        observation = {self.input_name[0]: mimic_obs_tensor}
        outputs_result = self.ort_session.run(None, observation)
        self.action = outputs_result[0]

        target_dof_pos_mj = np.zeros(self.motor_nums, dtype=np.float32)
        target_dof_pos_mj_inner = np.zeros(self.num_actions, dtype=np.float32)
        target_dof_pos_lab = self.action * self.action_scale_lab + self.default_angles_lab
        if target_dof_pos_lab.ndim > 1:
            target_dof_pos_lab = np.squeeze(target_dof_pos_lab, axis=0)

        target_dof_pos_mj_inner[self.mj2lab] = target_dof_pos_lab
        target_dof_pos_mj[self.locked_joint_map] = target_dof_pos_mj_inner

        for i in range(self.motor_nums):
            joint_idx = 35 - self.motor_nums + i
            self.robot_data_.q_d_[joint_idx] = target_dof_pos_mj[i]
            self.robot_data_.q_dot_d_[joint_idx] = 0.0
            self.robot_data_.tau_d_[joint_idx] = 0.0

        self.counter_step += 1
        if self.counter_step >= self.motion_length:
            self.is_motion_end = True

    @classmethod
    def get_state_dir(cls) -> str:
        return os.path.dirname(os.path.abspath(inspect.getfile(cls)))

    @staticmethod
    def resolve_onnx_path(onnx_path: str, current_dir: str) -> str:
        if os.path.isabs(onnx_path):
            return onnx_path
        if onnx_path.startswith(".") or "/" in onnx_path or "\\" in onnx_path:
            return os.path.abspath(os.path.join(current_dir, onnx_path))
        return os.path.abspath(os.path.join(current_dir, "model", onnx_path))
