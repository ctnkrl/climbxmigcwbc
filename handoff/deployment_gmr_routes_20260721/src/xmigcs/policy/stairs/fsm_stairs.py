from xmigcs.FSM.fsm_base import FSMState, FSMStateName
import numpy as np
import yaml
import os
from types import SimpleNamespace
from typing import Optional
import onnxruntime
from xmigcs.common.robot_data import RobotData
from xmigcs.common.control_flag import FSMControlFlag
import time

from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)


DEFAULT_BODY_NAMES = [
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
DEFAULT_JOINT_XML_23 = [
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
DEFAULT_JOINT_XML_29 = [
    "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
]


def _build_onnx_session(onnx_path: str):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        onnx_path,
        sess_options=options,
        providers=['CPUExecutionProvider'],
    )


def _parse_numeric_list(value, dtype=float):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [dtype(float(x)) for x in value]
    return [dtype(float(x.strip())) for x in str(value).split(",") if x.strip()]


def _parse_string_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [x.strip() for x in str(value).split(",") if x.strip()]


class FSMStateSTAIRS(FSMState):
    default_config_name = "stairs.yaml"

    def __init__(self, robot_data: RobotData, config_path: Optional[str] = None, variant_name: str = "default"):
        super().__init__(robot_data)
        self.motion_phase = 0
        self.counter_step = 0
        self.ref_motion_phase = 0
        self.variant_name = variant_name

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if config_path is None:
            config_path = os.path.join(current_dir, "config", "stairs.yaml")
        self.config_path = os.path.abspath(config_path)
        with open(self.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            # 兼容多策略：模型路径默认指向 policy/stairs/model 下
            self.onnx_path = config["onnx_path"]
            if not os.path.isabs(self.onnx_path):
                self.onnx_path = os.path.join(current_dir, "model", self.onnx_path)
            self.physical_dt = config["physical_dt"]
            self.decimation_ = config["decimation"]
            self.num_actions = config["num_actions"]
            self.motor_nums = config["motor_nums"]
            self.warm_start_time = config["warm_start_time"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.hold_final_reference = config.get("hold_final_reference", False)
            legacy_realtime_gmr = bool(config.get("use_realtime_gmr", False))
            self.reference_source = str(
                config.get(
                    "reference_source",
                    "realtime_gmr" if legacy_realtime_gmr else "embedded_npz",
                )
            ).strip().lower()
            if self.reference_source not in ("embedded_npz", "realtime_gmr"):
                raise ValueError(
                    "reference_source must be 'embedded_npz' or 'realtime_gmr', "
                    f"got {self.reference_source!r}."
                )
            self.use_realtime_gmr = self.reference_source == "realtime_gmr" or legacy_realtime_gmr
            gmr_ref_cfg = config.get("gmr_reference", {}) or {}
            self.gmr_reference_max_age_s = float(gmr_ref_cfg.get("max_age_s", 0.25))
            self.motion_id = int(config.get("motion_id", 0))
            self.motion_name = config.get("motion_name")
            self.motion_offsets = _parse_numeric_list(config.get("motion_offsets"), int)
            self.motion_lengths = _parse_numeric_list(config.get("motion_lengths"), int)
            self.motion_names = _parse_string_list(config.get("motion_names"))
            self.motion_files = _parse_string_list(config.get("motion_files"))
            self.motion_offset = 0
            self.motion_total_length = None
            self._warned_missing_gmr = False
            self._gmr_ref_active = False
            terrain_scan_cfg = config.get("terrain_scan", {}) or {}
            self.terrain_scan_dim = tuple(
                terrain_scan_cfg.get("scan_dim", [33, 21, 3]))
            if len(self.terrain_scan_dim) == 2:
                self.terrain_scan_dim = (*self.terrain_scan_dim, 3)
            self.terrain_scan_size = int(np.prod(self.terrain_scan_dim))
            self.terrain_scan_enabled_cfg = bool(
                terrain_scan_cfg.get("enabled", False))
            self._terrain_scan_missing_warned = False
            self.motion_length: Optional[int] = config.get("motion_length")
            self.body_names = config.get("body_names", DEFAULT_BODY_NAMES)
            locked_map_cfg = config.get("locked_joint_map")
            if locked_map_cfg is None:
                locked_map_cfg = list(range(self.num_actions))
            self.locked_joint_map = np.array(locked_map_cfg, dtype=np.int32)
            if len(self.locked_joint_map) != self.num_actions:
                raise ValueError(
                    f"locked_joint_map length {len(self.locked_joint_map)} must match num_actions {self.num_actions}"
                )
            joint_xml_cfg = config.get("joint_xml")
            if joint_xml_cfg is None:
                if self.num_actions == 29:
                    joint_xml_cfg = DEFAULT_JOINT_XML_29
                else:
                    joint_xml_cfg = DEFAULT_JOINT_XML_23
            self.joint_xml = list(joint_xml_cfg)
            if len(self.joint_xml) != self.num_actions:
                raise ValueError(
                    f"joint_xml length {len(self.joint_xml)} must match num_actions {self.num_actions}"
                )
            self.anchor_body_name = config.get("anchor_body", "pelvis")
            if self.anchor_body_name not in self.body_names:
                raise ValueError(f"Anchor body {self.anchor_body_name} missing from body list.")
            self.anchor_body_index = self.body_names.index(self.anchor_body_name)
            self.num_bodies = len(self.body_names)
            self._warm_start_from_lab = np.zeros(self.num_actions, dtype=np.float32)
            self._warm_start_to_lab = np.zeros(self.num_actions, dtype=np.float32)
            self._warm_start_prev_target = np.zeros(self.num_actions, dtype=np.float32)

            self.last_run_time = time.perf_counter()

            self.num_obs = None  # set after loading onnx
            self.action = np.zeros(self.num_actions, dtype=np.float32)
            self.proprio_history_length = max(1, int(config.get("proprio_history_length", config.get("history_length", 1))))
            self.command_includes_anchor_velocity = bool(config.get("command_includes_anchor_velocity", False))
            self._proprio_history_terms = (
                "command",
                "motion_anchor_ori_b",
                "base_ang_vel",
                "joint_pos",
                "joint_vel",
                "actions",
            )
            self._proprio_history = {}

            self.ref_joint_pos = np.zeros(self.num_actions, dtype=np.float32)
            self.ref_joint_vel = np.zeros(self.num_actions, dtype=np.float32)
            self.ref_body_pos_w = np.zeros((1, self.num_bodies, 3), dtype=np.float32)
            self.ref_body_quat_w = np.zeros((1, self.num_bodies, 4), dtype=np.float32)
            self.ref_body_lin_vel_w = np.zeros((1, self.num_bodies, 3), dtype=np.float32)
            self.ref_body_ang_vel_w = np.zeros((1, self.num_bodies, 3), dtype=np.float32)
            # load policy
            self.ort_session = _build_onnx_session(self.onnx_path)
            model_meta = self.ort_session.get_modelmeta()
            custom_map = getattr(model_meta, "custom_metadata_map", {})
            metadata_props = [SimpleNamespace(key=k, value=v) for k, v in custom_map.items()]

            input = self.ort_session.get_inputs()
            self.input_name = []
            for i, inpt in enumerate(input):
                self.input_name.append(inpt.name)
            obs_input = self.ort_session.get_inputs()[0]
            last_dim = obs_input.shape[-1]
            if isinstance(last_dim, int):
                self.num_obs = last_dim
            else:
                self.num_obs = config.get("num_obs", 154)

            # 从ONNX模型中读取参数
            self.joint_seq = None
            self.joint_pos_array_seq = None
            self.action_scale = None
            self.observation_names = []

            for prop in metadata_props:
                if prop.key == "joint_names":
                    self.joint_seq = prop.value.split(",")
                if prop.key == "default_joint_pos":
                    self.joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
                if prop.key == "action_scale":
                    self.action_scale = np.array([float(x) for x in prop.value.split(",")])
                if prop.key == "observation_names":
                    self.observation_names = [
                        x.strip() for x in prop.value.split(",") if x.strip()
                    ]
                if prop.key in ("motion_length", "time_step_total"):
                    try:
                        self.motion_total_length = int(float(prop.value))
                        if self.motion_length is None:
                            self.motion_length = self.motion_total_length
                    except (TypeError, ValueError):
                        logger.warning(f"[FSMStateSTAIRS] Invalid motion_length metadata value: {prop.value}")
                if prop.key == "motion_offsets" and not self.motion_offsets:
                    self.motion_offsets = _parse_numeric_list(prop.value, int)
                if prop.key == "motion_lengths" and not self.motion_lengths:
                    self.motion_lengths = _parse_numeric_list(prop.value, int)
                if prop.key == "motion_names" and not self.motion_names:
                    self.motion_names = _parse_string_list(prop.value)
                if prop.key == "motion_files" and not self.motion_files:
                    self.motion_files = _parse_string_list(prop.value)

            self._onnx_uses_motion_id_input = "motion_id" in self.input_name
            self._resolve_motion_selection()
            if self.motion_length is not None:
                try:
                    self.motion_length = int(self.motion_length)
                except (TypeError, ValueError):
                    logger.warning(f"[FSMStateSTAIRS] Invalid motion_length config value: {self.motion_length}")
                    self.motion_length = None

            self.terrain_scan_enabled = (
                self.terrain_scan_enabled_cfg
                or "height_scan" in self.observation_names
            )
            if self.terrain_scan_enabled and hasattr(self.robot_data_, "configure_terrain_scan"):
                self.robot_data_.configure_terrain_scan(self.terrain_scan_dim)
            expected_policy_obs_dim = self._expected_policy_obs_dim()
            if (
                self.terrain_scan_enabled
                or self.proprio_history_length > 1
                or self.command_includes_anchor_velocity
            ) and self.num_obs != expected_policy_obs_dim:
                raise ValueError(
                    f"Policy observation dimension mismatch: model/config expects {self.num_obs}, "
                    f"but deployment obs layout builds {expected_policy_obs_dim}. "
                    "Check proprio_history_length, command_includes_anchor_velocity, and terrain_scan settings."
                )

            # 设置从序列到实验室顺序的映射（从 MjXML 顺序映射到实验室顺序）
            try:
                self.mj2lab = np.array([self.joint_xml.index(joint) for joint in self.joint_seq])
            except ValueError as exc:
                raise ValueError(f"Joint name mismatch between ONNX metadata and joint_xml: {exc}") from exc

            # 从实验室顺序映射到 MjXML 顺序
            self.joint_pos_array = np.array([self.joint_pos_array_seq[self.joint_seq.index(joint)] for joint in self.joint_xml])

            self.default_angles_lab = self.joint_pos_array_seq
            self.action_scale_lab = self.action_scale

            logger.info("STAIRS policy initializing ...")
            self._warmup_inference_counter = 0
            self.warm_start_steps = 0
            # Cache for the last motion frame so we can keep sending it after motion ends.
            self._final_ref_cached = False
            self._final_ref_joint_pos = np.zeros_like(self.ref_joint_pos)
            self._final_ref_joint_vel = np.zeros_like(self.ref_joint_vel)
            self._final_ref_body_pos_w = np.zeros_like(self.ref_body_pos_w)
            self._final_ref_body_quat_w = np.zeros_like(self.ref_body_quat_w)
            self._final_ref_body_lin_vel_w = np.zeros_like(self.ref_body_lin_vel_w)
            self._final_ref_body_ang_vel_w = np.zeros_like(self.ref_body_ang_vel_w)

    def on_enter(self):
        self.ref_motion_phase = 0.
        self.motion_time = 0
        self.counter_step = 0
        self._warmup_inference_counter = 0
        logger.info(
            f"[FSMStateSTAIRS] enter variant={self.variant_name}, config={self.config_path}, "
            f"reference_source={self.reference_source}, motion_id={self.motion_id}, "
            f"motion_offset={self.motion_offset}, motion_length={self.motion_length}"
        )
        if self.warm_start_time > 0:
            step = self.decimation_ * self.physical_dt
            self.warm_start_steps = max(1, int(self.warm_start_time / step))
        else:
            self.warm_start_steps = 0

        observation = {}
        observation[self.input_name[0]] = np.zeros((1, self.num_obs), dtype=np.float32)
        observation.update(self._build_aux_onnx_inputs(time_index=0))
        outputs_result = self.ort_session.run(None, observation)
        if self.use_realtime_gmr:
            self.action = outputs_result[0]
            self._sync_ref_from_gmr(require=True)
        else:
            (
                # self.action,
                _,
                self.ref_joint_pos,
                self.ref_joint_vel,
                self.ref_body_pos_w,
                self.ref_body_quat_w,
                self.ref_body_lin_vel_w,
                self.ref_body_ang_vel_w,
            ) = outputs_result
        current_q_lab = self._get_current_joint_pos_lab()
        safe_scale = np.where(self.action_scale_lab == 0, 1.0, self.action_scale_lab)
        self.action = ((current_q_lab - self.default_angles_lab) / safe_scale).astype(np.float32)
        self._final_ref_cached = False
        self._warm_start_from_lab = current_q_lab
        first_ref_joint_pos = np.asarray(self.ref_joint_pos, dtype=np.float32).reshape(-1)
        if first_ref_joint_pos.shape[0] == self.num_actions:
            self._warm_start_to_lab = first_ref_joint_pos
        else:
            self._warm_start_to_lab = self._get_onnx_first_frame_lab()
        self._warm_start_prev_target = np.array(self._warm_start_from_lab, copy=True)
        self._reset_proprio_history()

        pass

    def quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        # perform multiplication
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        return np.array([w, x, y, z])

    def matrix_from_quat(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])

    def yaw_quat(self, q):
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    def euler_single_axis_to_quat(self, angle, axis, degrees=False):
        """
        将单个欧拉角转换为四元数

        参数:
            angle: 旋转角度
            axis: 旋转轴，可以是 'x', 'y', 'z' 或者单位向量 [x, y, z]
            degrees: 如果为True，输入角度为度数；如果为False，输入角度为弧度

        返回:
            四元数 (w, x, y, z)
        """
        # 转换角度为弧度
        if degrees:
            angle = np.radians(angle)

        # 计算半角
        half_angle = angle * 0.5
        cos_half = np.cos(half_angle)
        sin_half = np.sin(half_angle)

        # 根据旋转轴确定四元数分量
        if isinstance(axis, str):
            if axis.lower() == 'x':
                return np.array([cos_half, sin_half, 0.0, 0.0])
            elif axis.lower() == 'y':
                return np.array([cos_half, 0.0, sin_half, 0.0])
            elif axis.lower() == 'z':
                return np.array([cos_half, 0.0, 0.0, sin_half])
            else:
                raise ValueError("axis must be 'x', 'y', 'z' or a 3D unit vector")
        else:
            # 假设axis是一个3D向量 [x, y, z]
            axis = np.array(axis, dtype=np.float32)
            # 归一化轴向量
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0:
                raise ValueError("axis vector cannot be zero")
            axis = axis / axis_norm

            # 计算四元数分量
            w = cos_half
            x = sin_half * axis[0]
            y = sin_half * axis[1]
            z = sin_half * axis[2]

            return np.array([w, x, y, z])

    def inner_run(self):
        robot_quat = self.robot_data_.get_robot_quat()
        qj = self.robot_data_.get_joint_pos()
        # 将关节数据映射到策略使用的自由度顺序
        qj = qj[self.locked_joint_map]

        qj = qj[self.mj2lab]
        qj = (qj - self.default_angles_lab)

        if self.use_realtime_gmr:
            self._sync_ref_from_gmr(require=True)

        # IMU mounted on pelvis, so directly use measured orientation.
        ref_anchor_pos_w, ref_anchor_ori_w = self._get_ref_anchor_pose()
        ref_anchor_lin_vel_w, ref_anchor_ang_vel_w = self._get_ref_anchor_velocity()

        # 在第一帧提取当前机器人yaw方向，与参考动作yaw方向做差（与beyond mimic一致）
        if(self.counter_step < 2):
            init_to_anchor = self.matrix_from_quat(self.yaw_quat(ref_anchor_ori_w))
            world_to_anchor = self.matrix_from_quat(self.yaw_quat(robot_quat))
            self.init_to_world = world_to_anchor @ init_to_anchor.T
            self.counter_step += 1
            return

        robot_rot_mat = self.matrix_from_quat(robot_quat)
        motion_anchor_ori_b = robot_rot_mat.T @ self.init_to_world @ self.matrix_from_quat(ref_anchor_ori_w)
        ref_anchor_lin_vel_b = robot_rot_mat.T @ self.init_to_world @ ref_anchor_lin_vel_w
        ref_anchor_ang_vel_b = robot_rot_mat.T @ self.init_to_world @ ref_anchor_ang_vel_w

        ang_vel = self.robot_data_.get_angular_velocity()

        dqj = self.robot_data_.get_joint_vel()
        dqj = dqj[self.locked_joint_map]

        use_warmstart = (
            self.warm_start_steps > 0
            and self._warmup_inference_counter < self.warm_start_steps
        )
        blended_target = None
        if use_warmstart:
            blend = (self._warmup_inference_counter + 1) / self.warm_start_steps
            blended_target = (1.0 - blend) * self._warm_start_from_lab + blend * self._warm_start_to_lab
            blended_vel = (blended_target - self._warm_start_prev_target) / (
                self.decimation_ * self.physical_dt
            )
            self._warm_start_prev_target = blended_target

            command_joint_pos = blended_target.reshape(1, -1)
            command_joint_vel = blended_vel.reshape(1, -1)
            safe_scale = np.where(self.action_scale_lab == 0, 1.0, self.action_scale_lab)
            action_for_history = (blended_target - self.default_angles_lab) / safe_scale
        else:
            command_joint_pos = self.ref_joint_pos
            command_joint_vel = self.ref_joint_vel
            action_for_history = self.action

        command_root = self.matrix_from_quat(ref_anchor_ori_w)
        command_parts = [command_joint_pos.squeeze(0), command_joint_vel.squeeze(0)]
        if self.command_includes_anchor_velocity:
            command_parts.extend([ref_anchor_lin_vel_b, ref_anchor_ang_vel_b])
        # ref_anchor_pos_w and command_root are intentionally not part of the terrain WoSE policy observation.
        command_vec = np.concatenate(command_parts, dtype=np.float32)

        proprio_obs_buf = self._build_proprio_obs(
            {
                "command": command_vec,
                "motion_anchor_ori_b": motion_anchor_ori_b[:, :2].reshape(-1),
                "base_ang_vel": ang_vel,
                "joint_pos": qj,
                "joint_vel": dqj[self.mj2lab],
                "actions": np.asarray(action_for_history, dtype=np.float32).reshape(-1),
            }
        )
        expected_scan_size = self.num_obs - proprio_obs_buf.shape[0]
        if expected_scan_size > 0:
            terrain_scan = self._get_terrain_scan(expected_scan_size)
            mimic_obs_buf = np.concatenate(
                (proprio_obs_buf, terrain_scan),
                axis=-1,
                dtype=np.float32,
            )
        else:
            mimic_obs_buf = proprio_obs_buf
        if mimic_obs_buf.shape[0] != self.num_obs:
            raise RuntimeError(f"Observation length mismatch. Expected {self.num_obs}, got {mimic_obs_buf.shape[0]}.")

        mimic_obs_tensor = np.expand_dims(mimic_obs_buf, axis=0)
        observation = {}

        # obs0 是网络观测，obs1 是当前时间步，用于输出参考动作信息
        observation[self.input_name[0]] = mimic_obs_tensor
        time_index = max(self.counter_step - self.warm_start_steps, 0)

        if (
            self.hold_final_reference
            and self.motion_length is not None
            and self.motion_length > 0
        ):
            if self.motion_length is not None and self.motion_length > 0:
                time_index = min(time_index, self.motion_length - 1)

        observation.update(self._build_aux_onnx_inputs(time_index=time_index))
        outputs_result = self.ort_session.run(None, observation)
        if self.use_realtime_gmr:
            self.action = outputs_result[0]
        else:
            (
                self.action,
                self.ref_joint_pos,
                self.ref_joint_vel,
                self.ref_body_pos_w,
                self.ref_body_quat_w,
                self.ref_body_lin_vel_w,
                self.ref_body_ang_vel_w,
            ) = outputs_result

        if (
            self.hold_final_reference
            and self.motion_length is not None
            and self.motion_length > 0
        ):
            if time_index == self.motion_length - 1 and not self._final_ref_cached:
                self._cache_final_ref()
            elif self.counter_step >= self.motion_length and self._final_ref_cached:
                self._apply_final_ref()
        target_dof_pos_mj = np.zeros(self.motor_nums)
        target_dof_pos_policy = np.zeros(self.num_actions)
        if use_warmstart and blended_target is not None:
            target_dof_pos_lab = blended_target
            # Keep action history aligned with the inserted warm trajectory.
            self.action = np.asarray(action_for_history, dtype=np.float32).reshape(1, -1)
        else:
            target_dof_pos_lab = self.action * self.action_scale_lab + self.default_angles_lab
            if target_dof_pos_lab.ndim > 1:
                target_dof_pos_lab = np.squeeze(target_dof_pos_lab, axis=0)

        if self.warm_start_steps > 0:
            self._warmup_inference_counter += 1
            if self._warmup_inference_counter <= self.warm_start_steps:
                blend = self._warmup_inference_counter / self.warm_start_steps
                if not use_warmstart:
                    target_dof_pos_lab = (1.0 - blend) * self._warm_start_from_lab + blend * self._warm_start_to_lab

        target_dof_pos_policy[self.mj2lab] = target_dof_pos_lab
        target_dof_pos_mj[self.locked_joint_map] = target_dof_pos_policy

        # Set joint commands exactly like C++
        for i in range(self.motor_nums):
            # C++: robot_data_->q_d_(35 - motor_num_ + i)
            joint_idx = 35 - self.motor_nums + i
            self.robot_data_.q_d_[joint_idx] = target_dof_pos_mj[i]
            self.robot_data_.q_dot_d_[joint_idx] = 0.0
            self.robot_data_.tau_d_[joint_idx] = 0.0

        # update motion phase
        self.counter_step += 1

    def _cache_final_ref(self):
        if not self.hold_final_reference:
            return
        self._final_ref_cached = True
        self._final_ref_joint_pos = np.array(self.ref_joint_pos, copy=True)
        self._final_ref_joint_vel = np.array(self.ref_joint_vel, copy=True)
        self._final_ref_body_pos_w = np.array(self.ref_body_pos_w, copy=True)
        self._final_ref_body_quat_w = np.array(self.ref_body_quat_w, copy=True)
        self._final_ref_body_lin_vel_w = np.array(self.ref_body_lin_vel_w, copy=True)
        self._final_ref_body_ang_vel_w = np.array(self.ref_body_ang_vel_w, copy=True)

    def _apply_final_ref(self):
        if not self.hold_final_reference or not self._final_ref_cached:
            return
        self.ref_joint_pos = np.array(self._final_ref_joint_pos, copy=True)
        self.ref_joint_vel = np.array(self._final_ref_joint_vel, copy=True)
        self.ref_body_pos_w = np.array(self._final_ref_body_pos_w, copy=True)
        self.ref_body_quat_w = np.array(self._final_ref_body_quat_w, copy=True)
        self.ref_body_lin_vel_w = np.array(self._final_ref_body_lin_vel_w, copy=True)
        self.ref_body_ang_vel_w = np.array(self._final_ref_body_ang_vel_w, copy=True)

    def run(self, flag: FSMControlFlag):
        if self.robot_data_.control_step_ % self.decimation_ == 0:
            current_time = time.perf_counter()
            # logger.debug(f"Inference hz: {1/(current_time - self.last_run_time)}")
            self.last_run_time = current_time
            self.inner_run()
        self.set_kp_kd()

    def set_kp_kd(self):
        # Set kp/kd gains
        self.robot_data_.joint_kp_p_[:self.motor_nums] = self.kps
        self.robot_data_.joint_kd_p_[:self.motor_nums] = self.kds

    def on_exit(self):
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.ref_motion_phase = 0.
        self.motion_time = 0
        self.counter_step = 0
        self._final_ref_cached = False
        self._reset_proprio_history()

        logger.info("exited")

    def check_transition(self, *args, **kwargs):
        """检查状态转换"""
        pass

    def _get_ref_anchor_pose(self):
        ref_pos = self.ref_body_pos_w[:, self.anchor_body_index].squeeze(0)
        ref_quat = self.ref_body_quat_w[:, self.anchor_body_index].squeeze(0)
        return ref_pos.astype(np.float32), ref_quat.astype(np.float32)

    def _get_ref_anchor_velocity(self):
        ref_lin_vel = self.ref_body_lin_vel_w[:, self.anchor_body_index].squeeze(0)
        ref_ang_vel = self.ref_body_ang_vel_w[:, self.anchor_body_index].squeeze(0)
        return ref_lin_vel.astype(np.float32), ref_ang_vel.astype(np.float32)

    def _reset_proprio_history(self):
        self._proprio_history = {}

    def _append_history_term(self, term_name: str, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        expected_shape = (self.proprio_history_length, value.shape[0])
        history = self._proprio_history.get(term_name)
        if history is None or history.shape != expected_shape:
            history = np.repeat(value[None, :], self.proprio_history_length, axis=0)
        else:
            history = np.roll(history, shift=-1, axis=0)
            history[-1] = value
        self._proprio_history[term_name] = history
        return history.reshape(-1)

    def _build_proprio_obs(self, term_values: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [self._append_history_term(term_name, term_values[term_name]) for term_name in self._proprio_history_terms],
            axis=-1,
            dtype=np.float32,
        )

    def _expected_policy_obs_dim(self) -> int:
        command_dim = self.num_actions * 2
        if self.command_includes_anchor_velocity:
            command_dim += 6
        proprio_frame_dim = command_dim + 6 + 3 + self.num_actions * 3
        obs_dim = proprio_frame_dim * self.proprio_history_length
        if self.terrain_scan_enabled:
            obs_dim += self.terrain_scan_size
        return obs_dim

    def _build_aux_onnx_inputs(self, time_index: int) -> dict[str, np.ndarray]:
        aux_inputs = {}
        for input_name in self.input_name[1:]:
            if input_name == "time_step":
                value = time_index if self._onnx_uses_motion_id_input else self._global_time_index(time_index)
            elif input_name == "motion_id":
                value = self.motion_id
            else:
                logger.warning(f"[FSMStateSTAIRS] Unknown ONNX auxiliary input {input_name}; feeding zero.")
                value = 0
            aux_inputs[input_name] = np.array([[value]], dtype=np.float32)
        return aux_inputs

    def _global_time_index(self, local_time_index: int) -> int:
        local_time_index = max(int(local_time_index), 0)
        if self.motion_length is not None and self.motion_length > 0:
            local_time_index = min(local_time_index, self.motion_length - 1)
        return int(self.motion_offset) + local_time_index

    def _resolve_motion_selection(self):
        if self.motion_names and self.motion_name is not None:
            if self.motion_name in self.motion_names:
                self.motion_id = self.motion_names.index(self.motion_name)
            else:
                matches = [i for i, name in enumerate(self.motion_names) if self.motion_name in name]
                if len(matches) == 1:
                    self.motion_id = matches[0]
                else:
                    raise ValueError(
                        f"motion_name {self.motion_name!r} not found in ONNX motion_names. "
                        f"Available examples: {self.motion_names[:5]}"
                    )
        if self.motion_files and self.motion_name is not None and not self.motion_names:
            basenames = [os.path.splitext(os.path.basename(path))[0] for path in self.motion_files]
            if self.motion_name in basenames:
                self.motion_id = basenames.index(self.motion_name)

        if self.motion_offsets or self.motion_lengths:
            if len(self.motion_offsets) != len(self.motion_lengths):
                raise ValueError(
                    f"motion_offsets length {len(self.motion_offsets)} must match "
                    f"motion_lengths length {len(self.motion_lengths)}."
                )
            if not (0 <= self.motion_id < len(self.motion_lengths)):
                raise ValueError(
                    f"motion_id {self.motion_id} is out of range for {len(self.motion_lengths)} embedded motions."
                )
            self.motion_offset = int(self.motion_offsets[self.motion_id])
            self.motion_length = int(self.motion_lengths[self.motion_id])
        elif self.motion_length is not None:
            self.motion_offset = 0

        if self.motion_files and self.motion_names and len(self.motion_files) != len(self.motion_names):
            logger.warning(
                f"[FSMStateSTAIRS] motion_files length {len(self.motion_files)} differs from "
                f"motion_names length {len(self.motion_names)}."
            )

    def _sync_ref_from_gmr(self, require: bool = False) -> bool:
        """Use runtime GMR reference when this variant is configured for online motion."""
        if not self.use_realtime_gmr:
            self._gmr_ref_active = False
            return False

        ref = None
        if hasattr(self.robot_data_, "get_gmr_reference"):
            ref = self.robot_data_.get_gmr_reference(
                max_age_s=self.gmr_reference_max_age_s
            )
        if ref is not None:
            ref_body_pos = ref["body_pos_w"]
            ref_body_quat = ref["body_quat_w"]
            ref_joint_pos = ref["joint_pos"]
            ref_joint_vel = ref["joint_vel"]
            ref_body_lin_vel = ref.get("body_lin_vel_w")
            ref_body_ang_vel = ref.get("body_ang_vel_w")
        else:
            ref_body_pos = getattr(self.robot_data_, "ref_body_pos_w", None)
            ref_body_quat = getattr(self.robot_data_, "ref_body_quat_w", None)
            ref_joint_pos = getattr(self.robot_data_, "ref_joint_pos", None)
            ref_joint_vel = getattr(self.robot_data_, "ref_joint_vel", None)
            ref_body_lin_vel = getattr(self.robot_data_, "ref_body_lin_vel_w", None)
            ref_body_ang_vel = getattr(self.robot_data_, "ref_body_ang_vel_w", None)
        if (
            ref_body_pos is None
            or ref_body_quat is None
            or ref_joint_pos is None
            or ref_joint_vel is None
        ):
            message = (
                "[FSMStateSTAIRS] No /gmr_info reference available for realtime GMR stairs."
            )
            if require:
                raise RuntimeError(message)
            if not self._warned_missing_gmr:
                logger.warning(message)
                self._warned_missing_gmr = True
            self._gmr_ref_active = False
            return False

        ref_body_pos = np.asarray(ref_body_pos, dtype=np.float32)
        ref_body_quat = np.asarray(ref_body_quat, dtype=np.float32)
        ref_joint_pos = np.asarray(ref_joint_pos, dtype=np.float32).reshape(1, -1)
        ref_joint_vel = np.asarray(ref_joint_vel, dtype=np.float32).reshape(1, -1)
        if ref_body_pos.size != self.num_bodies * 3:
            raise RuntimeError(
                f"[FSMStateSTAIRS] /gmr_info body_pos size {ref_body_pos.size} "
                f"does not match expected {self.num_bodies * 3}."
            )
        if ref_body_quat.size != self.num_bodies * 4:
            raise RuntimeError(
                f"[FSMStateSTAIRS] /gmr_info body_quat size {ref_body_quat.size} "
                f"does not match expected {self.num_bodies * 4}."
            )
        if ref_joint_pos.shape[1] != self.num_actions or ref_joint_vel.shape[1] != self.num_actions:
            raise RuntimeError(
                f"[FSMStateSTAIRS] /gmr_info joint size "
                f"{ref_joint_pos.shape[1]}/{ref_joint_vel.shape[1]} does not match num_actions {self.num_actions}."
            )

        self.ref_joint_pos = np.array(ref_joint_pos, copy=True)
        self.ref_joint_vel = np.array(ref_joint_vel, copy=True)
        self.ref_body_pos_w = np.array(ref_body_pos.reshape(1, self.num_bodies, 3), copy=True)
        self.ref_body_quat_w = np.array(ref_body_quat.reshape(1, self.num_bodies, 4), copy=True)
        if ref_body_lin_vel is not None:
            self.ref_body_lin_vel_w = np.array(
                np.asarray(ref_body_lin_vel, dtype=np.float32).reshape(1, self.num_bodies, 3),
                copy=True,
            )
        else:
            self.ref_body_lin_vel_w = np.zeros_like(self.ref_body_pos_w, dtype=np.float32)
        if ref_body_ang_vel is not None:
            self.ref_body_ang_vel_w = np.array(
                np.asarray(ref_body_ang_vel, dtype=np.float32).reshape(1, self.num_bodies, 3),
                copy=True,
            )
        else:
            self.ref_body_ang_vel_w = np.zeros_like(self.ref_body_pos_w, dtype=np.float32)

        self._warned_missing_gmr = False
        self._gmr_ref_active = True
        return True

    def _get_current_joint_pos_lab(self) -> np.ndarray:
        try:
            current_q = self.robot_data_.get_joint_pos()
            current_q = current_q[self.locked_joint_map]
            current_q_lab = current_q[self.mj2lab]
            return current_q_lab.astype(np.float32)
        except Exception as exc:
            logger.warning(f"[FSMStateSTAIRS] Failed to read current joint pose: {exc}")
            return np.array(self.default_angles_lab, copy=True)

    def _get_terrain_scan(self, scan_size: int) -> np.ndarray:
        if scan_size <= 0:
            return np.zeros(0, dtype=np.float32)
        if hasattr(self.robot_data_, "get_terrain_scan"):
            terrain_scan = self.robot_data_.get_terrain_scan(scan_size)
            valid = getattr(self.robot_data_, "terrain_scan_valid_", False)
            if not valid and not self._terrain_scan_missing_warned:
                logger.warning(
                    "[FSMStateSTAIRS] Terrain scan has not been received; "
                    "using zeros until PointCloud2 arrives."
                )
                self._terrain_scan_missing_warned = True
            return terrain_scan.astype(np.float32, copy=False)
        if not self._terrain_scan_missing_warned:
            logger.warning("[FSMStateSTAIRS] RobotData has no terrain scan cache; using zeros.")
            self._terrain_scan_missing_warned = True
        return np.zeros(scan_size, dtype=np.float32)

    def _get_onnx_first_frame_lab(self) -> np.ndarray:
        try:
            action = self.action
            if action is None:
                raise RuntimeError("ONNX action output is None")
            if action.ndim > 1:
                action = np.squeeze(action, axis=0)
            first_frame = action * self.action_scale_lab + self.default_angles_lab
            return first_frame.astype(np.float32)
        except Exception as exc:
            logger.warning(f"[FSMStateSTAIRS] Failed to read ONNX first frame: {exc}")
            return np.array(self.default_angles_lab, copy=True)
