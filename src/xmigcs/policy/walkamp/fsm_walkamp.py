"""
FSM State Implementations
Concrete implementations of different FSM states
"""

import numpy as np
import onnxruntime as ort

from xmigcs.FSM.fsm_base import FSMState, FSMStateName
from xmigcs.common.control_flag import FSMControlFlag
from xmigcs.common.robot_data import RobotData
import os
import time
import yaml
from scipy.spatial.transform import Rotation

from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

class FSMStateWALKAMP(FSMState):
    """WALKAMP策略状态实现"""

    def _reset_internal_state(self):
        """把所有随时间变化的内部状态重置成初始值"""

        # 1) 清空 obs / hist / actions
        self.observations_.fill(0.0)
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)
        self.smoothed_command_.fill(0.0)

        # 2) 标志位重置
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self.is_first_step_ = True

        # 3) 期望关节 / 期望速度 / 力矩重置为“初始姿态”
        # 你已经有 self.joint_pos_array（mj 顺序，长度 len(self.joint_xml)）
        base = self.robot_data_.q_d_.shape[0] - self.motor_num_
        # 期望角 = 初始角
        self.robot_data_.q_d_[base:base + len(self.joint_xml)] = self.joint_pos_array
        # 期望速度 = 0
        self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
        # 期望力矩 = 0（位置控制）
        self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)

        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "walk_amp.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)
        # Load configuration exactly like C++
        self.action_num_ = policy_config.get('actions_size')
        self.motor_num_ = policy_config.get('motor_num')
        self.dt_ = policy_config.get('dt')

        # Size configuration
        size_config = policy_config.get('size', {})
        self.num_hist_ = size_config.get('num_hist')
        self.obs_size_ = size_config.get('observations_size')

        # Control configuration
        control_config = policy_config.get('control', {})
        self.action_scale_ = control_config.get('action_scale')
        # self.gait_cycle_period_ = control_config.get('gait_cycle_period', 1.0)
        self.decimation_ = control_config.get('decimation')
        self.warm_start_time_ = control_config.get('warm_start_time', 0.3)
        self.inference_period_s_ = float(self.decimation_ if self.decimation_ else 1) * float(self.dt_)
        self.inference_hz_nominal_ = 1.0 / self.inference_period_s_ if self.inference_period_s_ > 0 else 0.0
        timing_config = policy_config.get('inference_timing') or {}
        self.delay_check_enabled_ = bool(timing_config.get('enabled', True))
        self._infer_iter_start = None
        self.time_sleep = self.inference_period_s_

        command_config = policy_config.get('commands', {})
        command_ranges = command_config.get('ranges', {})
        self.command_lin_vel_x_range_ = tuple(command_ranges.get('lin_vel_x', [-0.8, 1.1]))
        self.command_lin_vel_y_range_ = tuple(command_ranges.get('lin_vel_y', [-0.5, 0.5]))
        self.command_ang_vel_z_range_ = tuple(command_ranges.get('ang_vel_z', [-3.14, 3.14]))
        self.yaw_only_lin_vel_range_ = tuple(command_config.get('yaw_only_lin_vel_range', [-0.03, 0.03]))
        self.yaw_only_ang_vel_deadzone_ = float(command_config.get('yaw_only_ang_vel_deadzone', 0.2))
        self.straight_lateral_vel_range_ = tuple(command_config.get('straight_lateral_vel_range', [-0.05, 0.05]))
        self.straight_ang_vel_range_ = tuple(command_config.get('straight_ang_vel_range', [-0.30, 0.30]))
        self.lateral_forward_vel_range_ = tuple(command_config.get('lateral_forward_vel_range', [-0.05, 0.05]))
        self.lateral_ang_vel_range_ = tuple(command_config.get('lateral_ang_vel_range', [-0.5, 0.5]))
        self.lateral_vel_deadzone_ = float(command_config.get('lateral_vel_deadzone', 0.2))
        self.command_zero_epsilon_ = float(command_config.get('zero_epsilon', 1e-6))
        self.policy_cmd_smooth_alpha_ = float(command_config.get('policy_cmd_smooth_alpha', 0.2))
        self.policy_cmd_smooth_alpha_ = float(np.clip(self.policy_cmd_smooth_alpha_, 0.0, 1.0))
        self.policy_cmd_smooth_snap_epsilon_ = float(command_config.get('policy_cmd_smooth_snap_epsilon', 1e-3))

        # Normalization configuration
        norm_config = policy_config.get('normalization', {})
        clip_config = norm_config.get('clip_scales', {})
        obs_config = norm_config.get('obs_scales', {})

        self.clip_obs_ = clip_config.get('clip_observations', 100.0)
        self.clip_act_ = clip_config.get('clip_actions', 100.0)
        self.lin_vel_scale_ = obs_config.get('lin_vel')
        self.ang_vel_scale_ = obs_config.get('ang_vel')
        self.dof_pos_scale_ = obs_config.get('dof_pos')
        self.dof_vel_scale_ = obs_config.get('dof_vel')


        # Initialize buffers and actions
        self.observations_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.proprio_hist_buf_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.smoothed_command_ = np.zeros(3, dtype=np.float32)
        self._warm_start_pose = np.zeros(self.motor_num_, dtype=np.float32)


        # Flags matching C++
        self.is_first_obs_ = True
        self.is_first_action_ = True
        # self.phase_locked = False
        self.is_first_step_ = True
        step = (self.decimation_ if self.decimation_ else 1) * self.dt_
        if self.warm_start_time_ > 0 and step > 0:
            self._warm_start_steps = max(1, int(self.warm_start_time_ / step))
        else:
            self._warm_start_steps = 0
        self._warmup_inference_counter = 0


        # Initialize ONNX session
        self.model_path = os.path.join(current_dir, "model", policy_config["model_path"]) 
        self._init_onnx_session()

        # ====== 从 ONNX metadata 里读 lab 侧关节信息 ======
        # self.onnx_model = onnx.load(self.model_path)
        # self.onnx_model = onnxruntime.InferenceSession(self.model_path)
        meta = self.ort_session_.get_modelmeta().custom_metadata_map

        # 从ONNX模型中读取参数
        self.joint_seq = None
        self.joint_pos_array_seq = None
        self.action_scale = None
        self.stiffness_array_seq = None
        self.damping_array_seq = None
        
        self.joint_seq = meta["joint_names"].split(",")
        self.joint_pos_array_seq = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
        self.stiffness_array_seq = np.array([float(x) for x in meta["joint_stiffness"].split(",")])
        self.damping_array_seq = np.array([float(x) for x in meta["joint_damping"].split(",")])
        self.action_scale = np.array([float(x) for x in meta["action_scale"].split(",")])
        # for prop in self.onnx_model.metadata_props:
        #     if prop.key == "joint_names":
        #         self.joint_seq = prop.value.split(",")
        #     if prop.key == "default_joint_pos":   
        #         self.joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
        #     if prop.key == "joint_stiffness":
        #         self.stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
        #     if prop.key == "joint_damping":
        #         self.damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
        #     if prop.key == "action_scale":
        #         self.action_scale = np.array([float(x) for x in prop.value.split(",")])
        
        # # 设置从序列到实验室顺序的映射
        self.joint_xml = [
            "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
            "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
            "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
            "head_pitch_joint", "head_yaw_joint",
        ]

        # 从MjXUML顺序映射到实验室顺序
        # self.mj2lab = np.array([self.joint_xml.index(joint) for joint in self.joint_seq])
        self.lab2mj = []
        for name in self.joint_seq:
            if name not in self.joint_xml:
                raise ValueError(f"[FSMStateWALKAMP] joint '{name}' from ONNX not found in joint_xml!")
            self.lab2mj.append(self.joint_xml.index(name))
        self.lab2mj = np.array(self.lab2mj, dtype=int)

        # 从实验室顺序映射到MjXUML顺序
        # self.joint_pos_array = np.array([self.joint_pos_array_seq[self.joint_seq.index(joint)] for joint in self.joint_xml])
        # self.stiffness_array = np.array([self.stiffness_array_seq[self.joint_seq.index(joint)] for joint in self.joint_xml])
        # self.damping_array = np.array([self.damping_array_seq[self.joint_seq.index(joint)] for joint in self.joint_xml])
                # ====== 把 23 个 lab 关节 scatter 到 29 个 xml 里，多的 6 个保持默认 ======
        n_mj = len(self.joint_xml)

        # 29 长度，mujoco XML 顺序，先全 0 或者你想要的默认值
        self.joint_pos_array = np.zeros(n_mj, dtype=np.float32)
        self.stiffness_array = np.zeros(n_mj, dtype=np.float32)
        self.damping_array = np.zeros(n_mj, dtype=np.float32)

        # joint_pos_array_seq / stiffness_array_seq / damping_array_seq 是 23 长度，lab 顺序
        for lab_idx, mj_idx in enumerate(self.lab2mj):
            self.joint_pos_array[mj_idx] = self.joint_pos_array_seq[lab_idx]
            self.stiffness_array[mj_idx] = self.stiffness_array_seq[lab_idx]
            self.damping_array[mj_idx] = self.damping_array_seq[lab_idx]


        # 设置其他参数
        self.kps_lab = self.stiffness_array_seq
        self.kds_lab = self.damping_array_seq
        self.default_angles_lab = self.joint_pos_array_seq
        self.action_scale_lab = self.action_scale

        # n_mj = len(self.joint_xml)
        # self.kp_array = np.zeros(n_mj, dtype=np.float32)
        # self.kd_array = np.zeros(n_mj, dtype=np.float32)

        # for lab_idx, mj_idx in enumerate(self.lab2mj):
        #     self.kp_array[mj_idx] = self.kps_lab[lab_idx]
        #     self.kd_array[mj_idx] = self.kds_lab[lab_idx]

    def _init_onnx_session(self):
        """初始化ONNX推理会话"""
        try:
            # 配置SessionOptions
            options = ort.SessionOptions()

            # 启用图优化，使用所有可用的优化（包括算子融合等）
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # 设置执行模式（可选，默认执行模式是顺序执行，但图优化会改变计算图）
            # 设置线程数（根据CPU核心数调整）
            # 建议设置为CPU物理核心数（非超线程数），因为超线程可能不会带来线性提升
            options.intra_op_num_threads = 1  # 设置计算图中的运算符内部并行线程数
            options.inter_op_num_threads = 1  # 设置多个运算符之间的并行线程数（如果模型有多个分支）

            # 启用内存优化（避免重复分配内存）
            options.enable_mem_pattern = False  # 对于固定输入大小，可以设为False以避免内存规划的开销
            options.enable_mem_reuse = True # 启用内存重用机制

            self.ort_session_ = ort.InferenceSession(self.model_path, options, providers=['CPUExecutionProvider'])
            
        except Exception as e:
            self.ort_session_ = None

    def _reset_infer_timing_state(self):
        self._infer_iter_start = None
        self.time_sleep = self.inference_period_s_

    def _begin_delay_check(self):
        if self.delay_check_enabled_:
            self._infer_iter_start = time.perf_counter()

    def delay_check(self):
        """更新动态 time_sleep，并在推理超过名义周期时提示延迟。"""
        if not self.delay_check_enabled_ or self._infer_iter_start is None:
            return

        time_until_next_step = self.inference_period_s_ - (
            time.perf_counter() - self._infer_iter_start
        )
        self.time_sleep = max(0.0, time_until_next_step)
        if time_until_next_step <= 0:
            logger.info(f"[FSMStateMLP] Time until next step is negative: {time_until_next_step:.6f}")

    def on_enter(self):
        """进入WALKAMP状态"""
        self._reset_internal_state()
        self._reset_infer_timing_state()
        # logger.info("[FSMStateMLP] enter")
        # logger.info(
        #     "[FSMStateMLP] nominal policy inference: "
        #     f"{self.inference_hz_nominal_:.3f} Hz "
        #     f"(period {self.inference_period_s_ * 1000.0:.3f} ms = "
        #     f"decimation={self.decimation_} x dt={self.dt_})"
        # )
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self.smoothed_command_.fill(0.0)
        self._warmup_inference_counter = 0
        if self.robot_data_ is not None:
            try:
                self._warm_start_pose = self.robot_data_.get_serial_joint_pos_desired()
            except Exception:
                self._warm_start_pose.fill(0.0)
        else:
            self._warm_start_pose.fill(0.0)

    def run(self, flag: FSMControlFlag):
        """运行WALKAMP状态 - 与C++版本完全一致"""
        # Only run policy inference every decimation_ steps
        if self.robot_data_.control_step_ % self.decimation_ == 0:
            self.compute_observation(flag)
            self.compute_actions()


                    # ====== 临时测试：强制动作为 0 ======
            # self.actions_.fill(0.0)
            # q_des = self.robot_data_.get_joint_pos()
            # q_des = q_des[self.mj2lab]
            # q_des = (q_des - self.default_angles_lab)

            # lab 顺序目标角 23 维
            target_dof_pos_lab = self.actions_ * self.action_scale_lab + self.default_angles_lab

            # 拿一份当前 mj 顺序的关节角（或你原来用的 default 也行）
            target_dof_pos_mj = self.robot_data_.get_joint_pos().copy()

            # 只更新 23 个受控 DOF
            target_dof_pos_mj[self.lab2mj] = target_dof_pos_lab
            # target_dof_pos_mj = np.zeros(29)
            # target_dof_pos_lab = self.actions_ * self.action_scale_lab + self.default_angles_lab
            # target_dof_pos_mj[self.mj2lab] = target_dof_pos_lab
            commanded_pos = target_dof_pos_mj
            # if self._warm_start_steps > 0 and self._warmup_inference_counter < self._warm_start_steps:
            #     self._warmup_inference_counter += 1
            #     blend = self._warmup_inference_counter / float(self._warm_start_steps)
            #     commanded_pos = (1.0 - blend) * self._warm_start_pose + blend * target_dof_pos_mj

            base = self.robot_data_.q_d_.shape[0] - self.motor_num_
            self.robot_data_.q_d_[base:base + len(self.joint_xml)] = commanded_pos

            self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
            self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0
            # q_des_lab = self.default_angles_lab + self.actions_ * self.action_scale_lab
            # qdot_des_lab = np.zeros_like(q_des_lab, dtype=np.float32)

            self.last_actions_[:] = self.actions_

        self.robot_data_.joint_kp_p_[:len(self.joint_xml)] = self.stiffness_array
        self.robot_data_.joint_kd_p_[:len(self.joint_xml)] = self.damping_array
        self.robot_data_.joint_kp_p_[27] = 80
        self.robot_data_.joint_kd_p_[27] = 5
        self.robot_data_.joint_kp_p_[26] = 80
        self.robot_data_.joint_kd_p_[26] = 5
        self.robot_data_.joint_kp_p_[28] = 80
        self.robot_data_.joint_kd_p_[28] = 5
        self.robot_data_.q_d_[33] = 0
        self.robot_data_.q_d_[34] = 0
        self.robot_data_.q_d_[32] = 0

        self.robot_data_.joint_kp_p_[21] = 80
        self.robot_data_.joint_kd_p_[21] = 5
        self.robot_data_.joint_kp_p_[20] = 80
        self.robot_data_.joint_kd_p_[20] = 5
        self.robot_data_.joint_kp_p_[19] = 80
        self.robot_data_.joint_kd_p_[19] = 5
        self.robot_data_.q_d_[27] = 0
        self.robot_data_.q_d_[26] = 0
        self.robot_data_.q_d_[25] = 0


    def compute_observation(self, flag: FSMControlFlag):

        roll, pitch, yaw = (
                        float(self.robot_data_.imu_data_[2]),
                        float(self.robot_data_.imu_data_[1]),
                        float(self.robot_data_.imu_data_[0]),
                    )
        quat_wxyz = self.euler_to_quaternion_scipy(roll, pitch, yaw)
        q_xyzw    = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        gravity_init   = self.quat_rotate_inverse_numpy(q_xyzw, np.array([0.,0.,-1.], dtype=np.float32))
        

        # Command vector exactly like C++
        raw_command = self.robot_data_.get_walk_cmd()
        # raw_command = np.array([
        #     flag.x_speed_command,
        #     flag.y_speed_command,
        #     flag.yaw_speed_command
        # ], dtype=np.float32)
        target_command, target_mode = self._project_command_to_xwalk_modes_numpy(raw_command)
        command, command_mode = self._smooth_policy_command(target_command)
        # logger.debug(
        #     f'Input command raw={raw_command}, target_mode={target_mode}, '
        #     f'target={target_command}, policy_mode={command_mode}, policy={command}'
        # )

        # IMU data exactly like C++
        # rpy = np.array([
        #     self.robot_data_.imu_data_[2],  # roll
        #     self.robot_data_.imu_data_[1],  # pitch
        #     self.robot_data_.imu_data_[0]   # yaw
        # ], dtype=np.float32) * 1.0

        gyro = np.array([
            self.robot_data_.imu_data_[3],
            self.robot_data_.imu_data_[4],
            self.robot_data_.imu_data_[5]
        ], dtype=np.float32) * self.ang_vel_scale_

        q_mj = self.robot_data_.get_joint_pos()
        qdot_mj = self.robot_data_.get_joint_vel()



        ang_vel = self.robot_data_.get_angular_velocity()
        q_mj = self.robot_data_.get_joint_pos()   # mj 顺序，长度 29
        dq_mj = self.robot_data_.get_joint_vel()

        # 只取 23 个受控关节，变成 lab 顺序
        qj = q_mj[self.lab2mj]
        dqj = dq_mj[self.lab2mj]

        qj = qj - self.default_angles_lab


        amp_condition_onehot = self._compute_amp_condition_from_command(command)

        # Concatenate exactly like C++
        proprio = np.concatenate([
            ang_vel ,              # 3 elements
            gravity_init,
            command,
            amp_condition_onehot,
            qj,         # 12 elements
            dqj,         # 12 elements
            self.last_actions_, # 12 elements
            [1],
        ])  # Total: 47 elements

        # History buffer management exactly like C++
        if self.is_first_obs_:
            for i in range(self.num_hist_):
                start_idx = i * self.obs_size_
                end_idx = start_idx + self.obs_size_
                self.proprio_hist_buf_[start_idx:end_idx] = proprio
            self.is_first_obs_ = False
        else:
            # Shift history: head((num_hist-1)*obs_size) = tail((num_hist-1)*obs_size)
            shift_size = (self.num_hist_ - 1) * self.obs_size_
            self.proprio_hist_buf_[:shift_size] = self.proprio_hist_buf_[self.obs_size_:]
            self.proprio_hist_buf_[shift_size:] = proprio

        # Clip observations exactly like C++
        self.observations_ = np.clip(self.proprio_hist_buf_, -self.clip_obs_, self.clip_obs_)


    def _smooth_policy_command(self, target_command: np.ndarray) -> tuple[np.ndarray, str]:
        alpha = self.policy_cmd_smooth_alpha_
        self.smoothed_command_[:] = alpha * target_command + (1.0 - alpha) * self.smoothed_command_
        if np.max(np.abs(self.smoothed_command_ - target_command)) <= self.policy_cmd_smooth_snap_epsilon_:
            self.smoothed_command_[:] = target_command

        # Re-project after filtering so the policy still only sees xwalk-supported command modes.
        return self._project_command_to_xwalk_modes_numpy(self.smoothed_command_)


    def _project_command_to_xwalk_modes_numpy(self, raw_command: np.ndarray) -> tuple[np.ndarray, str]:
        """Project teleop command into the xwalk command-mode support used for training."""
        command = np.zeros(3, dtype=np.float32)
        if raw_command.shape[0] < 3:
            return command, "idle"

        vx = float(raw_command[0])
        vy = float(raw_command[1])
        wz = float(raw_command[2])
        if max(abs(vx), abs(vy), abs(wz)) <= self.command_zero_epsilon_:
            return command, "idle"

        mode = self._select_xwalk_command_mode(vx, vy, wz)
        lin_x_low, lin_x_high = self.command_lin_vel_x_range_
        lin_y_low, lin_y_high = self.command_lin_vel_y_range_
        ang_low, ang_high = self.command_ang_vel_z_range_

        if mode == "yaw_only":
            command[0] = self._clip(vx, *self.yaw_only_lin_vel_range_)
            command[1] = self._clip(vy, *self.yaw_only_lin_vel_range_)
            command[2] = self._clip_signed_with_deadzone(wz, ang_low, ang_high, self.yaw_only_ang_vel_deadzone_)
        elif mode == "forward":
            command[0] = self._clip(vx, max(0.0, float(lin_x_low)), lin_x_high)
            command[1] = self._clip(vy, *self.straight_lateral_vel_range_)
            command[2] = self._clip(wz, *self.straight_ang_vel_range_)
        elif mode == "backward":
            command[0] = self._clip(vx, lin_x_low, min(0.0, float(lin_x_high)))
            command[1] = self._clip(vy, *self.straight_lateral_vel_range_)
            command[2] = self._clip(wz, *self.straight_ang_vel_range_)
        elif mode == "lateral":
            command[0] = self._clip(vx, *self.lateral_forward_vel_range_)
            command[1] = self._clip_signed_with_deadzone(vy, lin_y_low, lin_y_high, self.lateral_vel_deadzone_)
            command[2] = self._clip(wz, *self.lateral_ang_vel_range_)
        else:
            mode = "idle"
        return command, mode

    def _compute_amp_condition_from_command(self, command: np.ndarray) -> np.ndarray:
        """Linearly blend AMP condition channels from the policy command magnitude."""
        amp_condition = np.zeros(6, dtype=np.float32)
        if command.shape[0] < 3:
            amp_condition[4] = 1.0
            return amp_condition

        vx = float(command[0])
        vy = float(command[1])
        wz = float(command[2])

        lin_x_low, lin_x_high = self.command_lin_vel_x_range_
        lin_y_low, lin_y_high = self.command_lin_vel_y_range_
        ang_low, ang_high = self.command_ang_vel_z_range_

        forward_den = max(float(lin_x_high), self.command_zero_epsilon_)
        backward_den = max(abs(float(lin_x_low)), self.command_zero_epsilon_)
        lateral_den = max(abs(float(lin_y_low)), abs(float(lin_y_high)), self.command_zero_epsilon_)
        yaw_den = max(abs(float(ang_low)), abs(float(ang_high)), self.command_zero_epsilon_)

        amp_condition[0] = np.clip(max(vx, 0.0) / forward_den, 0.0, 1.0)
        amp_condition[1] = np.clip(max(-vx, 0.0) / backward_den, 0.0, 1.0)
        amp_condition[2] = np.clip(abs(vy) / lateral_den, 0.0, 1.0)
        amp_condition[3] = np.clip(abs(wz) / yaw_den, 0.0, 1.0)

        active_sum = float(np.sum(amp_condition[:4]))
        if active_sum > 1.0:
            amp_condition[:4] /= active_sum
        else:
            amp_condition[4] = 1.0 - active_sum

        amp_condition[5] = 0.0
        return amp_condition

    def _select_xwalk_command_mode(self, vx: float, vy: float, wz: float) -> str:
        abs_vx = abs(vx)
        abs_vy = abs(vy)
        abs_wz = abs(wz)
        yaw_deadzone = self.yaw_only_ang_vel_deadzone_
        lateral_deadzone = self.lateral_vel_deadzone_

        if abs_wz >= yaw_deadzone and abs_vx <= 0.05 and abs_vy <= 0.05:
            return "yaw_only"
        if abs_vy >= lateral_deadzone and abs_vy >= abs_vx:
            return "lateral"
        if vx > 0.0:
            return "forward"
        if vx < 0.0:
            return "backward"
        if abs_wz >= yaw_deadzone:
            return "yaw_only"
        return "idle"

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(float(value), float(low), float(high)))

    @staticmethod
    def _clip_signed_with_deadzone(value: float, low: float, high: float, deadzone: float) -> float:
        low = float(low)
        high = float(high)
        max_abs = max(abs(low), abs(high))
        min_abs = min(max(float(deadzone), 0.0), max_abs)
        sign = -1.0 if value < 0.0 else 1.0
        magnitude = float(np.clip(abs(float(value)), min_abs, max_abs))
        return float(np.clip(sign * magnitude, low, high))


    @staticmethod
    def euler_to_quaternion_scipy(roll, pitch, yaw, degrees=False):
        r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
        q_xyzw = r.as_quat()
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    @staticmethod
    def quat_rotate_inverse_numpy(q_xyzw, v):
        q_w = q_xyzw[3]
        q_v = q_xyzw[:3]
        a = v * (2.0 * q_w * q_w - 1.0)
        b = np.cross(q_v, v) * (2.0 * q_w)
        c = q_v * (2.0 * np.dot(q_v, v))
        return a - b + c
    def compute_actions(self):
        """使用ONNX模型计算动作 - 与C++版本完全一致"""
        if self.ort_session_ is None:
            return

        try:
            # Prepare input tensor
            input_data = self.observations_.reshape(1, -1).astype(np.float32)

            # ONNX inference
            input_name = self.ort_session_.get_inputs()[0].name
            outputs = self.ort_session_.run(None, {input_name: input_data})

            # Extract and clip actions exactly like C++
            output_data = outputs[0][0]
            for i in range(self.action_num_):
                self.actions_[i] = np.clip(output_data[i], -self.clip_act_, self.clip_act_)

            if self.is_first_action_:
                for i in range(self.obs_size_):
                    pass
                self.is_first_action_ = False

        except Exception as e:
            pass

    def on_exit(self):
        """退出WALKAMP状态"""
        # self._reset_infer_timing_state()
        # 关掉 obs 日志文件
        if getattr(self, "obs_log_file", None) is not None:
            try:
                self.obs_log_file.flush()
                self.obs_log_file.close()
            except Exception as e:
                pass
            self.obs_log_file = None


        # if getattr(self, "log_file_", None) is not None:
        #     self.log_file_.flush()
        #     self.log_file_.close()
        #     self.log_file_ = None
        #     logger.info(f"[FSMStateWALKAMP] log saved to {self.log_path_}")

    def check_transition(self, *args, **kwargs):
        """检查状态转换"""
        result = {}
        result['allow_transition'] = True
        if kwargs.get("target_state", None) != FSMStateName['WALKAMP']:
            walk_cmd = self.robot_data_.get_walk_cmd()
            #TODO: 可通过Imu加速度判断是否可以转换
        return result