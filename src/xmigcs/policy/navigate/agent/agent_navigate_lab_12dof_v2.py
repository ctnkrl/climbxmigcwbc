# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np
import onnxruntime as ort
import time
import os.path as osp
from xmigcs.utils.exp_filter import expFilter
from xmigcs.utils.file_logger import FileLogger
from xmigcs.policy.navigate.model import MODEL_DIR
from xmigcs import XMIGCS_ROOT_DIR
from xmigcs.utils.cmd_filter import CmdFilter
from xmigcs.policy.navigate.agent import agent_navigate_lab_12dof
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

class agent_navigate_lab_12dof_v2(agent_navigate_lab_12dof):
    "有ref_ang_vel和endless_flag的版本"
    def __init__(self):
        self.num_actions = 12
        self.prop_obs_history_len = 10
        self.num_prop_obs = 48

        self.dt = 0.01
        self.decimation = 1

        policy_path = osp.join(MODEL_DIR, "Dex_navigate_12dof_amp_sym_20260514_140415_model_64000.onnx")

        self.action_scale = 0.25

        self.clip_observation = 100.
        self.clip_action = 100
        self.obs_scale = {
            "dof_pos": 1.0,
            "dof_vel": 0.25,
            "ang_vel": 0.25,
            "lin_vel": 1.0,
            "commands": 1.0,
            "projected_gravity": 1.0,
            "action": 0.25,
        }

        providers = [
            "CUDAExecutionProvider",  # 优先使用GPU
            "CPUExecutionProvider"    # 回退到CPU
        ] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]

        # 启用线程优化配置
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # 设置计算图中的运算符内部并行线程数
        options.inter_op_num_threads = 1  # 设置多个运算符之间的并行线程数（如果模型有多个分支）
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # 创建推理会话
        self.onnx_session = ort.InferenceSession(
            policy_path,
            providers=providers,
            sess_options=options
        )
        logger.info(f"{__class__.__name__} loaded")
        logger.info(f"inputs: {[n.name for n in self.onnx_session.get_inputs()]}")
        logger.info(f"outputs: {[n.name for n in self.onnx_session.get_outputs()]}")

        self.prop_obs_history = np.zeros((self.prop_obs_history_len, self.num_prop_obs), dtype=np.float32)
        self.last_actions_buf = np.zeros(self.num_actions, dtype=np.float32)
        self.exp_filter = expFilter(tau=20)
        self.exp_filter_navigate = expFilter(tau=20)
        self.filter_walk = CmdFilter(low=None, high=None, rate_limit=5.0, alpha=0.5)
        self.filter_rotate = CmdFilter(low=None, high=None, rate_limit=10.0, alpha=1.0)  # 不用滤波
        self.filter_overall = CmdFilter(low=None, high=None, rate_limit=1.0, alpha=1.0)
        self.last_timestamp = time.perf_counter()
        self.navigate_mode_switch_time = 0.
        self.last_navigate_mode = False

        self.file_logger = FileLogger(self.dt, f"{XMIGCS_ROOT_DIR}/logs/navigate", "commands")

        self.agent_names = [
            "hip_pitch_l_joint", "hip_pitch_r_joint",
            "hip_roll_l_joint", "hip_roll_r_joint",
            "hip_yaw_l_joint", "hip_yaw_r_joint",
            "knee_pitch_l_joint", "knee_pitch_r_joint",
            "ankle_pitch_l_joint", "ankle_pitch_r_joint",
            "ankle_roll_l_joint", "ankle_roll_r_joint",
        ]
        self.env_names = [
            "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint", "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint", "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint"
        ]
        self.env_to_agent = [self.env_names.index(name) for name in self.agent_names]
        self.default_dof_pos = np.array([
            -0.2, 0, 0, 0.42, -0.23, 0.0,  # hip pitch/roll/yaw, knee, ankle pitch/roll
            -0.2, 0, 0, 0.42, -0.23, 0.0,
        ])[self.env_to_agent]

        self.default_obs_group = {
            "dof_pos": np.zeros(self.num_actions, dtype=np.float32),
            "dof_vel": np.zeros(self.num_actions, dtype=np.float32),
            "angular_velocity": np.zeros(3, dtype=np.float32),
            "commands": np.zeros(3, dtype=np.float32),
            "navigate_commands": np.zeros(3, dtype=np.float32),
            "navigate_mode": True,
            "projected_gravity": np.array([0., 0., -1.], dtype=np.float32),
        }
        self.fps = 0.

    def build_observations_one_step(self, obs_group):
        for key, val in obs_group.items():
            if key == "timestamp": continue  # 时间戳不要clip
            if isinstance(val, np.ndarray):
                obs_group[key] = val.clip(-self.clip_observation, self.clip_observation)

        walk_cmd = obs_group["commands"]
        navigate_cmd = obs_group["navigate_commands"]
        navigate_mode = obs_group["navigate_mode"]
        navigate_cmd_raw = navigate_cmd.copy()

        # navigate switch
        if self.last_navigate_mode != navigate_mode:
            self.navigate_mode_switch_time = 0.
            self.last_navigate_mode = navigate_mode
        else:
            self.navigate_mode_switch_time += self.dt

        if not navigate_mode:
            # 走路模式
            # 只响应walk_cmd
            # 根据lin_vel_x, lin_vel_y计算单位圆上的bcs_x, bcs_y
            # walk_cmd_xy_f = self.filter_walk.filter(walk_cmd[:2], self.dt)
            walk_cmd_xy_f = walk_cmd[:2]
            walk_speed_ref = np.linalg.norm(walk_cmd_xy_f[:2])
            if walk_speed_ref < 0.05:
                walk_speed_ref *= 0.
                bcs_pos_xy = np.zeros(2, dtype=np.float32)
            else:
                vel_vector_angle = np.arctan2(walk_cmd[1], walk_cmd[0])
                bcs_pos_xy = np.array([np.cos(vel_vector_angle), np.sin(vel_vector_angle)], dtype=np.float32)
                dist = (walk_speed_ref * 3.0).clip(max=1.0)
                bcs_pos_xy *= dist

            rotate_cmd_f = self.filter_rotate.filter(walk_cmd[2], self.dt)
            if np.abs(rotate_cmd_f) < 0.1:
                rotate_speed_ref = np.zeros_like(rotate_cmd_f)
                heading_cmd = np.zeros_like(rotate_cmd_f)
            else:
                rotate_speed_ref = np.abs(rotate_cmd_f)
                heading_cmd = rotate_cmd_f * 3.0

            heading_cmd = heading_cmd.clip(min=-1.0, max=1.0)
            endless_flag = 0.0

            self.filter_overall.last_value = walk_speed_ref
        else:
            # 导航模式
            # 只响应navigate_cmd
            # vel_ref由lin_vel_x, lin_vel_y计算得到
            # navigate_cmd = self.exp_filter_navigate.filter(navigate_cmd)
            walk_speed_ref = np.linalg.norm(walk_cmd[:2]).clip(max=1.0)
            # walk_speed_ref = np.array(0.8)
            bcs_pos_xy = navigate_cmd[:2]
            # 如果target_distance超出threshold，就clip到threshold，方向不变
            target_distance = np.linalg.norm(bcs_pos_xy)

            if self.navigate_mode_switch_time < 1.0:
                walk_speed_ref = self.filter_overall.filter(walk_speed_ref, self.dt)
            threshold = (walk_speed_ref * 3.0).clip(max=1.0)
            if target_distance > threshold:
                scale = target_distance / threshold
                bcs_pos_xy /= scale
            endless_flag = 0.0

            rotate_speed_ref = np.abs(walk_cmd[2])
            # rotate_speed_ref = np.array(1.0)
            heading_cmd = navigate_cmd[2]
            heading_cmd = heading_cmd.clip(min=-1.0, max=1.0)  # NOTE: planning走路切旋转有heading指令突变, 多限制一下

            # 小于临界距离的时候speed_ref=0
            near_target = target_distance < walk_speed_ref * 0.5
            if near_target:
                walk_speed_ref *= 0.
            near_target_rotate = np.abs(heading_cmd) < rotate_speed_ref * 0.5
            if near_target_rotate:
                rotate_speed_ref *= 0.

        obs_commands = np.concatenate((
            walk_speed_ref[None],
            bcs_pos_xy,
            rotate_speed_ref[None],
            heading_cmd[None],
            np.array(endless_flag)[None],
        ), dtype=np.float32) * self.obs_scale["commands"]

        # np.set_printoptions(formatter={"float": "{:.3f}".format})
        logger.debug("agent obs_command: %s", obs_commands)

        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        obs_projected_gravity = obs_group["projected_gravity"] * self.obs_scale["projected_gravity"]

        obs_dof_pos = (obs_group["dof_pos"] - self.default_dof_pos) * self.obs_scale["dof_pos"]  # 注意要减掉default
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_action = self.last_actions_buf * self.obs_scale["action"]

        self.file_logger.data_log(
            [self.fps]
            + walk_cmd.tolist()
            + navigate_cmd_raw.tolist()
            + [navigate_mode] + obs_commands.tolist()
            + ["switch"] + [self.navigate_mode_switch_time]
            + ["filter"] + [self.filter_overall.last_value]
        )
        prop_obs = np.concatenate([
            obs_base_ang_vel,  # 3
            obs_projected_gravity,  # 3
            obs_commands,  # 4
            obs_dof_pos,  # 23
            obs_dof_vel,  # 23
            obs_last_action,  # 23
        ])  # 78

        prop_obs = np.clip(prop_obs, -self.clip_observation, self.clip_observation)
        return prop_obs


if __name__ == "__main__":
    a = agent_navigate_lab_12dof_v2()
    np.set_printoptions(formatter={"float": "{:.2f}".format})
    for i in range(100):
        logger.debug(a.inference(a.default_obs_group))
