# Copyright (c) 2026 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np
import onnxruntime as ort
import time
import os.path as osp
from xmigcs.utils.exp_filter import expFilter
from xmigcs.utils.file_logger import FileLogger
from xmigcs.policy.stand.model import MODEL_DIR
from xmigcs import XMIGCS_ROOT_DIR
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

class agent_stand:
    def __init__(self):
        self.num_actions = 14
        self.prop_obs_history_len = 10
        self.num_prop_obs = 72

        self.dt = 0.01
        self.decimation = 1

        policy_path = osp.join(MODEL_DIR, "policy_2025-12-31_10-29-01_515000_pd.onnx")

        self.action_scale = 0.25

        self.clip_observation = 100.
        self.clip_action = 100
        self.obs_scale = {
            "dof_pos": 1.0,
            "dof_vel": 1.0,
            "ang_vel": 1.0,
            "commands": 1.0,
            "projected_gravity": 1.0,
            "action": 1.0,  # obs中没有乘action_scale缩放
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
        self.last_timestamp = time.perf_counter()

        self.file_logger = FileLogger(self.dt, f"{XMIGCS_ROOT_DIR}/logs/stand", "commands")

        self.agent_obs_names = [
            "hip_pitch_l_joint", "hip_pitch_r_joint", "waist_yaw_joint",
            "hip_roll_l_joint", "hip_roll_r_joint", "waist_roll_joint",
            "hip_yaw_l_joint", "hip_yaw_r_joint", "waist_pitch_joint",
            "knee_pitch_l_joint", "knee_pitch_r_joint",
            "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",
            "ankle_pitch_l_joint", "ankle_pitch_r_joint",
            "shoulder_roll_l_joint", "shoulder_roll_r_joint",
            "ankle_roll_l_joint", "ankle_roll_r_joint",
            "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_l_joint", "elbow_pitch_r_joint",
        ]
        self.agent_control_names = [
            "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "waist_roll_joint", "waist_pitch_joint",
        ]
        self.env_names = [
            "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint", "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint", "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint"
        ]
        self.env_to_agent_obs = [self.env_names.index(name) for name in self.agent_obs_names]
        self.env_to_agent_control = [self.env_names.index(name) for name in self.agent_control_names]

        self.default_dof_pos_dict = {
            "hip_yaw_l_joint": -0.0,
            "hip_roll_l_joint": 0.0,
            "hip_pitch_l_joint": -0.4,
            "knee_pitch_l_joint": 0.8,
            "ankle_pitch_l_joint": -0.4,
            "ankle_roll_l_joint": 0.0,
            "hip_yaw_r_joint": -0.0,
            "hip_roll_r_joint": 0.0,
            "hip_pitch_r_joint": -0.4,
            "knee_pitch_r_joint": 0.8,
            "ankle_pitch_r_joint": -0.4,
            "ankle_roll_r_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "shoulder_pitch_l_joint": 0.35,
            "shoulder_roll_l_joint": 0.18,
            "shoulder_yaw_l_joint": 0.0,
            "elbow_pitch_l_joint": -0.87,
            "shoulder_pitch_r_joint": 0.35,
            "shoulder_roll_r_joint": -0.18,
            "shoulder_yaw_r_joint": 0.0,
            "elbow_pitch_r_joint": -0.87,
        }
        self.default_dof_pos_obs = np.array([self.default_dof_pos_dict[name] for name in self.agent_obs_names])
        self.default_dof_pos_control = np.array([self.default_dof_pos_dict[name] for name in self.agent_control_names])

        self.default_obs_group = {
            "dof_pos": np.zeros(len(self.agent_obs_names), dtype=np.float32),
            "dof_vel": np.zeros(len(self.agent_obs_names), dtype=np.float32),
            "angular_velocity": np.zeros(3, dtype=np.float32),
            "commands": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.965], dtype=np.float32),
            "projected_gravity": np.array([0., 0., -1.], dtype=np.float32),
        }
        self.fps = 0.

    def bootstrap(self):
        "预热用"
        self.inference(self.default_obs_group)

    def build_observations_one_step(self, obs_group):
        for key, val in obs_group.items():
            if isinstance(val, np.ndarray):
                obs_group[key] = val.clip(-self.clip_observation, self.clip_observation)

        obs_command = obs_group["commands"] * self.obs_scale["commands"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        obs_projected_gravity = obs_group["projected_gravity"] * self.obs_scale["projected_gravity"]

        obs_dof_pos = (obs_group["dof_pos"] - self.default_dof_pos_obs) * self.obs_scale["dof_pos"]
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_action = self.last_actions_buf * self.obs_scale["action"]

        self.file_logger.data_log([self.fps] + obs_command.tolist())
        prop_obs = np.concatenate([
            obs_base_ang_vel,  # 3
            obs_projected_gravity,  # 3
            obs_command,  # 6
            obs_dof_pos,  # 23
            obs_dof_vel,  # 23
            obs_last_action,  # 14
        ])  # 72

        prop_obs = np.clip(prop_obs, -self.clip_observation, self.clip_observation)
        return prop_obs

    def build_observations(self, obs_group):
        prop_obs = self.build_observations_one_step(obs_group)

        # [history, obs_length]
        # 历史观测更新：向左滚动，新观测放在最后
        self.prop_obs_history = np.roll(self.prop_obs_history, shift=-1, axis=0)
        self.prop_obs_history[-1, :] = prop_obs

        return self.prop_obs_history.copy()

    def inference(self, obs_group):
        obs = self.build_observations(obs_group)
        obs_f = obs.flatten()[None, :].astype(np.float32)

        input_feed = {
            "obs": obs_f,
        }
        actions = np.squeeze(self.onnx_session.run(["actions"], input_feed))

        actions = np.clip(actions, -self.clip_action, self.clip_action)

        self.last_actions_buf = actions

        dof_pos_target_urdf = actions * self.action_scale + self.default_dof_pos_control

        # dof_pos_target_urdf = self.exp_filter.filter(dof_pos_target_urdf)
        return dof_pos_target_urdf

    def reset(self, first_obs_group=None):
        self.last_actions_buf.fill(0.0)
        if first_obs_group is None:
            first_obs_group = self.default_obs_group
        norminal_obs = self.build_observations_one_step(first_obs_group)
        self.prop_obs_history[:, :] = norminal_obs[None, :]
        self.exp_filter.reset()
        self.last_run_time = time.perf_counter()

if __name__ == "__main__":
    a = agent_stand()
    np.set_printoptions(formatter={"float": "{:.2f}".format})
    for i in range(100):
        logger.debug(a.inference(a.default_obs_group))
