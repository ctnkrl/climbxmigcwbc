# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np
import onnxruntime as ort
import time
import os.path as osp
from xmigcs.utils.exp_filter import expFilter
from xmigcs.policy.hbwalk.model import MODEL_DIR
from xmigcs.utils.sym_tiangong_23dof_matrix_renteng import sym_tiangong_23dof_matrix_renteng_numpy
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

class agent_mlp_23dof_amp_lab_sym_renteng:
    def __init__(self):
        self.num_actions = 23
        self.prop_obs_history_len = 10
        self.num_prop_obs = 79

        self.dt = 0.01
        self.decimation = 1

        policy_path = osp.join(MODEL_DIR, "policy_14.onnx")

        self.action_scale = 0.25

        self.clip_observation = 100.
        self.clip_action = 100
        self.obs_scale = {
            "dof_pos": 1.0,
            "dof_vel": 1.0,
            "ang_vel": 1.0,
            "lin_vel": 1.0,
            "commands": 1.0,
            "projected_gravity": 1.0,
            "action": 1.0,
        }

        providers = [
            "CUDAExecutionProvider",  # 优先使用GPU
            "CPUExecutionProvider"    # 回退到CPU
        ] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]

        # 启用线程优化配置
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # 设置计算线程数
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # 创建推理会话
        self.onnx_session = ort.InferenceSession(
            policy_path,
            providers=providers,
            sess_options=options
        )

        self.prop_obs_history = np.zeros((self.prop_obs_history_len, self.num_prop_obs))
        self.last_actions_buf = np.zeros(self.num_actions)
        self.exp_filter = expFilter(0.3)
        self.last_timestamp = time.perf_counter()
        self.inference_count = 0

        self.agent_names = [
            "hip_pitch_l_joint", "hip_pitch_r_joint",
            "waist_yaw_joint",
            "hip_roll_l_joint", "hip_roll_r_joint",
            "waist_roll_joint",
            "hip_yaw_l_joint", "hip_yaw_r_joint",
            "waist_pitch_joint",
            "knee_pitch_l_joint", "knee_pitch_r_joint",
            "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",
            "ankle_pitch_l_joint", "ankle_pitch_r_joint",
            "shoulder_roll_l_joint", "shoulder_roll_r_joint",
            "ankle_roll_l_joint", "ankle_roll_r_joint",
            "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_l_joint", "elbow_pitch_r_joint",
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
            -0.15, 0, 0, 0.3, -0.17, 0.0,  # hip pitch/roll/yaw, knee, ankle pitch/roll
            -0.15, 0, 0, 0.3, -0.17, 0.0,
            0.0, 0.0, 0.0,                   # 腰
            0.2, 0.1, 0.0, -0.5, 0.0, 0.0, 0.0,  # shoulder pitch/roll/yaw, elbow pitch/yaw, wrist pitch/roll
            0.2, -0.1, 0.0, -0.5, 0.0, 0.0, 0.0,  # 右臂
        ])[self.env_to_agent]

        self.default_obs_group = {
            "dof_pos": np.zeros(self.num_actions, dtype=float),
            "dof_vel": np.zeros(self.num_actions, dtype=float),
            "angular_velocity": np.zeros(3, dtype=float),
            "commands": np.zeros(3, dtype=float),
            "projected_gravity": np.array([0., 0., -1.], dtype=float),
        }

        self.symmetry_module = sym_tiangong_23dof_matrix_renteng_numpy()
        self.symmetry_inference = True
        self.start_time = 0.

    def bootstrap(self):
        "预热用"
        self.inference(self.default_obs_group)

    def build_observations_one_step(self, obs_group):
        for key, val in obs_group.items():
            if isinstance(val, np.ndarray):
                obs_group[key] = val.clip(-self.clip_observation, self.clip_observation)

        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        obs_projected_gravity = obs_group["projected_gravity"] * self.obs_scale["projected_gravity"]
        obs_commands = obs_group["commands"] * self.obs_scale["commands"]

        obs_dof_pos = (obs_group["dof_pos"] - self.default_dof_pos) * self.obs_scale["dof_pos"]  # 注意要减掉default
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_action = self.last_actions_buf * self.obs_scale["action"]

        prop_obs = np.concatenate([
            obs_base_ang_vel,  # 3
            obs_projected_gravity,  # 3
            obs_commands,  # 3
            obs_dof_pos,  # 23
            obs_dof_vel,  # 23
            obs_last_action,  # 23
            [0.],
        ])  # 78

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

        commands = obs_group["commands"]
        cmd_small = np.linalg.norm(commands) < 0.1
        if not cmd_small:
            self.start_time += self.dt
        else:
            self.start_time = 0.

        if self.symmetry_inference:
            if (self.inference_count % 2 == 0) and self.start_time > 2.0:
                input_feed = {
                    "obs": obs_f,
                }
                actions = np.squeeze(self.onnx_session.run(["actions"], input_feed))
            else:
                # mirror
                obs_m = self.symmetry_module.mirror_obs_history(obs_f, self.symmetry_module.policy_history, self.symmetry_module.mirror_policy_obs)
                input_feed = {
                    "obs": obs_m,
                }
                actions_m = np.squeeze(self.onnx_session.run(["actions"], input_feed))
                actions = self.symmetry_module.mirror_action(actions_m)
        else:
            input_feed = {
                "obs": obs_f,
            }
            actions = np.squeeze(self.onnx_session.run(["actions"], input_feed))

        actions = np.clip(actions, -self.clip_action, self.clip_action)

        self.last_actions_buf = actions

        dof_pos_target_urdf = actions * self.action_scale + self.default_dof_pos

        # dof_pos_target_urdf = self.exp_filter.filter(dof_pos_target_urdf)
        self.inference_count += 1
        return dof_pos_target_urdf

    def reset(self, first_obs_group=None):
        self.last_actions_buf = np.zeros(self.num_actions)
        if first_obs_group is None:
            first_obs_group = self.default_obs_group
        norminal_obs = self.build_observations_one_step(first_obs_group)
        self.prop_obs_history[:, :] = norminal_obs[None, :]
        self.exp_filter.reset()
        self.last_run_time = time.perf_counter()
        self.inference_count = 0
        self.start_time = 0.

if __name__ == "__main__":
    a = agent_mlp_23dof_amp_lab_sym_renteng()
    np.set_printoptions(formatter={"float": "{:.2f}".format})
    for i in range(100):
        logger.debug(a.inference(a.default_obs_group))
