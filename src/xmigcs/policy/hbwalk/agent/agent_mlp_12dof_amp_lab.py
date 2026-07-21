# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np
import onnxruntime as ort
import time
import os.path as osp
from xmigcs.policy.hbwalk.agent import agent_mlp_23dof_amp_lab
from xmigcs.utils.exp_filter import expFilter
from xmigcs.policy.hbwalk.model import MODEL_DIR
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

class agent_mlp_12dof_amp_lab(agent_mlp_23dof_amp_lab):
    def __init__(self):
        self.num_actions = 12
        self.prop_obs_history_len = 10
        self.num_prop_obs = 45

        self.dt = 0.01
        self.decimation = 1

        # policy_path = osp.join(MODEL_DIR, "2026-01-14_16-05-44_resume_supervise_pitch_25Nm_add_feet_align_m2_model_41250.onnx")  # 普通拟人半身
        # policy_path = osp.join(MODEL_DIR, "20260123_221205_flat_large_plus_pitch_torque_feet_height_penalty.onnx")  # 搬箱子用 踏脚重
        # policy_path = osp.join(MODEL_DIR, "Dex_walk_12dof_amp_sym_20260624_123543_baseline_new_urdf_model_99998.onnx")
        policy_path = osp.join(MODEL_DIR, "Dex_walk_12dof_with_arms_amp_sym_20260626_155156_add_arm_rand_model_104000.onnx")

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
        options.intra_op_num_threads = 1  # 设置计算线程数
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

        self.prop_obs_history = np.zeros((self.prop_obs_history_len, self.num_prop_obs))
        self.last_actions_buf = np.zeros(self.num_actions)
        self.exp_filter = expFilter(0.3)
        self.last_timestamp = time.perf_counter()

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
            -0.179, 0, 0, 0.323, -0.144, 0.0,  # hip pitch/roll/yaw, knee, ankle pitch/roll
            -0.179, 0, 0, 0.323, -0.144, 0.0,
        ])[self.env_to_agent]

        self.default_obs_group = {
            "dof_pos": np.zeros(self.num_actions, dtype=float),
            "dof_vel": np.zeros(self.num_actions, dtype=float),
            "angular_velocity": np.zeros(3, dtype=float),
            "commands": np.zeros(3, dtype=float),
            "projected_gravity": np.array([0., 0., -1.], dtype=float),
        }


if __name__ == "__main__":
    a = agent_mlp_12dof_amp_lab()
    np.set_printoptions(formatter={"float": "{:.2f}".format})
    for i in range(100):
        logger.debug(a.inference(a.default_obs_group))
