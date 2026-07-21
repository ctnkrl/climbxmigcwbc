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

class agent_navigate_lab(agent_navigate_lab_12dof):
    def __init__(self):
        self.num_actions = 23
        self.prop_obs_history_len = 10
        self.num_prop_obs = 79

        self.dt = 0.01
        self.decimation = 1

        policy_path = osp.join(MODEL_DIR, "23dof_20260324_203333_cmd_resample_m2c_dist0d044_heading0d066.onnx")

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
        self.filter_overall = CmdFilter(low=None, high=None, rate_limit=5.0, alpha=1.0)
        self.last_timestamp = time.perf_counter()
        self.navigate_mode_switch_time = 0.
        self.last_navigate_mode = False

        self.file_logger = FileLogger(self.dt, f"{XMIGCS_ROOT_DIR}/logs/navigate", "commands")

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
            "dof_pos": np.zeros(self.num_actions, dtype=np.float32),
            "dof_vel": np.zeros(self.num_actions, dtype=np.float32),
            "angular_velocity": np.zeros(3, dtype=np.float32),
            "commands": np.zeros(3, dtype=np.float32),
            "navigate_commands": np.zeros(3, dtype=np.float32),
            "navigate_mode": True,
            "projected_gravity": np.array([0., 0., -1.], dtype=np.float32),
        }
        self.fps = 0.

if __name__ == "__main__":
    a = agent_navigate_lab()
    np.set_printoptions(formatter={"float": "{:.2f}".format})
    for i in range(100):
        logger.debug(a.inference(a.default_obs_group))
