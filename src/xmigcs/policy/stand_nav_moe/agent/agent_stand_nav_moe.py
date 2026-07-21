# Copyright (c) 2026 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np
import onnxruntime as ort
import time
import os.path as osp
from xmigcs.utils.exp_filter import expFilter
from xmigcs.utils.file_logger import FileLogger
from xmigcs.policy.stand_nav_moe.model import MODEL_DIR
from xmigcs import XMIGCS_ROOT_DIR
import xmigcs.policy.stand_nav_moe.agent.joint_info as joint_info
# from tabulate import tabulate
from xmigcs.utils.cmd_filter import CmdFilter
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

class agent_stand_nav_moe:
    def __init__(self):
        self.num_actions = 15

        self.dt = 0.01
        self.decimation = 1

        policy_path = osp.join(MODEL_DIR, "Dex_stand_navigate_moe_add_blank_supervise_20260612_164943_w0d5_cmd_all_resample_6s_model_2000.onnx")

        self.action_scale = 0.25

        self.clip_observation = 100.
        self.clip_action = 100
        self.obs_scale_stand = {
            "dof_pos": 1.0,
            "dof_vel": 1.0,
            "ang_vel": 1.0,
            "commands": 1.0,
            "projected_gravity": 1.0,
            "action": 1.0,  # stand_obs中没有乘action_scale缩放
        }
        self.obs_scale_navigate = {
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

        self.prop_obs_history_len = 10
        self.num_prop_obs_stand = 73
        self.num_prop_obs_navigate = 57
        self.num_prop_obs_gating = 80
        self.prop_obs_history_stand = np.zeros((self.prop_obs_history_len, self.num_prop_obs_stand), dtype=np.float32)
        self.prop_obs_history_navigate = np.zeros((self.prop_obs_history_len, self.num_prop_obs_navigate), dtype=np.float32)
        self.prop_obs_history_gating = np.zeros((self.prop_obs_history_len, self.num_prop_obs_gating), dtype=np.float32)

        self.last_actions_buf = np.zeros(self.num_actions, dtype=np.float32)
        self.exp_filter = expFilter(tau=20)
        self.exp_filter_navigate = expFilter(tau=20)
        self.filter_walk = CmdFilter(low=None, high=None, rate_limit=5.0, alpha=0.5)
        self.filter_rotate = CmdFilter(low=None, high=None, rate_limit=10.0, alpha=1.0)  # 不用滤波
        self.filter_overall = CmdFilter(low=None, high=None, rate_limit=1.0, alpha=0.6)
        self.filter_overall_rot = CmdFilter(low=None, high=None, rate_limit=1.0, alpha=0.6)
        self.last_timestamp = time.perf_counter()
        self.navigate_mode_switch_time = 0.
        self.last_navigate_mode = False

        self.file_logger_stand = FileLogger(self.dt, f"{XMIGCS_ROOT_DIR}/logs/stand_nav_moe", "commands_stand")
        self.file_logger_navigate = FileLogger(self.dt, f"{XMIGCS_ROOT_DIR}/logs/stand_nav_moe", "commands_navigate")
        self.file_logger_transition = FileLogger(self.dt, f"{XMIGCS_ROOT_DIR}/logs/stand_nav_moe", "transition")

        self.default_obs_group = {
            "current_active_cmd": 1,
            "dof_pos": np.zeros(29, dtype=np.float32),
            "dof_vel": np.zeros(29, dtype=np.float32),
            "angular_velocity": np.zeros(3, dtype=np.float32),
            "stand_commands": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.965], dtype=np.float32),
            "walk_commands": np.zeros(3, dtype=np.float32),
            "navigate_commands": np.zeros(3, dtype=np.float32),
            "navigate_mode": True,
            "timestamp": np.zeros(1, dtype=np.float32),
            "projected_gravity": np.array([0., 0., -1.], dtype=np.float32),
        }
        self.fps = 0.

        self.env_names = joint_info.env_names
        self.default_dof_pos_obs_stand = joint_info.default_dof_pos_obs_stand
        self.default_dof_pos_obs_navigate = joint_info.default_dof_pos_obs_navigate
        self.default_dof_pos_control_stand = joint_info.default_dof_pos_control_stand
        self.default_dof_pos_control_navigate = joint_info.default_dof_pos_control_navigate
        self.env_to_agent_obs_stand = joint_info.env_to_agent_obs_stand
        self.env_to_agent_obs_navigate = joint_info.env_to_agent_obs_navigate
        self.env_to_agent_control_stand = joint_info.env_to_agent_control_stand
        self.env_to_agent_control_navigate = joint_info.env_to_agent_control_navigate
        self.action_indices_nav_in_stand = [joint_info.agent_control_names_stand.index(name) for name in joint_info.agent_control_names_navigate]

        # action转换
        stand_default_pos = np.array(joint_info.default_dof_pos_control_stand, dtype=np.float32)
        self.d_s = stand_default_pos[self.action_indices_nav_in_stand]
        self.d_v = np.array(joint_info.default_dof_pos_control_navigate, dtype=np.float32)
        self.c_s = 0.25  # action_scale_stand
        self.c_v = 0.25  # action_scale_navigate

        self.time_since_switch = np.zeros(1)
        self.last_active_command = None
        self.transition_window = 1.0  # secs
        self.transition_percent = 0.

        np.set_printoptions(formatter={"float": "{:.3f}".format})

    def action_navigate_to_stand(self, a_v: np.ndarray) -> np.ndarray:
        "navigate和stand的default_pos不同, 转换为等效action"
        # processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos
        return self.c_v / self.c_s * a_v + 1.0 / self.c_s * (self.d_v - self.d_s)

    def action_stand_to_navigate(self, a_s: np.ndarray) -> np.ndarray:
        "default_pos不同, 转换为等效action"
        # processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos
        return self.c_s / self.c_v * (a_s - 1.0 / self.c_s * (self.d_v - self.d_s))

    def bootstrap(self):
        "预热用"
        self.inference(self.default_obs_group)

    def build_observations_one_step(self, obs_group):
        for key, val in obs_group.items():
            if key == "timestamp": continue  # 时间戳不要clip
            if isinstance(val, np.ndarray):
                obs_group[key] = val.clip(-self.clip_observation, self.clip_observation)

        aligned_navigate_commands = self.get_aligned_navigate_cmd(
            obs_group["walk_commands"].copy(),
            obs_group["navigate_commands"].copy(),
            obs_group["navigate_mode"],
        )
        obs_group["aligned_navigate_commands"] = aligned_navigate_commands
        self.transition_percent = self.get_transition_percent(obs_group)
        obs_group["transition_percent"] = self.transition_percent

        self.file_logger_stand.data_log([self.fps] + obs_group["stand_commands"].tolist())
        self.file_logger_navigate.data_log(
            [self.fps]
            + obs_group["walk_commands"].tolist()
            + obs_group["navigate_commands"].tolist()
            + [obs_group["navigate_mode"]]
            + aligned_navigate_commands.tolist()
        )
        self.file_logger_transition.data_log(self.transition_percent.tolist())
        # logger.debug(f"stand_command: {obs_group['stand_commands']}")
        # logger.debug(f"walk_commands: {obs_group['walk_commands']}")
        # logger.debug(f"navigate_commands_raw: {obs_group['navigate_commands']}")
        logger.debug(f"aligned_navigate_command: {obs_group['aligned_navigate_commands']}")
        current_active_cmd = obs_group["current_active_cmd"]
        # if current_active_cmd == 0:
        #     logger.debug("current_state: stand")
        # if current_active_cmd == 1:
        #     logger.debug("current_state: navigate")
        logger.debug(f"transition_percent: {self.transition_percent}")

        stand_obs = self.build_observations_one_step_stand(obs_group)
        navigate_obs = self.build_observations_one_step_navigate(obs_group)
        gating_obs = self.build_observations_one_step_gating(obs_group)
        return stand_obs, navigate_obs, gating_obs

    def get_transition_percent(self, obs_group):
        """
        stand(cmd=0), p=0 -> 1. 稳定后=1.
        navigate(cmd=1), p=1 -> 0. 稳定后=0.
        """
        current_active_cmd = obs_group["current_active_cmd"]
        if self.last_active_command != current_active_cmd:
            self.last_active_command = current_active_cmd
            self.time_since_switch[:] = 0.
        else:
            # logger.debug(f"current_time: {obs_group['timestamp']}")
            # logger.debug(f"last_timestamp: {self.last_timestamp}")
            self.time_since_switch += obs_group["timestamp"] - self.last_timestamp
            self.last_timestamp = obs_group["timestamp"]
        percent = self.time_since_switch / self.transition_window
        percent = percent.clip(min=0.0, max=1.0)
        percent_inv = 1.0 - percent
        if current_active_cmd == 0:
            p = percent
        else:
            p = percent_inv

        # logger.debug(f"last_timestamp: {self.last_timestamp}")
        # logger.debug(f"last_command: {self.last_active_command}")
        # logger.debug(f"current_command: {current_active_cmd}")
        # logger.debug(f"time_since_switch: {self.time_since_switch}")
        # logger.debug(f"percent: {percent}")
        # logger.debug(f"percent_inv: {percent_inv}")
        # logger.debug(f"p: {p}")
        return p

    def get_aligned_navigate_cmd(self, walk_cmd, navigate_cmd, navigate_mode):
        "获得walk/navigate cmd的统一表示"
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
            self.filter_overall_rot.last_value = rotate_speed_ref
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

            # if self.navigate_mode_switch_time < 1.0:
            walk_speed_ref = self.filter_overall.filter(walk_speed_ref, self.dt)
            threshold = (walk_speed_ref * 3.0).clip(max=1.0, min=0.3)
            if target_distance > threshold:
                scale = target_distance / threshold
                bcs_pos_xy /= scale
            endless_flag = 0.0

            rotate_speed_ref = np.abs(walk_cmd[2])
            rotate_speed_ref = self.filter_overall_rot.filter(rotate_speed_ref, self.dt)
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

        aligned_navigate_commands = np.concatenate((
            walk_speed_ref[None],
            bcs_pos_xy,
            rotate_speed_ref[None],
            heading_cmd[None],
            np.array(endless_flag)[None],
        ), dtype=np.float32)
        return aligned_navigate_commands

    def build_observations_one_step_stand(self, obs_group):
        obs_commands = obs_group["stand_commands"] * self.obs_scale_stand["commands"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale_stand["ang_vel"]
        obs_projected_gravity = obs_group["projected_gravity"] * self.obs_scale_stand["projected_gravity"]

        obs_dof_pos = (obs_group["dof_pos"][self.env_to_agent_obs_stand] - self.default_dof_pos_obs_stand) * self.obs_scale_stand["dof_pos"]
        obs_dof_vel = obs_group["dof_vel"][self.env_to_agent_obs_stand] * self.obs_scale_stand["dof_vel"]
        obs_last_action = self.last_actions_buf * self.obs_scale_stand["action"]

        prop_obs = np.concatenate([
            obs_base_ang_vel,  # 3
            obs_projected_gravity,  # 3
            obs_commands,  # 6
            obs_dof_pos,  # 23
            obs_dof_vel,  # 23
            obs_last_action,  # 15
        ])  # 73

        prop_obs = np.clip(prop_obs, -self.clip_observation, self.clip_observation)
        return prop_obs

    def build_observations_one_step_navigate(self, obs_group):
        obs_commands = obs_group["aligned_navigate_commands"] * self.obs_scale_navigate["commands"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale_navigate["ang_vel"]
        obs_projected_gravity = obs_group["projected_gravity"] * self.obs_scale_navigate["projected_gravity"]

        obs_dof_pos = (obs_group["dof_pos"][self.env_to_agent_obs_navigate] - self.default_dof_pos_obs_navigate) * self.obs_scale_navigate["dof_pos"]  # 注意要减掉default
        obs_dof_vel = obs_group["dof_vel"][self.env_to_agent_obs_navigate] * self.obs_scale_navigate["dof_vel"]

        last_actions_converted = self.action_stand_to_navigate(self.last_actions_buf[self.action_indices_nav_in_stand])
        obs_last_action = last_actions_converted * self.obs_scale_navigate["action"]

        prop_obs = np.concatenate([
            obs_base_ang_vel,  # 3
            obs_projected_gravity,  # 3
            obs_commands,  # 6
            obs_dof_pos,  # 15
            obs_dof_vel,  # 15
            obs_last_action,  # 15
        ])  # 57

        prop_obs = np.clip(prop_obs, -self.clip_observation, self.clip_observation)
        return prop_obs

    def build_observations_one_step_gating(self, obs_group):
        obs_commands_navigate = obs_group["aligned_navigate_commands"] * self.obs_scale_navigate["commands"]
        obs_commands_stand = obs_group["stand_commands"] * self.obs_scale_navigate["commands"]  # FIXME: gating_obs中训练中用的就是navigate_scale 后续在训练端修改了再修改这里
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale_navigate["ang_vel"]
        obs_projected_gravity = obs_group["projected_gravity"] * self.obs_scale_navigate["projected_gravity"]

        obs_dof_pos = (obs_group["dof_pos"][self.env_to_agent_obs_stand] - self.default_dof_pos_obs_stand) * self.obs_scale_navigate["dof_pos"]  # 注意要减掉default
        obs_dof_vel = obs_group["dof_vel"][self.env_to_agent_obs_stand] * self.obs_scale_navigate["dof_vel"]

        obs_last_action = self.last_actions_buf * self.obs_scale_navigate["action"]
        prop_obs = np.concatenate([
            obs_base_ang_vel,  # 3
            obs_projected_gravity,  # 3
            obs_commands_stand,  # 6
            obs_commands_navigate,  # 6
            obs_dof_pos,  # 23
            obs_dof_vel,  # 23
            obs_last_action,  # 15
            obs_group["transition_percent"],  # 1
        ])  # 80

        prop_obs = np.clip(prop_obs, -self.clip_observation, self.clip_observation)
        return prop_obs

    def build_observations(self, obs_group):
        stand_obs, navigate_obs, gating_obs = self.build_observations_one_step(obs_group)

        # [history, obs_length]
        # 历史观测更新：向左滚动，新观测放在最后
        self.prop_obs_history_stand = np.roll(self.prop_obs_history_stand, shift=-1, axis=0)
        self.prop_obs_history_stand[-1, :] = stand_obs
        self.prop_obs_history_navigate = np.roll(self.prop_obs_history_navigate, shift=-1, axis=0)
        self.prop_obs_history_navigate[-1, :] = navigate_obs
        self.prop_obs_history_gating = np.roll(self.prop_obs_history_gating, shift=-1, axis=0)
        self.prop_obs_history_gating[-1, :] = gating_obs

        return self.prop_obs_history_stand.copy(), self.prop_obs_history_navigate.copy(), self.prop_obs_history_gating.copy()

    def inference(self, obs_group):
        obs_stand, obs_navigate, obs_gating = self.build_observations(obs_group)
        obs_stand_f = obs_stand.flatten()[None, :].astype(np.float32)
        obs_navigate_f = obs_navigate.flatten()[None, :].astype(np.float32)
        obs_gating_f = obs_gating.flatten()[None, :].astype(np.float32)

        input_feed = {
            "obs_mlp_stand": obs_stand_f,
            "obs_mlp_navigate": obs_navigate_f,
            "obs_moe_gating": obs_gating_f,
        }

        moe_actions, gating_weights, expert_outputs = self.onnx_session.run(
            ["actions", "gating_weights", "expert_outputs"], input_feed
        )
        stand_actions = expert_outputs[..., 0]
        navigate_actions = expert_outputs[..., 1]
        blank_actions = expert_outputs[..., 2]

        # logger.debug(f"stand_actions: {np.squeeze(stand_actions)}")
        # logger.debug(f"navigate_actions: {np.squeeze(navigate_actions)}")
        # logger.debug(f"blank_actions: {np.squeeze(blank_actions)}")
        # logger.debug(f"gating_weight: {np.squeeze(gating_weights)}")
        expert_names = ["stand", "navigate", "blank"]
        weights = np.squeeze(gating_weights)
        # table = []
        # for joint, weight in zip(joint_info.agent_control_names_stand, weights):
        #     table.append([joint] + weight.tolist())
        # logger.debug(tabulate(table, headers=["joint_name"] + expert_names, tablefmt="simple", floatfmt=".4f", numalign="right", stralign="left"))

        # if self.time_since_switch > self.transition_window + 1.0:
        #     # 切换窗口外强制gating输出one-hot，保证单个expert输出
        #     gating_weights = np.zeros_like(gating_weights)
        #     gating_weights[..., self.last_active_command] = 1.0
        #     # 手动混合
        #     moe_actions = np.sum(expert_outputs * gating_weights, axis=-1)

        actions = np.squeeze(moe_actions)
        current_active_cmd = obs_group["current_active_cmd"]
        if (current_active_cmd == 0) and (self.time_since_switch > 1.0):
            actions = np.squeeze(stand_actions)
            # logger.debug("current_state: stand")
        elif (current_active_cmd == 1) and (self.time_since_switch > 1.0):
            actions = np.squeeze(navigate_actions)
            # logger.debug("current_state: navigate")

        actions = np.clip(actions, -self.clip_action, self.clip_action)

        self.last_actions_buf = actions

        dof_pos_target_urdf = actions * self.action_scale + self.default_dof_pos_control_stand

        # dof_pos_target_urdf = self.exp_filter.filter(dof_pos_target_urdf)
        return dof_pos_target_urdf

    def reset(self, first_obs_group=None):
        self.last_actions_buf.fill(0.0)
        if first_obs_group is None:
            first_obs_group = self.default_obs_group
        norminal_stand_obs, norminal_navigate_obs, norminal_gating_obs = self.build_observations_one_step(first_obs_group)
        self.prop_obs_history_stand[:, :] = norminal_stand_obs[None, :]
        self.prop_obs_history_navigate[:, :] = norminal_navigate_obs[None, :]
        self.prop_obs_history_gating[:, :] = norminal_gating_obs[None, :]
        self.exp_filter.reset()
        self.exp_filter_navigate.reset()
        self.filter_walk.reset()
        self.filter_rotate.reset()
        self.filter_overall.reset()
        self.filter_overall_rot.reset()
        self.last_timestamp = time.perf_counter()

if __name__ == "__main__":
    a = agent_stand_nav_moe()
    np.set_printoptions(formatter={"float": "{:.2f}".format})
    for i in range(100):
        logger.debug(a.inference(a.default_obs_group))
