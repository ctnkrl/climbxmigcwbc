# Copyright (c) 2026 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
from typing import Dict
import numpy as np
from xmigcs.FSM.fsm_base import FSMState, FSMStateName
from xmigcs.common.control_flag import FSMControlFlag
from xmigcs.common.robot_data import RobotData
import os
import yaml
import time
from xmigcs.policy.stand_nav_moe.agent import agent_stand_nav_moe
from xmigcs.utils.cmd_filter import CmdFilter
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)


class FSMStateSTAND_NAV_MOE(FSMState):
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # NOTE: select policy here
        self.agent = agent_stand_nav_moe()
        config_path = os.path.join(current_dir, "config", "stand_nav.yaml")

        with open(config_path, "r") as f:
            policy_config = yaml.safe_load(f)

        self.env_to_agent_control = self.agent.env_to_agent_control_stand

        self.dt = 0.01
        self.physics_dt = 0.01
        self.decimation = round(self.dt / self.physics_dt)
        self.kp = np.array(policy_config["kp"])
        self.kd_stand = np.array(policy_config["kd_stand"])
        self.kd_navigate = np.array(policy_config["kd_navigate"])
        self.kd = self.kd_navigate

        # 从yaml读取滤波器参数
        cmd_filter = policy_config["cmd_filter"]
        fc_roll = cmd_filter["roll"]
        self.filter_roll = CmdFilter(
            low=float(fc_roll["low"]),
            high=float(fc_roll["high"]),
            rate_limit=float(fc_roll["rate_limit"]),
            alpha=float(fc_roll["alpha"]),
        )
        fc_pitch = cmd_filter["pitch"]
        self.filter_pitch = CmdFilter(
            low=float(fc_pitch["low"]),
            high=float(fc_pitch["high"]),
            rate_limit=float(fc_pitch["rate_limit"]),
            alpha=float(fc_pitch["alpha"]),
        )
        fc_yaw = cmd_filter["yaw"]
        self.filter_yaw = CmdFilter(
            low=float(fc_yaw["low"]),
            high=float(fc_yaw["high"]),
            rate_limit=float(fc_yaw["rate_limit"]),
            alpha=float(fc_yaw["alpha"]),
        )
        fc_height = cmd_filter["height"]
        self.filter_height = CmdFilter(
            low=float(fc_height["low"]),
            high=float(fc_height["high"]),
            rate_limit=float(fc_height["rate_limit"]),
            alpha=float(fc_height["alpha"]),
        )

        self.waist_yaw_env_idx = self.agent.env_names.index("waist_yaw_joint")
        self.stand_default_height = float(policy_config["stand_default_height"])

        self.init_flag = False

        self.last_run_time = time.perf_counter()
        self.warm_start_pose = None
        self.warm_start_time = None
        self.warm_time = 0.5
        self.last_state = None
        self.time_since_enter = np.zeros(1)

    def on_enter(self):
        logger.info("[FSMStateSTAND_NAV_MOE] enter")
        self.init_flag = True
        self.filter_roll.reset()
        self.filter_pitch.reset()
        self.filter_yaw.reset()
        self.filter_height.reset(self.stand_default_height)
        self.warm_start_pose = self.robot_data_.get_serial_joint_pos_desired()[self.env_to_agent_control]
        self.warm_start_time = self.robot_data_.time_now_
        self.agent.last_timestamp = time.perf_counter()
        self.agent.last_active_command = self.robot_data_.nav_stand_mode  # 0=stand, 1=navigate
        self.time_since_enter[:] = 0.

    def on_exit(self):
        logger.info("[FSMStateSTAND_NAV_MOE] exit")

    def run(self, flag: FSMControlFlag):
        if int(self.robot_data_.time_now_ / self.physics_dt) % self.decimation == 0:
            current_time = time.perf_counter()
            fps = 1 / (current_time - self.last_run_time)
            logger.debug("Inference hz: %s", fps)
            self.last_run_time = current_time

            obs_group = self.get_observation_group()
            self.time_since_enter += self.agent.dt
            current_state = self.current_state
            logger.info(f"moe inner state: {current_state}")
            if self.init_flag:
                self.init_flag = False
                self.agent.reset(obs_group)
            self.agent.fps = fps
            output_joint_pos = self.agent.inference(obs_group)
            # output_joint_pos = self.interpolate_actions(output_joint_pos)
            self.write_to_robot_data(output_joint_pos)

        self.set_kp_kd()

    def interpolate_actions(self, action):
        if self.warm_start_pose is not None and self.warm_start_time is not None:
            # 计算插值
            alpha = (self.robot_data_.time_now_ - self.warm_start_time) / self.warm_time
            alpha = np.clip(alpha, 0.0, 1.0)
            action = self.warm_start_pose * (1 - alpha) + action * alpha
        return action

    def set_kp_kd(self):
        # Set kp/kd gains
        if self.agent.time_since_switch > 1.0:
            # 切换后5秒 把kd换成对应的
            if self.agent.last_active_command == 0:
                self.kd = self.kd_stand
            elif self.agent.last_active_command == 1:
                self.kd = self.kd_navigate
        else:
            # transition时使用stand
            self.kd = self.kd_stand
        self.robot_data_.joint_kp_p_[self.env_to_agent_control] = self.kp
        self.robot_data_.joint_kd_p_[self.env_to_agent_control] = self.kd

    def write_to_robot_data(self, output_joint_pos):
        """将输出写入机器人数据"""
        # [6:]跳过前6维浮动基
        # NOTE: 只写入agent控制的 其余不写入
        self.robot_data_.q_d_[6:][self.env_to_agent_control] = output_joint_pos
        self.robot_data_.q_dot_d_[6:][self.env_to_agent_control] = 0.0
        self.robot_data_.tau_d_[6:][self.env_to_agent_control] = 0.0

        current_active_cmd = self.robot_data_.nav_stand_mode  # 0=stand, 1=navigate
        if current_active_cmd == 0:
            # waist_yaw 不经网络，直接用滤波后的命令值驱动
            yaw_raw = float(self.robot_data_.floating_base_cmd_[2])
            yaw_target = self.filter_yaw.filter(yaw_raw, self.dt * self.decimation)
            self.robot_data_.q_d_[6 + self.waist_yaw_env_idx] = yaw_target
            self.robot_data_.q_dot_d_[6 + self.waist_yaw_env_idx] = 0.0
            self.robot_data_.tau_d_[6 + self.waist_yaw_env_idx] = 0.0

    def get_observation_group(self) -> Dict[str, np.ndarray]:
        dof_pos = self.robot_data_.get_joint_pos()
        dof_vel = self.robot_data_.get_joint_vel()

        angular_velocity = self.robot_data_.get_angular_velocity()

        roll_raw = self.robot_data_.floating_base_cmd_[0]
        pitch_raw = self.robot_data_.floating_base_cmd_[1]
        yaw_raw = self.robot_data_.floating_base_cmd_[2]
        height_raw = self.robot_data_.floating_base_cmd_[3]
        if height_raw == 0.0:
            # 0指令转换成默认高度
            height_raw = self.stand_default_height
        # np.set_printoptions(formatter={"float": "{:.3f}".format})
        logger.debug("rpy_command: %s", np.array([roll_raw, pitch_raw, yaw_raw, height_raw]))

        dt = self.dt * self.decimation
        roll_cmd = self.filter_roll.filter(roll_raw, dt)
        pitch_cmd = self.filter_pitch.filter(pitch_raw, dt)
        # yaw_cmd = self.filter_yaw.filter(yaw_raw, dt)  # not used in network
        height_cmd = self.filter_height.filter(height_raw, dt)

        stand_commands = np.array([
            0.,
            0.,
            0.,
            roll_cmd,
            pitch_cmd,
            height_cmd,
        ], dtype=np.float32)
        walk_commands = np.array([
            self.robot_data_.get_walk_cmd()[0],  # x_speed
            self.robot_data_.get_walk_cmd()[1],  # y_speed
            self.robot_data_.get_walk_cmd()[2],  # yaw_speed
        ], dtype=np.float32).clip(max=1.2)
        logger.debug("walk commands: %s", walk_commands)
        navigate_commands = np.array([
            self.robot_data_.get_footpoint_cmd()[1],  # bcs_x
            self.robot_data_.get_footpoint_cmd()[2],  # bcs_y
            self.robot_data_.get_footpoint_cmd()[3],  # target_heading
        ], dtype=np.float32)
        navigate_mode = self.robot_data_.get_footpoint_cmd()[0]
        logger.debug("navigate commands: %s, navigate mode: %s", navigate_commands, navigate_mode)
        euler_rad = np.array([
            self.robot_data_.imu_data_[2],  # roll
            self.robot_data_.imu_data_[1],  # pitch
            self.robot_data_.imu_data_[0],  # yaw
        ], dtype=np.float32)

        projected_gravity = self.robot_data_.get_project_gravity()

        current_active_cmd = self.robot_data_.nav_stand_mode  # 0=stand, 1=navigate
        if self.last_state != current_active_cmd:
            if current_active_cmd == 0:
                logger.info("current_state: stand")
            if current_active_cmd == 1:
                logger.info("current_state: navigate")
            self.last_state = current_active_cmd
        # 只允许active command 其他置零
        if current_active_cmd == 0:
            # stand active
            walk_commands[:] = 0.
            navigate_commands[:] = 0.
        else:
            # navigate active
            stand_commands[:] = 0.
            stand_commands[-1] = self.stand_default_height
        obs_group = {
            "current_active_cmd": current_active_cmd,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "angular_velocity": angular_velocity,
            "stand_commands": stand_commands,
            "walk_commands": walk_commands,
            "navigate_commands": navigate_commands,
            "navigate_mode": navigate_mode,  # NOTE: bool type
            "height_cmd": np.array([self.robot_data_.walk_height_cmd_]),  # not used
            "timestamp": np.array([time.perf_counter()]),
            "euler_rad": euler_rad,
            "projected_gravity": projected_gravity,
        }

        return obs_group

    @property
    def current_state(self):
        current_active_cmd = self.robot_data_.nav_stand_mode  # 0=stand, 1=navigate
        if self.time_since_enter < self.agent.transition_window:
            # first enter 返回和当前cmd一致的状态
            if current_active_cmd == 0:
                return "STAND"
            else:
                return "NAVIGATE"
        else:
            p = self.agent.transition_percent
            if p > 0.99:
                return "STAND"
            elif p < 0.01:
                return "NAVIGATE"
            else:
                # 切换中 返回上一个状态
                if current_active_cmd == 0:
                    return "NAVIGATE"
                else:
                    return "STAND"

    def check_transition(self, *args, **kwargs):
        """检查状态转换"""
        pass
