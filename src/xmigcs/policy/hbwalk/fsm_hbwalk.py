from typing import Dict
import numpy as np
from xmigcs.FSM.fsm_base import FSMState, FSMStateName
from xmigcs.common.control_flag import FSMControlFlag
from xmigcs.common.robot_data import RobotData
import os
import yaml
import time

from xmigcs.policy.hbwalk.agent import (
    agent_mlp_12dof_amp_lab,
    agent_mlp_15dof_amp_lab,
    agent_mlp_23dof_amp_lab,
    agent_mlp_23dof_amp_lab_sym,
    agent_mlp_23dof_amp_lab_sym_renteng,
)
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

class FSMStateHBWALK(FSMState):
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # NOTE: select policy here
        self.agent = agent_mlp_12dof_amp_lab()
        # self.agent = agent_mlp_15dof_amp_lab()
        # self.agent = agent_mlp_23dof_amp_lab()
        # self.agent = agent_mlp_23dof_amp_lab_sym()
        config_path = os.path.join(current_dir, "config", "hbwalk_lab.yaml")

        with open(config_path, "r") as f:
            policy_config = yaml.safe_load(f)

        self.agent_dof_indices = self.agent.env_to_agent

        self.dt = 0.01
        self.physics_dt = 0.01
        self.decimation_ = round(self.dt / self.physics_dt)
        self.kp = np.array(policy_config.get("kp"))
        self.kd = np.array(policy_config.get("kd"))

        self.init_flag = False

        self.last_run_time = time.perf_counter()
        self.warm_start_pose = None
        self.warm_start_time = None
        self.warm_time = 1.0

    def on_enter(self):
        logger.info("[FSMStateHBWALK] enter")
        self.init_flag = True
        self.warm_start_pose = self.robot_data_.get_serial_joint_pos_desired()[self.agent_dof_indices]
        self.warm_start_time = self.robot_data_.time_now_

    def on_exit(self):
        logger.info("[FSMStateHBWALK] exit")

    def run(self, flag: FSMControlFlag):
        # Only run policy inference every decimation_ steps
        if self.robot_data_.control_step_ % self.decimation_ == 0:
            current_time = time.perf_counter()
            self.last_run_time = current_time

            obs_group = self.get_observation_group()
            if self.init_flag:
                self.init_flag = False
                self.agent.reset(obs_group)
            output_joint_pos = self.agent.inference(obs_group)
            # output_joint_pos = self.interpolate_actions(output_joint_pos)
            self.write_to_robot_data(output_joint_pos)

        self.set_kp_kd()

    def interpolate_actions(self, action):
        if self.warm_start_pose is not None and self.warm_start_time is not None:
            # 计算插值
            alpha = (self.robot_data_.time_now_ - self.warm_start_time) / self.warm_time  # 1.5秒内完成插值
            alpha = np.clip(alpha, 0.0, 1.0)
            action = self.warm_start_pose * (1 - alpha) + action * alpha
        return action

    def set_kp_kd(self):
        # Set kp/kd gains
        self.robot_data_.joint_kp_p_[self.agent_dof_indices] = self.kp[self.agent_dof_indices]
        self.robot_data_.joint_kd_p_[self.agent_dof_indices] = self.kd[self.agent_dof_indices]

    def write_to_robot_data(self, output_joint_pos):
        """将输出写入机器人数据"""
        # [6:]跳过前6维浮动基
        # NOTE: 只写入agent控制的 其余不写入
        self.robot_data_.q_d_[6:][self.agent_dof_indices] = output_joint_pos
        self.robot_data_.q_dot_d_[6:][self.agent_dof_indices] = 0.0
        self.robot_data_.tau_d_[6:][self.agent_dof_indices] = 0.0
        # 写入估计的基座线速度
        if hasattr(self.agent, "base_lin_vel_estimated"):
            self.robot_data_.q_dot_a_[:3] = self.agent.base_lin_vel_estimated

    def get_observation_group(self) -> Dict[str, np.ndarray]:
        dof_pos = self.robot_data_.get_joint_pos()[self.agent_dof_indices]
        dof_vel = self.robot_data_.get_joint_vel()[self.agent_dof_indices]

        angular_velocity = self.robot_data_.get_angular_velocity()

        commands = np.array([
            self.robot_data_.get_walk_cmd()[0],  # x_speed
            self.robot_data_.get_walk_cmd()[1],  # y_speed
            self.robot_data_.get_walk_cmd()[2],  # yaw_speed
        ], dtype=np.float32)
        # np.set_printoptions(formatter={"float": "{:.3f}".format})
        logger.debug("command: %s", commands)
        # commands = np.array([
        #     0.0,  # x_speed
        #     0.0,  # y_speed
        #     0.0,  # yaw_speed
        # ], dtype=np.float32)
        euler_rad = np.array([
            self.robot_data_.imu_data_[2],  # roll
            self.robot_data_.imu_data_[1],  # pitch
            self.robot_data_.imu_data_[0],  # yaw
        ], dtype=np.float32)

        projected_gravity = self.robot_data_.get_project_gravity()

        obs_group = {
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "angular_velocity": angular_velocity,
            "commands": commands,
            "height_cmd": np.array([self.robot_data_.walk_height_cmd_]),  # not used
            "timestamp": np.array([self.robot_data_.time_now_]),
            "euler_rad": euler_rad,
            "projected_gravity": projected_gravity,
        }

        return obs_group

    def check_transition(self, *args, **kwargs):
        """检查状态转换"""
        pass
