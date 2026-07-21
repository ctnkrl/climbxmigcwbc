"""
FSM State Implementations
Concrete implementations of different FSM states
"""

import numpy as np

from xmigcs.FSM.fsm_base import FSMState, FSMStateName
from xmigcs.common.control_flag import FSMControlFlag
from xmigcs.common.robot_data import RobotData
import yaml
import os
import pinocchio as pin
import numpy as np


class FSMStateNOGRAVITY(FSMState):
    """无重力状态实现 - 与C++版本完全一致"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "nogravity.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)

        self.motor_num_ = policy_config["motor_num"]
        self.kp_pos_ = np.zeros(self.motor_num_)
        self.kd_pos_ = np.zeros(self.motor_num_)
        # Load kp and kd gains from config
        for i in range(self.motor_num_):
            self.kp_pos_[i] = policy_config["kp_pos"][i]
            self.kd_pos_[i] = policy_config["kd_pos"][i]
        self.joint_index_ = policy_config["joint_index"]
        
        xmigcs_path = os.path.dirname(os.path.dirname(current_dir))
        urdf_model_path = os.path.join(xmigcs_path, "config", "tiangong3.urdf")

        self.model = pin.buildModelFromUrdf(
            urdf_model_path)
        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])
        self.data = self.model.createData()
        
        # Initialize low-pass filter for gravity
        self.gravity_filter_coeff = 0.1  # Filter coefficient (adjustable)
        self.filtered_gravity = np.array([0.0, 0.0, -9.81])  # Start with standard gravity
        self.floating_base_dof_ = 6
        self.head_motor_num_ = 2

    def on_enter(self):
        """进入零重力状态"""

    def run(self, flag: FSMControlFlag):
        """运行零重力状态"""
        if self.robot_data_ is None:
            return
        # Enforce the hold position for every frame (equivalent to tail(motor_num_))
        self.robot_data_.q_d_[-self.motor_num_:] = self.robot_data_.q_a_[
            -self.motor_num_:].copy()
        # Set desired joint velocities to zero
        self.robot_data_.q_dot_d_[-self.motor_num_:] = 0.0
        # wxyz = self.robot_data_.get_robot_quat()
        # q = np.concatenate(
        #     (np.array([0, 0, 0, wxyz[1], wxyz[2], wxyz[3], wxyz[0]]), self.robot_data_.q_a_[-self.motor_num_:]))
        q = self.robot_data_.q_a_[self.floating_base_dof_: -self.head_motor_num_]
        v = pin.utils.zero(self.model.nv)
        a = pin.utils.zero(self.model.nv)

        # Get the current gravity estimate from sensor
        current_gravity = self.robot_data_.get_project_gravity() * 9.81
        
        # Apply low-pass filter to smooth the gravity vector
        self.filtered_gravity = (1 - self.gravity_filter_coeff) * self.filtered_gravity + \
                               self.gravity_filter_coeff * current_gravity
        
        self.model.gravity.linear = self.filtered_gravity


        tau = pin.rnea(self.model, self.data, q, v, a)

        # Set desired torques to gravity compensation torques
        self.robot_data_.tau_d_[-self.motor_num_:][self.joint_index_] = tau[-self.motor_num_:][self.joint_index_]*0.7

        self.robot_data_.joint_kp_p_[self.joint_index_] = self.kp_pos_[self.joint_index_]
        self.robot_data_.joint_kd_p_[self.joint_index_] = self.kd_pos_[self.joint_index_]


    def on_exit(self):
        """退出无重力状态 - 与C++版本完全一致"""

    def check_transition(self, *args, **kwargs):
        """检查状态转换"""
        pass