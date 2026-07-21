from typing import Optional

from xmigcs.common.robot_data import RobotData
from xmigcs.policy.niukua.fsm_niukua import FSMStateNIUKUA
import os
import time
import xlog
from xmigcs.FSM.fsm_base import FSMState, FSMStateName

class FSMStateLAYUP(FSMStateNIUKUA):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(current_dir, "config", "layup.yaml")
    left_hand_id16_motor_indices = (20,)
    left_hand_id16_targets = (-1.2,)
    right_hand_id26_motor_indices = (27,)
    right_hand_id26_targets = (-1.2,)
    
    def __init__(self, robot_data: RobotData, config_path: Optional[str] = None):
        super().__init__(
            robot_data,
            config_path=config_path or self.default_config_path,
            variant_name="layup",
        )
        self._last_exit_blocked_log_time = 0.0


    def inner_run(self):
        super().inner_run()
        self._override_left_hand_id16_joint()
        self._override_right_hand_id26_joint()

    def _override_left_hand_id16_joint(self):
        floating_base_dof = self.robot_data_.q_d_.shape[0] - self.robot_data_.motor_num
        for motor_index, target in zip(
            self.left_hand_id16_motor_indices,
            self.left_hand_id16_targets,
        ):
            qd_index = floating_base_dof + motor_index
            self.robot_data_.q_d_[qd_index] = target
            self.robot_data_.q_dot_d_[qd_index] = 0.0
            self.robot_data_.tau_d_[qd_index] = 0.0

    def _override_right_hand_id26_joint(self):
        floating_base_dof = self.robot_data_.q_d_.shape[0] - self.robot_data_.motor_num
        for motor_index, target in zip(
            self.right_hand_id26_motor_indices,
            self.right_hand_id26_targets,
        ):
            qd_index = floating_base_dof + motor_index
            self.robot_data_.q_d_[qd_index] = target
            self.robot_data_.q_dot_d_[qd_index] = 0.0
            self.robot_data_.tau_d_[qd_index] = 0.0


    def check_transition(self, *args, **kwargs):
        """检查是否允许切入/退出 layup。"""
        result = {}
        result['allow_transition'] = True
        action = kwargs.get("action")
        target_state = kwargs.get("target_state")
        white_list = kwargs.get("white_list")
        if action == "enter":
            #TODO: 判断是否允许切入 layup
            return result
        else :
            # 如果目标状态为停止，则允许退出 layup
            if target_state is not None and target_state.name in white_list.values():
                return result
            # 如果当前机器人未执行完成 layup 动作，则禁止退出 layup
            if not self.is_motion_end:
                current_time = time.perf_counter()
                if current_time - self._last_exit_blocked_log_time >= 5.0:
                    xlog.warning("[FSMStateLAYUP] 当前机器人未执行完成 layup 动作，禁止退出 layup")
                    self._last_exit_blocked_log_time = current_time
                result['allow_transition'] = False
        return result