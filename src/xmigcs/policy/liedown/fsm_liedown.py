from typing import Optional

from xmigcs.common.robot_data import RobotData
from xmigcs.policy.niukua.fsm_niukua import FSMStateNIUKUA
import os
import numpy as np
import time
import xlog


class FSMStateLIEDOWN(FSMStateNIUKUA):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(current_dir, "config", "liedown.yaml")
    right_wrist_pitch_motor_index = 27
    right_wrist_interp_frames = 100
    right_wrist_start_offset = -1.2
    right_wrist_return_start_frame = 1200 #1000 700
    
    def __init__(self, robot_data: RobotData, config_path: Optional[str] = None):
        super().__init__(
            robot_data,
            config_path=config_path or self.default_config_path,
            variant_name="liedown",
        )
        self._right_wrist_base_pos: Optional[float] = None
        self._right_wrist_final_start_pos: Optional[float] = None
        self._last_exit_blocked_log_time = 0.0

    def on_enter(self):
        self._right_wrist_base_pos = None
        self._right_wrist_final_start_pos = None
        super().on_enter()

    def inner_run(self):
        motion_frame = self.counter_step
        motor_q_a = self.robot_data_.q_a_[-self.robot_data_.motor_num:]
        right_wrist_current_pos = float(motor_q_a[self.right_wrist_pitch_motor_index])

        super().inner_run()

        self._apply_right_wrist_liedown_interp(motion_frame, right_wrist_current_pos)

    def on_exit(self):
        self._right_wrist_base_pos = None
        self._right_wrist_final_start_pos = None
        super().on_exit()

    def _apply_right_wrist_liedown_interp(self, motion_frame: int, right_wrist_current_pos: float):
        if motion_frame < 0:
            return
        if self._right_wrist_base_pos is None:
            self._right_wrist_base_pos = right_wrist_current_pos

        if motion_frame >= self.right_wrist_return_start_frame:
            if self._right_wrist_final_start_pos is None:
                self._right_wrist_final_start_pos = right_wrist_current_pos

            return_frame = motion_frame - self.right_wrist_return_start_frame
            alpha = self._interp_alpha(return_frame)
            right_wrist_target = (1.0 - alpha) * self._right_wrist_final_start_pos
        else:
            self._right_wrist_final_start_pos = None
            alpha = self._interp_alpha(motion_frame)
            right_wrist_target = (
                self._right_wrist_base_pos
                + self.right_wrist_start_offset * alpha
            )

        motor_num = self.robot_data_.motor_num
        right_wrist_index = self.right_wrist_pitch_motor_index
        motor_q_d = self.robot_data_.q_d_[-motor_num:]
        motor_q_dot_d = self.robot_data_.q_dot_d_[-motor_num:]
        motor_tau_d = self.robot_data_.tau_d_[-motor_num:]

        motor_q_d[right_wrist_index] = right_wrist_target
        motor_q_dot_d[right_wrist_index] = 0.0
        motor_tau_d[right_wrist_index] = 0.0

    def _interp_alpha(self, frame: int) -> float:
        if self.right_wrist_interp_frames <= 1:
            return 1.0
        return float(np.clip(frame / (self.right_wrist_interp_frames - 1), 0.0, 1.0))

    def check_transition(self, *args, **kwargs):
        """检查是否允许切入/退出 liedown。"""
        result = {}
        result['allow_transition'] = True
        action = kwargs.get("action")
        target_state = kwargs.get("target_state")
        white_list = kwargs.get("white_list")
        if action == "enter":
            #TODO: 判断是否允许切入 liedown
            return result
        else :
            # 如果目标状态为停止，则允许退出 liedown
            if target_state is not None and target_state.name in white_list.values():
                return result
            # 如果当前机器人未执行完成 liedown 动作，则禁止退出 liedown
            if not self.is_motion_end:
                current_time = time.perf_counter()
                if current_time - self._last_exit_blocked_log_time >= 5.0:
                    xlog.warning("[FSMStateLIEDOWN] 当前机器人未执行完成 liedown 动作，禁止退出 liedown")
                    self._last_exit_blocked_log_time = current_time
                result['allow_transition'] = False
        return result
