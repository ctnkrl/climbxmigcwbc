"""
FSM State Implementations
Concrete implementations of different FSM states
"""
import numpy as np
from xmigcs.FSM.fsm_base import FSMState, FSMStateName
from xmigcs.common.control_flag import FSMControlFlag
from xmigcs.utils.logging_utils import get_logger
from xmigcs.common.robot_data import RobotData
import os
import time
import yaml
from xmigcs.utils.xlog_utils import xlog

logger = get_logger(__name__)


class FSMStateSTANDZERO(FSMState):
    """站立零位状态实现"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.q_factor_ = 0.0
        self._last_far_from_zero_log_time = 0.0
        self._last_not_standing_log_time = 0.0
        self._last_exit_blocked_log_time = 0.0
        self.is_motion_end = False
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "standzero.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)
        self.motor_num_ = policy_config["motor_num"]
        self.zero_positions_ = np.array(policy_config["zero_positions"],
                                        dtype=float)
        self.kp_pos_ = np.array(policy_config["kp_pos"], dtype=float)
        self.kd_pos_ = np.array(policy_config["kd_pos"], dtype=float)
        self.interp_step_ = float(policy_config["interp_step"])
        self.interp_max_ = float(policy_config["interp_max"])
        self.close_threshold_ = float(policy_config.get("close_threshold"))

    def on_enter(self):
        self.q_factor_ = 0.0
        self.is_motion_end = False

    def run(self, flag: FSMControlFlag):
        # q_est = self.robot_data_.q_a_[-self.motor_num_:].copy()  # numpy数组切片    
        q_est = self.robot_data_.get_serial_joint_pos_desired()
        if self.q_factor_ < self.interp_max_:
            pos_cmd = (1.0 - self.q_factor_
                       ) * q_est + self.q_factor_ * self.zero_positions_
            self.q_factor_ = min(self.q_factor_ + self.interp_step_,
                                 self.interp_max_)
        else:
            pos_cmd = self.zero_positions_
        self.robot_data_.q_d_[-self.motor_num_:] = pos_cmd
        self.robot_data_.q_dot_d_[-self.motor_num_:] = 0
        self.robot_data_.tau_d_[-self.motor_num_:] = 0
        self.robot_data_.joint_kp_p_[:self.motor_num_] = self.kp_pos_
        self.robot_data_.joint_kd_p_[:self.motor_num_] = self.kd_pos_
        is_close, q_est, position_diff, max_diff, max_idx, max_diff_id = self.is_close_to_zero_positions(threshold=self.close_threshold_ / 180 * np.pi)
        if is_close:
            self.is_motion_end = True

    def on_exit(self):
        self.is_motion_end = False

    def check_transition(self, *args, **kwargs):
        """检查状态转换"""
        result = {}
        result['allow_transition'] = True

        action = kwargs.get("action")
        target_state = kwargs.get("target_state")
        white_list = kwargs.get("white_list")
        if action == "exit":
            #TODO: 判断是否允许退出 standzero
            if target_state is not None and white_list is not None and target_state.name in white_list.values():
                return result
            if not self.is_motion_end:
                current_time = time.perf_counter()
                if current_time - self._last_exit_blocked_log_time >= 5.0:
                    xlog.warning("[FSMStateSTANDZERO] 当前机器人未执行完成 standzero 动作，禁止退出 standzero")
                    is_close, q_est, position_diff, max_diff, max_idx, max_diff_id = self.is_close_to_zero_positions(threshold=self.close_threshold_ / 180 * np.pi)
                    if not is_close:
                        xlog.warning("[FSMStateSTANDZERO] 当前距离站立零位位置过远")
                        xlog.warning(f"  各关节位置差值：{position_diff}")
                        xlog.warning(
                            f"  最大差值：{max_diff:.4f} (关节 ID: {max_diff_id}, 索引: {max_idx})"
                        )
                        xlog.warning(f"  当前估计位置：{q_est.tolist()}")
                        xlog.warning(f"  目标零位位置：{self.zero_positions_.tolist()}")
                    self._last_exit_blocked_log_time = current_time
                result['allow_transition'] = False
            return result
        else :
            q_est = self.robot_data_.q_a_[-self.motor_num_:].copy()  # numpy数组切片
            projected_gravity = self.robot_data_.get_project_gravity()
            is_body_upright = projected_gravity[2] < -0.8
            # left_leg = q_est[0:4]
            # right_leg = q_est[6:10]
            # leg_symmetry_sign = np.array([1.0, -1.0, -1.0, 1.0])
            # leg_symmetry_diff = np.abs(left_leg - leg_symmetry_sign * right_leg)
            # is_leg_symmetric = np.max(leg_symmetry_diff) < 0.25
            hip_pitch_l, knee_pitch_l = q_est[0], q_est[3]
            hip_pitch_r, knee_pitch_r = q_est[6], q_est[9]
            # has_sitting_leg_pose = (
            #     (knee_pitch_l > 1.0 and knee_pitch_r > 1.0)
            #     and (hip_pitch_l < -0.8 and hip_pitch_r < -0.8)
            # )
            # is_sitting = has_sitting_leg_pose and is_leg_symmetric

            has_standing_leg_pose = (
                (knee_pitch_l < 1.0 and knee_pitch_r < 1.0)
                and (hip_pitch_l > - 0.7 and hip_pitch_r > - 0.7)
            )
            is_standing = is_body_upright and has_standing_leg_pose
            if not is_standing:
                current_time = time.perf_counter()
                if current_time - self._last_not_standing_log_time >= 5.0:
                    xlog.warning("[FSMStateSTANDZERO] 当前机器人未处于站立状态，禁止切入站立零位状态")
                    xlog.warning(f"  当前重力投影：{projected_gravity.tolist()}")
                    xlog.warning(f"  当前髋/膝关节：{hip_pitch_l:.4f}, {knee_pitch_l:.4f}, {hip_pitch_r:.4f}, {knee_pitch_r:.4f}")
                    self._last_not_standing_log_time = current_time
                result['allow_transition'] = False
            is_close, q_est, position_diff, max_diff, max_idx, max_diff_id = self.is_close_to_zero_positions(threshold=120.0 / 180 * np.pi)
            if not is_close:
                current_time = time.perf_counter()
                if current_time - self._last_far_from_zero_log_time >= 5.0:
                    xlog.warning("[FSMStateSTANDZERO] 当前距离站立零位位置过远")
                    xlog.warning(f"  各关节位置差值：{position_diff}")
                    xlog.warning(
                        f"  最大差值：{max_diff:.4f} (关节 ID: {max_diff_id}, 索引: {max_idx})"
                    )
                    xlog.warning(f"  当前估计位置：{q_est.tolist()}")
                    xlog.warning(f"  目标零位位置：{self.zero_positions_.tolist()}")
                    self._last_far_from_zero_log_time = current_time
                result['allow_transition'] = False
            return result

    def is_close_to_zero_positions(self, threshold: float = 120.0 / 180 * np.pi) -> bool:
        q_est = self.robot_data_.q_a_[-self.motor_num_:].copy()  # numpy数组切片
        motor_ids = [51, 52, 53, 54, 55, 56,  # 左腿
                    61, 62, 63, 64, 65, 66, # 右腿
                    33, 32, 31, # 腰
                    11, 12, 13, 14, 15, 16, 17, # 左臂
                    21, 22, 23, 24, 25, 26, 27, # 右臂
                    1, 2] # 头部
        position_diff = np.abs(q_est - self.zero_positions_)
        max_diff = np.max(position_diff)
        max_idx = int(np.argmax(position_diff))
        ids_for_slice = motor_ids[: self.motor_num_]
        max_diff_id = ids_for_slice[max_idx] if max_idx < len(ids_for_slice) else max_idx
        return max_diff < threshold, q_est, position_diff, max_diff, max_idx, max_diff_id
