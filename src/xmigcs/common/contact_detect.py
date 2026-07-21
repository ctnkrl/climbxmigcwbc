import pinocchio as pin
import numpy as np
import time
import os
from xmigcs.common.robot_data import RobotData
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

# class ContactBasedLiftDetector:
#     """
#     基于接触力估计的悬空检测器

#     步骤：
#     1. 估计接触力（基于力矩差异）
#     2. 检查接触力是否小于阈值
#     3. 确认是否持续悬空
#     """

#     def __init__(self, robot_data: RobotData):
#         """
#         初始化悬空检测器

#         参数:
#         """
#         # 加载模型
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         parent_dir = os.path.dirname(current_dir)
#         urdf_path = os.path.join(parent_dir, "config", "dex_evt_hand_29.urdf")
#         self.model = pin.buildModelFromUrdf(urdf_path)
#         self.data = self.model.createData()
#         self.model.gravity.linear = np.array([0.0, 0.0, -9.81])
#         self.dt = 0.01
#         self.robot_data_ = robot_data

#         # 检测参数
#         self.contact_threshold = 80.0  # 每只脚的接触力阈值
#         self.time_threshold = 0.5

#         # 定义脚部接触点（需要根据你的URDF调整）
#         self.foot_frames = self._setup_foot_frames()

#         # 状态变量
#         self.lift_timer = None
#         self.q_prev = None
#         self.v_prev = None
#         self.t_prev = None

#         # 滤波器
#         self.force_history = {}
#         for foot in self.foot_frames.keys():
#             self.force_history[foot] = []
#         self.history_size = 5

#         # Initialize low-pass filter for gravity
#         self.gravity_filter_coeff = 0.1  # Filter coefficient (adjustable)
#         self.filtered_gravity = np.array([0.0, 0.0, -9.81])  # Start with standard gravity

#         logger.debug(f"初始化完成:")
#         logger.debug(f"  接触力阈值: {self.contact_threshold}N")
#         logger.debug(f"  时间阈值: {self.time_threshold}s")
#         logger.debug(f"  检测脚部: {list(self.foot_frames.keys())}")

#     def _setup_foot_frames(self):
#         """设置脚部帧"""
#         foot_frames = {}

#         # 尝试查找左右脚帧
#         left_names = ['ankle_roll_l_link', ]
#         right_names = ['ankle_roll_r_link', ]

#         # 查找左脚
#         for name in left_names:
#             frame_id = self.model.getFrameId(name)
#             if frame_id < self.model.nframes:
#                 foot_frames['left_foot'] = frame_id
#                 logger.debug(f"找到左脚帧: {name} (ID: {frame_id})")
#                 break

#         # 查找右脚
#         for name in right_names:
#             frame_id = self.model.getFrameId(name)
#             if frame_id < self.model.nframes:
#                 foot_frames['right_foot'] = frame_id
#                 logger.debug(f"找到右脚帧: {name} (ID: {frame_id})")
#                 break

#         # 如果没找到，使用所有可能的帧
#         if not foot_frames:
#             logger.debug("警告: 未找到标准脚部帧，使用所有末端帧")
#             for i in range(self.model.nframes):
#                 frame_name = self.model.frames[i].name
#                 if 'foot' in frame_name.lower() or 'ankle' in frame_name.lower():
#                     foot_frames[frame_name] = i

#         return foot_frames

#     def estimate_contact_forces(self, q, v, tau):
#         """
#         估计接触力（基于力矩差异）

#         原理: tau_measured = tau_model + Σ J_i^T * f_i
#         """
#         # 1. 估计加速度
#         a_estimated = self._estimate_acceleration(v)
#         # a_estimated =  np.zeros(self.model.nv)
#         # v = np.zeros(self.model.nv)

#         # 2. 计算理论力矩（无外部接触力）
#         tau_expected = pin.rnea(self.model, self.data, q, v, a_estimated)

#         # 3. 计算力矩差异
#         tau_diff = tau - tau_expected
#         # logger.debug(f"tau_diff: {tau_diff[-self.robot_data_.motor_num:]}")
#         total_mass = pin.computeTotalMass(self.model)
#         # logger.debug(f"total_mass: {total_mass}")

#         # 4. 更新运动学
#         pin.forwardKinematics(self.model, self.data, q)
#         pin.updateFramePlacements(self.model, self.data)
#         pin.computeJointJacobians(self.model, self.data, q)

#         # 5. 为每个脚部帧估计接触力
#         contact_forces = {}

#         # for foot_name, frame_id in self.foot_frames.items():
#         #     # 获取雅可比矩阵
#         #     J = pin.computeFrameJacobian(self.model, self.data, q,
#         #                                 frame_id, pin.LOCAL_WORLD_ALIGNED)

#         #     # 简化假设：主要接触力在垂直方向
#         #     # 我们可以尝试求解完整的6维力，但这里简化处理

#         #     # 方法1：使用伪逆估算垂直力
#         #     try:
#         #         # 构建选择垂直力的方程
#         #         # 假设 f_z 是主要分量，其他分量为0
#         #         J_T = J.T
#         #         # 取雅可比矩阵的Z方向行（第3行对应垂直力）
#         #         if J.shape[0] >= 3:
#         #             J_z = J[2:3, :]  # Z方向力的雅可比行
#         #             J_z_T = J_z.T

#         #             # 求解 f_z
#         #             if np.linalg.matrix_rank(J_z_T) > 0:
#         #                 # 最小二乘求解
#         #                 f_z = - np.linalg.lstsq(J_z_T, tau_diff, rcond=None)[0][0]
#         #                 # f_z = max(0, f_z)  # 接触力不能为负
#         #             else:
#         #                 f_z = 0
#         #         else:
#         #             f_z = 0
#         #     except:
#         #         f_z = 0

#         #     # # 方法2：使用总力矩差异的比例分配（更稳定）
#         #     # total_tau_norm = np.linalg.norm(tau_diff)
#         #     # if total_tau_norm > 0:
#         #     #     # 简单分配：根据雅可比矩阵的范数分配
#         #     #     J_norm = np.linalg.norm(J)
#         #     #     if J_norm > 0:
#         #     #         # 假设30%的力矩差异来自垂直接触力
#         #     #         f_z_estimated = 0.3 * total_tau_norm / (J_norm + 1e-6)
#         #     #     else:
#         #     #         f_z_estimated = 0
#         #     # else:
#         #     #     f_z_estimated = 0

#         #     # 使用两个估计的平均值
#         #     # f_z = 0.5 * f_z + 0.5 * f_z_estimated


#         #     # 滤波处理
#         #     self.force_history[foot_name].append(f_z)
#         #     if len(self.force_history[foot_name]) > self.history_size:
#         #         self.force_history[foot_name].pop(0)

#         #     # # 使用滤波后的值
#         #     if self.force_history[foot_name]:
#         #         f_z_filtered = np.mean(self.force_history[foot_name])
#         #     else:
#         #         f_z_filtered = f_z
#         #     f_z_filtered = f_z

#         #     # 获取脚部位置
#         #     foot_pos = self.data.oMf[frame_id].translation

#         #     contact_forces[foot_name] = {
#         #         'vertical_force': f_z_filtered,
#         #         'is_contact': f_z_filtered > self.contact_threshold,
#         #         'frame_id': frame_id,
#         #         'position': foot_pos,
#         #         'height': foot_pos[2]
#         #     }
#         J_left = pin.computeFrameJacobian(self.model, self.data, q,
#                                          self.foot_frames['left_foot'],
#                                          pin.LOCAL_WORLD_ALIGNED)
#         J_right = pin.computeFrameJacobian(self.model, self.data, q,
#                                           self.foot_frames['right_foot'],
#                                           pin.LOCAL_WORLD_ALIGNED)

#         # 垂直方向
#         J_left_z = J_left[2:3, :]  # (1, nv)
#         J_right_z = J_right[2:3, :]

#         # 组合矩阵
#         A = np.hstack([J_left_z.T, J_right_z.T])  # (nv, 2)
#         # f_z = - np.linalg.lstsq(A, tau_diff, rcond=None)[0]

#         from scipy.optimize import minimize

#         def objective(f):
#             f_left, f_right = f
#             tau_est = A @ np.array([f_left, f_right])
#             return np.linalg.norm(tau_est - tau_diff)**2

#         # 初始猜测：平均分配重量（负值）
#         self.robot_weight = total_mass * self.model.gravity.linear[2] * -1  # 机器人重量（正值）
#         x0 = [-self.robot_weight/2, -self.robot_weight/2]

#         # 约束：力不能为正（不能向上拉地面）
#         bounds = [(-self.robot_weight*2, 0),  # 左脚下限，上限0
#                  (-self.robot_weight*2, 0)]   # 右脚下限，上限0

#         f_z = - minimize(objective, x0, bounds=bounds, method='L-BFGS-B').x

#         for index, (foot_name, frame_id) in enumerate(self.foot_frames.items()):
#             # 获取脚部位置
#             foot_pos = self.data.oMf[frame_id].translation
#             contact_forces[foot_name] = {
#                     'vertical_force': f_z[index],
#                     'is_contact': f_z[index] > self.contact_threshold,
#                     'frame_id': frame_id,
#                     'position': foot_pos,
#                     'height': foot_pos[2]
#                 }
#         return contact_forces, tau_diff, tau_expected

#     def _estimate_acceleration(self, v):
#         """估计关节加速度"""
#         current_time = time.perf_counter()

#         # 首次调用
#         if self.v_prev is None or self.t_prev is None:
#             self.v_prev = v.copy()
#             self.t_prev = current_time
#             return np.zeros_like(v)

#         # 计算时间间隔
#         dt = current_time - self.t_prev
#         if dt <= 0:
#             dt = self.dt

#         # 数值差分
#         a_raw = (v - self.v_prev) / dt

#         # 低通滤波
#         if not hasattr(self, 'a_filtered'):
#             self.a_filtered = a_raw.copy()
#         else:
#             alpha = 0.3
#             self.a_filtered = alpha * a_raw + (1 - alpha) * self.a_filtered

#         # 更新状态
#         self.v_prev = v.copy()
#         self.t_prev = current_time

#         return self.a_filtered

#     def detect_lifted(self,):
#         """
#         检测机器人是否被吊起（基于接触力）

#         返回:
#         - is_lifted: 是否悬空
#         - duration: 悬空持续时间(如果悬空)
#         - contact_forces: 估计的接触力
#         - info: 额外信息
#         """
#         current_time = time.perf_counter()

#         # 1. 估计接触力
#         wxyz = self.robot_data_.get_robot_quat()
#         omega_xyz = self.robot_data_.get_angular_velocity()
#         # Get the current gravity estimate from sensor
#         current_gravity = self.robot_data_.get_project_gravity() * 9.81

#         # Apply low-pass filter to smooth the gravity vector
#         self.filtered_gravity = (1 - self.gravity_filter_coeff) * self.filtered_gravity + \
#                                self.gravity_filter_coeff * current_gravity

#         self.model.gravity.linear = self.filtered_gravity

#         # q = np.concatenate((np.array([0, 0, 0, wxyz[1], wxyz[2], wxyz[3], wxyz[0]]), self.robot_data_.q_a_[-self.robot_data_.motor_num:]))
#         # v = np.concatenate((np.array([0, 0, 0, *omega_xyz]), self.robot_data_.q_dot_a_[-self.robot_data_.motor_num:]))
#         # tau = np.concatenate((np.array([0, 0, 0, 0, 0, 0]), self.robot_data_.tau_a_[-self.robot_data_.motor_num:]))
#         q = self.robot_data_.q_a_[-self.robot_data_.motor_num:]
#         v = self.robot_data_.q_dot_a_[-self.robot_data_.motor_num:]
#         tau = self.robot_data_.tau_a_[-self.robot_data_.motor_num:]
#         contact_forces, tau_diff, tau_expected = self.estimate_contact_forces(q, v, tau)

#         # 2. 检查每只脚的接触状态
#         foot_contact_states = {}
#         for foot_name, force_info in contact_forces.items():
#             foot_contact_states[foot_name] = force_info['is_contact']

#         # 3. 判断是否悬空（两只脚都无接触）
#         both_feet_lifted = True
#         for foot_name, is_contact in foot_contact_states.items():
#             if is_contact:  # 如果有任意一只脚接触
#                 both_feet_lifted = False
#                 break

#         # 4. 计时逻辑
#         if both_feet_lifted:
#             if self.lift_timer is None:
#                 self.lift_timer = current_time

#             lifted_duration = current_time - self.lift_timer

#             # 只有持续悬空超过阈值才确认
#             if lifted_duration >= self.time_threshold:
#                 return True, lifted_duration, contact_forces, {
#                     'status': 'LIFTED',
#                     'tau_diff_norm': np.linalg.norm(tau_diff),
#                     'tau_expected_norm': np.linalg.norm(tau_expected),
#                     'foot_states': foot_contact_states
#                 }
#             else:
#                 return False, lifted_duration, contact_forces, {
#                     'status': 'LIFTING',
#                     'duration_so_far': lifted_duration,
#                     'foot_states': foot_contact_states
#                 }
#         else:
#             # 有脚接触地面，重置计时器
#             self.lift_timer = None
#             return False, 0, contact_forces, {
#                 'status': 'GROUNDED',
#                 'foot_states': foot_contact_states,
#                 'tau_diff_norm': np.linalg.norm(tau_diff)
#             }

#     def get_force_statistics(self):
#         """获取力统计信息"""
#         stats = {}
#         for foot_name, force_history in self.force_history.items():
#             if force_history:
#                 stats[foot_name] = {
#                     'current': force_history[-1] if force_history else 0,
#                     'mean': np.mean(force_history),
#                     'max': np.max(force_history) if force_history else 0,
#                     'min': np.min(force_history) if force_history else 0,
#                     'threshold': self.contact_threshold
#                 }
#             else:
#                 stats[foot_name] = {
#                     'current': 0,
#                     'mean': 0,
#                     'max': 0,
#                     'min': 0,
#                     'threshold': self.contact_threshold
#                 }
#         return stats

#     def reset(self):
#         """重置检测器"""
#         self.lift_timer = None
#         self.q_prev = None
#         self.v_prev = None
#         self.t_prev = None
#         for foot in self.force_history.keys():
#             self.force_history[foot] = []
#         if hasattr(self, 'a_filtered'):
#             delattr(self, 'a_filtered')


import time
from collections import deque
import numpy as np


class ContactBasedLiftDetector:
    """
    人形机器人触地与飞车检测器。
    基于左右膝关节反馈力矩：
    1. 触地检测：力矩在指定时间阈值内持续大于指定值，判定为触地。
    2. 飞车检测：维护历史峰值列表，若当前峰值远超历史峰值，判定为飞车。
    """

    def __init__(
            self,
            robot_data: RobotData,
            torque_threshold=10.0,  # 触地力矩阈值(Nm)
            time_threshold=1.0,  # 触地持续时间阈值(s)
            history_window=5.0,  # 历史峰值记录窗口(s)
            peak_ratio_threshold=2.0,  # 飞车判定倍数(当前峰值/历史平均峰值)
            sample_rate=100.0):  # 控制频率/采样频率(Hz)
        """
        初始化检测器参数。
        """
        self.robot_data = robot_data
        self.torque_threshold = torque_threshold
        self.time_threshold = time_threshold
        self.history_window = history_window
        self.peak_ratio_threshold = peak_ratio_threshold
        self.dt = 1.0 / sample_rate
        self.max_history_len = int(history_window / self.dt)

        # 左右膝触地状态机
        self.is_lifted_timer = None

        # 历史峰值记录（双端队列，按时间排序）
        self.left_peak_history = deque(maxlen=self.max_history_len)
        self.right_peak_history = deque(maxlen=self.max_history_len)

        # 当前峰值（用于飞车判断）
        self.left_current_peak = 0.0
        self.right_current_peak = 0.0
        self.left_peak_decay_counter = 0
        self.right_peak_decay_counter = 0
        self.peak_decay_frames = int(0.5 / self.dt)  # 峰值保持0.5秒后衰减
        self.max_agv_peak_threshold = 50

    def reset(self):
        """重置所有状态"""
        self.left_contact_start_time = None
        self.right_contact_start_time = None
        self.left_contact_state = False
        self.right_contact_state = False
        self.left_peak_history.clear()
        self.right_peak_history.clear()
        self.left_current_peak = 0.0
        self.right_current_peak = 0.0
        self.left_peak_decay_counter = 0
        self.right_peak_decay_counter = 0

    def update(self):
        """
        更新检测器状态。
        Args:
            
        Returns:
            dict: {
                'is_lifted': bool,
                'left_lift_off': bool,   # 飞车检测
                'right_lift_off': bool,
                'any_lift_off': bool
            }
        """
        current_time = time.perf_counter()
        left_torque = abs(
            self.robot_data.tau_a_[self.robot_data.whole_joint_num -
                                   self.robot_data.motor_num + 3])  # 左膝力矩
        right_torque = abs(
            self.robot_data.tau_a_[self.robot_data.whole_joint_num -
                                   self.robot_data.motor_num + 9])  # 右膝力矩
        # 1. 更新当前峰值（带衰减）
        self._update_peak('left', left_torque)
        self._update_peak('right', right_torque)

        # 2. 触地检测
        is_lifted = self._detect_lifted(left_torque, right_torque, current_time)
        # 3. 飞车检测（基于峰值）
        left_lift = self._detect_lift_off('left', current_time)
        right_lift = self._detect_lift_off('right', current_time)

        return {
            'is_lifted': is_lifted,
            'left_lift_off': left_lift,
            'right_lift_off': right_lift,
            'any_lift_off': left_lift or right_lift
        }

    def _update_peak(self, side, torque):
        """更新当前峰值，带指数衰减或计数衰减"""
        # if side == 'left':
        #     if torque > self.left_current_peak:
        #         self.left_current_peak = torque
        #         self.left_peak_decay_counter = 0
        #     else:
        #         self.left_peak_decay_counter += 1
        #         if self.left_peak_decay_counter >= self.peak_decay_frames:
        #             self.left_current_peak *= 0.95  # 缓慢衰减
        # else:
        #     if torque > self.right_current_peak:
        #         self.right_current_peak = torque
        #         self.right_peak_decay_counter = 0
        #     else:
        #         self.right_peak_decay_counter += 1
        #         if self.right_peak_decay_counter >= self.peak_decay_frames:
        #             self.right_current_peak *= 0.95
        if side == 'left':
            self.left_current_peak = torque
        else:
            self.right_current_peak = torque

    def _detect_lifted(self, left_torque, right_torque, current_time):
        """
        基于力矩阈值和持续时间的触地检测。
        """
        left_lifted = (left_torque < self.torque_threshold)
        right_lifted = (right_torque < self.torque_threshold)
        is_lifted = left_lifted and right_lifted
        if is_lifted:
            if self.is_lifted_timer is None:
                # 首次低于阈值，记录开始时间
                self.is_lifted_timer = current_time
            else:
                # 检查持续时间
                if current_time - self.is_lifted_timer >= self.time_threshold:
                    return True
                else:
                    return False
        else:
            # 力矩高于阈值，重置状态
            self.is_lifted_timer = None
            return False

    def _detect_lift_off(self, side, current_time):
        """
        飞车检测：当前峰值是否远超历史峰值。
        同时将当前峰值记录到历史峰值队列中。
        """
        if side == 'left':
            peak = self.left_current_peak
            history = self.left_peak_history
        else:
            peak = self.right_current_peak
            history = self.right_peak_history

        # 将当前峰值加入历史队列（带有时间戳）
        history.append((current_time, peak))

        # 移除过期的历史数据
        while history and current_time - history[0][0] > self.history_window:
            history.popleft()

        # 如果没有足够的历史数据，不触发飞车
        if len(history) < 10:  # 至少10个采样点
            return False

        # 计算历史平均峰值（忽略当前峰值？不，历史包含当前，但影响不大）
        avg_peak = np.mean([p for _, p in history])
        

        # 飞车判定：当前峰值远大于历史平均峰值
        if avg_peak > 1e-6 and avg_peak > self.max_agv_peak_threshold * self.peak_ratio_threshold:
            return True

        return False

# =============== 使用示例 ===============
if __name__ == "__main__":
    # 创建检测器
    detector = ContactBasedLiftDetector(
        torque_threshold=50.0,
        time_threshold=0.1,
        history_window=5.0,
        peak_ratio_threshold=3.0,
        sample_rate=100.0
    )

    # 模拟力矩数据
    import time
    import random

    t = 0.0
    for i in range(1000):
        t += 0.01

        # 正常力矩：20~60 Nm
        left_torque = 30 + 10 * np.sin(t) + random.uniform(-5, 5)
        right_torque = 28 + 12 * np.sin(t + 0.5) + random.uniform(-5, 5)

        # 模拟飞车：第500帧时左膝突然出现200Nm
        if i == 500:
            left_torque = 200

        result = detector.update(left_torque, right_torque, current_time=t)

        if i % 50 == 0:
            pass
