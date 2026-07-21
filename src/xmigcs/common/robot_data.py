"""
Robot Data Structure
Python equivalent of the C++ RobotData class
"""
import numpy as np
from scipy.spatial.transform import Rotation
import xlog
import copy
import threading
import time


class RobotData:
    """机器人状态数据结构"""

    def __init__(self, motor_num: int = 31, whole_joint_num: int = 37):
        self.motor_num = motor_num
        self.whole_joint_num = whole_joint_num

        # Joint states (actual)
        self.q_a_ = np.zeros(whole_joint_num)  # Joint positions
        self.q_dot_a_ = np.zeros(whole_joint_num)  # Joint velocities
        self.tau_a_ = np.zeros(whole_joint_num)  # Joint torques
        self.temperature_a = np.zeros(motor_num)  # Joint temperatures
        self.q_a_last = np.zeros(whole_joint_num)  # 上一时刻关节位置
        self.qdot_a_last = np.zeros(whole_joint_num)  # 上一时刻关节速度
        self.tor_a_last = np.zeros(whole_joint_num)  # 上一时刻关节力矩

        # Joint commands (desired)
        self.q_d_ = np.zeros(whole_joint_num)  # Desired joint positions
        self.q_d_s_ = np.zeros(
            whole_joint_num)  # Desired serial joint positions
        self.q_dot_d_ = np.zeros(whole_joint_num)  # Desired joint velocities
        self.tau_d_ = np.zeros(whole_joint_num)  # Desired joint torques

        # Control gains
        self.joint_kp_p_ = np.zeros(motor_num)  # Proportional gains
        self.joint_kd_p_ = np.zeros(motor_num)  # Derivative gains

        self.joint_kp_s_ = np.zeros(motor_num)  # Serial proportional gains
        self.joint_kd_s_ = np.zeros(motor_num)  # Serial derivative gains

        # IMU data: [yaw, pitch, roll, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
        self.imu_data_ = np.zeros(13)

        # Timing
        self.time_now_ = 0.0
        self.control_step_ = 0

        # Configuration
        self.config_file_ = ""

        # control cmd
        # walk command
        self.walk_cmd_ = np.zeros(3)  # x_speed, y_speed, yaw_speed
        # walk height command
        self.walk_height_cmd_ = 0.0
        # floating base command
        self.floating_base_cmd_ = np.zeros(
            4)  # waist_roll, waist_pitch, waist_yaw, height_cmd
        # footpoint command
        self.footpoint_cmd_ = np.array([False, 0, 0, 0],
                                       dtype=object)  # footflag,x,y,yaw
        # navigation or stand flag
        self.nav_stand_mode = 1  # 0=stand, 1=navigate
        # 状态切换时间
        self.trans_start_time = 0.0
        # 上一次状态的最后命令
        self.q_d_s_last_state = np.zeros(whole_joint_num)

        # 诊断信息: 电量，关节温度等
        self.diagnostic_info = {'battery_level': 100.0, 'joint_temperatures': self.temperature_a, 'motor_status': {}}

        # Terrain scan cache for policies that consume PointCloud2-derived observations.
        self._terrain_scan_lock = threading.Lock()
        self.terrain_scan_dim = (33, 21, 3)
        self.terrain_scan_size = int(np.prod(self.terrain_scan_dim))
        self.terrain_scan_ = np.zeros(self.terrain_scan_size, dtype=np.float32)
        self.terrain_scan_valid_ = False
        self.terrain_scan_stamp_ = 0.0

        # Runtime GMR reference cache. This is optional: policies that use
        # embedded references never read it, and realtime policies can fail fast
        # when the cache has not received /gmr_info yet.
        self._gmr_ref_lock = threading.Lock()
        self.gmr_ref_joint_dim = 0
        self.gmr_ref_body_dim = 0
        self.gmr_ref_valid_ = False
        self.gmr_ref_stamp_ = 0.0
        self.gmr_ref_seq_ = 0
        self.ref_joint_pos = None
        self.ref_joint_vel = None
        self.ref_body_pos_w = None
        self.ref_body_quat_w = None
        self.ref_body_lin_vel_w = None
        self.ref_body_ang_vel_w = None

    def copy_from(self, other: 'RobotData'):
        """从另一个RobotData对象复制数据"""
        self.q_a_[:] = other.q_a_[:]
        self.q_dot_a_[:] = other.q_dot_a_[:]
        self.tau_a_[:] = other.tau_a_[:]
        self.q_d_[:] = other.q_d_[:]
        self.q_dot_d_[:] = other.q_dot_d_[:]
        self.tau_d_[:] = other.tau_d_[:]
        self.joint_kp_p_[:] = other.joint_kp_p_[:]
        self.joint_kd_p_[:] = other.joint_kd_p_[:]
        self.imu_data_[:] = other.imu_data_[:]
        self.time_now_ = other.time_now_
        self.control_step_ = other.control_step_
        self.config_file_ = other.config_file_
        self.walk_cmd_[:] = other.walk_cmd_[:]
        self.floating_base_cmd_[:] = other.floating_base_cmd_[:]
        if hasattr(other, "get_terrain_scan"):
            other_dim = getattr(other, "terrain_scan_dim", self.terrain_scan_dim)
            self.configure_terrain_scan(other_dim)
            other_scan = other.get_terrain_scan(self.terrain_scan_size)
            with self._terrain_scan_lock:
                self.terrain_scan_[:] = other_scan
                self.terrain_scan_valid_ = getattr(other, "terrain_scan_valid_", False)
                self.terrain_scan_stamp_ = getattr(other, "terrain_scan_stamp_", 0.0)
        if hasattr(other, "get_gmr_reference"):
            ref = other.get_gmr_reference()
            if ref is not None:
                self.set_gmr_reference(
                    ref["joint_pos"],
                    ref["body_pos_w"],
                    ref["body_quat_w"],
                    stamp=ref["stamp"],
                    joint_vel=ref["joint_vel"],
                    body_lin_vel_w=ref["body_lin_vel_w"],
                    body_ang_vel_w=ref["body_ang_vel_w"],
                )

    def configure_terrain_scan(self, scan_dim):
        """Configure cached terrain scan dimensions."""
        dims = tuple(int(x) for x in scan_dim)
        if len(dims) == 2:
            dims = (dims[0], dims[1], 3)
        if len(dims) != 3:
            raise ValueError(f"terrain scan_dim must have 2 or 3 entries, got {scan_dim}")
        size = int(np.prod(dims))
        if size <= 0:
            raise ValueError(f"terrain scan_dim must be positive, got {scan_dim}")
        with self._terrain_scan_lock:
            if size != self.terrain_scan_size:
                self.terrain_scan_ = np.zeros(size, dtype=np.float32)
                self.terrain_scan_valid_ = False
            self.terrain_scan_dim = dims
            self.terrain_scan_size = size

    def set_terrain_scan(self, scan, scan_dim=None, valid=True, stamp=0.0):
        """Cache flattened xyz terrain scan in policy observation order."""
        if scan_dim is not None:
            self.configure_terrain_scan(scan_dim)
        arr = np.asarray(scan, dtype=np.float32).reshape(-1)
        with self._terrain_scan_lock:
            if arr.size != self.terrain_scan_size:
                padded = np.zeros(self.terrain_scan_size, dtype=np.float32)
                n = min(arr.size, self.terrain_scan_size)
                padded[:n] = arr[:n]
                arr = padded
            else:
                arr = arr.astype(np.float32, copy=True)
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.terrain_scan_ = arr
            self.terrain_scan_valid_ = bool(valid)
            self.terrain_scan_stamp_ = float(stamp)

    def update_terrain_scan(self, scan: np.ndarray, stamp: float = 0.0) -> None:
        """Compatibility wrapper for callers using the older cache method."""
        self.set_terrain_scan(scan, valid=True, stamp=stamp)

    def get_terrain_scan(self, scan_size=None) -> np.ndarray:
        """Return a flattened terrain scan, padded/truncated if requested."""
        with self._terrain_scan_lock:
            if scan_size is None or int(scan_size) == self.terrain_scan_size:
                return self.terrain_scan_.astype(np.float32, copy=True)
            scan_size = int(scan_size)
            result = np.zeros(scan_size, dtype=np.float32)
            copy_size = min(scan_size, self.terrain_scan_.size)
            if copy_size > 0:
                result[:copy_size] = self.terrain_scan_[:copy_size]
            return result

    def configure_gmr_reference(self, joint_dim: int, body_dim: int) -> None:
        """Configure expected realtime GMR reference dimensions."""
        joint_dim = int(joint_dim)
        body_dim = int(body_dim)
        if joint_dim <= 0 or body_dim <= 0:
            raise ValueError(
                f"gmr reference dims must be positive, got joints={joint_dim}, bodies={body_dim}"
            )
        with self._gmr_ref_lock:
            if joint_dim != self.gmr_ref_joint_dim or body_dim != self.gmr_ref_body_dim:
                self.gmr_ref_joint_dim = joint_dim
                self.gmr_ref_body_dim = body_dim
                self.gmr_ref_valid_ = False
                self.ref_joint_pos = np.zeros(joint_dim, dtype=np.float32)
                self.ref_joint_vel = np.zeros(joint_dim, dtype=np.float32)
                self.ref_body_pos_w = np.zeros((body_dim, 3), dtype=np.float32)
                self.ref_body_quat_w = np.zeros((body_dim, 4), dtype=np.float32)
                self.ref_body_lin_vel_w = np.zeros((body_dim, 3), dtype=np.float32)
                self.ref_body_ang_vel_w = np.zeros((body_dim, 3), dtype=np.float32)

    def set_gmr_reference(
        self,
        joint_pos,
        body_pos_w,
        body_quat_w,
        stamp: float = 0.0,
        joint_vel=None,
        body_lin_vel_w=None,
        body_ang_vel_w=None,
    ) -> None:
        """Cache one realtime GMR frame and finite-difference missing velocities."""
        joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        body_pos_w = np.asarray(body_pos_w, dtype=np.float32).reshape(-1, 3)
        body_quat_w = np.asarray(body_quat_w, dtype=np.float32).reshape(-1, 4)
        if joint_pos.size == 0 or body_pos_w.shape[0] == 0:
            raise ValueError("empty realtime GMR reference")
        if body_quat_w.shape[0] != body_pos_w.shape[0]:
            raise ValueError(
                f"body_pos/body_quat size mismatch: {body_pos_w.shape[0]} vs {body_quat_w.shape[0]}"
            )

        self.configure_gmr_reference(joint_pos.size, body_pos_w.shape[0])
        stamp = float(stamp if stamp is not None else 0.0)
        if stamp <= 0.0:
            stamp = time.perf_counter()

        body_quat_w = self._normalize_quat_wxyz(body_quat_w)
        with self._gmr_ref_lock:
            prev_joint_pos = None if not self.gmr_ref_valid_ else self.ref_joint_pos.copy()
            prev_body_pos = None if not self.gmr_ref_valid_ else self.ref_body_pos_w.copy()
            prev_body_quat = None if not self.gmr_ref_valid_ else self.ref_body_quat_w.copy()
            prev_stamp = self.gmr_ref_stamp_

            if joint_vel is None:
                if prev_joint_pos is not None and stamp > prev_stamp:
                    dt = max(stamp - prev_stamp, 1.0e-3)
                    joint_vel_arr = (joint_pos - prev_joint_pos) / dt
                else:
                    joint_vel_arr = np.zeros_like(joint_pos)
            else:
                joint_vel_arr = np.asarray(joint_vel, dtype=np.float32).reshape(-1)
                if joint_vel_arr.size != joint_pos.size:
                    joint_vel_arr = self._pad_or_truncate(joint_vel_arr, joint_pos.size)

            if body_lin_vel_w is None:
                if prev_body_pos is not None and stamp > prev_stamp:
                    dt = max(stamp - prev_stamp, 1.0e-3)
                    body_lin_vel_arr = (body_pos_w - prev_body_pos) / dt
                else:
                    body_lin_vel_arr = np.zeros_like(body_pos_w)
            else:
                body_lin_vel_arr = np.asarray(body_lin_vel_w, dtype=np.float32).reshape(-1, 3)
                body_lin_vel_arr = self._pad_or_truncate_rows(body_lin_vel_arr, body_pos_w.shape[0], 3)

            if body_ang_vel_w is None:
                if prev_body_quat is not None and stamp > prev_stamp:
                    dt = max(stamp - prev_stamp, 1.0e-3)
                    body_ang_vel_arr = self._quat_angular_velocity_wxyz(prev_body_quat, body_quat_w, dt)
                else:
                    body_ang_vel_arr = np.zeros_like(body_pos_w)
            else:
                body_ang_vel_arr = np.asarray(body_ang_vel_w, dtype=np.float32).reshape(-1, 3)
                body_ang_vel_arr = self._pad_or_truncate_rows(body_ang_vel_arr, body_pos_w.shape[0], 3)

            np.nan_to_num(joint_pos, copy=False)
            np.nan_to_num(joint_vel_arr, copy=False)
            np.nan_to_num(body_pos_w, copy=False)
            np.nan_to_num(body_quat_w, copy=False)
            np.nan_to_num(body_lin_vel_arr, copy=False)
            np.nan_to_num(body_ang_vel_arr, copy=False)

            self.ref_joint_pos = joint_pos.copy()
            self.ref_joint_vel = joint_vel_arr.astype(np.float32, copy=True)
            self.ref_body_pos_w = body_pos_w.copy()
            self.ref_body_quat_w = body_quat_w.copy()
            self.ref_body_lin_vel_w = body_lin_vel_arr.astype(np.float32, copy=True)
            self.ref_body_ang_vel_w = body_ang_vel_arr.astype(np.float32, copy=True)
            self.gmr_ref_stamp_ = stamp
            self.gmr_ref_seq_ += 1
            self.gmr_ref_valid_ = True

    def get_gmr_reference(self, max_age_s: float | None = None):
        """Return a thread-safe copy of the latest realtime GMR frame."""
        with self._gmr_ref_lock:
            if not self.gmr_ref_valid_:
                return None
            now = time.perf_counter()
            if max_age_s is not None and self.gmr_ref_stamp_ > 0.0:
                if now - self.gmr_ref_stamp_ > float(max_age_s):
                    return None
            return {
                "joint_pos": self.ref_joint_pos.copy(),
                "joint_vel": self.ref_joint_vel.copy(),
                "body_pos_w": self.ref_body_pos_w.copy(),
                "body_quat_w": self.ref_body_quat_w.copy(),
                "body_lin_vel_w": self.ref_body_lin_vel_w.copy(),
                "body_ang_vel_w": self.ref_body_ang_vel_w.copy(),
                "stamp": float(self.gmr_ref_stamp_),
                "seq": int(self.gmr_ref_seq_),
            }

    @staticmethod
    def _pad_or_truncate(arr: np.ndarray, size: int) -> np.ndarray:
        out = np.zeros(size, dtype=np.float32)
        n = min(arr.size, size)
        if n > 0:
            out[:n] = arr[:n]
        return out

    @staticmethod
    def _pad_or_truncate_rows(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
        out = np.zeros((rows, cols), dtype=np.float32)
        copy_rows = min(arr.shape[0], rows)
        copy_cols = min(arr.shape[1], cols)
        if copy_rows > 0 and copy_cols > 0:
            out[:copy_rows, :copy_cols] = arr[:copy_rows, :copy_cols]
        return out

    @staticmethod
    def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32)
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        safe = norm > 1.0e-6
        normalized = np.zeros_like(quat, dtype=np.float32)
        normalized[..., 0] = 1.0
        np.divide(quat, np.where(safe, norm, 1.0), out=normalized, where=safe)
        return normalized.astype(np.float32, copy=False)

    @staticmethod
    def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return np.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            axis=-1,
        ).astype(np.float32)

    @classmethod
    def _quat_angular_velocity_wxyz(cls, prev_quat: np.ndarray, curr_quat: np.ndarray, dt: float) -> np.ndarray:
        prev_quat = cls._normalize_quat_wxyz(prev_quat)
        curr_quat = cls._normalize_quat_wxyz(curr_quat)
        inv_prev = prev_quat.copy()
        inv_prev[..., 1:] *= -1.0
        delta = cls._quat_mul_wxyz(curr_quat, inv_prev)
        delta = cls._normalize_quat_wxyz(delta)
        sign = np.where(delta[..., :1] < 0.0, -1.0, 1.0)
        delta *= sign
        vec = delta[..., 1:]
        vec_norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        angle = 2.0 * np.arctan2(vec_norm, np.clip(delta[..., :1], -1.0, 1.0))
        axis = np.divide(vec, np.where(vec_norm > 1.0e-6, vec_norm, 1.0))
        return (axis * angle / max(float(dt), 1.0e-3)).astype(np.float32)

    def get_joint_pos(self) -> np.ndarray:
        joint_start_idx = self.whole_joint_num - self.motor_num
        joint_pos = self.q_a_[joint_start_idx:].astype(np.float32)
        return joint_pos

    def get_desired_joint_pos(self) -> np.ndarray:
        joint_start_idx = self.whole_joint_num - self.motor_num
        desired_joint_pos = self.q_d_[joint_start_idx:].astype(np.float32)
        return desired_joint_pos

    def get_serial_joint_pos_desired(self) -> np.ndarray:
        joint_start_idx = self.whole_joint_num - self.motor_num
        joint_pos_desired = self.q_d_s_[joint_start_idx:].astype(np.float32)
        return joint_pos_desired

    def get_joint_vel(self) -> np.ndarray:
        joint_start_idx = self.whole_joint_num - self.motor_num
        joint_vel = self.q_dot_a_[joint_start_idx:].astype(np.float32)
        return joint_vel

    def get_angular_velocity(self) -> np.ndarray:
        omega_xyz = np.array(
            [self.imu_data_[3], self.imu_data_[4], self.imu_data_[5]],
            dtype=np.float32)
        return omega_xyz

    def get_robot_quat(self):
        rpy = np.array(
            [
                self.imu_data_[2],  # roll
                self.imu_data_[1],  # pitch
                self.imu_data_[0]  # yaw
            ],
            dtype=np.float32) * 1.0
        robot_quat_wxyz = self.euler_to_quaternion_scipy(
            rpy[0], rpy[1], rpy[2])
        return robot_quat_wxyz

    def euler_to_quaternion_scipy(self, roll, pitch, yaw, degrees=False):
        """
        使用SciPy进行欧拉角转四元数
        参数:
            roll: 绕x轴的旋转角度
            pitch: 绕y轴的旋转角度  
            yaw: 绕z轴的旋转角度
            degrees: 输入角度是否为度，默认为弧度
        返回:
            [w, x, y, z]: 四元数分量 (w为实部)
        """
        # 创建旋转对象 (顺序: 'xyz' 对应 roll, pitch, yaw)
        rotation = Rotation.from_euler('xyz', [roll, pitch, yaw],
                                       degrees=degrees)

        # 转换为四元数 (顺序: [x, y, z, w])
        quaternion = rotation.as_quat()

        return [quaternion[3], quaternion[0], quaternion[1],
                quaternion[2]]  # 返回 w, x, y, z

    def get_waist_yrp(self) -> np.ndarray:
        joint_pos = self.get_joint_pos()
        waist_yaw, waist_roll, waist_pitch = joint_pos[12], joint_pos[
            13], joint_pos[14]
        return np.array([waist_yaw, waist_roll, waist_pitch], dtype=np.float32)

    def get_base_linear_acceleration(self) -> np.ndarray:
        lin_acc = np.array(
            [self.imu_data_[6], self.imu_data_[7], self.imu_data_[8]],
            dtype=np.float32)
        return lin_acc

    def get_project_gravity(self) -> np.ndarray:
        """根据机器人姿态重力投影(待完善)

        Args:
            None
        """
        robot_quat_wxyz = self.get_robot_quat()
        robot_quat_xyzw = np.array([
            robot_quat_wxyz[1], robot_quat_wxyz[2], robot_quat_wxyz[3],
            robot_quat_wxyz[0]
        ])
        g = np.array([0., 0., -1.])
        projected_gravity = self.quat_rotate_inverse_numpy(robot_quat_xyzw, g)
        return projected_gravity

    def quat_rotate_inverse_numpy(self, q, v):
        """
        q: [x, y, z, w], shape=(4)\\
        v: [x, y, z], shape=(3)
        """
        q_w = q[3]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def get_walk_cmd(self) -> np.ndarray:
        """获取行走命令: [x_speed, y_speed, yaw_speed]"""
        return self.walk_cmd_.copy()

    def get_floating_base_cmd(self) -> np.ndarray:
        """获取浮动基座命令: [x, y, z, roll, pitch, yaw]"""
        return self.floating_base_cmd_.copy()

    def get_footpoint_cmd(self) -> np.ndarray:
        """获取足点命令: [footflag, x, y, yaw]"""
        return self.footpoint_cmd_.copy()

    def record_transition(self):
        """设置转换开始时间
        """
        self.trans_start_time = self.time_now_
        self.q_d_s_last_state = self.q_d_s_.copy()
        self.joint_kp_s_last_state = self.joint_kp_s_.copy()
        self.joint_kd_s_last_state = self.joint_kd_s_.copy()

    def get_last_state_serial_qd(self) -> np.ndarray:
        """获取上一个状态输出的串联关节指令"""
        return self.q_d_s_last_state[self.whole_joint_num - self.motor_num:].astype(np.float32)
    
    def get_last_state_serial_kp(self) -> np.ndarray:
        """获取上一个状态输出的串联关节刚度"""
        return self.joint_kp_s_last_state.copy()
    
    def get_last_state_serial_kd(self) -> np.ndarray:
        """获取上一个状态输出的串联关节阻尼"""
        return self.joint_kd_s_last_state.copy()
    
    def get_joint_kp(self) -> np.ndarray:
        """获取关节刚度"""
        return self.joint_kp_p_.copy()
    
    def get_joint_kd(self) -> np.ndarray:
        """获取关节阻尼"""
        return self.joint_kd_p_.copy()
    
    def get_nav_stand_mode(self) -> int:
        """获取导航或站立模式 0=stand, 1=navigate"""
        return self.nav_stand_mode
    
    def set_walk_cmd(self, walk_cmd: np.ndarray, xyyaw_speed_limits: dict, trans_flag: bool):
        """设置行走命令: [x_speed, y_speed, yaw_speed]"""
        if trans_flag:
            # xlog.warning("trans_flag is True, walk_cmd is not updated")
            self.walk_cmd_[:] = np.zeros(3)
            return
        self.walk_cmd_[:] = walk_cmd[:]
        self.walk_cmd_[0] = np.clip(self.walk_cmd_[0] + xyyaw_speed_limits["x_command_offset"], -xyyaw_speed_limits["max_x_minus"], xyyaw_speed_limits["max_x_plus"])
        self.walk_cmd_[1] = np.clip(self.walk_cmd_[1] + xyyaw_speed_limits["y_command_offset"], -xyyaw_speed_limits["max_y"], xyyaw_speed_limits["max_y"])
        self.walk_cmd_[2] = np.clip(self.walk_cmd_[2] + xyyaw_speed_limits["yaw_command_offset"], -xyyaw_speed_limits["max_yaw"], xyyaw_speed_limits["max_yaw"])
    
    def set_walk_height_cmd(self, walk_height_cmd: float, trans_flag: bool):
        """设置行走高度命令"""
        if trans_flag:
            # xlog.warning("trans_flag is True, walk_height_cmd is not updated")
            return
        self.walk_height_cmd_ = walk_height_cmd
    
    def set_floating_base_cmd(self, floating_base_cmd: np.ndarray, trans_flag: bool):
        """设置浮动基座命令: [roll, pitch, yaw, height]"""
        if trans_flag:
            # xlog.warning("trans_flag is True, floating_base_cmd is not updated")
            return
        self.floating_base_cmd_[:] = floating_base_cmd[:]
    
    def set_footpoint_cmd(self, footpoint_cmd: np.ndarray, trans_flag: bool):
        """设置足点命令: [footflag, relative_x, relative_y, relative_yaw]"""
        if trans_flag:
            # xlog.warning("trans_flag is True, footpoint_cmd is not updated")
            return
        self.footpoint_cmd_[:] = footpoint_cmd[:]
   
    def set_moe_state_command(self, moe_state_command: dict):
        """设置MOE状态命令"""
        for attr, value in moe_state_command.items():
            if not hasattr(self, attr):
                xlog.warning(
                    f"robot_data has no attribute '{attr}', skip moe mapping")
                continue
            setattr(self, attr, value)
