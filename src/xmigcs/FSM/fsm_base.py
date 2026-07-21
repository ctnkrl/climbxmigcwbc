"""
Finite State Machine (FSM) Module
Python equivalent of the C++ FSM system
"""
from abc import ABC, abstractmethod
from enum import Enum

from xmigcs.common.robot_data import RobotData
from xmigcs.common.control_flag import FSMControlFlag


class FSMStateName(Enum):
    """FSM状态枚举"""
    STOP = 0  # 停止状态

    @classmethod
    def extend(cls, name: str, value: int):
        """动态添加或更新状态"""
        if name in cls.__members__:
            # 更新已存在的状态值
            member = cls.__members__[name]
            old_value = member.value
            member._value_ = value
            # 更新值映射
            del cls._value2member_map_[old_value]
            cls._value2member_map_[value] = member
        else:
            # 添加新状态
            member = object.__new__(cls)
            member._value_ = value
            member._name_ = name
            cls._member_map_[name] = member
            cls._value2member_map_[value] = member


class FSMState(ABC):
    """FSM状态抽象基类"""

    def __init__(self, robot_data: RobotData):
        self.robot_data_ = robot_data

    @abstractmethod
    def on_enter(self):
        """进入状态时的行为"""
        pass

    @abstractmethod
    def run(self, flag: FSMControlFlag):
        """运行状态的正常行为"""
        pass

    @abstractmethod
    def on_exit(self):
        """退出状态时的行为"""
        pass

    @abstractmethod
    def check_transition(self, *args, **kwargs):
        """检查状态转换"""
        pass

class RobotFSM(ABC):
    """机器人FSM抽象基类"""

    def __init__(self, robot_data: RobotData):
        self.robot_data_ = robot_data
        # self.disable_joints_ = False

    @abstractmethod
    def run_fsm(self, flag: FSMControlFlag):
        """运行FSM"""
        pass

    @abstractmethod
    def get_current_state(self) -> FSMStateName:
        """获取当前FSM状态"""
        pass

    @abstractmethod
    def get_current_state_class(self) -> FSMState:
        """获取当前FSM状态类"""
        pass