"""
FSM State Implementations
Concrete implementations of different FSM states
"""
from typing import Dict

import numpy as np
import onnxruntime as ort

from xmigcs.FSM.fsm_base import FSMState, FSMStateName
from xmigcs.common.control_flag import FSMControlFlag
from xmigcs.common.robot_data import RobotData
import math
import onnxruntime
import os
import yaml
from scipy.spatial.transform import Rotation
from xmigcs.policy.swr.fsm_swr import FSMStateSWR


class FSMStateSW(FSMStateSWR):
    """SW站走策略状态实现"""
    def get_config_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "config", "sw.yaml")