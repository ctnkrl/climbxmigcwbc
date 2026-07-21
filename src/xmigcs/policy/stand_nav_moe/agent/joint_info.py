import numpy as np
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

# NOTE: stand策略观测上肢 但是只控制下肢 所以obs和control不一样
agent_obs_names_stand = [
    "hip_pitch_l_joint", "hip_pitch_r_joint", "waist_yaw_joint",
    "hip_roll_l_joint", "hip_roll_r_joint", "waist_roll_joint",
    "hip_yaw_l_joint", "hip_yaw_r_joint", "waist_pitch_joint",
    "knee_pitch_l_joint", "knee_pitch_r_joint",
    "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",
    "ankle_pitch_l_joint", "ankle_pitch_r_joint",
    "shoulder_roll_l_joint", "shoulder_roll_r_joint",
    "ankle_roll_l_joint", "ankle_roll_r_joint",
    "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_l_joint", "elbow_pitch_r_joint",
]
agent_control_names_stand = [
    "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

# navigate obs和action一致 (15-dof Isaac Lab交错顺序)
agent_obs_names_navigate = [
    "hip_pitch_l_joint", "hip_pitch_r_joint",
    "waist_yaw_joint",
    "hip_roll_l_joint", "hip_roll_r_joint",
    "waist_roll_joint",
    "hip_yaw_l_joint", "hip_yaw_r_joint",
    "waist_pitch_joint",
    "knee_pitch_l_joint", "knee_pitch_r_joint",
    "ankle_pitch_l_joint", "ankle_pitch_r_joint",
    "ankle_roll_l_joint", "ankle_roll_r_joint",
]
agent_control_names_navigate = agent_obs_names_navigate

# xmigcs内部顺序
env_names = [
    "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint", "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint", "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint"
]

# default pos
default_dof_pos_dict_stand = {
    "hip_yaw_l_joint": -0.0,
    "hip_roll_l_joint": 0.0,
    "hip_pitch_l_joint": -0.4,
    "knee_pitch_l_joint": 0.8,
    "ankle_pitch_l_joint": -0.4,
    "ankle_roll_l_joint": 0.0,
    "hip_yaw_r_joint": -0.0,
    "hip_roll_r_joint": 0.0,
    "hip_pitch_r_joint": -0.4,
    "knee_pitch_r_joint": 0.8,
    "ankle_pitch_r_joint": -0.4,
    "ankle_roll_r_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "shoulder_pitch_l_joint": 0.35,
    "shoulder_roll_l_joint": 0.18,
    "shoulder_yaw_l_joint": 0.0,
    "elbow_pitch_l_joint": -0.87,
    "shoulder_pitch_r_joint": 0.35,
    "shoulder_roll_r_joint": -0.18,
    "shoulder_yaw_r_joint": 0.0,
    "elbow_pitch_r_joint": -0.87,
}
default_dof_pos_dict_navigate = {
    "hip_yaw_l_joint": -0.0,
    "hip_roll_l_joint": 0.0,
    "hip_pitch_l_joint": -0.179,
    "knee_pitch_l_joint": 0.323,
    "ankle_pitch_l_joint": -0.144,
    "ankle_roll_l_joint": 0.0,
    "hip_yaw_r_joint": -0.0,
    "hip_roll_r_joint": 0.0,
    "hip_pitch_r_joint": -0.179,
    "knee_pitch_r_joint": 0.323,
    "ankle_pitch_r_joint": -0.144,
    "ankle_roll_r_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
}


env_to_agent_obs_stand = [env_names.index(name) for name in agent_obs_names_stand]
env_to_agent_control_stand = [env_names.index(name) for name in agent_control_names_stand]
env_to_agent_obs_navigate = [env_names.index(name) for name in agent_obs_names_navigate]
env_to_agent_control_navigate = env_to_agent_obs_navigate

default_dof_pos_obs_stand = [default_dof_pos_dict_stand[name] for name in agent_obs_names_stand]
default_dof_pos_control_stand = [default_dof_pos_dict_stand[name] for name in agent_control_names_stand]
default_dof_pos_obs_navigate = [default_dof_pos_dict_navigate[name] for name in agent_obs_names_navigate]
default_dof_pos_control_navigate = default_dof_pos_obs_navigate

if __name__ == "__main__":
    logger.debug(env_to_agent_obs_stand)
    logger.debug(env_to_agent_control_stand)
    logger.debug(env_to_agent_obs_navigate)
    logger.debug(env_to_agent_control_navigate)

    logger.debug(default_dof_pos_obs_stand)
    logger.debug(default_dof_pos_control_stand)
    logger.debug(default_dof_pos_obs_navigate)
    logger.debug(default_dof_pos_control_navigate)
