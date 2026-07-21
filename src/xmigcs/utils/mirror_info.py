import numpy as np
joint_names = [  # 注意: isaaclab顺序和urdf顺序不同
    "hip_pitch_l_joint", "hip_pitch_r_joint",
    "waist_yaw_joint",
    "hip_roll_l_joint", "hip_roll_r_joint",
    "waist_roll_joint",
    "hip_yaw_l_joint", "hip_yaw_r_joint",
    "waist_pitch_joint",
    "knee_pitch_l_joint", "knee_pitch_r_joint",
    "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",
    "ankle_pitch_l_joint", "ankle_pitch_r_joint",
    "shoulder_roll_l_joint", "shoulder_roll_r_joint",
    "ankle_roll_l_joint", "ankle_roll_r_joint",
    "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_l_joint", "elbow_pitch_r_joint",
]
num_dofs = len(joint_names)

mirrored_names = [  # left <-> right
    "hip_pitch_r_joint", "hip_pitch_l_joint",
    "waist_yaw_joint",
    "hip_roll_r_joint", "hip_roll_l_joint",
    "waist_roll_joint",
    "hip_yaw_r_joint", "hip_yaw_l_joint",
    "waist_pitch_joint",
    "knee_pitch_r_joint", "knee_pitch_l_joint",
    "shoulder_pitch_r_joint", "shoulder_pitch_l_joint",
    "ankle_pitch_r_joint", "ankle_pitch_l_joint",
    "shoulder_roll_r_joint", "shoulder_roll_l_joint",
    "ankle_roll_r_joint", "ankle_roll_l_joint",
    "shoulder_yaw_r_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_r_joint", "elbow_pitch_l_joint",
]
mirror_negative_dofs = [
    "hip_roll_l_joint", "hip_roll_r_joint",
    "hip_yaw_l_joint", "hip_yaw_r_joint",
    "ankle_roll_l_joint", "ankle_roll_r_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "shoulder_roll_l_joint", "shoulder_roll_r_joint",
    "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
]
mirror_origin_indices = np.arange(num_dofs).tolist()
mirrored_indices = [joint_names.index(name) for name in mirrored_names]
negative_dof_indices = [joint_names.index(name) for name in mirror_negative_dofs]
negative_dof_indices.sort()
negative_flag = [True if name in mirror_negative_dofs else False for name in joint_names]

if __name__ == "__main__":
    pass
