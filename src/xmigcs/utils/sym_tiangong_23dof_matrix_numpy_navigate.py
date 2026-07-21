# Copyright (c) 2025 Xuxin @ 747302550. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np
import xmigcs.utils.mirror_info as mirror_info
from xmigcs.utils.sym_tiangong_23dof_matrix_numpy import sym_tiangong_23dof_matrix_numpy

class sym_tiangong_23dof_matrix_numpy_navigate(sym_tiangong_23dof_matrix_numpy):
    def __init__(self):
        self.policy_history = 10
        self.critic_history = 10

        dof_mirror_matrix = self.mirror_rule_to_matrix(mirror_info.mirrored_indices, negative_flag=mirror_info.negative_flag)
        self.dof_mirror_matrix = dof_mirror_matrix.astype(np.float32)

        # ang_vel * self.obs_scales.ang_vel, # 3
        # projected_gravity * self.obs_scales.projected_gravity, # 3
        # command * self.obs_scales.commands, # 3
        # joint_pos * self.obs_scales.joint_pos, # 23
        # joint_vel * self.obs_scales.joint_vel, # 23
        # action * self.obs_scales.actions, # 23
        policy_obs_terms = {
            "ang_vel": (False, [True, False, True]),  # [-wx,wy,-wz]
            "projected_gravity": (False, [False, True, False]),  # [gx,-gy,gz]
            "command": (False, [False, False, True, True]),  # [vel_ref, bcs_x, -bcs_y, -heading_b]
            "joint_pos": (mirror_info.mirrored_indices, mirror_info.negative_flag),
            "joint_vel": (mirror_info.mirrored_indices, mirror_info.negative_flag),
            "action": (mirror_info.mirrored_indices, mirror_info.negative_flag),
        }

        self.policy_obs_mirror_matrix = self.get_mirror_matrix(policy_obs_terms).astype(np.float32)


if __name__ == "__main__":
    s1 = sym_tiangong_23dof_matrix_numpy_navigate()
    num_envs = 1000

    num_dofs = 23
    num_policy_obs = 10 + 3 * num_dofs
    history_length = 10
    policy_obs = np.random.rand(num_envs, history_length, num_policy_obs)
    policy_obs = policy_obs.reshape(num_envs, -1)
    actions = np.random.rand(num_envs, num_dofs)

    # warmup
    m2 = s1.mirror_obs_history(policy_obs, history_length, s1.mirror_policy_obs)

    import time
    num_tests = 100

    time1 = 0
    for i in range(num_tests):
        policy_obs = np.random.rand(num_envs, history_length, num_policy_obs)
        policy_obs = policy_obs.reshape(num_envs, -1)

        start = time.time()
        m2 = s1.mirror_obs_history(policy_obs, history_length, s1.mirror_policy_obs)
        end = time.time()
        time1 += end - start

  # 0.05
