# Copyright (c) 2025 Xuxin @ 747302550. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np
from typing import Dict
import xmigcs.utils.mirror_info as mirror_info

class sym_tiangong_23dof_matrix_numpy:
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
            "command": (False, [False, True, True]),  # [vx,-vy,-wz]
            "joint_pos": (mirror_info.mirrored_indices, mirror_info.negative_flag),
            "joint_vel": (mirror_info.mirrored_indices, mirror_info.negative_flag),
            "action": (mirror_info.mirrored_indices, mirror_info.negative_flag),
        }

        self.policy_obs_mirror_matrix = self.get_mirror_matrix(policy_obs_terms).astype(np.float32)

    def get_mirror_matrix(self, obs_terms: Dict):
        total_mirror_matrix = []
        for key, rule_tuple in obs_terms.items():
            swapped_indices, negative_flag = rule_tuple
            mirror_matrix = self.mirror_rule_to_matrix(swapped_indices, negative_flag)
            total_mirror_matrix.append(mirror_matrix)

        # 手动构建块对角矩阵
        result = total_mirror_matrix[0]
        for mat in total_mirror_matrix[1:]:
            rows1, cols1 = result.shape
            rows2, cols2 = mat.shape
            new_mat = np.zeros((rows1 + rows2, cols1 + cols2), dtype=result.dtype)
            new_mat[:rows1, :cols1] = result
            new_mat[rows1:, cols1:] = mat
            result = new_mat
        return result

    @staticmethod
    def mirror_rule_to_matrix(swapped_indices: list, negative_flag: list):
        n = len(negative_flag)
        mirror_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            if swapped_indices:
                target_idx = swapped_indices[i]
            else:
                target_idx = i
            # 如果不需要取负，系数为1；需要取负，系数为-1
            coeff = -1.0 if negative_flag[i] else 1.0
            mirror_matrix[target_idx, i] = coeff
            mirror_matrix[i, target_idx] = coeff
        return mirror_matrix

    def mirror_policy_obs(self, obs: np.ndarray):
        # [batch, time, length] * [length, length]
        mirrored_obs = np.einsum("btl,lm->btm", obs, self.policy_obs_mirror_matrix)
        return mirrored_obs

    def mirror_action(self, action: np.ndarray):
        mirrored_obs = action @ self.dof_mirror_matrix
        return mirrored_obs

    def mirror_obs_history(self, obs: np.ndarray, history_length: int, mirror_func: callable):
        batch_num = obs.shape[0]
        obs = obs.reshape(batch_num, history_length, -1)  # [num_env, history, num_obs]
        mirrored_obs = mirror_func(obs)
        out = mirrored_obs.reshape(batch_num, -1)
        return out


if __name__ == "__main__":
    s1 = sym_tiangong_23dof_matrix_numpy()
    num_envs = 1000

    num_dofs = 23
    num_policy_obs = 9 + 3 * num_dofs
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
