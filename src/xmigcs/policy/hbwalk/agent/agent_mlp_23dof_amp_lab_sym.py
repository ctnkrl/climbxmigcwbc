# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np
from xmigcs.policy.hbwalk.agent import agent_mlp_23dof_amp_lab
from xmigcs.utils.sym_tiangong_23dof_matrix_numpy import sym_tiangong_23dof_matrix_numpy
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

class agent_mlp_23dof_amp_lab_sym(agent_mlp_23dof_amp_lab):
    def __init__(self):
        super().__init__()
        self.inference_count = 0
        self.symmetry_module = sym_tiangong_23dof_matrix_numpy()
        self.symmetry_inference = True
        self.start_time = 0.

    def inference(self, obs_group):
        obs = self.build_observations(obs_group)
        obs_f = obs.flatten()[None, :].astype(np.float32)

        commands = obs_group["commands"]
        cmd_small = np.linalg.norm(commands) < 0.1
        if not cmd_small:
            self.start_time += self.dt
        else:
            self.start_time = 0.

        if self.symmetry_inference:
            if (self.inference_count % 2 == 0) and self.start_time > 2.0:
                logger.debug("inference: original")
                input_feed = {
                    "obs": obs_f,
                }
                actions = np.squeeze(self.onnx_session.run(["actions"], input_feed))
            else:
                logger.debug("inference: sym")
                # mirror
                obs_m = self.symmetry_module.mirror_obs_history(obs_f, self.symmetry_module.policy_history, self.symmetry_module.mirror_policy_obs)
                input_feed = {
                    "obs": obs_m,
                }
                actions_m = np.squeeze(self.onnx_session.run(["actions"], input_feed))
                actions = self.symmetry_module.mirror_action(actions_m)
        else:
            input_feed = {
                "obs": obs_f,
            }
            actions = np.squeeze(self.onnx_session.run(["actions"], input_feed))

        actions = np.clip(actions, -self.clip_action, self.clip_action)

        self.last_actions_buf = actions

        dof_pos_target_urdf = actions * self.action_scale + self.default_dof_pos

        # dof_pos_target_urdf = self.exp_filter.filter(dof_pos_target_urdf)
        self.inference_count += 1
        return dof_pos_target_urdf

    def reset(self, first_obs_group=None):
        super().reset(first_obs_group)
        self.inference_count = 0
        self.start_time = 0.

if __name__ == "__main__":
    a = agent_mlp_23dof_amp_lab_sym()
    np.set_printoptions(formatter={"float": "{:.2f}".format})
    for i in range(100):
        logger.debug(a.inference(a.default_obs_group))
