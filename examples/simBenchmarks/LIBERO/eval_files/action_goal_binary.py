"""Synchronize the bounded native OSC reference pose before converting goals."""
import numpy as np
from scipy.spatial.transform import Rotation
from starVLA.dataloader.libero_action_contract import OSC_SCALE


def controller_reference(env):
    robot = env.robots[0]
    c = robot.controller
    if not bool(getattr(c, 'use_delta', False)) or not bool(getattr(c, 'use_ori', False)) or c.impedance_mode != 'fixed':
        raise ValueError('Goal-binary v1 requires the official fixed-impedance delta OSC_POSE controller')
    if not (np.allclose(c.input_min, -1, rtol=0, atol=1e-12) and np.allclose(c.input_max, 1, rtol=0, atol=1e-12)
            and np.allclose(c.output_min, -OSC_SCALE, rtol=0, atol=1e-12)
            and np.allclose(c.output_max, OSC_SCALE, rtol=0, atol=1e-12)):
        raise ValueError('OSC scaling does not match the versioned training contract')
    # forward() changes no physical time; it matches env.step's pre-action forward.
    env.sim.forward()
    c.update(force=True)
    body = np.asarray(env.sim.data.get_body_xmat(robot.robot_model.eef_name), dtype=np.float64).reshape(3, 3)
    body_to_site = body.T @ np.asarray(c.ee_ori_mat, dtype=np.float64)
    if c.goal_ori is None:
        raise ValueError('Native controller orientation goal is not initialized')
    goal_body = np.asarray(c.goal_ori, dtype=np.float64) @ body_to_site.T
    return dict(controller_eef_pos=np.asarray(c.ee_pos, dtype=np.float64).copy(),
                controller_eef_quat=Rotation.from_matrix(body).as_quat(),
                controller_goal_quat=Rotation.from_matrix(goal_body).as_quat())
