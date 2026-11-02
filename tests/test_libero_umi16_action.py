"""Checks the LIBERO observation-window adapter against UMI anchor semantics."""

import sys

import numpy as np

from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP


libero_config = ROBOT_TYPE_CONFIG_MAP["libero_franka_umi16"]
adapter = sys.modules[type(libero_config).__module__].libero_state_window_to_umi16


def test_anchor_relative_pose_and_mask():
    state = np.array([
        [0, 0, 0, 0, 0, np.pi / 2, 0.04, -0.04],
        [0, 1, 0, 0, 0, np.pi, 0.03, -0.03],
        [0, 2, 0, 0, 0, np.pi, 0.01, -0.01],
    ], dtype=np.float64)
    action, mask = adapter(state)
    assert action.shape == mask.shape == (2, 16)
    np.testing.assert_allclose(action[:, :3], [[1, 0, 0], [2, 0, 0]], atol=1e-6)
    np.testing.assert_allclose(action[:, 3:7],
                               [[0, 0, 2**-0.5, 2**-0.5]] * 2, atol=1e-6)
    np.testing.assert_allclose(action[:, 7], [0.06, 0.02], atol=1e-6)
    assert mask[:, :8].all() and not mask[:, 8:].any()
    assert not action[:, 8:].any()


def test_identity_rotation_and_invalid_values():
    state = np.zeros((3, 8), dtype=np.float64)
    action, _ = adapter(state)
    np.testing.assert_allclose(action[:, 3:7], [[0, 0, 0, 1]] * 2)
    state[1, 0] = np.nan
    try:
        adapter(state)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN state should be rejected")


if __name__ == "__main__":
    test_anchor_relative_pose_and_mask()
    test_identity_rotation_and_invalid_values()
    print("LIBERO UMI16 adapter checks passed")
