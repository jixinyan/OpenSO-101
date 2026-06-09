# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Contact-confirmed grasp predicate and reward for the pick-place task.

The grasp predicate is True for an env only when BOTH SO-101 jaw bodies
report nonzero filtered contact force against the cube prim simultaneously.
This is the ground-truth grasp signal: the policy cannot satisfy it by
closing the gripper in mid-air or by pushing the cube without pinching it.

Used in two places in the pick-and-lift reward stack:
- As its own dense ``RewTerm`` (``grasped_reward``) so the policy gets
  immediate gradient the moment it first achieves a real grasp.
- As a success gate so reward and termination only credit progress while
  the cube is actually held.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_GRIPPER_JAW_SENSOR = "gripper_jaw_contact"
_MOVING_JAW_SENSOR = "moving_jaw_contact"


def _jaw_force_magnitude(sensor) -> torch.Tensor:
    """Return per-env force magnitude for one single-body single-filter sensor.

    Reads ``sensor.data.force_matrix_w`` of shape (num_envs, num_bodies, num_filters, 3),
    flattens the middle dims, takes the L2 norm of each force vector, and reduces
    by ``max`` along the body/filter axis so a sensor with multiple body/filter
    rows still produces a single magnitude per env.
    """
    fm = sensor.data.force_matrix_w
    flat = fm.reshape(fm.shape[0], -1, 3)
    norms = torch.linalg.vector_norm(flat, dim=-1)
    return norms.max(dim=-1).values


def object_grasped_by_jaws(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
) -> torch.Tensor:
    """Return a per-env bool tensor: True iff both jaws are in contact with the cube.

    Parameters
    ----------
    env:
        Manager-based RL env carrying the ``gripper_jaw_contact`` and
        ``moving_jaw_contact`` sensors.
    force_threshold:
        Newtons. The default 0.5 N is well above PhysX numerical noise and
        well below the static grasp force the SO-101 actuator can hold.

    Returns
    -------
    torch.Tensor
        Bool tensor of shape (num_envs,).
    """
    g_force = _jaw_force_magnitude(env.scene[_GRIPPER_JAW_SENSOR])
    m_force = _jaw_force_magnitude(env.scene[_MOVING_JAW_SENSOR])
    return (g_force > force_threshold) & (m_force > force_threshold)


def grasped_reward(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
) -> torch.Tensor:
    """Float version of :func:`object_grasped_by_jaws`, suitable as a RewTerm function."""
    return object_grasped_by_jaws(env, force_threshold).float()


def object_grasped_obs(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
) -> torch.Tensor:
    """Grasp-state observation: ``[N, 1]`` float, 1.0 iff both jaws pinch the cube.

    Wraps :func:`object_grasped_by_jaws` (per-env bool) into the column shape
    Isaac Lab's observation manager concatenates: ``.float().unsqueeze(-1)``.
    Wiring this as a ``grasp_state`` ``ObsTerm`` gives the policy direct access
    to the contact-confirmed grasp signal that the reward stack already uses,
    so it doesn't have to infer "am I holding the cube?" from raw proprio.
    """
    return object_grasped_by_jaws(env, force_threshold).float().unsqueeze(-1)


__all__ = [
    "object_grasped_by_jaws",
    "grasped_reward",
    "object_grasped_obs",
]
