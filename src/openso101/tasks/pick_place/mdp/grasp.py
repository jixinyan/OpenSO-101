# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Backwards-compatible re-export of the shared grasp helpers.

The implementation moved to ``openso101.tasks.shared.grasp`` so lift can use
the same contact-confirmed grasp signal without duplicating the function.
"""

from openso101.tasks.shared.grasp import grasped_reward, object_grasped_by_jaws  # noqa: F401

__all__ = ["grasped_reward", "object_grasped_by_jaws"]
