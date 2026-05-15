# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Built-in example tasks for OpenSO-101.

Importing this package triggers gym registration of every task module.
Custom user tasks should use `from openso101.envs import register_task`.
"""

from . import lift  # noqa: F401
from . import pick_place  # noqa: F401
