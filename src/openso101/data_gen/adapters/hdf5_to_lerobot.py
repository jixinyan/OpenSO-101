# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Standalone HDF5-to-LeRobot format conversion adapter.

DEFERRED as a standalone entry point. The conversion is currently performed
inline at the end of every `openso101 il record` session — see
`_push_convert_hdf5_to_lerobot` in `openso101.cli.il`, which iterates the
HDF5 directory, drops short / leading frames, and pushes a LeRobotDataset.
This adapter module exists for future extraction of that path into a
re-usable, batch-friendly tool; until that is needed, callers should use
`openso101 il record --convert-lerobot` (the default).
"""

from __future__ import annotations


def convert_hdf5_to_lerobot(*args, **kwargs):
    """Convert HDF5 demonstration files to LeRobot dataset format."""
    raise NotImplementedError(
        "Standalone HDF5→LeRobot conversion is deferred. Use the inline "
        "converter that runs at the end of `openso101 il record`."
    )


__all__ = ["convert_hdf5_to_lerobot"]
