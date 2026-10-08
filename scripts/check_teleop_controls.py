import argparse
import json
import math
import os
from pathlib import Path

import h5py
import pytest
import torch

from openso101.scenes.models import file_digest
from openso101.teleop.controls import _TeleopResumeHold, _TeleopTargetRateLimiter
from openso101.teleop.recorder.hdf5 import validate_hdf5_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("遥操目标检查需要禁止 CUDA")
    if args.output.exists():
        raise FileExistsError(args.output)
    validate_hdf5_episode(args.episode)
    digest = file_digest(args.episode)
    with h5py.File(args.episode, "r") as source:
        actions = torch.from_numpy(source["action"][:])
        positions = torch.from_numpy(source["observations/qpos"][:])
        dt = 1 / int(source.attrs["fps"])
    cases = []
    for max_delta in (0.005, 0.03):
        limiter = _TeleopTargetRateLimiter(max_delta)
        limiter.reset(positions[0])
        limited = torch.stack([limiter.apply(action) for action in actions])
        deltas = torch.cat([limited[:1] - positions[:1], torch.diff(limited, dim=0)]).abs()
        maximum = float(deltas.max())
        if maximum > max_delta + 0.000001 or not torch.isfinite(limited).all():
            raise RuntimeError("实际记录目标超过每步控制变化限制")
        settling_steps = math.ceil(float((actions[-1] - limited[-1]).abs().max()) / max_delta) + 2
        for _ in range(settling_steps):
            settled = limiter.apply(actions[-1])
        if not torch.allclose(settled, actions[-1], atol=0.000001, rtol=0):
            raise RuntimeError("每步变化控制未到达实际最终目标")
        cases.append({"frames": len(actions), "max_delta_rad": max_delta,
                      "maximum_actual_delta_rad": maximum, "maximum_command_rate_rad_s": maximum / dt,
                      "settling_steps": settling_steps, "settled_at_final_target": True})
    disabled = _TeleopTargetRateLimiter(0)
    first_action = actions[0]
    if disabled.apply(first_action) is not first_action:
        raise RuntimeError("关闭变化限制时需要直接使用输入目标")
    reference = positions[0].clone()
    selected = int((actions - reference).abs().amax(dim=-1).argmax())
    far_target = actions[selected]
    if float((far_target - reference).abs().max()) <= 0.05:
        raise ValueError("实际来源记录没有覆盖保持与恢复所需的关节变化")
    holds = []
    for device in ("leader", "keyboard"):
        hold = _TeleopResumeHold(0.05, device=device)
        captured = reference.clone()
        hold.activate(captured)
        captured.copy_(far_target)
        blocked = hold.apply(far_target)
        released = hold.apply(reference)
        if (not blocked.holding or blocked.released or not torch.equal(blocked.targets, reference)
                or released.holding or not released.released or hold.active):
            raise RuntimeError("实际目标的保持、数组复制或恢复状态不一致")
        holds.append({"device": device, "actual_frame": selected, "holding_error_rad": blocked.error,
                      "released_error_rad": released.error, "captured_target_preserved": True})
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="非负有限数值"):
            _TeleopResumeHold(value)
        with pytest.raises(ValueError, match="非负有限数值"):
            _TeleopTargetRateLimiter(value)
    invalid = actions[0].clone()
    invalid[0] = float("nan")
    with pytest.raises(ValueError, match="有限数值"):
        disabled.apply(invalid)
    with pytest.raises(ValueError, match="有限数值"):
        _TeleopResumeHold(0.05).activate(invalid)
    if file_digest(args.episode) != digest:
        raise RuntimeError("遥操控制检查更改了来源文件")
    report = {"status": "actual_recorded_teleop_controls_verified", "rate_limits": cases, "holds": holds,
              "source_episode": str(args.episode), "source_sha256": digest, "source_unchanged": True,
              "invalid_numeric_inputs_rejected": True, "gpu_tests_started": False,
              "physical_robot_control_verified": False, "native_restore_verified": False,
              "controls_source_sha256": file_digest(Path("src/openso101/teleop/controls.py")),
              "validation_source_sha256": file_digest(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "source_unchanged", "gpu_tests_started")}))


if __name__ == "__main__":
    main()
