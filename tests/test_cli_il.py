import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("arguments, message", [
    (["record", "--task", "OpenSO101-PickPlace-v0", "--num-envs", "0"], "num_envs 必须为正整数"),
    (["record", "--task", "OpenSO101-PickPlace-v0", "--num-envs", "2"], "交互录制要求 num_envs 为 1"),
    (["record", "--task", "OpenSO101-PickPlace-v0", "--fps", "0"], "fps 必须为正整数"),
    (["play", "--task", "OpenSO101-PickPlace-v0", "--policy-path", "missing-model", "--steps", "0"], "steps 必须为正整数"),
    (["play", "--task", "OpenSO101-PickPlace-v0", "--policy-path", "missing-model", "--num-envs", "2"], "il play 要求 num_envs 为 1"),
    (["play", "--task", "OpenSO101-PickPlace-v0", "--policy-path", "missing-model", "--action-mode", "rl"], "绝对关节位置控制需要 teleop"),
    (["eval", "--task", "OpenSO101-PickPlace-v0", "--policy-path", "missing-model", "--n-episodes", "0"], "n_episodes 必须为正整数"),
    (["eval", "--task", "OpenSO101-PickPlace-v0", "--policy-path", "missing-model", "--num-envs", "0"], "num_envs 必须为正整数"),
    (["replay", "--episode", "missing.hdf5", "--max-steps", "0"], "max_steps 必须为正整数"),
    (["train", "--policy", "act", "--dataset", "missing", "--steps", "0"], "steps 必须为正整数"),
    (["train", "--policy", "act", "--dataset", "missing", "--batch-size", "0"], "batch_size 必须为正整数"),
    (["train", "--policy", "act", "--dataset", "./absent-dataset", "--prepare-only"], "LeRobot 数据集目录不存在"),
    (["train", "--policy", "act", "--dataset", "missing", "--output-dir", "."], "训练输出目录必须尚未存在"),
    (["train", "--policy", "act", "--dataset", "missing", "--", "--policy.device=cuda:1"], "额外参数不能重复指定"),
    (["train", "--policy", "act", "--dataset", "missing", "--", "--output_dir=outputs/unexpected"], "额外参数不能重复指定"),
    (["train", "--policy", "act", "--dataset", "missing", "--", "--policy.chunk_size", "20"], "额外参数需要使用 --name=value"),
    (["train", "--policy", "act", "--dataset", "user/project", "--gpu", "1", "--prepare-only"], "超出允许范围"),
    (["train", "--policy", "act", "--dataset", "missing", "--preparation-output", "outputs/check"], "需要同时使用 prepare_only"),
])
def test_invalid_input_before_native_startup(arguments, message):
    result = subprocess.run([sys.executable, "-m", "openso101.cli.main", "il", *arguments],
                            capture_output=True, text=True, timeout=30,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OPENSO101_SKIP_ISAAC": "1"})
    assert result.returncode != 0
    assert message in result.stderr
    assert "AppLauncher" not in result.stderr


@pytest.mark.parametrize("command", ["play", "eval"])
def test_missing_policy_before_native_startup(tmp_path, command):
    result = subprocess.run([sys.executable, "-m", "openso101.cli.main", "il", command,
                             "--task", "OpenSO101-PickPlace-v0", "--policy-path", str(tmp_path / "absent")],
                            capture_output=True, text=True, timeout=30,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OPENSO101_SKIP_ISAAC": "1"})
    assert result.returncode != 0
    assert "policy checkpoint not found" in result.stderr
    assert "AppLauncher" not in result.stderr


@pytest.mark.parametrize("command", ["random", "zero", "preview"])
@pytest.mark.parametrize("argument", ["--steps", "--num-envs"])
def test_invalid_environment_input_before_native_startup(command, argument):
    result = subprocess.run([sys.executable, "-m", "openso101.cli.main", "envs", command,
                             "--task", "OpenSO101-PickPlace-v0", "--headless", argument, "0"],
                            capture_output=True, text=True, timeout=30,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OPENSO101_SKIP_ISAAC": "1"})
    assert result.returncode != 0
    assert "steps 和 num_envs 必须为正整数" in result.stderr
    assert "AppLauncher" not in result.stderr
