import argparse
import ast
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path, sources):
    sources[str(path)] = digest(path)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--squint", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = {}
    source = args.squint / "train_squint.py"
    sources[str(source)] = digest(source)
    # 使用 Python AST 读取默认参数，训练模块保持未执行。
    tree = ast.parse(source.read_text())
    config_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Args")
    defaults = {node.target.id: ast.literal_eval(node.value) for node in config_class.body
                if isinstance(node, ast.AnnAssign) and node.value is not None}
    published_csv = args.squint / "results/SO101LiftCube-v1.csv"
    sources[str(published_csv)] = digest(published_csv)
    with published_csv.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["algorithm"] == "Squint" and row["task"] == "SO101LiftCube-v1"]
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) != 5:
        raise ValueError("Squint 公开结果需要包含五个 seed")
    published = {}
    for seed in seeds:
        values = np.array([[float(row[name]) for name in ("env_samples", "wall_time", "success_rate")]
                           for row in rows if int(row["seed"]) == seed])
        if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0):
            raise ValueError("公开 CSV 的数值或采样顺序检查失败")
        published[seed] = values

    evaluations = {"lift": "evaluation-20260930T010530915791Z.json",
                   "pick_place": "evaluation-20260930T010445856266Z.json"}
    tasks = {}
    curves_for_plot = {}
    jaws_for_plot = {}
    for task, filename in evaluations.items():
        previous = root / f"docs/validation/2026-09-29/{task}_seed42"
        config = read_json(previous / "train.json", sources)
        curves = read_json(previous / "learning_curves.json", sources)
        curve_report = {}
        for name, series in curves.items():
            values = np.array([point["value"] for point in series["points"]])
            if len(values) != 200 or not np.isfinite(values).all():
                raise ValueError(f"{task} 的 {name} 数量或数值检查失败")
            curve_report[name] = {"count": len(values), "first": float(values[0]), "last": float(values[-1]),
                                  "minimum": float(values.min()), "maximum": float(values.max())}
        evaluation = read_json(root / "docs/validation/2026-09-29" / filename, sources)
        if len(evaluation["episodes"]) != 100 or evaluation["task_profile"] != "grasp_v2":
            raise ValueError("独立评估数量或 profile 检查失败")
        success_rate = float(np.mean([episode["success"] for episode in evaluation["episodes"]]))
        if success_rate != evaluation["success_rate"]:
            raise ValueError("独立评估成功率检查失败")
        backend = read_json(args.backend_dir / f"{task}_grasp_v2_backend.json", sources)
        folder = root / f"outputs/rl_progress/{task}_grasp_v2_portable_50"
        metadata = read_json(folder / "policy.json", sources)
        validation = read_json(folder / "validation.json", sources)
        trace_path = folder / "isaac_validation.hdf5"
        if digest(trace_path) != validation["trace_sha256"]:
            raise ValueError("原生 Isaac 轨迹 SHA256 检查失败")
        sources[str(trace_path)] = validation["trace_sha256"]
        if metadata["checkpoint_sha256"] != evaluation["checkpoint_sha256"]:
            raise ValueError("策略与评估的 checkpoint 来源检查失败")
        with h5py.File(trace_path, "r") as trace:
            actions = trace["raw_action"][:]
            targets = trace["joint_targets"][:]
            rewards = trace["weighted_reward"][:]
            forces = trace["jaw_forces"][:]
        if actions.shape != (500, 4, 6) or targets.shape != actions.shape:
            raise ValueError("原生 Isaac 动作记录数量检查失败")
        if any(not np.isfinite(value).all() for value in (actions, targets, rewards, forces)):
            raise ValueError("原生 Isaac 动作、reward 或接触数值检查失败")
        jaw_mapping = metadata["action_mapping"][5]
        predicted_jaw = np.clip(actions[:, :, 5] * jaw_mapping["scale"] + jaw_mapping["offset"],
                                *jaw_mapping["processed_clip"])
        error = float(np.abs(predicted_jaw - targets[:, :, 5]).max())
        if error > 1e-6:
            raise ValueError("连续夹爪动作转换检查失败")
        totals = rewards.astype(np.float64).sum(axis=(0, 1))
        reward_totals = {term["name"]: float(value) for term, value in zip(metadata["reward_terms"], totals, strict=True)}
        for name, value in reward_totals.items():
            if not np.isclose(value, validation["weighted_reward_totals"][name], atol=1e-4, rtol=1e-5):
                raise ValueError("每项 reward 与原生验证报告检查失败")
        gamma = backend["algorithm"]["gamma"]
        tasks[task] = {"baseline_config": config, "baseline_scalar_summary": curve_report,
                       "grasp_v2_backend": backend, "grasp_v2_success_rate": success_rate,
                       "grasp_v2_progress_rates": evaluation["progress_rates"],
                       "completed_transitions": evaluation["completed_transitions"],
                       "checkpoint_sha256": evaluation["checkpoint_sha256"],
                       "control_dt": metadata["control_dt"], "episode_length_s": metadata["episode_length_s"],
                       "discount_time_constant_s": -metadata["control_dt"] / math.log(gamma),
                       "jaw_raw_action_range": [float(actions[:, :, 5].min()), float(actions[:, :, 5].max())],
                       "jaw_open_clip_fraction": float(np.mean(actions[:, :, 5] >= 1)),
                       "jaw_close_clip_fraction": float(np.mean(actions[:, :, 5] <= -1)),
                       "jaw_target_range_rad": [float(targets[:, :, 5].min()), float(targets[:, :, 5].max())],
                       "jaw_mapping_maximum_error_rad": error,
                       "bilateral_contact_fraction": float(np.mean((forces > .5).all(axis=-1))),
                       "weighted_reward_totals": reward_totals}
        curves_for_plot[task] = curves["Train/mean_reward"]["points"]
        jaws_for_plot[task] = targets[:, :, 5]

    result = {"status": "saved_training_records_audited", "rl_training_started": False,
              "squint_git_sha": subprocess.check_output(["git", "-C", str(args.squint), "rev-parse", "HEAD"], text=True).strip(),
              "squint_defaults": defaults,
              "squint_public_results_independently_reproduced": False,
              "squint_published_lift_final_success_percent": {str(seed): float(values[-1, 2]) for seed, values in published.items()},
              "squint_critic_updates_per_transition": defaults["num_updates"] / defaults["num_envs"],
              "squint_replay_sample_draws_per_transition": defaults["num_updates"] * defaults["batch_size"] / defaults["num_envs"],
              "squint_gamma_at_50hz_for_equal_time_discount": defaults["gamma"] ** (.02 / .1),
              "tasks": tasks, "source_sha256": sources, "audit_source_sha256": digest(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    font = FontProperties(fname=str(args.font))
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), constrained_layout=True)
    for seed, values in published.items():
        axes[0, 0].plot(values[:, 0] / 1e6, values[:, 2], label=f"seed {seed}")
    axes[0, 0].set(title="Squint Lift：作者公开结果，尚未复现", xlabel="采集 transitions（百万）", ylabel="成功率（%）", ylim=(-3, 103))
    colors = {"lift": "#2469be", "pick_place": "#ba5727"}
    labels = {"lift": "Lift", "pick_place": "PickPlace"}
    for task, points in curves_for_plot.items():
        batch = tasks[task]["baseline_config"]["rollout_steps"] * 1024
        axes[0, 1].plot([(point["step"] + 1) * batch / 1e6 for point in points],
                        [point["value"] for point in points], label=labels[task], color=colors[task])
    axes[0, 1].set(title="OpenSO-101 default：实际训练记录", xlabel="采集 transitions（百万）", ylabel="记录的 mean reward")
    stages = ("reached", "grasped", "held_above_table")
    x = np.arange(len(stages))
    for index, task in enumerate(tasks):
        rates = [100 * tasks[task]["grasp_v2_progress_rates"][stage] for stage in stages]
        bars = axes[1, 0].bar(x + (index - .5) * .34, rates, .34, label=labels[task], color=colors[task])
        axes[1, 0].bar_label(bars, labels=[f"{rate:.0f}%" for rate in rates], padding=3)
        jaw = jaws_for_plot[task]
        axes[1, 1].plot(np.arange(len(jaw)) * .02, jaw.mean(axis=1), label=labels[task], color=colors[task])
    axes[1, 0].set(title="grasp_v2 第 50 次迭代：各 100 episode", xticks=x,
                   xticklabels=("接近物体", "形成抓取", "保持提升"), ylabel="episode 比例（%）", ylim=(0, 115))
    axes[1, 1].set(title="grasp_v2：四环境实际夹爪目标", xlabel="记录时间（秒，包含 reset）", ylabel="夹爪目标（rad）", ylim=(-.05, .9))
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.18)
    fig.savefig(args.output / "training_audit.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"status": result["status"], "tasks": {name: {key: value[key] for key in
          ("jaw_raw_action_range", "jaw_open_clip_fraction", "grasp_v2_success_rate", "bilateral_contact_fraction")}
          for name, value in tasks.items()}}, indent=2))


if __name__ == "__main__":
    main()
