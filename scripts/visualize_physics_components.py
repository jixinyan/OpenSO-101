import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from openso101.rl.config import digest


CONDITIONS = ("baseline", "bodies", "gravity", "armature", "friction", "combined")
LABELS = ("Baseline", "Body parameters", "Gravity", "Armature", "Joint friction", "Combined")
TASKS = ("lift", "pick_place")
COLORS = {"Isaac": "#172b4d", "baseline": "#e08c24", "combined": "#1679b8"}


def visualize(root, run_id, output, summary_path):
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads(summary_path.read_text())
    assert summary["paired_inputs_verified"] and summary["parameter_readback_verified"]
    output.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.16, "figure.facecolor": "#f6f8fc"})
    reports, traces = {}, {}
    for task in TASKS:
        reports[task], traces[task] = {}, {}
        for condition in CONDITIONS:
            folder = root / f"{task}_physics_components_{run_id}_{condition}"
            report = json.loads((folder / "report.json").read_text())
            assert digest(folder / "comparison.hdf5") == summary["tasks"][task][condition]["trajectory_sha256"]
            reports[task][condition] = report
            if condition in ("baseline", "combined"):
                with h5py.File(folder / "comparison.hdf5", "r") as file:
                    traces[task][condition] = [{name: group[name][:] for name in (
                        "isaac/joint_position", "isaac/joint_velocity", "isaac/object_position_root",
                        "mujoco/joint_position", "mujoco/joint_velocity", "mujoco/object_position_root")}
                        for group in file.values()]

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    metrics = (("joint_position_rmse_rad", "Joint position RMSE (rad)", 1),
               ("joint_position_max_error_rad", "Maximum joint position error (rad)", 1),
               ("physics_max_joint_speed_rad_s", "Maximum physics-step speed (rad/s)", 1),
               ("object_position_max_error_m", "Maximum object position error (mm)", 1000))
    for axis, (metric, title, scale) in zip(axes.flat, metrics):
        for index, task in enumerate(TASKS):
            x = np.arange(len(CONDITIONS)) + (index - 0.5) * 0.36
            values = [summary["tasks"][task][condition][metric] * scale for condition in CONDITIONS]
            bars = axis.bar(x, values, width=0.34, label="Lift" if task == "lift" else "PickPlace",
                            color=("#1679b8", "#e08c24")[index])
            axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks(np.arange(len(CONDITIONS)), LABELS, rotation=20, ha="right")
        axis.margins(y=0.2)
        if metric == "physics_max_joint_speed_rad_s":
            axis.axhline(2, color="#ba3345", linestyle="--", label="Recorded speed limit: 2 rad/s")
        axis.legend(fontsize=8)
    figure.suptitle("SO-101 paired physics experiment | same Isaac actions, 4 environments per task", fontsize=15)
    figure.savefig(output / "physics_overview.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for axis, task in zip(axes, TASKS):
        matrix = np.asarray([np.max([e["joint_position_max_error_rad"] for e in reports[task][condition]["environments"]], axis=0)
                             for condition in CONDITIONS])
        plot = axis.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(
            summary["tasks"][name][condition]["joint_position_max_error_rad"] for name in TASKS for condition in CONDITIONS), aspect="auto")
        axis.set_xticks(range(6), reports[task]["baseline"]["joint_names"], rotation=35, ha="right")
        axis.set_yticks(range(6), LABELS)
        axis.grid(False)
        axis.set_title("Lift" if task == "lift" else "PickPlace", fontweight="bold")
        for row in range(6):
            for column in range(6):
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", fontsize=9)
    figure.colorbar(plot, ax=axes, label="Maximum joint position error (rad)", shrink=0.85)
    figure.savefig(output / "joint_error_matrix.png", dpi=160)
    plt.close(figure)

    for task in TASKS:
        # 选择 Combined 最大关节误差所在的环境，展示相同环境的两项 MuJoCo 条件。
        records = reports[task]["combined"]["environments"]
        environment = int(np.argmax([max(e["joint_position_max_error_rad"]) for e in records]))
        base, combined = traces[task]["baseline"][environment], traces[task]["combined"][environment]
        time = np.arange(len(base["isaac/joint_position"])) * reports[task]["baseline"]["control_dt"]
        figure, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
        for joint, axis in enumerate(axes.flat):
            for label, key, source in (("Isaac", "isaac", base), ("baseline", "mujoco", base), ("combined", "mujoco", combined)):
                axis.plot(time, source[f"{key}/joint_position"][:, joint], label=label, color=COLORS[label], linewidth=1.5)
            axis.set_title(reports[task]["baseline"]["joint_names"][joint], loc="left")
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Position (rad)")
        axes[0, 0].legend()
        figure.suptitle(f"{task} | environment {environment} | recorded-action joint trajectories", fontsize=15)
        figure.savefig(output / f"{task}_joint_trajectories.png", dpi=160)
        plt.close(figure)
        figure, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
        for component, axis in enumerate(axes.flat[:3]):
            for label, key, source in (("Isaac", "isaac", base), ("baseline", "mujoco", base), ("combined", "mujoco", combined)):
                axis.plot(time, source[f"{key}/object_position_root"][:, component] * 1000, label=label, color=COLORS[label])
            axis.set_title(f"Object {'XYZ'[component]} in robot-root coordinates (mm)", loc="left")
            axis.set_xlabel("Time (s)")
        for label, source in (("baseline", base), ("combined", combined)):
            error = np.linalg.norm(source["mujoco/object_position_root"] - source["isaac/object_position_root"], axis=-1)
            axes[1, 1].plot(time, error * 1000, label=label, color=COLORS[label])
        axes[1, 1].set_title("Object position error (mm)", loc="left")
        axes[1, 1].set_xlabel("Time (s)")
        axes[0, 0].legend()
        figure.suptitle(f"{task} | environment {environment} | object motion", fontsize=15)
        figure.savefig(output / f"{task}_object_trajectories.png", dpi=160)
        plt.close(figure)

    titles = [*reports["lift"]["baseline"]["joint_names"], "Object Z (mm)", "Maximum joint position error (rad)"]
    interactive = make_subplots(rows=4, cols=2, subplot_titles=titles, vertical_spacing=0.065)
    groups = []
    for task in TASKS:
        for environment in range(4):
            start = len(interactive.data)
            base, combined = traces[task]["baseline"][environment], traces[task]["combined"][environment]
            time = np.arange(len(base["isaac/joint_position"])) * reports[task]["baseline"]["control_dt"]
            for joint in range(6):
                for label, key, source in (("Isaac", "isaac", base), ("baseline", "mujoco", base), ("combined", "mujoco", combined)):
                    interactive.add_trace(go.Scatter(x=time, y=source[f"{key}/joint_position"][:, joint], name=label,
                        legendgroup=label, showlegend=joint == 0, visible=not groups,
                        line={"color": COLORS[label]}), row=joint // 2 + 1, col=joint % 2 + 1)
            for label, key, source in (("Isaac", "isaac", base), ("baseline", "mujoco", base), ("combined", "mujoco", combined)):
                interactive.add_trace(go.Scatter(x=time, y=source[f"{key}/object_position_root"][:, 2] * 1000, name=label,
                    legendgroup=label, showlegend=False, visible=not groups, line={"color": COLORS[label]}), row=4, col=1)
            for label, source in (("baseline", base), ("combined", combined)):
                errors = np.max(np.abs(source["mujoco/joint_position"] - source["isaac/joint_position"]), axis=1)
                interactive.add_trace(go.Scatter(x=time, y=errors, name=label, legendgroup=label,
                    showlegend=False, visible=not groups, line={"color": COLORS[label]}), row=4, col=2)
            groups.append((f"{task} / environment {environment}", start, len(interactive.data)))
    buttons = [{"label": label, "method": "update", "args": [
        {"visible": [start <= index < end for index in range(len(interactive.data))]},
        {"title": f"SO-101 sim2sim | {label} | 相同动作记录"}]} for label, start, end in groups]
    interactive.update_xaxes(title_text="Time (s)")
    interactive.update_layout(title="SO-101 sim2sim | lift / environment 0 | 相同动作记录", height=1250,
        template="plotly_white", updatemenus=[{"buttons": buttons, "x": 0, "y": 1.1}],
        legend={"orientation": "h", "x": 0.55, "y": 1.08}, margin={"t": 140})
    interactive.write_html(output / "physics_explorer.html", include_plotlyjs=True, full_html=True)
    manifest = {"source_summary_sha256": digest(summary_path), "script_sha256": digest(Path(__file__)),
                "interactive_trace_count": len(interactive.data), "interactive_environment_count": len(groups),
                "source": "实际 Isaac 轨迹与 MuJoCo comparison.hdf5", "task_success_verified": False,
                "files": {}}
    for path in sorted(output.iterdir()):
        if path.suffix == ".png":
            with Image.open(path) as pixels:
                pixels.verify()
            with Image.open(path) as pixels:
                assert pixels.width > 1000 and pixels.height > 600
                assert np.std(np.asarray(pixels)) > 1
                manifest["files"][path.name] = {"sha256": digest(path), "size": list(pixels.size)}
        else:
            manifest["files"][path.name] = {"sha256": digest(path)}
    (output / "visualization_report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/rl_progress"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    visualize(args.root, args.run_id, args.output, args.summary)
