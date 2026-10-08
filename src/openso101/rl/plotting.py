import json
import math
from datetime import UTC, datetime
from pathlib import Path

from fontTools.ttLib import TTFont
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from openso101.rl.config import digest


def load_scalar_data(log_dir: Path):
    log_dir = log_dir.resolve()
    if not log_dir.is_dir():
        raise NotADirectoryError(log_dir)
    files = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"目录没有 TensorBoard event 文件：{log_dir}")
    if len({path.parent for path in files}) != 1:
        raise ValueError("曲线需要单个 event 目录；请指定目标模型的实际 event 目录")
    sources = {}
    retained = {}
    samples = 0
    duplicates = 0
    for path in files:
        checksum = digest(path)
        accumulator = event_accumulator.EventAccumulator(
            str(path), size_guidance={event_accumulator.SCALARS: 0}, purge_orphaned_data=False,
        )
        accumulator.Reload()
        count = 0
        for tag in accumulator.Tags()["scalars"]:
            values = retained.setdefault(tag, {})
            for event in accumulator.Scalars(tag):
                if event.step < 0 or not math.isfinite(event.value) or not math.isfinite(event.wall_time):
                    raise ValueError(f"TensorBoard scalar 的 step 或数值无效：{path}，{tag}")
                count += 1
                if event.step in values:
                    duplicates += 1
                if event.step not in values or event.wall_time >= values[event.step].wall_time:
                    values[event.step] = event
        if digest(path) != checksum:
            raise ValueError(f"读取期间 TensorBoard 文件发生改变：{path}")
        sources[path.relative_to(log_dir).as_posix()] = {"sha256": checksum, "scalar_events": count}
        samples += count
    if not retained:
        raise ValueError(f"TensorBoard 文件没有 scalar 记录：{log_dir}")
    data = {}
    for tag, values in sorted(retained.items()):
        steps = sorted(values)
        data[tag] = (np.asarray(steps, dtype=np.int64), np.asarray([values[step].value for step in steps]))
    report = {
        "log_directory": str(log_dir), "files": sources, "scalar_events": samples,
        "duplicate_steps": duplicates, "duplicate_selection": "latest_wall_time_per_tag_and_step",
        "series": {tag: {"samples": len(steps), "first_step": int(steps[0]), "last_step": int(steps[-1])}
                   for tag, (steps, _) in data.items()},
    }
    return data, report


def smooth_values(values, window: int):
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("smooth 必须为正整数")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("曲线需要非空的有限一维数值")
    size = min(window, values.size)
    padded = np.pad(values, (size // 2, (size - 1) // 2), mode="edge")
    return np.convolve(padded, np.ones(size) / size, mode="valid")


def _latest_run(task: str) -> Path:
    root = Path("logs/rsl_rl") / task
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"任务没有训练目录：{root}")
    return candidates[-1]


def _draw_panel(axis, data, tags, title, window, font, *, fraction=False):
    for tag in tags:
        steps, values = data[tag]
        line, = axis.plot(steps, smooth_values(values, window), label=tag, linewidth=1.4)
        axis.plot(steps, values, color=line.get_color(), alpha=0.2, linewidth=0.6)
    axis.set_title(title, fontproperties=font)
    axis.set_xlabel("记录中的 step", fontproperties=font)
    axis.grid(True, alpha=0.25)
    if tags:
        axis.legend(fontsize=6)
    else:
        axis.text(0.5, 0.5, "没有对应的 scalar 记录", ha="center", va="center", transform=axis.transAxes,
                  fontproperties=font)
    if fraction:
        axis.set_ylim(-0.05, 1.05)


def plot_training(args) -> int:
    if args.smooth < 1:
        raise ValueError("smooth 必须为正整数")
    supplied = getattr(args, "log_dir", None) or getattr(args, "run", None)
    log_dir = Path(supplied).resolve() if supplied is not None else _latest_run(args.task).resolve()
    output = getattr(args, "output", None)
    destination = Path(output).resolve() if output is not None else log_dir
    save = args.save or output is not None
    image_path = destination / "training_curves.png"
    report_path = destination / "training_curves.json"
    if output is not None and destination.exists():
        raise FileExistsError(destination)
    if save and (image_path.exists() or report_path.exists()):
        raise FileExistsError(f"图表或报告已经存在：{destination}")
    font_file = getattr(args, "font_file", None)
    if font_file is None:
        font_file = Path(findfont(FontProperties(family="Noto Sans CJK SC"), fallback_to_default=False))
    font_file = Path(font_file).resolve()
    if not font_file.is_file():
        raise FileNotFoundError(font_file)
    font = FontProperties(fname=font_file)
    data, report = load_scalar_data(log_dir)
    tags = sorted(data)
    panels = [
        ("平均 episode return", [tag for tag in tags if tag in {
            "Train/mean_reward", "rollout/ep_rew_mean", "Reward / Total reward (mean)", "rewards/step"}], False),
        ("平均 episode 长度", [tag for tag in tags if tag in {
            "Train/mean_episode_length", "rollout/ep_len_mean", "Episode / Total timesteps (mean)", "episode_lengths/step"}], False),
        ("日志中的 reward 分量", [tag for tag in tags if tag.startswith("Episode_Reward/")], False),
        ("优化 loss", [tag for tag in tags if "learning_rate" not in tag.lower()
                       and (tag.startswith("Loss/") or "loss" in tag.lower())], False),
        ("策略探索数值", [tag for tag in tags if tag.startswith("Policy/") or "entropy" in tag.lower()], False),
        ("训练中的任务成功终止比例", [tag for tag in tags if tag.startswith("Episode_Termination/") and "success" in tag.lower()], True),
        ("物体到目标的距离", [tag for tag in tags if tag.startswith("Metrics/") and ("distance" in tag.lower() or "position_error" in tag.lower())], False),
        ("其他 episode 终止比例", [tag for tag in tags if tag.startswith("Episode_Termination/") and "success" not in tag.lower()], True),
        ("学习率", [tag for tag in tags if "learning_rate" in tag.lower()], False),
    ]
    title = f"已保存的训练记录：{log_dir.name}"
    labels = title + "记录中的 step没有对应的 scalar 记录" + "".join(item[0] for item in panels)
    with TTFont(font_file, fontNumber=0) as selected_font:
        available = selected_font.getBestCmap()
    if any(ord(character) not in available for character in labels if not character.isspace()):
        raise ValueError("字体缺少图表所需字符，请指定包含这些字符的 font-file")
    figure, axes = plt.subplots(3, 3, figsize=(20, 15))
    try:
        figure.suptitle(title, fontproperties=font, fontsize=14)
        for axis, (title, selected, fraction) in zip(axes.flat, panels):
            _draw_panel(axis, data, selected, title, args.smooth, font, fraction=fraction)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        if save:
            if output is not None:
                destination.mkdir(parents=True, exist_ok=False)
            figure.savefig(image_path, dpi=150, bbox_inches="tight")
            report.update(
                status="recorded_tensorboard_curves_generated", created_at=datetime.now(UTC).isoformat(),
                source_sha256=digest(Path(__file__)), image_sha256=digest(image_path),
                font_sha256=digest(font_file),
                smooth_window=args.smooth, panels={title: selected for title, selected, _ in panels},
                independent_task_success_verified=False,
            )
            with report_path.open("x") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({"status": report["status"], "image": str(image_path), "report": str(report_path),
                              "event_files": len(report["files"]), "scalar_events": report["scalar_events"]}, ensure_ascii=False))
        else:
            plt.show()
    finally:
        plt.close(figure)
    return 0
