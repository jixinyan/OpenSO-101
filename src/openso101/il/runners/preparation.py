import argparse
import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path

from openso101.scenes.models import file_digest


def prepare(arguments: list[str], report_dir: Path, physical_gpu: int):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("IL 准备检查需要禁止 CUDA")
    if report_dir.exists():
        raise FileExistsError(report_dir)

    import draccus
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.factory import make_dataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.factory import make_pre_post_processors

    from openso101.il.datasets.validation import validate_lerobot_metadata
    from openso101.il.runners.model_validation import validate_model_graph
    from openso101.il.policies.simulation import dataset_simulation

    cpu_arguments = ["--policy.device=cpu" if value == "--policy.device=cuda:0" else value
                     for value in arguments]
    cfg = draccus.parse(TrainPipelineConfig, args=cpu_arguments)
    cfg.validate()
    if cfg.policy.device != "cpu" or cfg.env is not None or cfg.resume or cfg.dataset.streaming:
        raise ValueError("IL 准备检查需要 CPU policy、本地读取模式和新的离线训练配置")
    if cfg.rename_map:
        raise ValueError("SO-101 IL 使用导出数据集的原始 observation 名称")
    if not cfg.save_checkpoint:
        raise ValueError("IL 训练需要保留模型和 optimizer checkpoint")
    if cfg.policy.n_obs_steps < 1 or cfg.policy.n_action_steps < 1 or len(cfg.policy.action_delta_indices) < 1:
        raise ValueError("IL observation 和 action 序列长度必须为正整数")
    for name in ("steps", "batch_size", "log_freq", "save_freq"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须为正整数")
    for name in ("num_workers", "eval_freq"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} 必须为非负整数")
    if cfg.dataset.root is not None:
        validate_lerobot_metadata(Path(cfg.dataset.root))
    dataset = make_dataset(cfg)
    if len(dataset) < 1:
        raise ValueError("LeRobot 数据集必须包含实际数据帧")
    root = Path(dataset.root)
    validate_lerobot_metadata(root)
    dataset_simulation(root, dataset.meta.fps, dataset.num_episodes)
    source_files = sorted(path for folder in ("meta", "data", "videos")
                          for path in (root / folder).rglob("*") if path.is_file())
    source_hashes = {str(path.relative_to(root)): file_digest(path) for path in source_files}
    features = dataset_to_policy_features(dataset.features)
    outputs = {key: value for key, value in features.items() if value.type is FeatureType.ACTION}
    inputs = {key: value for key, value in features.items() if key not in outputs}
    if cfg.policy.input_features and cfg.policy.input_features != inputs:
        raise ValueError("IL policy input_features 与实际 LeRobot 数据集不一致")
    if cfg.policy.output_features and cfg.policy.output_features != outputs:
        raise ValueError("IL policy output_features 与实际 LeRobot 数据集不一致")
    cfg.policy.input_features, cfg.policy.output_features = inputs, outputs
    cfg.policy.validate_features()
    preprocessor, postprocessor = make_pre_post_processors(cfg.policy, dataset_stats=dataset.meta.stats)
    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    loader = DataLoader(dataset, batch_size=min(cfg.batch_size, len(dataset)), num_workers=0)
    batches = [("first_batch", next(iter(loader)))]
    batches.extend((f"frame_{index}", default_collate([dataset[index]])) for index in indices)
    checked_batches, maximum_error = [], 0.0
    for label, batch in batches:
        raw_action = batch["action"].clone()
        for key in ("action", "observation.state", *cfg.policy.image_features):
            values = batch[key]
            if values.device.type != "cpu" or not torch.isfinite(values).all():
                raise ValueError(f"IL 原始数据需要有限 CPU Tensor: {key}")
        normalized = preprocessor(batch)
        for key in ("action", "observation.state", *cfg.policy.image_features):
            if not torch.isfinite(normalized[key]).all():
                raise ValueError(f"IL normalization 产生非有限数值: {key}")
        recovered = postprocessor(normalized["action"])
        error = float((recovered - raw_action).abs().max())
        maximum_error = max(maximum_error, error)
        if error > 0.0002:
            raise ValueError("IL action normalization 与还原结果不一致")
        checked_batches.append({"sample": label, "shapes": {
            key: list(normalized[key].shape) for key in ("action", "observation.state", *cfg.policy.image_features)},
            "action_roundtrip_maximum_error": error,
            "action_padding_frames": int(batch["action_is_pad"].sum())})
    model = validate_model_graph(cfg, dataset, report_dir / "pretrained_model")
    if {str(path.relative_to(root)): file_digest(path) for path in source_files} != source_hashes:
        raise RuntimeError("IL 准备检查期间的数据来源 SHA256 发生变化")
    if cfg.output_dir.exists():
        raise RuntimeError("IL 准备检查不能创建训练输出目录")
    repo = Path(__file__).resolve().parents[4]
    report = {"status": "actual_lerobot_training_preparation_verified",
              "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "lerobot_version": version("lerobot"), "policy": cfg.policy.type,
              "validation_device": "cpu", "planned_device": "cuda:0", "physical_gpu": physical_gpu,
              "dataset_root": str(root), "frames": len(dataset), "episodes": dataset.num_episodes,
              "source_files": source_hashes, "source_files_unchanged": True,
              "cpu_config": cfg.to_dict(), "training_arguments": arguments,
              "checked_batches": checked_batches, "sampled_frame_indices": indices,
              "action_roundtrip_maximum_error": maximum_error,
              "output_directory_created": False, "gpu_tests_started": False,
              "model_graph": model, "model_forward_verified": True,
              "training_started": False, "task_success_verified": False,
              "source_sha256": {name: file_digest(repo / name) for name in (
                  "src/openso101/il/runners/preparation.py", "src/openso101/il/runners/trainer.py",
                  "src/openso101/il/runners/worker.py", "src/openso101/il/runners/model_validation.py",
                  "src/openso101/il/policies/factory.py", "src/openso101/il/policies/validation.py",
                  "src/openso101/il/policies/simulation.py",
                  "src/openso101/il/datasets/validation.py")}}
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "policy", "frames", "action_roundtrip_maximum_error",
                                                 "gpu_tests_started", "training_started")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    prepare(arguments, args.report_dir, args.physical_gpu)


if __name__ == "__main__":
    main()
