# IL 配置与模型检查

ACT 与 Diffusion 使用 LeRobot 0.4.0 的训练入口。数据输入支持 Hub repo_id 和本地 LeRobot 导出目录；本地目录可以直接包含 `meta/info.json`，也可以包含 `lerobot_dataset/`。尚未存在的本地目录使用 `./`、绝对路径或 Python `Path` 指定。

## CPU 准备

`--prepare-only` 通过 LeRobot 与 Draccus 解析配置，读取实际 metadata、Parquet 和双相机视频，检查时间窗口、边界 padding 与 normalization。完整模型使用实际首帧数据执行 CPU forward、backward 和 inference，并构造实际 optimizer 与 scheduler；检查保存 loss、gradient、模型状态 SHA256、来源文件 SHA256 与配置。该流程执行零次 optimizer 更新，训练输出目录保持尚未创建状态。

```bash
OPENSO101_REPO="$PWD" bash scripts/run_cpu_python.sh native \
  -m openso101.cli.main il train \
  --policy act \
  --dataset outputs/rl_progress/pickplace_cohort_dataset_20261007/lerobot \
  --prepare-only \
  --preparation-output outputs/rl_progress/il_preparation/act_check
```

Diffusion 使用相同命令并指定 `--policy diffusion`。检查使用完整模型结构、实际图像尺寸，以及配置指定的 torchvision backbone 权重。必需的权重通过 torchvision 下载到其 cache；文件与检查记录均予以保留。

`--preparation-output` 必须位于 `outputs/`，使用数据集和训练输出之外的新目录。检查报告保存 CPU 配置与计划使用的 GPU 参数。`model_forward_verified`、`model_backward_verified`、`model_inference_verified` 分别记录模型程序检查；任务成功需要独立运行验收。

## 训练参数与执行

训练输出默认位于 `outputs/rl_progress/il/<policy>/<timestamp>/`。已有输出目录立即终止启动。`--steps` 和 `--batch-size` 必须为正整数；`--gpu` 使用物理设备编号，范围由 `configs/runtime/gpu_scope.json` 指定。

额外 LeRobot 参数放在 `--` 后，使用 `--name=value`。配置中的 `policy.type`、`policy.device`、数据源、输出目录与训练入口已提供的参数由入口管理。模型结构、序列长度、crop、normalization 和 optimizer 配置在 CPU 准备阶段接受实际库检查。模型和 optimizer checkpoint 需要保留。

`train_il_policy` 完成 CPU 准备后，通过 `gpu_guard.py` 检查设备与进程。设备需要独占且空闲；发现其它作业时停止本项目的启动或运行。worker 使用设备 UUID，并要求 Torch 只看到一张 CUDA GPU。训练输出、CPU 准备记录、终端日志和 GPU 监督记录分别保存明确路径。`TrainResult` 提供 `output_dir`、`last_checkpoint`、`preparation_report` 和 `gpu_report`。

当前用户要求 GPU 计算保持停止。GPU worker 的实际训练、设备检查与任务成功继续等待对应验收。

## Python API

```python
from pathlib import Path
from openso101.il.runners import prepare_il_policy

report = prepare_il_policy(
    policy="act",
    dataset=Path("outputs/my_lerobot_dataset"),
    report_dir=Path("outputs/rl_progress/il_preparation/my_dataset"),
)
```

实现位置为 `il/runners/trainer.py`、`preparation.py`、`model_validation.py` 和 `worker.py`。真实数据与完整模型的批量 CPU 检查使用 `scripts/check_il_training.py`；统一验收中的阶段名称为 `il_training_preparation`。
