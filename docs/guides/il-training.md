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

准备目录的 `pretrained_model/` 保存完整模型参数和 observation/action processors。模型使用实际 CPU 初始化权重，optimizer 更新次数为零，保存的模型配置包含计划使用的 `cuda:0`。检查通过共享加载入口明确指定 CPU，逐项核查模型状态 SHA256、normalization Tensor、processor device 和实际动作推理结果。`model_checkpoint_verified` 与 `checkpoint_roundtrip` 保存对应结果。该模型文件用于程序验证，任务成功需要训练及独立评估。

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

## 保存模型的读取

`il/policies/factory.py` 接收 `pretrained_model/`、训练输出目录或 Hub repo_id。本地 `Path` 始终按照目录读取，Hub 名称通过 `huggingface_hub` 的官方检查。完整模型需要 `config.json`、`model.safetensors` 和两个 processor 配置及引用的状态文件。指定 `device` 时，模型构建、权重加载与 preprocessing 使用该设备；postprocessing 返回 CPU 动作。缺少文件或设备不可用时立即终止。

## 仿真评估

`il play` 与 `il eval` 共用 `il/policies/inference.py` 的 observation 检查、processors 与动作转换。输入具有明确的环境数量，输出为 `[N, 6]` 的 SO-101 绝对关节目标。非有限数值和错误形状立即终止推理。

准备检查在模型目录保存 `openso101_simulation.json`，包含采集 FPS、双相机尺寸、SO-101 关节名称、动作单位和来源 metadata SHA256。训练进程结束后，同一设置保存到全部已产生的 `pretrained_model/`。模型文件、optimizer 与已有记录全部保留。仿真入口读取保存的频率和 `input_features` 对应的相机尺寸，创建环境之前检查名称与尺寸。

已有 ACT、Diffusion 模型需要明确提供 `--control-fps`；新模型自动读取保存设置。指定频率与保存频率不一致时立即终止。student 的独立仿真验证使用 `rl student-eval`。

`il eval` 接收 `--n-episodes`、`--num-envs` 和 `--episode-length-s`，默认 episode 时间为 20 秒。评估配置设置实际 timeout、物体掉下桌面的结束条件和任务成功条件。PickPlace 使用释放后的 0.5 秒稳定条件；Lift 与 Stack 使用对应任务的成功检查；自定义场景使用 bundle 中的任务条件。

每个批次中的环境各执行一个待统计 episode。较早结束的环境完成统计后保持恢复后的关节姿态，等待当前批次的其它环境结束；下一批次统一恢复环境并重置 policy。ACT 的 action queue 和 Diffusion 的 observation history 在整个 episode 中持续保存。最后一个批次按逐环境配额执行统计，报告保存精确的请求数量、每个环境的 episode、结束状态、步数和 Wilson 成功率区间。

成功与 timeout 在物理步骤结束、环境恢复之前通过 Isaac Lab recorder 回调保存，包含最后一个控制步骤。报告写入发生在应用与环境关闭之后；中断执行保留已完成的 episode，并返回未完成状态。

`scripts/check_il_evaluation.py` 读取已保存的四环境轨迹，验证不同请求数量、较早结束的环境与最后一批次的统计。完整 ACT、Diffusion 模型在实际双相机记录上各执行三环境、两个批次和 16 次序列推理，并与 LeRobot 的实际推理结果比较。该检查使用 CPU，模型 optimizer 更新次数为零。统一 GPU 阶段包含两个完整模型的原生推理与 timeout 检查；GPU 运行保持待执行状态。
