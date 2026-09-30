# Squint 训练设计与 OpenSO-101 的采纳条件

## 来源与运行范围

阅读范围包含完整 README、`train_squint.py`、Lift 环境、SO101 controller、随机化环境和图像处理代码。源码版本为 `7086fd516eda7df585a5261c541e0670a6d916b4`。论文与公开 CSV 属于作者提供的结果，本项目尚未独立复现。

- [完整训练说明](https://github.com/aalmuzairee/squint/blob/7086fd516eda7df585a5261c541e0670a6d916b4/README.md)
- [训练代码](https://github.com/aalmuzairee/squint/blob/7086fd516eda7df585a5261c541e0670a6d916b4/train_squint.py)
- [Lift 环境与 reward](https://github.com/aalmuzairee/squint/blob/7086fd516eda7df585a5261c541e0670a6d916b4/envs/lift.py)
- [SO101 controller](https://github.com/aalmuzairee/squint/blob/7086fd516eda7df585a5261c541e0670a6d916b4/envs/robot/so101.py)
- [随机化与控制频率](https://github.com/aalmuzairee/squint/blob/7086fd516eda7df585a5261c541e0670a6d916b4/envs/base_random_env.py)
- [图像处理](https://github.com/aalmuzairee/squint/blob/7086fd516eda7df585a5261c541e0670a6d916b4/utils.py)
- [论文](https://arxiv.org/html/2602.21203v1)

本次检查只读取已经保存的真实训练 scalar、独立评估和原生 Isaac 推理轨迹。RL 训练保持停止，新增训练与恢复训练需要用户重新授权。报告、文件 SHA256 和图表见 [记录核查](../validation/2026-09-30/squint_training/README.md)。

## Squint 的完整训练流程

1. 在独立环境安装 ManiSkill、PyTorch、TorchRL、TensorDict 与 LeRobot。原仓库使用 `mani_skill_nightly`；复现实验需要保存实际安装版本及环境清单。
2. 创建 1024 个训练环境和 16 个评估环境。默认任务为 `SO101LiftCube-v1`，物理频率 100 Hz，控制频率 10 Hz，episode 长度 50 个控制步骤，共五秒。
3. 输入包含 wrist RGB、关节位置和 controller 状态。默认视觉模式为 `rgb+segmentation`；segmentation 用于替换图像背景，actor 接收处理后的 RGB。图像从 128×128 使用 area interpolation 转换为 16×16，再进行 ColorJitter 和数值归一化。
4. actor 输出连续关节目标增量，采用 `tanh` 限制动作范围，并在 `log_prob` 中计算对应变换。五个手臂关节每步目标增量为 ±0.1 rad，夹爪为 ±0.2 rad；目标从上一次 controller 目标累积，并受关节范围约束。
5. 前约 5000 transitions 使用随机动作采样。此后将真实当前观测、动作、reward、最终观测和终止信息保存到容量一百万的 GPU replay buffer。
6. 每次 1024 环境同步前进一步后，执行 256 次 critic 更新，每次抽取 512 transitions。actor 每四次 critic 更新执行一次更新。每个新 transition 对应 0.25 次 critic 更新，以及平均 128 次 replay 样本抽取；这个比例属于 off-policy 数据重用。
7. critic 使用两个 C51 网络、101 个 atoms、范围 `[-20, 20]`；actor 优化使用两个 Q 值的平均值。共享 CNN 由 critic 更新，actor 使用停止 encoder 梯度的特征。MLP 使用 LayerNorm，entropy temperature 自动调整。
8. actor、critic 和 temperature 的 learning rate 均为 `3e-4`，`gamma=0.9`，target 更新系数 `tau=0.01`。默认使用 `torch.compile`、CUDA Graphs 和 bfloat16 AMP。
9. 每约十万 transitions 使用确定性 actor 评估、记录 `success_once`、`success_at_end`、return 和视频，同时保存 checkpoint。训练与评估默认均启用 domain randomization 和 ColorJitter。成功不会立即终止默认 episode；终止观测用于 bootstrap。

README 的训练命令调用上述完整流程，默认总量为 150 万 transitions。论文的多 seed 实验使用 RTX 3090；训练时间依赖图像处理、并行环境、更新次数和编译开销。作者的分钟数不能作为本项目的运行承诺。

## 动作与时间尺度

| 项目 | Squint Lift | 本项目实际 grasp_v2 |
|---|---|---|
| 控制周期 | 0.1 秒 | 0.02 秒 |
| episode | 50 步，五秒 | Lift 250 步，五秒；PickPlace 400 步，八秒 |
| 手臂目标 | 上一次目标＋关节增量 | 默认关节姿态＋`0.5 × action` |
| 夹爪目标 | 连续目标增量 | `clip(0.4 × action + 0.4, 0, 0.8)` |
| actor 输出 | 经 `tanh` 限幅 | RSL-RL Gaussian 输出，夹爪目标在环境中裁剪 |
| 折扣系数 | 0.9 | 第 50 次迭代 snapshot 为 0.995 |
| 折扣时间常数 | 约 0.949 秒 | 约 3.990 秒 |
| 训练算法 | visual SAC、C51、replay buffer | state PPO，96 步 rollout，5 epochs、4 minibatches |

`gamma_50Hz = gamma_10Hz ** (0.02 / 0.1)`。Squint 的 0.9 换算为相同时间折扣后的 50 Hz 数值约为 0.97915。本项目已有 snapshot 的时间折扣覆盖更长，当前数据不足以判定延长折扣能够改善抓取。

采纳增量动作时，所有六个关节需要共同定义目标积累、动作限幅、速度约束、reset 初始化和 controller 状态观测。现有 2 rad/s 限制在 50 Hz 下对应每步最大目标增量 0.04 rad。目标增量约束与物理采样中的实际速度检查分别执行；PhysX 接触期间的实际速度仍需验证。策略导出、MuJoCo、遥操作与部署必须保存并使用相同动作定义。

## Lift reward 与任务标准

Squint 的 dense reward 为：

```text
r = (1 - tanh(5 × tcp_distance))
  + grasped
  + exp(-2 × rest_joint_distance) × grasped
  - 3 × jaw_touching_table
  - not_lifted
```

normalized reward 为 `r / 3`。接近项持续提供方向信息，返回预设关节姿态的奖励由真实双侧抓取状态启用，夹爪与桌面接触受到惩罚。

其 Lift success 同时检查双侧抓取、controller 目标距预设姿态小于 0.2 rad，以及物体中心超过初始几何高度至少 1 mm。本项目 Lift 要求双侧接触力均大于 0.5 N、物体高度超过 4 cm、进入目标半径 5 cm，并保持 0.25 秒；目标位于物体初始位置上方约 15–20 cm。结果评估保留本项目任务标准。

Isaac RewardManager 将各项乘以控制周期。采纳 reward 设计需要核查每步实际尺度、事件奖励与持续奖励的差异，以及接近、闭合、接触保持、提升和目标保持之间的收益关系。直接使用 Squint 的 C51 范围时，还需要核查 reward 与折扣决定的 Q 值范围。例如每步 0.8、`gamma=0.995` 对应持续奖励上界 160，超过其默认 support；该计算用于配置检查，未代表实际 critic 数值。

## 保存模型中的抓取行为

第 50 次迭代 snapshot 各包含 5,013,504 transitions，使用实际 `backend.json` 核查为固定 learning rate `1e-4`、初始 Gaussian std 0.5 和 `gamma=0.995`。独立评估每项 100 episode，Lift 接近率 53%，PickPlace 接近率 100%，双侧抓取和成功率均为零。

原生 Isaac 的确定性 actor 轨迹每项为 500 步、四环境，共 2000 个环境步骤：

| 实际指标 | Lift | PickPlace |
|---|---:|---:|
| 夹爪原始 action 最小值 | 1.27608 | 1.36604 |
| 夹爪原始 action 最大值 | 1.60658 | 1.75822 |
| 原始 action ≥ 1 的比例 | 100% | 100% |
| 实际夹爪目标 | 全部 0.8 rad | 全部 0.8 rad |
| 双侧接触比例 | 0% | 0% |
| closure、grasp_hold、held_height、held_goal reward 合计 | 全部为零 | 全部为零 |

动作映射与原生 ActionManager 的检查通过；多种超过范围的原始输出对应同一个打开夹爪目标。由此推断，动作裁剪区域和抓取前的探索是需要检查的学习因素。`tanh`、连续小幅目标动作和完整 controller 状态值得采纳，收益需要独立训练实验确认。环境内的动作裁剪不会自动使 actor 的 Gaussian 分布满足动作范围。

default profile 的 200 次迭代日志中，每项 scalar 均有限，Lift mean reward 从约 0.303 增至 3.092，PickPlace 从约 -0.00381 增至 0.03110。与失败的独立任务评估共同观察，这些记录表明 reward 增长未形成成功抓取；数值有限和任务收敛使用独立指标。

## 随机化与物理检查

Squint 训练使用物体尺寸与摩擦、夹爪 stiffness/damping、相机、照明和关节观测噪声。本项目还包含全部机器人 link 质量、关节摩擦与 armature、每次 reset 的全部关节 PD、物体质量、重力，以及动作延迟。

受控实验需要保存每个环境实际使用的参数，并使用明确的固定物理条件和完整随机化条件分别评估。物体位置随机化继续覆盖规定工作区。以相同输入检查控制、接触和任务路径，再解释不同条件下的学习结果。

Squint SO101 controller 的 nominal stiffness/damping 为 1000/100，force limit 为 100；夹爪材质摩擦为 2，并使用自身 URDF 与碰撞几何。本项目保留经过检查的机器人模型和动作范围，PD、力矩、碰撞与摩擦通过实际物理记录验证。已有原生 Isaac 夹爪接触诊断的最大速度为 38.163 rad/s，2 rad/s 速度验收尚未通过。MuJoCo scripted IK 的持续抓取已通过，它单独证明对应 MuJoCo 场景的操作路径。

## 采纳与实验要求

| 工作 | 完整要求 | 核查指标 |
|---|---|---|
| 连续目标增量动作 | 保留关节范围和 2 rad/s 要求；reset 初始化目标；观测提供 controller 目标；actor 的分布与限幅操作一致；导出与两个模拟器共享定义 | 原始动作、实际目标、裁剪比例、闭合时间、实际速度、reset 后首步动作 |
| 抓取阶段 reward | 接近、抓取中心、闭合、双侧接触、持续提升和目标保持分别记录；维持本项目 success；固定每步尺度和事件计数 | 每项 reward、实际接触、物体高度、目标距离、持续保持时间 |
| Squint 外部训练基准 | 使用该版本的完整 upstream 实现、真实 ManiSkill 环境、实际依赖清单、图像处理和作者配置；保存模型、配置和 seed | 持续成功率、末步成功率、实际采集 transitions、梯度更新数量、训练时间 |
| 本项目算法检查 | 固定任务、controller、观测、物理参数、随机化条件和评估初始状态；相同物理时间折扣；保留 PPO 与 SAC 的各自更新统计 | 每个阶段成功率、critic 数值、entropy、动作范围和目标速度 |
| sim2sim 验证 | 同一成功模型，关节映射、观测处理、动作定义、控制周期、reset 状态和任务标准完整一致 | Isaac 与 MuJoCo 的独立成功率、接触保持和动作轨迹 |

训练验收沿用 v2 Spec：三个 seed、每次 100 episode，连续三次独立评估成功率至少 90%。每种条件保存所有 checkpoint 和评估结果，模型选择依据成功率与保持指标。PPO、普通 SB3 SAC 与完整 Squint 使用各自准确的算法名称和配置；本项目现有 SB3 SAC 尚未包含 Squint 的 C51、视觉 encoder 和 GPU replay 设计。

本轮交付为源码研究、保存记录核查和实验要求。动作实现、reward 修改、Squint 训练与收敛验证保持待完成状态。
