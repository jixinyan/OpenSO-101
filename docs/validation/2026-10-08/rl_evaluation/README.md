# CPU、评估统计与打包验证

被检查的代码为 `1a07ae4679850dfac15f790eacf9c4b997d30906`。`verification.json` 保存全部报告的 SHA256；`suite.json` 保存源码、输入文件、阶段参数和运行状态。

统一 CPU 流程通过 11 个阶段。GitHub CI 通过 49 项几何检查和 34 项接口检查。源码报告覆盖 299 个 Python 文件、179 个模块、778 处项目 import 和 23 个 Shell 文件；七个命令组的实际帮助入口通过检查。Ruff 未定义名称检查的结果为空列表。实际 wheel 在 CI 与 `jd_B300` 分别通过文件 SHA256、metadata、入口和 ZIP CRC 验证。

RL 命令入口位于 `cli/rl.py`，原生执行位于 `rl/rsl_execution.py`，图表位于 `rl/plotting.py`，episode 分配和 Wilson 成功率区间位于 `rl/evaluation.py`。评估严格分配请求的 episode 数量，记录控制步骤之前的任务进度，保存独立命名的评估 JSON。原生应用和环境关闭代码已完成源码检查。

保存的训练记录 `lift_corrective_coverage_continue_20261007` 另行完成实际 TensorBoard 读取与 CLI 图表验证：2,296 个 scalar event、23 条曲线、2,985 × 2,211 的 PNG。报告与图表位于 `outputs/rl_progress/refactor_20261008/recorded_curves_verified/`，图像 SHA256 为 `4addc88c42ba788de2e5833b6a3c1c2eeb96e2b61b1bd086cc787170967e6ffa`。此次读取没有产生训练步骤。

MuJoCo 运行已保存策略，完成四个 episode 和 20,000 个物理步骤；任务成功为 0/4。最终 PPO 的既有独立任务评估为 0/100。`gpu_tests_started`、`native_physics_verified`、`full_v2_verified` 均为 `false`。人工键盘采集、原生 GPU 运行、成功策略 sim2sim、视觉 student 和真实视频恢复继续等待对应验收。
