# 脚本入口

| 操作 | 入口 |
|---|---|
| CPU 环境 | `run_cpu_python.sh native` 或 `run_cpu_python.sh mujoco` |
| GPU 环境与空闲监督 | `run_native_python.sh 2` |
| 统一验收 | `openso101 validate run`，配置为 `configs/validation/v2_preparation.json` |
| 源码检查 | `openso101 validate source` |
| 实际 wheel 检查 | `openso101 validate package` |
| 场景批量检查 | `check_scene_batch.py` |
| 自然语言场景修改检查 | `check_scene_edits.py` |
| 实际 MP4 标定检查 | `check_metric_video.py` |
| 实际 SO-101 键盘 IK 检查 | `check_keyboard_kinematics.py` |
| LeRobot 全部帧读取 | `validate_lerobot_dataset.py` |
| 保存模型的 previous actions 分析 | `audit_policy_history.py` |
| 保存 optimizer 的 CPU 更新检查 | `check_demonstration_updates.py` |
| 项目 GPU 查询与停止 | `audit_project_gpu.py` |
| 图表与运行记录可视化 | `visualization/` |

从仓库根目录执行脚本。输出使用 `outputs/rl_progress/`，中间文件使用 `outputs/tmp/`。GPU 入口需要指定 `OPENSO101_REPO`，并且使用唯一的空闲物理设备；启动与运行过程保存单独报告。

应用功能的实现位置见 [代码目录](../docs/guides/code-map.md)。
