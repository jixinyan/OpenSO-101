# 保存 RL 策略的实际双相机视频

训练 seed 43 的 `model_821.pt` 在 Isaac 中执行确定性推理，模型累计包含 161,611,776 transitions。任务为 nominal `OpenSO101-Lift-v0`，使用 `grasp_v4` 配置；评估 seed 为 10043，四个环境各执行一个完整 episode。

首个环境的 250 个控制步骤保存为双相机 HDF5，并编码为 5 秒、50 FPS、1024×512 的 MP4。左侧为 overhead，右侧为 wrist。原始录制保存实际关节目标、位置、速度、物体和任务状态。全部 250 帧视频完成解码检查，源文件 SHA256、模型 SHA256、控制时间、完整步骤数量与成功标记检查通过。

首个环境完成接近和双侧夹爪接触，抬升与任务成功均为 false。四个环境的接近、双侧接触分别为 4/4，抬升与同时持物抬升分别为 1/4，任务成功为 0/4。完整任务成功继续使用规定的终止条件。

录制进程的实际计算与渲染均使用物理 GPU 4，设备目录仅包含指定 GPU 与 NVIDIA 公共设备。已有三个训练进程持续运行。CLI 的八项本地检查通过，录制、完整 HDF5 校验与 MP4 编码使用实际模型和仿真数据。

原始文件统一保存在 `outputs/rl_progress/v4_seed43_policy_video/`，MP4 为 `policy.mp4`，HDF5 为 `episodes/episode_000000.hdf5`。模型位于 `outputs/rl_progress/v4_seed43_scoped_a7cab37/evaluations/iteration_000821/`。

- [视频验证与来源](video.json)
- [四环境独立评估](evaluation.json)
- [实际 GPU 检查](gpu_scope.json)
- [模型 metadata](checkpoint.json)

复现入口见 [保存策略运行视频](../../../guides/v2-training.md#保存策略运行视频)。
