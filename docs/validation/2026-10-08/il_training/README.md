# CPU 配置、完整 IL 模型与采集 checkpoint 验证

源码版本为 `a09f17980f094003628d76a5bd26611e53c4f5ac`。统一流程通过 16 个 CPU 阶段；独立 Linux CI 通过 123 项检查。源码检查覆盖 189 个模块、810 处项目 import；实际 wheel 与 sdist 检查通过。

验证索引保存在 [verification.json](verification.json)，每份原始报告保留 SHA256。完整阶段、输入与来源保存在 [suite.json](suite.json)。

## IL 模型与配置

实际双相机 LeRobot 数据包含 328 帧。ACT、Diffusion 使用完整配置执行 CPU forward、backward、动作 inference 和实际 optimizer、scheduler 构建。所检查的模型参数数量分别为 51,597,190 和 266,622,758；全部梯度保持有限数值。optimizer 更新次数为零，训练未启动。动作 normalization 的最大往返误差为 `3.814697265625e-6` motor units。

[ACT 报告](models/act.json) 和 [Diffusion 报告](models/diffusion.json) 保存完整 CPU 配置、未来训练参数、实际模型状态 SHA256、loss、gradient 与 inference action。八项无效配置均通过实际程序终止检查。来源数据的 metadata、Parquet 和双相机视频 SHA256 保持一致。

## 采集与回放

HDF5 的磁盘和内存 checkpoint 分别保存 16 帧、32 帧。来源数组在添加记录后被更新，保存的全部 29 个数据字段仍与来源帧一致。非法 FPS、未启动 episode 的 checkpoint 和无效恢复位置均终止运行。详细内容见 [checkpoint 报告](suite/recorder_checkpoint/report.json)。

同步和异步 LeRobot 导出均完成全部 328 帧读取，动作和状态转换误差为零，编码工作线程全部关闭。共享状态转换检查覆盖 23 个字段；已有视觉 student 在 243 帧记录上完成 CPU inference。

MuJoCo 的已有策略完成四个 episode、1,000 个控制步骤、20,000 个物理步骤，速度与力矩检查通过，任务成功为 0/4。CPU 程序检查、原生 Isaac 运行、人工遥操任务与独立策略成功分别记录；完整 v2 任务成功继续等待对应验收。GPU 计算与 RL 训练保持停止。
