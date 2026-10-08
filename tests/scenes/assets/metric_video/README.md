# CPU 视频测量数据

`scripts/check_metric_video.py` 使用 MuJoCo 3.14.0 与 CPU OSMesa 渲染 `landmarks.xml`，保存 30 帧 MP4。十二个 landmark 的像素位置来自实际 geom segmentation 图像，世界位置来自 `MjData.geom_xpos`。

`measurements.json` 保存八个拟合点与四个独立验证点。OpenCV 重投影的独立验证最大误差为 0.2901 px，平面位置恢复最大误差为 0.8222 mm。`report.json` 保存本次模型、视频、测量与程序的 SHA256。

测试范围为实际仿真视频的读取、标定与位置恢复。真实视频场景恢复和机器人任务分别验收。生成的模型与视频遵循项目 MIT 许可证。
