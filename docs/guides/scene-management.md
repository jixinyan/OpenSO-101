# 场景修改与能力检查

## 场景 revision

`scenes edit` 接受已有 bundle 与完整修改描述。实际模型服务配置使用现有 OpenAI compatible 入口；模型请求、修改字段和基础版本写入新 bundle 的 provenance。输出目录需要尚未存在。

```bash
openso101 scenes edit outputs/apple_bundle \
  --instruction "仅将 apple 初始位置沿 x 轴移动 0.02 米，保持任务与其他配置" \
  --codex-config outputs/provider/config.toml \
  --output outputs/rl_progress/scene_revision
```

提供 `--edit-file` 时使用完整 SceneEdit JSON。使用 `--catalog` 指定包含当前与新资产的目录。SceneStore 保存需要同时提供 `--store` 与 `--expected-revision`。目标条件变化保存在完整 goals 中；操作物体与流程变化需要完整 replacement_intent。

## 资产能力测量

GeometryProbe 指定实际场景尺寸、collision 与检查区域。graspable 需要夹爪最大开口及对应机器人文件 SHA256。container 与 support_surface 需要实体局部坐标中的三维检查区域。容器检查要求封闭实体 mesh、内部空间、底部支撑、四个方向的侧壁与向上的开口；collision 使用 convexDecomposition。

```bash
openso101 scenes inspect ASSET_UID --catalog outputs/assets \
  --probe outputs/probes/container.json --output outputs/container_geometry.json
```

报告中的 geometry_accepted 表示所指定样本的几何检查结果。物体质量、摩擦、完整运动路径、稳定性与成功采集由实际仿真验证。

`scenes generate --capability-probes FILE` 的 JSON 使用 entity_id 作为键，每个值为 GeometryProbe 列表。未提供检查的 required_capabilities 保存在 unverified_capabilities 中。

## BDDL 目标

```bash
openso101 scenes read-bddl outputs/tasks/problem.bddl \
  --scene-file outputs/tasks/scene.json --binding outputs/tasks/binding.json \
  --catalog outputs/assets --output outputs/tasks/bddl_report.json \
  --bundle-output outputs/rl_progress/bddl_bundle
```

BDDLBinding 保存原始 problem SHA256、对象到 entity_id 的完整对应关系及容器内部区域。官方 bddl 库执行量词和逻辑条件。支持的关系为 inside 与 ontop。未支持的 predicate、缺少对象绑定或内部区域时立即终止。源 problem 的全部初始条件继续保存在报告中，实际初始状态验收独立记录。

## 视频位置恢复

CameraMeasurements 保存输入视频 SHA256、测量帧、OpenCV pinhole intrinsic_matrix 与 distortion、至少六个拟合点及至少三个独立验证点。每个 ImagePoint 保存 identifier、实际米制 world_position_m 和 pixel_xy。拟合与验证使用不同的测量点。相机类型需要为 opencv_pinhole。

```bash
openso101 scenes calibrate-video outputs/input.mp4 \
  --measurements outputs/camera_measurements.json --output outputs/camera_calibration.json
openso101 scenes metric-layout outputs/scene.json --video outputs/input.mp4 \
  --calibration outputs/camera_calibration.json --observations outputs/object_pixels.json \
  --catalog outputs/assets --output outputs/metric_scene.json
```

ObjectImageMeasurement 的 reference_height_m 表示物体中心的实际 Z 坐标。位置恢复保留旋转、尺寸和任务条件，并保存视频、标定和物体测量的 SHA256。metric_scene.provenance.json 记录待执行的场景检查。

## 交互式预览

```bash
openso101 scenes preview outputs/apple_bundle --output outputs/preview.html \
  --robot-model outputs/so-arm100/Simulation/SO101/so101_old_calib.xml
```

预览使用 bundle 中的实际 GLB 与官方机器人 mesh。机器人关节位置可以通过六个 `--joint-positions` 数值指定。HTML 附带 preview.json 来源报告；物理任务状态保持待验收。
