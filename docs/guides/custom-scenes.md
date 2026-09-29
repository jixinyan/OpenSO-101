# Objaverse 与自定义场景

场景工具使用 Python 3.11。OpenSO-101 仿真环境安装 `scenes` extra：

```bash
uv pip install -e '.[scenes]'
```

资产查询、下载、JSON 配置和场景包检查可以在 CPU 运行。`compile` 使用已经配置完成的 Isaac Sim 环境。

## 获取资产

```bash
openso101 scenes search apple --limit 5
openso101 scenes fetch 4c19ae47dbe8468285ee53ff487fe51a
openso101 scenes list
```

`search` 查询 Objaverse 的 LVIS 类别，返回类别和对应 UID。`fetch` 使用官方 SDK 获取指定 UID 的 metadata 和 GLB，保存到 `outputs/assets`。`--catalog` 可以指定其他缓存目录。每项资产保存作者、原始许可证、来源 URL、SHA256、几何尺寸和面数。再次读取资产时检查文件内容是否与记录一致。

## 创建与编辑场景

```bash
openso101 scenes create 4c19ae47dbe8468285ee53ff487fe51a \
  --scene-id apple_table --dimensions 0.04 0.04 0.05 \
  --output outputs/apple_table.json
openso101 scenes validate outputs/apple_table.json
# CPU-only collision and relation checks
openso101 scenes diagnose outputs/apple_table.json
```

编辑生成的 JSON 可设置多个 `entities`、桌面尺寸、物体尺寸、位置、旋转、物理属性、机器人基座和任务目标。单位为米、千克、秒，坐标使用右手 Z-up，旋转使用 `quaternion_wxyz`。物体位置表示几何包围盒中心。`dimensions_m` 表示物体旋转前在场景坐标轴方向上的尺寸。原始 GLB 的 Y-up 在编译时转换为 Z-up。

`reset_translation_m` 保存三个方向的下限和上限。布局检查验证物体及其 reset 范围是否进入桌面或超出桌面。任务必须引用动态物体，目标位置按该物体的几何中心定义。未知字段、无效 quaternion、非有限数值、无效质量及失配的资产 hash 会直接导致执行失败。

`layout_valid` 表示配置、资产和桌面范围检查通过。报告中的 `pending_checks` 列出碰撞、稳定性、机器人可达性、相机和采集等尚未执行的检查。

## 场景包与 USD

```bash
openso101 scenes bundle outputs/apple_table.json --output outputs/apple_bundle
openso101 scenes verify outputs/apple_bundle
openso101 scenes compile outputs/apple_bundle --output outputs/apple_usd
```

场景包包含原始资产、metadata、配置、检查结果和完整文件 SHA256 清单。可以复制整个目录到其他主机，随后执行 `verify`。已存在的输出目录会被拒绝，修改后的配置使用新目录保存。

编译器调用 Isaac Sim 官方 GLB 转换器，使用 `UsdPreviewSurface` 材质，生成 `scene.usda`、资产 USD 和纹理依赖。OpenUSD 负责场景层级、尺寸、物体中心、碰撞、刚体质量、摩擦与恢复系数。编译后检查 USD 依赖是否完整包含在输出目录中。父进程读取并校验完成报告后返回成功。编译产物状态为 `compiled`；SO-101 载入、Isaac Lab 任务连接、相机与采集通过独立运行验收。

`environment.usda` 用于 Isaac Lab 并行环境，`scene.usda` 用于独立 PhysX 场景。任务支持 `at`、`inside` 和 `on_top`，可以同时要求多个物体满足目标。成功条件包括位置、完整包围盒关系、速度、夹爪释放和持续稳定时间。`inside` 必须提供经过测量的容器内部区域。

## 模型服务与独立工具

```bash
openso101 scenes generate --instruction "把苹果放置到桌面右侧" \
  --base-url "$SCENE_MODEL_BASE_URL" --model "$SCENE_MODEL_NAME" \
  --output outputs/apple_proposal
openso101 scenes inspect 4c19ae47dbe8468285ee53ff487fe51a
openso101 scenes layout outputs/apple_proposal/scene.json \
  --locked object --output outputs/apple_layout.json
```

模型接口为 OpenAI-compatible Chat Completions，服务地址包含其 API 前缀。默认从 `SCENE_MODEL_API_KEY` 读取密钥；`--api-key-env` 指定其他变量，无鉴权服务使用空字符串。模型名称与地址也可通过 `SCENE_MODEL_NAME` 和 `SCENE_MODEL_BASE_URL` 配置。

`generate` 将当前资产目录和场景 schema 发送给指定服务，校验返回配置并保存 proposal。目录需要预先导入所需资产。外部服务、超时、输出格式和场景校验错误立即终止。当前没有配置实际服务地址和模型，因此在线生成尚未验收；资产、布局、版本保存、编译与仿真检查均可独立调用。

## RGB 视频 real2sim agent loop

`agent-loop` 自动从 RGB 视频均匀抽帧，把帧和场景上下文交给 GPT-6 Astra，依次完成视频描述、Objaverse/LVIS 检索、场景编排、缺失 primitive/generated parts 资产生成、静态几何检查、物理合理性审查和 SO-101 数据采集 readiness 审查。模型发现问题时最多自动修复两轮，并把每轮反馈保存到结果 JSON。

```bash
openso101 scenes agent-loop \
  --video captures/pick_place.mp4 \
  --sample-count 8 --frames-dir outputs/pick_place_frames \
  --base-url "$SCENE_MODEL_BASE_URL" --model gpt-6-astra \
  --catalog outputs/assets --output outputs/agent_scene_bundle
```

也可以用 `--frame` 重用已经抽好的 JPEG/PNG；这时必须同时提供 `--fps`、`--frame-count`、`--width` 和 `--height`。`status=completed` 只表示静态检查和两个 Astra 审查通过；Isaac 的动态碰撞、可达性、接触、相机和成功采集仍由 `validate-runtime`、runtime check 或真机采集完成。

`inspect` 返回几何测量信息，未经验证的操作能力保持 `unknown`。`layout` 使用 SciPy MILP 求解桌面范围、reset 范围和实体间距，保留锁定实体的位置。机器人路径与接触验证在运行报告中单独记录。

```bash
openso101 scenes save outputs/apple_layout.json --expected-revision 0 --reason "创建场景"
openso101 scenes read apple_table --revision 1 --output outputs/apple_revision1.json
```

版本存储使用 SQLite。修改已有场景时提供读取时的 revision；过期版本不能覆盖当前版本。旧版本可以重新读取。编译产物与采集数据继续使用场景 SHA256 标识其内容。

## 外部资产与任务模板

```bash
openso101 scenes import model.glb --name apple --source local:apple --license CC0 --author owner
openso101 scenes import-usd source.usd --prim /World/Object --name object --license MIT --author owner
openso101 scenes import-robotwin /data/robotwin/objects/047_mouse --model-id 0 --license MIT --author RoboTwin
openso101 scenes robotwin-task mouse_pad --source /data/robotwin/objects \
  --license MIT --author RoboTwin --output outputs/mouse_bundle
```

RoboTwin 模板包括 `mouse_pad`、`stapler_pad`、`pillbottle_pad`、`object_scale`、`stack_blocks`。模板将资产配置到 SO-101 桌面任务中，保存来源 metadata，并使用释放后稳定放置的成功条件。任务尺寸属于模板配置，需要针对实际机器人操作能力继续验证。

USD 导入接受可读取的 mesh prim，保存 USDZ 及其依赖。包含 articulation 的资产会明确拒绝。B1K 加密资产、关节任务及 BDDL 任务转换尚未提供。

## 仿真、录制与回放

```bash
openso101 scenes validate-runtime outputs/apple_usd --output outputs/apple_runtime_report.json
openso101 il record --task OpenSO101-CustomScene-v0 --scene outputs/apple_usd \
  --teleop-device keyboard --repo-root outputs/apple_dataset
openso101 il replay --episode outputs/apple_dataset/episodes/episode_000000.hdf5
```

`validate-runtime` 默认运行四个环境、100 次 reset 和 200 个随机控制步骤，检查状态与数值有效性。`--cameras` 同时检查双相机图像。报告状态为 `runtime_verified`，任务完成、完整路径和成功采集保留独立检查要求。

键盘方向键控制平面移动，PageUp/PageDown 控制高度，A/D 控制旋转，Space 打开夹爪，Shift 关闭夹爪。damped least-squares IK 使用实际 Jacobian、关节限位与速度限制。leader 使用 `--teleop-device leader` 及原有设备参数。

自定义场景 HDF5 保存全部实体状态、场景 hash 与可移植场景副本。回放自动读取录制使用的场景版本。LeRobot 导出保留场景副本，并在 `meta/scenes.json` 保存 episode 对应关系。无显示环境回放使用 `--headless --no-camera-viewports`。

嵌入其他 Python 程序时，在 SimulationApp 启动后执行 `import openso101.tasks` 注册内置任务。`import openso101` 可用于独立资产处理程序。

## 测试

```bash
OPENSO101_SKIP_ISAAC=1 PYTHONPATH=src python -m pytest tests/test_cpu_regressions.py \
  -q
PYTHONPATH=src python -m pytest --confcutdir=tests/scenes tests/scenes \
  --basetemp=outputs/pytest-scenes -q
```

测试依赖为 `scenes` extra、pytest 与 usd-core。测试使用实际 GLB 文件、OpenUSD 场景和可计算尺寸的几何体，覆盖配置拒绝条件、视频抽帧、生成资产、agent loop 修复路径、文件修改检查、场景包迁移及 USD 属性。在线 Objaverse 下载和目标主机的 Isaac Sim 转换分别执行集成检查。

## 2026-09-26 运行记录

| 检查 | 结果 |
|---|---|
| 场景、控制、配置与 OpenUSD 测试 | 28 项通过 |
| 现有 RL CLI 回归 | 8 项通过 |
| 静态检查与安装包 | Ruff、diff 检查和 wheel 构建通过 |
| 在线资产 | Objaverse Apple `4c19ae47dbe8468285ee53ff487fe51a`，作者 `fzcdragun`，原始 license 字段为 `by` |
| 目标运行环境 | Linux、Python 3.11.16、Isaac Sim 5.1.0、报告型号为 NVIDIA H20G 的 GPU；转换器版本 5.0.17 |
| USD 编译 | 生成独立资产、物理属性与完整 SHA256 清单 |
| 跨主机加载 | 从 Linux 复制到 macOS，用 OpenUSD 加载成功，未解析依赖为 0 |
| 几何尺寸 | 指定 0.04 × 0.04 × 0.05 m，加载后包围盒尺寸符合要求 |
| PhysX 静置 | 1200 步、10 秒；最后 2 秒最大位移 `8.9968e-8 m`，旋转变化 `0°` |

SO-101 自定义场景已执行实际控制步骤；双相机生成了 12 帧 HDF5 数据，并通过 LeRobot 全部帧读取、场景 hash 校验和 Isaac 回放。五个 RoboTwin 模板分别通过四个并行环境、100 次 reset 和 200 个控制步骤检查。上述数据记录为失败 episode，用于检查采集流程。实际模型调用、成功任务示范、人工键盘操作与 leader 设备验收仍需对应运行条件。
