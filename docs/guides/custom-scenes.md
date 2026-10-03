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
  --output outputs/apple_job
openso101 scenes inspect 4c19ae47dbe8468285ee53ff487fe51a
openso101 scenes layout outputs/apple_job/bundle/scene.json \
  --locked object --output outputs/apple_layout.json
```

模型接口为 OpenAI-compatible Chat Completions，服务地址包含其 API 前缀。默认从 `SCENE_MODEL_API_KEY` 读取密钥；`--api-key-env` 指定其他变量，无鉴权服务使用空字符串。模型名称与地址也可通过 `SCENE_MODEL_NAME` 和 `SCENE_MODEL_BASE_URL` 配置。

`generate` 将任务转换为 `TaskIntent`，生成资产候选与 `SceneSpec`，编译 `TaskProgram`，执行静态布局检查，并保存 `job.json` 与 `bundle/`。TaskIntent 版本 2 保存每个实体的英文检索文本、数值尺寸、初始 Pose、操作顺序和最终条件。SceneSpec 必须保留任务文本、尺寸、初始 Pose 和全部最终目标。需要用户补充信息时返回 `needs_input`，并保存具体要求。

`TaskProgram` 按照顺序检查 pick、lift、move、place、release 和 wait 条件。抓取要求实际双侧接触力均超过 0.5 N；释放要求打开夹爪且双侧接触力均不超过 0.1 N。最终目标同时检查位置、速度、释放与连续稳定时间。具有 TaskProgram 的场景观测增加三个阶段数值；已有场景保持原有观测数量。

```bash
openso101 scenes index --catalog outputs/assets --output outputs/asset_index
openso101 scenes retrieve "red apple" --catalog outputs/assets --index outputs/asset_index
openso101 scenes generate --instruction "把苹果放置到桌面右侧" \
  --catalog outputs/assets --asset-index outputs/asset_index \
  --base-url "$SCENE_MODEL_BASE_URL" --model "$SCENE_MODEL_NAME" \
  --output outputs/apple_job --runtime-output outputs/apple_prepared
```

embedding 索引使用 Sentence Transformers，将模型固定到 Hugging Face commit，保存实际资产与 metadata 的 SHA256、embedding 数量和文件 SHA256。检索使用 normalized cosine similarity；场景中的实体只能选择对应检索结果中的资产。资产或 metadata 改变后，读取索引立即终止并要求重新生成。默认模型为 `sentence-transformers/all-MiniLM-L6-v2`，`--embedding-model` 与 `--model-revision` 可指定模型及版本。

默认预算为 12 次模型请求、最多三次场景修复、64 项候选、16 个实体和 600 秒实际运行检查。`job.json` 保存每项工具的输入、输出、用时、SHA256，以及实际模型请求与 tokens 用量。当前文本流程读取已有资产，下载量为零。成功的静态检查使用 `layout_valid`；实际编译与双相机运行通过后使用 `simulation_ready`。任务成功和数据集验收通过独立记录确认。

调用前配置本次使用的模型服务；输出格式和场景校验错误立即终止。`--codex-config` 可使用 Responses 服务设置，密钥只在当前进程中读取。资产、布局、版本保存、编译与仿真检查均可独立调用。

## RGB 视频 real2sim agent loop

`agent-loop` 自动从 RGB 视频均匀抽帧，把帧和场景上下文交给 GPT-6 Astra，依次完成视频描述、Objaverse/LVIS 检索、场景编排、缺失 primitive/generated parts 资产生成、静态几何检查、物理合理性审查和 SO-101 数据采集 readiness 审查。模型发现问题时最多自动修复两轮，并把每轮反馈保存到结果 JSON。

用户提供的 `--instruction` 在视频描述、首次编排及每轮修复中保持原文。没有显式任务文本时，使用视频描述中的任务文本。结果中的 `scene_revisions` 保存每个场景版本的编号、场景 SHA256、任务原文和文本来源；模型审查与修改建议保存在对应反馈字段。

```bash
openso101 scenes agent-loop \
  --video captures/pick_place.mp4 \
  --sample-count 8 --frames-dir outputs/pick_place_frames \
  --base-url "$SCENE_MODEL_BASE_URL" --model gpt-6-astra \
  --catalog outputs/assets --output outputs/agent_scene_bundle
```

`agent-loop` 会自动发现本机的
`/mnt/data/users/jixin/workspace/code/LitchiAgent/runtime/codex/config.toml`；也可以用
`--codex-config PATH` 或 `CODEX_CONFIG` 指定 Codex TOML。配置中的 `base_url`、模型、
`wire_api = "responses"`、reasoning effort 和 bearer token 只在当前进程中使用，token
不会写入场景或日志。也可以显式传 `--base-url`、`--model` 和 `--api-key-env` 覆盖地址、
模型和密钥环境变量。Responses API 使用 Codex 兼容的 SSE 流，收到完整结构化 JSON 后
立即结束本次请求。

默认会在线搜索 Objaverse；网络不可用或只需验证生成资产分支时使用
`--offline-objaverse`。`--image-limit`、`--reasoning-effort`、`--timeout-seconds` 和
`--max-revisions` 可分别控制每次请求的帧数、推理强度、单次超时和修复次数。

也可以用 `--frame` 重用已经抽好的 JPEG/PNG；这时必须同时提供 `--fps`、`--frame-count`、`--width` 和 `--height`。静态检查和两个 Astra 审查通过时使用 `review_passed`；Isaac 的动态碰撞、可达性、接触、相机和成功采集使用各自的实际记录。

`--runtime-output` 将生成 bundle 连接到 USD 编译和实际 Isaac 双相机检查，保存 `compiled/`、`runtime.json`、`preparation.json` 和 `agent_loop_result.json`。结果增加 `compiled_scene` 与 `runtime_validation`；`pending_checks` 使用实际运行报告中仍需完成的项目。模型审查结果、程序运行和任务成功分别保留其验证范围。

```bash
openso101 scenes agent-loop --video captures/pick_place.mp4 \
  --instruction "使用 SO-101 将苹果移动到桌面右侧" \
  --catalog outputs/assets --offline-objaverse \
  --output outputs/agent_scene_bundle --runtime-output outputs/agent_scene_ready \
  --runtime-num-envs 4 --runtime-resets 100 --runtime-steps 200
```

生成 bundle 和运行输出使用独立的新目录。调用模型之前检查运行输出路径和正数检查数量；编译、文件 hash 或运行检查失败时立即终止。

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

已有 bundle 可以通过统一入口执行相同的完整程序检查：

```bash
openso101 scenes prepare outputs/apple_bundle --output outputs/apple_prepared \
  --num-envs 4 --resets 100 --steps 200
openso101 il record --task OpenSO101-CustomScene-v0 \
  --scene outputs/apple_prepared/compiled --teleop-device keyboard \
  --repo-root outputs/apple_dataset
```

`prepare` 检查源 bundle、编译产物和场景 SHA256，执行全部 reset 范围检查，以及每个控制步骤的关节、实体、观测、reward 和双相机图像检查。相机报告包含形状、检查帧数和最小像素标准差；另行记录 episode 终止数量、控制周期及检查代码 SHA256。四个环境执行 200 个步骤时，每个相机检查 800 帧。

`preparation.json` 保留源 manifest、编译清单与 runtime 报告的 SHA256，并验证请求数量与实际数量一致。`runtime_verified` 表示上述程序检查通过；动态稳定性、完整路径、接触几何、任务物体可见性、任务完成和成功采集保留在 `pending_checks`，`task_success_verified` 与 `dataset_verified` 保持 `false`。

实际验证覆盖已有 Apple bundle 和真实 Astra 调用生成的 `so101_apple_move_right`。每项均完成四环境、100 次 reset、200 个控制步骤，每个相机检查 800 帧，关节、实体、观测和 reward 数值有效。生成场景的模型状态保持 `needs_review`；输入来自已保存仿真视频，度量重建与成功任务仍需验收。该生成场景另行录制双相机各 720 帧、512×512、60 FPS，供完整流程 MP4 演示使用；报告见 [运行记录](../validation/2026-09-29/README.md)。

完整流程演示已生成 35 秒、1080p、30 FPS 的 MP4，全部 1050 帧解码检查通过。录制与视频构建脚本、目录要求和报告见 [演示说明](../validation/2026-09-29/README.md#agentic-场景生成演示)。

键盘方向键控制平面移动，PageUp/PageDown 控制高度，A/D 控制旋转，Space 打开夹爪，Shift 关闭夹爪。damped least-squares IK 使用实际 Jacobian、关节限位与速度限制。leader 使用 `--teleop-device leader` 及原有设备参数。

自定义场景 HDF5 保存全部实体状态、双侧接触力、任务阶段、保持时间、成功状态、场景 hash 与可移植场景副本。录制检查点与回放恢复这些任务状态。LeRobot 导出保留场景副本，并在 `meta/scenes.json` 保存 episode 对应关系。无显示环境回放使用 `--headless --no-camera-viewports`。

嵌入其他 Python 程序时，在 SimulationApp 启动后执行 `import openso101.tasks` 注册内置任务。`import openso101` 可用于独立资产处理程序。

## 测试

```bash
OPENSO101_SKIP_ISAAC=1 TMPDIR="$PWD/outputs/tmp" PYTHONPATH=src \
  python -m pytest tests/scenes tests/test_cpu_regressions.py \
  -k 'not eagerly and not model_service and not agent_loop_materializes' \
  --basetemp=outputs/pytest-scenes -q
```

测试依赖为 `scenes` extra、pytest、usd-core、Torch 与 h5py。上述检查使用实际 GLB、OpenUSD、HDF5 和编码后的 MP4，覆盖配置拒绝条件、视频抽帧、控制计算、文件修改检查、场景包迁移及 USD 属性。模型服务、agent loop、在线 Objaverse 下载和 Isaac Sim 分别执行实际集成检查。当前结果见 [v2 状态记录](v2-status-2026-09-29.md)。

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

SO-101 自定义场景已执行实际控制步骤；双相机生成了 12 帧 HDF5 数据，并通过 LeRobot 全部帧读取、场景 hash 校验和 Isaac 回放。五个 RoboTwin 模板分别通过四个并行环境、100 次 reset 和 200 个控制步骤检查。上述数据记录为失败 episode，用于检查采集流程。成功任务示范、人工键盘操作与 leader 设备验收仍需对应运行条件。新增 Astra agent loop 的运行信息和验收范围见 [当前状态记录](v2-status-2026-09-29.md)。
