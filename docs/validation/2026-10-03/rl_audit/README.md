# RL 与场景服务运行检查

## 来源

原生物理检查来自固定提交 `5b46a2f`，物理周期为 0.001 秒，控制周期为 0.02 秒。每种条件包含四个真实 Isaac 环境、500 个控制步骤，共 40,000 个物理步骤；速度直接记录自 PhysX 每个物理步骤。完整报告位于 `native_default_order/`。

| 任务与条件 | 物理采样最高速度 rad/s | 目标转换最大误差 rad | 速度检查 |
|---|---:|---:|---|
| Lift nominal | 1.157624 | 0 | 通过 |
| PickPlace nominal | 1.714590 | 0 | 通过 |
| Lift randomized | 1.552026 | 0 | 通过 |
| PickPlace randomized | 1.727205 | 0 | 通过 |

完整 episode 的折扣 shaping 累计绝对值均低于 `5e-7`。检查报告保留实际 trace 和源码 SHA256。以上记录证明报告中的采样范围；任务成功未通过。`native/` 保留各自环境参数下的独立原始记录。

![原生物理检查](figures/native_physics.png)

## 训练、导出与 MuJoCo

训练来源为 `85af781`：Lift、32 个环境、两次 PPO 更新，共 6,144 transitions。独立评估使用 seed `10042`，100 episodes 的成功率为零，接近率为 19%，抓取与持续提升率为零。完整逐 episode 数据保存在 `evaluation_history.json`。

checkpoint SHA256 为 `fa287ff2614db8f93e9867ab9b491a799070a46175ec5120582a3adb4249f365`。导出的策略在四个原生 Isaac 环境执行 300 步，观测重建误差为零，actor 和目标转换最大误差均为 `1.1920928955e-7`。

同一策略使用官方 `so101_old_calib.xml`、实际源环境物理参数、CoACD 夹爪碰撞和 OSQP 受限 implicit PD，在 MuJoCo 3.14.0 执行四个 episode，共 10,000 个物理步骤。实际速度与力矩检查通过，成功率为零。1,200 个末端坐标的最大位置误差为 `1.9275e-6` m，最大旋转误差为 `1.0678e-5` rad。策略与 source trace SHA256、实体参数和每个 episode 的检查结果保存在 `mujoco_report.json`。

`8965174` 的固定 learning rate 检查包含 6,144 transitions 和 40 次梯度更新，actual learning rate 均为 `0.0001`。每次更新的 checkpoint 保存完成；100 episodes 独立评估为 `0/100`，接近物体比例为 17%，完整记录位于 `checked_ppo_evaluation.json`。`5e257c3` 的持续进展奖励训练运行于独立目录，原始模型和记录全部保留。

`training_scalars_32b58c6.json` 保存实际 TensorBoard 数值与已有 checkpoint 的参数有限性检查。图表直接使用原始数值，并通过 SHA256 关联文件。

![训练记录](figures/training_records.png)

## 模型服务

`gpt-6-astra` 的真实 Responses 请求经过完整 SSE 完成事件和 Pydantic schema 检查，保留输入任务与释放要求。请求用量为 153 input tokens、46 output tokens，共 199 tokens。请求和响应 SHA256 保存于 `model_completion.json`，记录不包含密钥。预算按实际 HTTP 请求次数计算。

CPU 检查为 44 项通过，四项使用替代服务或对象的测试未执行。

## TaskIntent、TaskProgram 与资产

`scenes/typed_intent_job.json` 保存真实 Astra 的两次请求，共 6,806 tokens、22.062 秒。TaskIntent 版本 2 保存完整文本、尺寸、初始 Pose 和最终目标。所用资产 UID 为 `811698689ddf00274a8abce36f4e159a`，来源为 `generated://trimesh/recipe`。场景 SHA256 为 `f8b5e582908483dbb33df65172903912eb751f426e4a582f4047d3285bb0a6a0`，TaskProgram SHA256 为 `9d7f6e141d7147069ff54e632e2b2ddb1b53f683be4514da4117cb3122e1684c`。该记录完成数值参数、条件、文件内容与静态布局检查，外观匹配和任务成功需要独立验收。

`scenes/program_runtime.json` 保存四环境、100 次 reset、200 步的实际 TaskProgram 检查。CPU 与 Isaac 检查器逐步读取相同的实体状态与双侧接触力，共 800 次结果比较，每台相机检查 800 帧。观测到的最大阶段为 0，双侧接触次数为 0，任务成功次数为 0。完整 HDF5 轨迹通过报告中的 SHA256 关联。

资产 embedding 的独立检查使用 Objaverse Apple `4c19ae47dbe8468285ee53ff487fe51a`，资产 SHA256 为 `709a61d29096e6d68730b9debc30c6816caf9b31315c1186017bd121814557f2`。模型为 `sentence-transformers/all-MiniLM-L6-v2`，固定 commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，384 维 normalized embedding；`red apple` 的 cosine similarity 为 `0.787699`。当前检查覆盖单个实际资产。

![任务阶段与接触记录](figures/scene_program.png)
