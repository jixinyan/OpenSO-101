import argparse
import os
from pathlib import Path

from openso101.rl.rsl_execution import (
    _ALGO_TO_ENTRY_POINT,
    _ALL_ALGOS,
    _cmd_eval,
    _cmd_play,
    _cmd_train,
)


def _cmd_snapshot(args):
    from openso101.rl.snapshot import snapshot

    return snapshot(args)


def _cmd_distill(args):
    from openso101.rl.execution import distill

    return distill(args)


def _cmd_student_eval(args):
    from openso101.rl.execution import evaluate_student

    return evaluate_student(args)


def _cmd_validate_loop(args):
    from openso101.rl.validation_loop import validate_loop

    return validate_loop(args)


def _cmd_campaign(args):
    from openso101.rl.campaign import campaign

    return campaign(args)


def _cmd_export(args):
    from openso101.rl.export import export

    return export(args)


def _cmd_plot(args):
    from openso101.rl.plotting import plot_training

    return plot_training(args)


def add_subparsers(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="rl_cmd", required=True)

    p_distill = sub.add_parser("distill", help="将 PPO teacher 蒸馏为双相机 student")
    p_distill.add_argument("--teacher-run", required=True)
    p_distill.add_argument("--output", required=True)
    p_distill.add_argument("--num-envs", dest="num_envs", type=int, default=16)
    p_distill.add_argument("--iterations", type=int, default=1500)
    p_distill.add_argument("--rollout-steps", type=int, default=16)
    p_distill.add_argument("--headless", action="store_true")
    p_distill.set_defaults(func=_cmd_distill)

    p_student_eval = sub.add_parser("student-eval", help="独立评估双相机与任务目标 student")
    p_student_eval.add_argument("--student", required=True)
    p_student_eval.add_argument("--num-envs", dest="num_envs", type=int, default=16)
    p_student_eval.add_argument("--n-episodes", dest="n_episodes", type=int, default=100)
    p_student_eval.add_argument("--seed", type=int, default=10042)
    p_student_eval.add_argument("--headless", action="store_true")
    p_student_eval.add_argument("--recording-output", help="保存完整 episode 的双相机与关节目标 HDF5")
    p_student_eval.set_defaults(func=_cmd_student_eval)

    p_loop = sub.add_parser("validate-loop", help="验证 teacher、MuJoCo、student 与实际采集")
    p_loop.add_argument("--teacher-run", required=True)
    p_loop.add_argument("--output", required=True)
    p_loop.add_argument("--robot-model", required=True)
    p_loop.add_argument("--collision-bundle", required=True)
    p_loop.add_argument("--mujoco-python", required=True)
    p_loop.add_argument("--num-envs", type=int, default=64)
    p_loop.add_argument("--student-num-envs", type=int, default=16)
    p_loop.add_argument("--distillation-iterations", type=int, default=1500)
    p_loop.add_argument("--seed", type=int, default=30042)
    p_loop.set_defaults(func=_cmd_validate_loop)

    p_train = sub.add_parser("train", help="训练 RL 策略")
    p_train.add_argument("--task", required=True, help="Gym ID")
    p_train.add_argument("--task-profile", choices=("default", "grasp_v2", "grasp_v3", "grasp_v4"))
    p_train.add_argument("--backend", choices=("rsl_rl", "sb3", "skrl", "rl_games"))
    p_train.add_argument("--train-config", help="TrainCfg JSON")
    p_train.add_argument("--output", help="新的模型与记录目录")
    p_train.add_argument("--scene", type=Path, help="已编译的自定义场景目录")
    p_train.add_argument("--source-revision", default=os.environ.get("OPENSO101_SOURCE_REVISION"))
    p_train.add_argument("--algo", required=True, choices=_ALL_ALGOS)
    p_train.add_argument("--teacher-checkpoint", help="distillation 所需的 teacher 模型目录或 .pt 文件")
    p_train.add_argument("--num_envs", type=int, default=None)
    p_train.add_argument("--seed", type=int, default=None, help="环境和策略使用的 seed，-1 选择随机 seed")
    p_train.add_argument("--max_iterations", type=int, default=None)
    p_train.add_argument("--resume", action="store_true", default=False)
    p_train.add_argument("--load_run", type=str, default=None)
    p_train.add_argument("--checkpoint", type=str, default=None)
    p_train.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--video_length", type=int, default=200)
    p_train.add_argument("--video_interval", type=int, default=2400)
    p_train.add_argument("--logger", choices=("wandb", "tensorboard", "neptune"), default=None)
    p_train.add_argument("--log_project_name", default="openso101")
    p_train.add_argument("--with-cameras", action="store_true")
    p_train.add_argument("--visual-dr", action="store_true", help="每次 reset 随机设置光照与物体颜色")
    p_train.add_argument("--headless", action="store_true")
    p_train.set_defaults(func=_cmd_train)

    p_snapshot = sub.add_parser("snapshot", help="保存 checkpoint 副本，用于独立评估")
    p_snapshot.add_argument("--run", required=True, type=Path)
    p_snapshot.add_argument("--checkpoint", required=True)
    p_snapshot.add_argument("--output", required=True, type=Path)
    p_snapshot.set_defaults(func=_cmd_snapshot)

    p_campaign = sub.add_parser("campaign", help="Lift 与 PickPlace 的三个 seed 独立训练和评估")
    p_campaign.add_argument("--train-config", required=True)
    p_campaign.add_argument("--output", required=True)
    p_campaign.add_argument("--seeds", type=int, nargs=3, default=[42, 43, 44])
    p_campaign.add_argument("--gpus", type=int, nargs="+", required=True)
    p_campaign.add_argument("--num-envs", dest="num_envs", type=int, default=2048)
    p_campaign.add_argument("--task-profile", choices=("grasp_v3", "grasp_v4"), default="grasp_v4")
    p_campaign.add_argument("--initial-runs", type=Path, help="每个任务与 seed 的已校验初始模型 JSON")
    p_campaign.add_argument("--validate-loop", action="store_true")
    p_campaign.add_argument("--robot-model")
    p_campaign.add_argument("--collision-bundle")
    p_campaign.add_argument("--mujoco-python")
    p_campaign.add_argument("--distillation-iterations", type=int, default=1500)
    p_campaign.set_defaults(func=_cmd_campaign)

    p_play = sub.add_parser("play", help="运行保存的策略")
    p_play.add_argument("--task", required=True)
    p_play.add_argument("--checkpoint", required=True)
    p_play.add_argument("--num_envs", type=int, default=None)
    p_play.add_argument("--with-cameras", action="store_true")
    p_play.add_argument("--deterministic", action="store_true", help="使用策略的 action mean")
    p_play.add_argument("--headless", action="store_true")
    p_play.set_defaults(func=_cmd_play)

    p_eval = sub.add_parser("eval", help="独立评估任务成功与接近、抓取、抬升")
    p_eval.add_argument("--task", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--num-envs", dest="num_envs", type=int, default=64)
    p_eval.add_argument("--n-episodes", dest="n_episodes", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--recording-output", help="保存完整 episode 的双相机与物理状态")
    p_eval.add_argument("--camera-resolution", type=int, default=256)
    p_eval.add_argument("--headless", action="store_true")
    p_eval.set_defaults(func=_cmd_eval)

    p_export = sub.add_parser("export", help="导出策略与实际 Isaac 数值诊断")
    p_export.add_argument("--task", required=True)
    p_export.add_argument("--checkpoint", required=True)
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--num-envs", dest="num_envs", type=int, default=4)
    p_export.add_argument("--validation-steps", type=int, default=256)
    p_export.add_argument("--seed", type=int)
    p_export.add_argument("--headless", action="store_true")
    p_export.set_defaults(func=_cmd_export)

    p_plot = sub.add_parser("plot", help="读取全部 TensorBoard 记录并生成训练曲线")
    group = p_plot.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="模型目录，与 --log_dir 相同")
    group.add_argument("--log_dir", help="模型目录")
    group.add_argument("--task", help="logs/rsl_rl 下任务的最新目录")
    p_plot.add_argument("--smooth", type=int, default=30)
    p_plot.add_argument("--save", action="store_true")
    p_plot.add_argument("--output", type=Path, help="新的图表和来源报告目录")
    p_plot.add_argument("--font-file", type=Path, help="包含中文的字体文件，默认使用 Noto Sans CJK SC")
    p_plot.set_defaults(func=_cmd_plot)
