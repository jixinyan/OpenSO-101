def add_subparsers(parser):
    sub = parser.add_subparsers(dest="command", required=True)
    evaluation = sub.add_parser("mujoco", help="在 MuJoCo 中运行实际 Isaac 初始场景与共享策略")
    evaluation.add_argument("--policy", required=True)
    evaluation.add_argument("--robot-model", required=True)
    evaluation.add_argument("--episodes", type=int, default=4)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(func=_evaluate)


def _evaluate(args):
    from openso101.sim2sim.mujoco import evaluate

    return evaluate(args)
