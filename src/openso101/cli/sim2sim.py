def add_subparsers(parser):
    sub = parser.add_subparsers(dest="command", required=True)
    evaluation = sub.add_parser("mujoco", help="在 MuJoCo 中运行实际 Isaac 初始场景与共享策略")
    evaluation.add_argument("--policy", required=True)
    evaluation.add_argument("--robot-model", required=True)
    evaluation.add_argument("--episodes", type=int, default=4)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--recorded-physics", action="store_true", help="使用源初始环境的实际实体、重力和关节参数")
    evaluation.add_argument("--constrained-drive", action="store_true", help="使用实际速度和力矩受限的 implicit PD 控制")
    evaluation.add_argument("--collision-bundle", help="使用已校验的 CoACD convex parts")
    evaluation.set_defaults(func=_evaluate)
    comparison = sub.add_parser("compare", help="比较相同 Isaac 关节目标驱动下的 MuJoCo 动力学")
    comparison.add_argument("--policy", required=True)
    comparison.add_argument("--robot-model", required=True)
    comparison.add_argument("--episodes", type=int, default=4)
    comparison.add_argument("--steps", type=int, default=500)
    comparison.add_argument("--recorded-pd", action="store_true", help="使用 Isaac 记录的每环境实际 PD 参数")
    comparison.add_argument("--velocity-servo", action="store_true", help="原生 velocity servo 使用受限速度目标，需要实际速度限制记录")
    comparison.add_argument("--constrained-drive", action="store_true", help="使用实际速度和力矩受限的 implicit PD 控制")
    comparison.add_argument("--collision-bundle", help="使用已校验的 CoACD convex parts")
    comparison.add_argument("--physics-components", nargs="+", choices=["bodies", "gravity", "armature", "friction"], default=[],
                            help="使用实际实体参数、重力、armature 或已验证的零关节摩擦")
    comparison.add_argument("--output", required=True)
    comparison.set_defaults(func=_compare)
    physics = sub.add_parser("physics", help="比较实际 Isaac 实体质量、惯性、COM 与 MuJoCo 参数")
    physics.add_argument("--policy", required=True)
    physics.add_argument("--robot-model", required=True)
    physics.add_argument("--output", required=True)
    physics.set_defaults(func=_physics)
    diagnostics = sub.add_parser("diagnose", help="检查实际 Isaac 与 MuJoCo 轨迹中的动作、夹爪接触和控制误差")
    diagnostics.add_argument("--policy", required=True)
    diagnostics.add_argument("--robot-model", required=True)
    diagnostics.add_argument("--evaluation", required=True, help="实际 MuJoCo 评估目录")
    diagnostics.add_argument("--output", required=True)
    diagnostics.set_defaults(func=_diagnose)


def _evaluate(args):
    from openso101.sim2sim.mujoco import evaluate

    return evaluate(args)


def _compare(args):
    from openso101.sim2sim.comparison import compare

    return compare(args)


def _physics(args):
    from openso101.sim2sim.body_physics import compare_body_physics

    return compare_body_physics(args)


def _diagnose(args):
    from openso101.sim2sim.diagnostics import diagnose

    return diagnose(args)
