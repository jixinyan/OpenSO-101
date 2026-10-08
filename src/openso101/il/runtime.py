from contextlib import ExitStack
from functools import wraps
from pathlib import Path


def _with_cleanup(command):
    @wraps(command)
    def run(args):
        with ExitStack() as cleanup:
            return command(args, cleanup)

    return run


def _launch_isaac_app(args, cleanup: ExitStack, enable_cameras: bool = True):
    if getattr(args, "scene", None):
        from openso101.scenes.isaaclab.usd import verify_compilation

        scene = Path(args.scene).expanduser().resolve()
        verify_compilation(scene)
        args.scene = str(scene)
    from openso101.rl.gpu_scope import configure_visible_gpu

    configure_visible_gpu()
    from isaaclab.app import AppLauncher

    args.enable_cameras = enable_cameras
    if not hasattr(args, "headless"):
        args.headless = False
    if not hasattr(args, "device"):
        args.device = "cuda:0"
    if not hasattr(args, "disable_fabric"):
        args.disable_fabric = False
    app = AppLauncher(args).app
    cleanup.callback(app.close)
    if getattr(args, "scene", None):
        from openso101.scenes.isaaclab.runtime import register_custom_scene

        register_custom_scene()
    return app


def resolve_policy_path(path: str) -> Path:
    from openso101.il.policies.factory import _resolve_checkpoint_dir, validate_checkpoint_files

    local = Path(path).expanduser()
    if (local / "student.json").is_file():
        if not (local / "student.pt").is_file():
            raise FileNotFoundError(f"student 权重文件不存在: {local / 'student.pt'}")
        return local.resolve()
    root = _resolve_checkpoint_dir(path)
    validate_checkpoint_files(root)
    return root


def validate_positive_count(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} 必须为正整数")
