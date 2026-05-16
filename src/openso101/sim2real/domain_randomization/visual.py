# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Visual domain randomization: lighting + material colors at episode reset.

Cheap perception sim2real wins flagged independently by both
``leisaac`` (``utils/domain_randomization.py``) and
``liorbenhorin/lerobot_so101_teleop`` (``tasks/base/base_env_cfg.py``).
Both reset light exposure and recolor scene assets per episode so a
single dataset covers a range of lighting conditions.

Each randomizer is a plain Isaac Lab event-manager function — callable
as ``func(env, env_ids, ...)`` — that you wire as
``EventTerm(func=..., mode="reset", params={...})``. The
:func:`attach_visual_dr` helper does that wiring for you on a task's
``EventCfg`` dataclass instance.

Wiring example::

    from openso101.sim2real.domain_randomization.visual import attach_visual_dr

    cfg = PickPlaceEnvCfg()
    attach_visual_dr(cfg.events)   # mutates cfg.events in-place
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Randomizer functions (called by Isaac Lab's EventManager each reset)
# ---------------------------------------------------------------------------


def randomize_dome_light_intensity(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    *,
    prim_path: str = "/World/light",
    intensity_range: tuple[float, float] = (1500.0, 4500.0),
) -> None:
    """Resample the dome light intensity uniformly each reset.

    The default ``DomeLightCfg`` ships at ``intensity=3000.0``; the
    ``(1500, 4500)`` range covers roughly ±50% exposure — broad enough
    to span "dim indoor" to "bright daylight" without crushing the cube
    rendering at either extreme.
    """
    intensity = _sample_uniform_scalar(env, intensity_range)
    _set_usd_attr(env, prim_path, "inputs:intensity", intensity)


def randomize_dome_light_color(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    *,
    prim_path: str = "/World/light",
    color_jitter: tuple[float, float, float] = (0.15, 0.15, 0.15),
    base_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
) -> None:
    """Add per-channel uniform jitter around the base dome-light color.

    Models the small color-temperature shift between real-world lighting
    setups (cool office fluorescent vs. warm halogen). Channels are
    independently sampled then clamped to [0, 1] so the renderer never
    sees an out-of-gamut color.
    """
    color = _sample_per_channel_color(env, base_color, color_jitter)
    _set_usd_attr(env, prim_path, "inputs:color", color, value_type="color3f")


def randomize_object_color(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    *,
    asset_name: str = "object",
    hue_jitter: float = 0.30,
    saturation_range: tuple[float, float] = (0.5, 1.0),
    value_range: tuple[float, float] = (0.5, 1.0),
) -> None:
    """Recolor the manipulated object's PreviewSurface per reset.

    Samples in HSV (perceptually-uniform-ish) and converts to RGB so a
    "green cube" task sweeps through every cube color the policy might
    see on a real lab table. ``hue_jitter=0.30`` covers ~108° of the
    color wheel — enough to flip green↔yellow↔red without staying in a
    single hue family.
    """
    rgb = _sample_jittered_hsv(env, hue_jitter, saturation_range, value_range)
    asset_prim = _scene_asset_prim(env, asset_name)
    if asset_prim is None:
        return
    # The PreviewSurface shader lives one level below the asset prim in
    # the canonical Isaac Lab spawn layout. We walk the prim tree to find
    # it rather than hardcoding a sub-path so this works for any spawner.
    for shader_path, attr_name in _iter_preview_surface_attributes(asset_prim):
        _set_usd_attr_absolute(shader_path, attr_name, rgb, value_type="color3f")


# ---------------------------------------------------------------------------
# Wiring helper
# ---------------------------------------------------------------------------


def attach_visual_dr(
    events,
    *,
    enable_light_intensity: bool = True,
    enable_light_color: bool = True,
    enable_object_color: bool = True,
    object_asset_name: str | tuple[str, ...] = "object",
    dome_light_prim_path: str = "/World/light",
) -> None:
    """Attach visual DR EventTerms to a task's ``EventCfg`` instance.

    All randomizers run in ``mode="reset"`` so they re-sample on every
    episode boundary. Flip the ``enable_*`` flags off to selectively
    disable individual terms — useful when ablating which DR axis
    contributes most to sim2real transfer.

    ``object_asset_name`` accepts either a single asset name (single-
    object tasks like Lift / PickPlace) or a tuple of names (multi-
    object tasks like Stack, where both ``cube_bottom`` and
    ``cube_top`` should be recolored independently).
    """
    from isaaclab.managers import EventTermCfg as EventTerm

    if enable_light_intensity:
        events.dr_dome_light_intensity = EventTerm(
            func=randomize_dome_light_intensity,
            mode="reset",
            params={"prim_path": dome_light_prim_path},
        )
    if enable_light_color:
        events.dr_dome_light_color = EventTerm(
            func=randomize_dome_light_color,
            mode="reset",
            params={"prim_path": dome_light_prim_path},
        )
    if enable_object_color:
        asset_names = (
            (object_asset_name,) if isinstance(object_asset_name, str) else tuple(object_asset_name)
        )
        for asset_name in asset_names:
            term_attr = f"dr_object_color_{asset_name}" if len(asset_names) > 1 else "dr_object_color"
            setattr(
                events,
                term_attr,
                EventTerm(
                    func=randomize_object_color,
                    mode="reset",
                    params={"asset_name": asset_name},
                ),
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_uniform_scalar(env, value_range: tuple[float, float]) -> float:
    lo, hi = float(value_range[0]), float(value_range[1])
    return float(torch.empty((), device=env.device).uniform_(lo, hi).item())


def _sample_per_channel_color(env, base, jitter) -> tuple[float, float, float]:
    base_t = torch.tensor(base, device=env.device, dtype=torch.float32)
    jit_t = torch.tensor(jitter, device=env.device, dtype=torch.float32)
    noise = (torch.rand(3, device=env.device) * 2.0 - 1.0) * jit_t
    out = (base_t + noise).clamp(0.0, 1.0)
    return (float(out[0]), float(out[1]), float(out[2]))


def _sample_jittered_hsv(
    env, hue_jitter: float, sat_range: tuple[float, float], val_range: tuple[float, float]
) -> tuple[float, float, float]:
    # Anchor hue at the existing cube color (a warm orange) and sweep
    # around it. This keeps the average look on-brand while still
    # exposing the policy to a wide swath of the color wheel.
    base_hue = 0.083  # ~30° on the wheel = orange in HSV
    h = (base_hue + (torch.rand((), device=env.device).item() * 2.0 - 1.0) * hue_jitter) % 1.0
    s = _sample_uniform_scalar(env, sat_range)
    v = _sample_uniform_scalar(env, val_range)
    return _hsv_to_rgb(h, s, v)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    # Standard HSV -> RGB; kept inline so we don't depend on colorsys
    # for what is essentially a one-line conversion. Mirrors the math in
    # ``matplotlib.colors.hsv_to_rgb`` for a single pixel.
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    table = ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))
    return table[i % 6]


def _scene_asset_prim(env, asset_name: str):
    """Return the Usd.Prim for a scene asset, or None if the asset is missing.

    The look-up goes through Isaac Lab's scene registry first (which
    works for both single- and multi-env scenes), then falls back to a
    direct USD stage query for assets that exist outside the regex
    namespace (e.g. world-fixed assets).
    """
    try:
        from pxr import Usd
    except ImportError:
        return None
    stage = _get_stage()
    if stage is None:
        return None
    asset = env.scene.get(asset_name)
    if asset is None:
        return None
    prim_paths = asset.cfg.prim_path
    # In multi-env scenes the prim_path contains "{ENV_REGEX_NS}"; touch
    # the env-0 instance — material attrs are shared across envs via
    # instancing so writing to env-0 propagates.
    prim_path = prim_paths.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    prim = stage.GetPrimAtPath(prim_path)
    return prim if prim and prim.IsValid() else None


def _iter_preview_surface_attributes(root_prim):
    """Yield (prim_path, attr_name) for each PreviewSurface diffuseColor under root."""
    try:
        from pxr import Usd, UsdShade
    except ImportError:
        return
    for descendant in Usd.PrimRange(root_prim):
        if not descendant.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(descendant)
        info_id = shader.GetIdAttr().Get()
        if info_id != "UsdPreviewSurface":
            continue
        yield str(descendant.GetPath()), "inputs:diffuseColor"


def _get_stage():
    try:
        import omni.usd
    except ImportError:
        return None
    ctx = omni.usd.get_context()
    return ctx.get_stage() if ctx is not None else None


def _set_usd_attr(env, prim_path: str, attr_name: str, value, value_type: str = "float"):
    """Set a USD attribute, resolving ``{ENV_REGEX_NS}`` to env_0 if present."""
    resolved = prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    _set_usd_attr_absolute(resolved, attr_name, value, value_type)


def _set_usd_attr_absolute(prim_path: str, attr_name: str, value, value_type: str = "float"):
    stage = _get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    attr = prim.GetAttribute(attr_name)
    if not attr:
        return
    if value_type == "color3f" and not isinstance(value, tuple):
        value = tuple(value)
    attr.Set(value)


__all__ = [
    "attach_visual_dr",
    "randomize_dome_light_intensity",
    "randomize_dome_light_color",
    "randomize_object_color",
]
