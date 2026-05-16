# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Visual DR attach-helper tests.

These tests cover the dependency-free portions of
``openso101.sim2real.domain_randomization.visual``: argument plumbing on
:func:`attach_visual_dr`, the deterministic HSV-RGB conversion, and the
per-channel color sampler. The actual USD stage writes are exercised
under Isaac Sim and have no off-platform equivalent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_attach_visual_dr_wires_all_three_event_terms():
    """Default attach call registers light intensity, light color, object color."""
    pytest.importorskip("isaaclab")
    from isaaclab.managers import EventTermCfg as EventTerm

    from openso101.sim2real.domain_randomization.visual import attach_visual_dr

    events = SimpleNamespace()
    attach_visual_dr(events)

    for attr in ("dr_dome_light_intensity", "dr_dome_light_color", "dr_object_color"):
        term = getattr(events, attr, None)
        assert isinstance(term, EventTerm), f"{attr} should be EventTerm"
        assert term.mode == "reset"


def test_attach_visual_dr_respects_disable_flags():
    """enable_*=False suppresses just that term — the others still wire."""
    pytest.importorskip("isaaclab")
    from openso101.sim2real.domain_randomization.visual import attach_visual_dr

    events = SimpleNamespace()
    attach_visual_dr(
        events,
        enable_light_intensity=False,
        enable_light_color=True,
        enable_object_color=False,
    )

    assert not hasattr(events, "dr_dome_light_intensity")
    assert hasattr(events, "dr_dome_light_color")
    assert not hasattr(events, "dr_object_color")


def test_attach_visual_dr_forwards_custom_object_asset_name():
    """The ``object_asset_name`` arg propagates into the EventTerm params."""
    pytest.importorskip("isaaclab")
    from openso101.sim2real.domain_randomization.visual import attach_visual_dr

    events = SimpleNamespace()
    attach_visual_dr(events, object_asset_name="cube_bottom")

    assert events.dr_object_color.params["asset_name"] == "cube_bottom"


def test_scene_asset_prim_returns_none_on_missing_asset(monkeypatch):
    """Regression: `InteractiveScene` has no `.get()` method, it uses bracket
    subscript and raises KeyError on miss. `_scene_asset_prim` must catch the
    KeyError so visual DR doesn't crash on a typo'd asset name.

    Caught a real bug: training with `--visual-dr` hung silently because
    `env.scene.get(asset_name)` raised AttributeError, the exception got
    swallowed in shutdown, and Isaac Sim sat in `simulation_app.close()`
    flushing renders for 30+s before exiting.
    """
    pytest.importorskip("isaaclab")  # __init__.py eager-imports physics → isaaclab
    from openso101.sim2real.domain_randomization import visual

    # Stub _get_stage to a non-None sentinel so we actually reach the
    # scene access. The real `pxr.Usd` import inside the function is
    # tried-and-skipped without Isaac Sim, so we monkey-patch around that.
    monkeypatch.setattr(visual, "_get_stage", lambda: object())

    class _SceneRaisesKeyError:
        def __getitem__(self, name):
            raise KeyError(name)

    class _SceneReturnsNone:
        def __getitem__(self, name):
            return None

    fake_env_missing = type("FakeEnv", (), {"scene": _SceneRaisesKeyError()})()
    fake_env_none = type("FakeEnv", (), {"scene": _SceneReturnsNone()})()

    # KeyError path (asset not registered at all)
    assert visual._scene_asset_prim(fake_env_missing, "no_such_asset") is None
    # None path (asset key present but value is None for some reason)
    assert visual._scene_asset_prim(fake_env_none, "object") is None


def test_scene_asset_prim_source_does_not_use_dict_get():
    """Belt-and-suspenders: the source MUST NOT call `env.scene.get(...)`
    — that method doesn't exist on InteractiveScene and the failure is
    silent under Isaac Sim's stdout hijack. Documenting via test."""
    pytest.importorskip("isaaclab")  # __init__.py eager-imports physics → isaaclab
    import inspect

    from openso101.sim2real.domain_randomization import visual

    src = inspect.getsource(visual._scene_asset_prim)
    assert "env.scene.get(" not in src, (
        "_scene_asset_prim must use `env.scene[name]` with KeyError fallback, "
        "NOT `env.scene.get(name)` — InteractiveScene has no such method."
    )


def test_hsv_to_rgb_pure_red_maps_to_canonical_triple():
    """Sanity check the inlined HSV→RGB conversion against known values."""
    pytest.importorskip("isaaclab")  # __init__.py eager-imports physics → isaaclab
    from openso101.sim2real.domain_randomization.visual import _hsv_to_rgb

    # Pure red (h=0, s=1, v=1) should round-trip to (1, 0, 0).
    r, g, b = _hsv_to_rgb(0.0, 1.0, 1.0)
    assert (r, g, b) == pytest.approx((1.0, 0.0, 0.0))

    # Pure green (h=1/3) and pure blue (h=2/3).
    assert _hsv_to_rgb(1.0 / 3.0, 1.0, 1.0) == pytest.approx((0.0, 1.0, 0.0))
    assert _hsv_to_rgb(2.0 / 3.0, 1.0, 1.0) == pytest.approx((0.0, 0.0, 1.0))
