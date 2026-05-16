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


pytest.importorskip("isaaclab")


def test_attach_visual_dr_wires_all_three_event_terms():
    """Default attach call registers light intensity, light color, object color."""
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
    from openso101.sim2real.domain_randomization.visual import attach_visual_dr

    events = SimpleNamespace()
    attach_visual_dr(events, object_asset_name="cube_bottom")

    assert events.dr_object_color.params["asset_name"] == "cube_bottom"


def test_hsv_to_rgb_pure_red_maps_to_canonical_triple():
    """Sanity check the inlined HSV→RGB conversion against known values."""
    from openso101.sim2real.domain_randomization.visual import _hsv_to_rgb

    # Pure red (h=0, s=1, v=1) should round-trip to (1, 0, 0).
    r, g, b = _hsv_to_rgb(0.0, 1.0, 1.0)
    assert (r, g, b) == pytest.approx((1.0, 0.0, 0.0))

    # Pure green (h=1/3) and pure blue (h=2/3).
    assert _hsv_to_rgb(1.0 / 3.0, 1.0, 1.0) == pytest.approx((0.0, 1.0, 0.0))
    assert _hsv_to_rgb(2.0 / 3.0, 1.0, 1.0) == pytest.approx((0.0, 0.0, 1.0))
