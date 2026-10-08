import pytest

from openso101.teleop.timing import control_rate_fps


@pytest.mark.parametrize("physics_dt,decimation,fps", [(1 / 120, 2, 60), (1 / 200, 4, 50)])
def test_recording_frequency_matches_control_period(physics_dt, decimation, fps):
    assert control_rate_fps(physics_dt, decimation) == fps


@pytest.mark.parametrize("physics_dt,decimation", [(0, 2), (-0.01, 2), (float("nan"), 2),
                                                  (float("inf"), 2), (0.01, 0), (0.01, 1.5),
                                                  (0.01, True), (0.003, 2)])
def test_invalid_recording_frequency(physics_dt, decimation):
    with pytest.raises(ValueError):
        control_rate_fps(physics_dt, decimation)
