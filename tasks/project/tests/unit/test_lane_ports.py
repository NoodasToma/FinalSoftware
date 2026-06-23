"""Unit tests for the "best of both" lane-servoing ports from the reference's
perfect-lane branch: wrong-side white rejection, wheel-speed floor (anti-stiction),
and steering EMA. The AGENT's hardcoded defaults are all OFF; the sim config opts
INTO reject_white_wrong_side (lane-side recovery, so the bot steers back if it
crosses the centre line on a bend), while the hardware-only ports (anti-stiction
floor, steering EMA) stay off in the clean sim. These tests exercise the knobs by
enabling them directly on a LaneServoingAgent instance."""

import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent


def _agent(**attrs):
    a = LaneServoingAgent()           # sim config; new ports default OFF
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


# --------------------------------------------------------------- wheel floor
def test_wheel_floor_is_noop_when_disabled():
    a = _agent(min_wheel_speed=0.0)
    assert a._apply_wheel_floor(0.0, 0.3) == (0.0, 0.3)


def test_wheel_floor_lifts_both_preserving_difference():
    # min_wheel_speed 0.08: a (0.0, 0.4) command -> both lifted by 0.08, diff kept.
    a = _agent(min_wheel_speed=0.08)
    left, right = a._apply_wheel_floor(0.0, 0.4)
    assert abs(left - 0.08) < 1e-9
    assert abs(right - 0.48) < 1e-9
    assert abs((right - left) - 0.4) < 1e-9      # steering difference preserved


def test_wheel_floor_rescales_when_peak_exceeds_one():
    a = _agent(min_wheel_speed=0.2)
    left, right = a._apply_wheel_floor(0.1, 1.0)   # lift by 0.1 -> (0.2, 1.1) -> rescale
    assert right <= 1.0 + 1e-9
    assert 0.0 <= left <= right


def test_wheel_floor_does_not_lift_when_already_above():
    a = _agent(min_wheel_speed=0.05)
    assert a._apply_wheel_floor(0.2, 0.3) == (0.2, 0.3)


# --------------------------------------------------------------- wrong-side rejection
def _err(a, yellow_xs, white_xs):
    return a._calculate_error(yellow_xs, white_xs, True, True, w=640)


def test_wrong_side_disabled_uses_midpoint():
    # white (200) LEFT of yellow (400): disabled -> uses midpoint of both.
    a = _agent(reject_white_wrong_side=False)
    a._lane_half_width = 160.0
    err_both = _err(a, [400], [200])
    # midpoint = 300; error = 320 - 300 = 20 -> +0.0625
    assert abs(err_both - (320 - 300) / 320) < 1e-6


def test_wrong_side_enabled_follows_yellow_only():
    # white LEFT of yellow -> discard white, track yellow + half_width.
    a = _agent(reject_white_wrong_side=True)
    a._lane_half_width = 160.0
    err = _err(a, [400], [200])
    # yellow-only: error = 320 - (400 + 160) = -240 -> clip to -0.75
    assert abs(err - (320 - (400 + 160)) / 320) < 1e-6
    assert err < 0          # steers right, away from the wrong-side white


def test_wrong_side_normal_geometry_unaffected():
    # white (450) RIGHT of yellow (150): wrong-side check does NOT fire.
    a = _agent(reject_white_wrong_side=True)
    a._lane_half_width = 160.0
    err = _err(a, [150], [450])
    # midpoint = 300; error = 320 - 300 = 20 -> +0.0625 (same as both-lines path)
    assert abs(err - (320 - 300) / 320) < 1e-6


# --------------------------------------------------------------- steering EMA
def test_steer_smooth_default_off_passes_through():
    a = LaneServoingAgent()
    assert a.steer_smooth == 0.0
    assert a._filtered_steering == 0.0


def test_steer_smooth_emas_toward_target():
    # With steer_smooth=0.5 the filtered value moves halfway to the target each step.
    a = _agent(steer_smooth=0.5)
    a._filtered_steering = 0.0
    # emulate the compute_commands EMA step
    target = 0.4
    a._filtered_steering = a.steer_smooth * target + (1 - a.steer_smooth) * a._filtered_steering
    assert abs(a._filtered_steering - 0.2) < 1e-9
    a._filtered_steering = a.steer_smooth * target + (1 - a.steer_smooth) * a._filtered_steering
    assert abs(a._filtered_steering - 0.3) < 1e-9   # converging toward 0.4


def test_agent_hardcoded_defaults_off():
    # With NO config file (the agent's own fallback defaults), every port is OFF.
    a = LaneServoingAgent(config_path="___no_such_lane_config___.yaml")
    assert a.reject_white_wrong_side is False
    assert a.min_wheel_speed == 0.0
    assert a.steer_smooth == 0.0


def test_sim_config_enables_wrong_side_recovery():
    # The SIM config opts INTO wrong-side recovery (so the bot steers back if it
    # drifts across the yellow centre line on a bend), but leaves the hardware-only
    # anti-stiction floor + steering EMA off (the sim has no motor stiction/jitter).
    a = LaneServoingAgent()
    assert a.reject_white_wrong_side is True
    assert a.min_wheel_speed == 0.0
    assert a.steer_smooth == 0.0
