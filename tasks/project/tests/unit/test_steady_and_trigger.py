"""Unit tests for the two fixes:
  1. Steady speed on turns — curve_boost=1.0 + curve_speed=base_speed => the forward
     speed is the SAME on a curve as on a straight (turning is steering-only).
  2. Tag action triggers only when the tag is VERY CLOSE or GONE — the red-band
     trigger is gated by use_red_band (OFF on the bot), so venue red/orange noise
     can't stop the bot early."""

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.project.packages.agent import _derive_event
from tasks.project.packages.states import State
from tasks.project.packages.perception.apriltags import TagObservation
from tasks.project.packages.sign_registry import SignSemantic


# =============================================================== steady speed
def _lane(**attrs):
    a = LaneServoingAgent()
    a.base_speed = 0.25
    a.max_steer = 0.55
    a.steering_threshold = 0.15
    a.min_wheel_speed = 0.0
    for k, v in attrs.items():
        setattr(a, k, v)
    return a


def test_steady_speed_curve_equals_straight():
    # curve_boost=1.0 + curve_speed==base_speed => same total forward speed on a
    # curve as on a straight; only the L/R difference (steering) changes.
    a = _lane(curve_boost=1.0, curve_speed=0.25)
    steer = 0.2  # > steering_threshold so the (now-disabled) boost path is reached
    straight = a._motor_commands(steer, recovery=False, is_curve=False, both_visible=True)
    curve = a._motor_commands(steer, recovery=False, is_curve=True, both_visible=True)
    assert abs(sum(straight) - sum(curve)) < 1e-9, "curve sped up vs straight"
    # steering differential preserved on both
    assert abs((straight[1] - straight[0]) - (curve[1] - curve[0])) < 1e-9


def test_boost_above_one_would_speed_up_curves():
    # Sanity: the OLD boost (>1) does increase curve speed — proving the fix matters.
    a = _lane(curve_boost=1.0, curve_speed=0.25)
    steady = sum(a._motor_commands(0.2, False, True, True))
    a.curve_boost = 1.3
    boosted = sum(a._motor_commands(0.2, False, True, True))
    assert boosted > steady + 1e-6


# =============================================================== trigger timing
def _tag(px):
    return TagObservation(id=1, center_xy=(320, 200), side_length_px=px,
                          est_distance_m=float('inf'), est_yaw_rad=0.0)


_STOP = SignSemantic(kind='stop', tag_type='TrafficSign', vehicle_name=None)


def _event(px, *, use_red_band, red_band, stop_min_px=65, react_min_px=35):
    return _derive_event(State.APPROACH, [(_tag(px), _STOP)], [], 480, None,
                         stop_min_px=stop_min_px, react_min_px=react_min_px,
                         red_band=red_band, use_red_band=use_red_band)


def test_red_band_off_does_not_trigger_early():
    # Bot: use_red_band False. A moderately-close tag (40px, past react 35) with a
    # red band present must NOT stop — that was the early trigger.
    assert _event(40, use_red_band=False, red_band=True) is None


def test_red_band_on_still_triggers_for_sim_clean_line():
    # Default/sim: use_red_band True. Same situation DOES trigger on the band.
    assert _event(40, use_red_band=True, red_band=True) == 'at_stop_line'


def test_very_close_triggers_regardless_of_band():
    # "Too close" (>= stop_min_px) stops whether or not the band path is enabled.
    assert _event(70, use_red_band=False, red_band=False) == 'at_stop_line'
    assert _event(70, use_red_band=True, red_band=False) == 'at_stop_line'


def test_not_close_and_band_off_waits_for_lost_or_close():
    # Below the very-close gate, band off, no band: no event — the bot keeps driving
    # until the tag is gone (lost-commit, handled in main) or reaches stop_min_px.
    assert _event(50, use_red_band=False, red_band=False) is None


def test_raised_stop_px_means_44_no_longer_triggers():
    # With sign_stop_px raised to 65, a 44px tag (the old stop size) no longer fires.
    assert _event(44, use_red_band=False, red_band=False, stop_min_px=65) is None
    # ...but it would have under the old 44 threshold:
    assert _event(44, use_red_band=False, red_band=False, stop_min_px=44) == 'at_stop_line'
