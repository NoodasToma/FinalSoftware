"""Unit tests — obstacle stop decision (obstacles.py).

Rubric: "(Stopping for obstacles) ... object detection should be used to drive and
stop on objects". ``should_stop_for_obstacle`` is the gate that turns a YOLO/HSV
duckie detection into a SOFT_STOP: stop if a duckie box is low in the frame (close)
OR large (close)."""

from tasks.project.packages.obstacles import should_stop_for_obstacle

FRAME_H = 480
DUCKIE = 0
OTHER = 1   # e.g. a truck class — not a pedestrian, not handled here


def det(bbox, score=0.9, cls=DUCKIE):
    return (bbox, score, cls)


def test_no_detections_no_stop():
    stop, reason = should_stop_for_obstacle([], FRAME_H)
    assert stop is False and reason == ""


def test_duckie_low_in_frame_stops():
    # y2 = 400 > 0.6*480 = 288  -> close duckie in our path.
    stop, reason = should_stop_for_obstacle([det((300, 350, 360, 400))], FRAME_H)
    assert stop is True
    assert "duckie" in reason


def test_duckie_high_and_small_is_ignored():
    # A roadside duckie: high in frame (y2 small) and tiny area.
    stop, _ = should_stop_for_obstacle([det((10, 10, 24, 24))], FRAME_H)
    assert stop is False


def test_duckie_large_area_stops_even_if_high():
    # Big box (area > 0.04*640*480 = 12288) means it's close, regardless of y2.
    # 200x120 = 24000 > 12288, y2=140 < 288.
    stop, reason = should_stop_for_obstacle([det((100, 20, 300, 140))], FRAME_H)
    assert stop is True
    assert "duckie" in reason


def test_non_duckie_class_is_not_a_pedestrian_stop():
    # A class-1 (truck) box low in the frame is NOT handled by this gate (the
    # pedestrian/duckie rule only fires on class 0). Documented limitation.
    stop, _ = should_stop_for_obstacle([det((300, 350, 360, 460), cls=OTHER)], FRAME_H)
    assert stop is False


def test_threshold_is_exclusive_just_below_does_not_stop():
    # y2 exactly at the boundary (0.6*480 = 288) must NOT stop (strict >).
    stop, _ = should_stop_for_obstacle([det((300, 250, 320, 288))], FRAME_H)
    assert stop is False
    # one pixel lower does stop
    stop2, _ = should_stop_for_obstacle([det((300, 250, 320, 289))], FRAME_H)
    assert stop2 is True


def test_any_qualifying_duckie_in_a_list_triggers_stop():
    dets = [det((10, 10, 20, 20)),                 # ignored (tiny, high)
            det((300, 350, 360, 460))]             # close -> stop
    stop, _ = should_stop_for_obstacle(dets, FRAME_H)
    assert stop is True


# ----------------------------------------------------------- "in the way" gate
FRAME_W = 640
# A close (low) duck hugging the RIGHT edge: close enough to stop on the plain rule,
# but off to the side, not in our lane.
_OFF_SIDE = det((590, 350, 630, 420))      # cx = 610 -> 0.95 of width
_CENTRED = det((300, 350, 360, 420))       # cx = 330 -> ~0.52 of width


def test_lateral_gate_default_off_is_byte_identical():
    # No frame_w / cx_margin_frac (the default): the close off-side duck still stops,
    # exactly as before the gate existed (no behaviour change for existing callers).
    assert should_stop_for_obstacle([_OFF_SIDE], FRAME_H)[0] is True


def test_lateral_gate_ignores_off_side_obstacle_when_enabled():
    # With the gate on, the off-side duck is NOT in our path -> no stop.
    assert should_stop_for_obstacle([_OFF_SIDE], FRAME_H, FRAME_W, 0.20)[0] is False


def test_lateral_gate_keeps_centred_obstacle_when_enabled():
    # A centred close duck IS in our path -> still stops with the gate on.
    assert should_stop_for_obstacle([_CENTRED], FRAME_H, FRAME_W, 0.20)[0] is True


def test_lateral_gate_needs_frame_w_to_engage():
    # cx_margin_frac > 0 but frame_w omitted: the gate stays OFF (this is what keeps
    # the far-left _OBSTACLE_BLOCK stop-line sentinel — passed with no frame_w —
    # working). The off-side duck stops because the gate never ran.
    assert should_stop_for_obstacle([_OFF_SIDE], FRAME_H, None, 0.20)[0] is True


def test_lateral_gate_does_not_rescue_a_high_small_box():
    # The gate only REMOVES off-side detections; it never turns a non-qualifying
    # (high+small) centred box into a stop.
    assert should_stop_for_obstacle([det((300, 10, 320, 24))], FRAME_H, FRAME_W, 0.20)[0] is False
