"""Unit tests — intersection logic (perception/intersection.py).

Two rubric pieces:
  * "signs in front of intersections indicate which turns are possible" — the
    legal-turn set is the INTERSECTION of every sign seen (merge_turn_constraints).
  * stopping AT the intersection — the red stop-line detectors and the tag-distance
    "at the line" check, exercised on synthetic frames that carry a real red band.
"""

import numpy as np

from tasks.project.packages.perception.intersection import (
    merge_turn_constraints, is_at_stop_line,
    red_line_pixel_count, red_line_threshold, detect_red_line, red_band_metrics,
)
from tasks.project.packages.perception.apriltags import TagObservation
from tasks.project.packages.sign_registry import SignSemantic, lookup

from conftest import blank_frame, add_red_stop_line


def sign(kind):
    return SignSemantic(kind=kind, tag_type="TrafficSign", vehicle_name=None)


# --------------------------------------------------------------- merge turns
def test_no_signs_all_turns_legal():
    assert merge_turn_constraints([]) == {"left", "right", "straight"}


def test_four_way_allows_all():
    assert merge_turn_constraints([sign("4-way-intersect")]) == {"left", "right", "straight"}


def test_t_intersection_has_no_straight():
    assert merge_turn_constraints([sign("T-intersection")]) == {"left", "right"}


def test_right_t_intersection():
    assert merge_turn_constraints([sign("right-T-intersect")]) == {"straight", "right"}


def test_oneway_left_only_left():
    assert merge_turn_constraints([sign("oneway-left")]) == {"left"}


def test_no_left_turn_removes_left():
    assert merge_turn_constraints([sign("no-left-turn")]) == {"right", "straight"}


def test_no_right_turn_removes_right():
    assert merge_turn_constraints([sign("no-right-turn")]) == {"left", "straight"}


def test_multiple_signs_intersect():
    # 4-way + no-left-turn -> {right, straight}
    legal = merge_turn_constraints([sign("4-way-intersect"), sign("no-left-turn")])
    assert legal == {"right", "straight"}


def test_do_not_enter_yields_empty_set():
    # No legal turn: the bot will not enter (held by design).
    assert merge_turn_constraints([sign("do-not-enter")]) == set()


def test_contradictory_signs_empty():
    # oneway-left AND no-left-turn -> nothing legal.
    assert merge_turn_constraints([sign("oneway-left"), sign("no-left-turn")]) == set()


def test_merge_uses_real_db_semantics():
    # End-to-end with registry lookups (not hand-built semantics).
    legal = merge_turn_constraints([lookup(8), lookup(4)])   # 4-way + no-left-turn
    assert legal == {"right", "straight"}


# --------------------------------------------------------------- at-stop-line (tag)
def _tag(dist=float("inf"), side=50):
    return TagObservation(id=1, center_xy=(320, 240), side_length_px=side,
                          est_distance_m=dist, est_yaw_rad=0.0)


def test_at_stop_line_metric_distance():
    mask = np.zeros((480, 640), np.uint8)
    assert is_at_stop_line(_tag(dist=0.2), mask, stop_distance_m=0.25) is True
    assert is_at_stop_line(_tag(dist=0.4), mask, stop_distance_m=0.25) is False


def test_at_stop_line_pixel_fallback_uncalibrated():
    # est inf + big tag + lane pixels in the bottom third -> at the line.
    mask = np.zeros((480, 640), np.uint8)
    mask[330:, 200:440] = 255
    assert is_at_stop_line(_tag(dist=float("inf"), side=60), mask) is True
    # small tag (far) -> not yet
    assert is_at_stop_line(_tag(dist=float("inf"), side=20), mask) is False


# --------------------------------------------------------------- red stop line
def test_red_line_detected_on_painted_band():
    frame = add_red_stop_line(blank_frame())
    assert red_line_pixel_count(frame) >= red_line_threshold()
    assert detect_red_line(frame) is True


def test_no_red_line_on_plain_road():
    frame = blank_frame(fill=(120, 120, 120))
    assert detect_red_line(frame) is False
    assert red_line_pixel_count(frame) == 0


def test_red_band_structure_present_and_absent():
    present, diag = red_band_metrics(add_red_stop_line(blank_frame()))
    assert present is True
    assert diag["row_max"] > 0.8     # red spans the lane width
    absent, _ = red_band_metrics(blank_frame())
    assert absent is False


def test_red_line_empty_frame_is_safe():
    assert red_line_pixel_count(None) == 0
    present, _ = red_band_metrics(None)
    assert present is False
