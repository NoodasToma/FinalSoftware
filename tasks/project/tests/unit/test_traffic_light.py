"""Unit tests — traffic-light perception + the yellow-brake rule (traffic_light.py).

Rubric (Task 2 carry-over used by stop/yield-light intersections): stop on red, go
on green, and don't dive through a settled yellow."""

import pytest

from tasks.project.packages.perception.traffic_light import (
    TrafficLightDetector, should_brake_for_yellow,
)
from conftest import blank_frame, add_light


# --------------------------------------------------------------- yellow rule
def test_brake_for_settled_yellow():
    # yellow for > 0.5 s -> brake (can't be sure of clearing the box).
    assert should_brake_for_yellow("yellow", 10.0, 10.6) is True


def test_dont_brake_for_fresh_yellow():
    assert should_brake_for_yellow("yellow", 10.0, 10.3) is False


@pytest.mark.parametrize("color", ["red", "green", None])
def test_other_colors_never_yellow_brake(color):
    assert should_brake_for_yellow(color, 10.0, 99.0) is False


def test_yellow_with_no_start_time_does_not_brake():
    assert should_brake_for_yellow("yellow", None, 99.0) is False


# --------------------------------------------------------------- detector arming
def test_detector_returns_none_when_disarmed():
    d = TrafficLightDetector()
    assert d.armed is False
    assert d.detect(add_light(blank_frame(), "red")) is None


def test_arm_disarm_toggles():
    d = TrafficLightDetector()
    d.arm(); assert d.armed is True
    d.disarm(); assert d.armed is False


@pytest.mark.parametrize("color", ["red", "yellow", "green"])
def test_detects_each_colour_when_armed(color):
    d = TrafficLightDetector()
    d.arm()
    frame = add_light(blank_frame(), color)
    assert d.detect(frame) == color


def test_blank_armed_frame_is_none():
    d = TrafficLightDetector()
    d.arm()
    assert d.detect(blank_frame()) is None


def test_light_only_read_in_top_half():
    # A green blob painted LOW in the frame must be ignored (lights sit high).
    d = TrafficLightDetector()
    d.arm()
    frame = blank_frame()
    add_light(frame, "green", center=(160, 430))   # bottom quarter
    assert d.detect(frame) is None


def test_empty_frame_is_safe():
    d = TrafficLightDetector()
    d.arm()
    assert d.detect(None) is None
