"""Unit tests for the ported reference SignBehaviorFSM (the working sign detection).

These pin the behaviours that make it work — and the one the user needs most: an
intersection sign makes the bot EXECUTE the chosen turn OPEN-LOOP (fixed wheel speeds,
lane ignored), so the sign decision wins over lane-following.

The FSM is time-based (time.monotonic). Two mechanics keep it deterministic without
real sleeping:
  * duration-gated states (PRE_TURN/TURNING/POST_STOP) compare elapsed FRAMES to a
    config threshold -> we set short/long thresholds so a state advances or persists
    on demand;
  * SLOWING decays by per-step `_dt` -> we rewind `_last_step_time` so the next step
    sees a large dt and decays in one step.
AprilTag/red-line detection is stubbed so we drive the FSM logic directly."""

import time

import numpy as np
import pytest

import tasks.project.packages.sign_detection.sign_behavior as sb
from tasks.project.packages.sign_detection.sign_behavior import SignBehaviorFSM
from tasks.project.packages.sign_detection.sign_behavior_config import (
    SignBehaviorConfig, State, TagID,
)

FRAME = np.zeros((480, 640, 3), np.uint8)
BASE = (0.4, 0.4)          # the lane-follow command handed to the FSM


@pytest.fixture
def fsm():
    cfg = SignBehaviorConfig(
        stop_hold_frames=1, slow_ramp_factor=0.5, stopped_speed_threshold=0.1,
        check_left_frames=1, check_right_frames=1, check_settle_frames=1,
        post_stop_frames=1, post_stop_speed=0.25,
        preturn_left_frames=0, preturn_right_frames=0, preturn_speed=0.2,
        # large turn windows so TURNING PERSISTS and we can read its open-loop speeds
        intersect_left_frames=1000, intersect_right_frames=1000, intersect_forward_frames=1000,
        intersect_left_speed=(0.2, 0.5), intersect_right_speed=(0.6, 0.05),
        intersect_forward_speed=(0.4, 0.4), exit_timeout_frames=1, exit_speed=0.4,
        approach_duration=0.0,
    )
    return SignBehaviorFSM(config=cfg)


def _step(fsm, monkeypatch, *, tags=(), red_line=False, detections=()):
    monkeypatch.setattr(sb, "detect_tags", lambda self, frame: [])
    monkeypatch.setattr(sb, "confirm_tags", lambda self, raw: list(tags))
    monkeypatch.setattr(sb, "detect_red_line", lambda self, frame: red_line)
    return fsm.step(FRAME, BASE[0], BASE[1], list(detections))


def _run_to_turning(fsm, monkeypatch, tag_id, choice):
    """Arm an intersection on `tag_id`, force the chosen turn, and step until TURNING.
    Returns (turning_wheels, chosen)."""
    monkeypatch.setattr(sb.random, "choice", lambda seq: choice)
    _step(fsm, monkeypatch, tags=(tag_id,))            # confirm tag (x2)
    _step(fsm, monkeypatch, tags=(tag_id,))
    _step(fsm, monkeypatch, tags=(tag_id,), red_line=True)   # red -> APPROACHING
    in_turning = False
    for _ in range(10):
        l, r, st = _step(fsm, monkeypatch)             # APPROACHING->INTERSECT->PRE_TURN->TURNING
        if st == State.TURNING:
            if in_turning:
                # the step that ENTERS turning returns (0,0); read the NEXT step's
                # wheels, which are the configured open-loop turn speeds.
                return (l, r), fsm._chosen_turn
            in_turning = True
    return None, fsm._chosen_turn


# ---------------------------------------------------------- red line WITHOUT a tag
def test_red_line_without_tag_is_ignored(fsm, monkeypatch):
    # THE key robustness rule: a red line alone (no saved sign) must NOT trigger
    # anything — it just keeps lane-following (returns the base command).
    l, r, st = _step(fsm, monkeypatch, tags=(), red_line=True)
    assert st == State.MOVING
    assert (l, r) == BASE


def test_moving_passes_lane_through_with_no_signs(fsm, monkeypatch):
    l, r, st = _step(fsm, monkeypatch, tags=(), red_line=False)
    assert st == State.MOVING and (l, r) == BASE


# ---------------------------------------------------------- STOP sign path
def test_stop_tag_then_red_line_brakes_to_stop(fsm, monkeypatch):
    _step(fsm, monkeypatch, tags=(1,))                 # id 1 -> STOP, saved (x2)
    _step(fsm, monkeypatch, tags=(1,))
    assert fsm._saved_tag == TagID.STOP
    _, _, st = _step(fsm, monkeypatch, tags=(1,), red_line=True)
    assert st == State.SLOWING
    # force a large per-step dt so the slow-ramp decays to a stop in one step
    fsm._last_step_time = time.monotonic() - 1.0
    l, r, st = _step(fsm, monkeypatch)
    assert st == State.STOPPED
    assert (l, r) == (0.0, 0.0)


# ---------------------------------------------------------- intersection turn (THE fix)
def test_right_turn_drives_open_loop_right_speeds(fsm, monkeypatch):
    # tag 9 -> {forward, right}; pick 'right' -> the TURNING wheels are the CONFIGURED
    # right-turn speeds (0.6, 0.05), driven OPEN-LOOP with the lane ignored.
    wheels, chosen = _run_to_turning(fsm, monkeypatch, 9, "right")
    assert chosen == "right"
    assert wheels == (0.6, 0.05)
    assert wheels[0] > wheels[1]          # left wheel faster = turning right


def test_left_turn_drives_open_loop_left_speeds(fsm, monkeypatch):
    # tag 10 = left-T -> {forward, left}; pick 'left' -> configured left-turn speeds (0.2, 0.5).
    wheels, chosen = _run_to_turning(fsm, monkeypatch, 10, "left")
    assert chosen == "left"
    assert wheels == (0.2, 0.5)
    assert wheels[1] > wheels[0]          # right wheel faster = turning left


def test_turn_speeds_ignore_the_lane_command(fsm, monkeypatch):
    # Same tag, but a wildly different base lane command must NOT change the turn —
    # proving the turn is authoritative (open-loop), not lane-derived.
    monkeypatch.setattr(sb.random, "choice", lambda seq: "right")
    _step(fsm, monkeypatch, tags=(9,))
    _step(fsm, monkeypatch, tags=(9,))
    _step(fsm, monkeypatch, tags=(9,), red_line=True)
    monkeypatch.setattr(sb, "detect_tags", lambda self, frame: [])
    monkeypatch.setattr(sb, "confirm_tags", lambda self, raw: [])
    monkeypatch.setattr(sb, "detect_red_line", lambda self, frame: False)
    wheels = None
    in_turning = False
    for _ in range(10):
        l, r, st = fsm.step(FRAME, 0.99, 0.01, [])     # hard-left lane command
        if st == State.TURNING:
            if in_turning:
                wheels = (l, r); break
            in_turning = True
    assert wheels == (0.6, 0.05)          # still the configured RIGHT turn, lane ignored


def test_turn_choice_is_always_legal_for_the_tag(fsm, monkeypatch):
    # tag 9 = {forward, right} -> the chosen turn is never 'left'
    monkeypatch.setattr(sb.random, "choice", lambda seq: seq[0])
    _step(fsm, monkeypatch, tags=(9,))
    _step(fsm, monkeypatch, tags=(9,))
    _step(fsm, monkeypatch, tags=(9,), red_line=True)
    assert fsm._chosen_turn in ("forward", "right")


def test_finish_returns_to_moving_and_ignores_red_briefly(fsm, monkeypatch):
    wheels, _ = _run_to_turning(fsm, monkeypatch, 9, "right")
    assert wheels is not None
    # large rewind ends TURNING -> EXITING -> MOVING
    for _ in range(10):
        fsm._state_started_at = time.monotonic() - 100.0
        _, _, st = _step(fsm, monkeypatch)
        if st == State.MOVING:
            break
    assert st == State.MOVING
    assert fsm._ignore_red_counter > 0.0          # just-crossed red line ignored a while
