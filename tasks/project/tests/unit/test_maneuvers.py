"""Unit tests — motion primitives (maneuvers.py): the smooth speed ramp and the
closed-loop turn/straight maneuvers, on both the encoder (tick) path and the
encoderless time-fallback path. Uses fake wheels that record every command."""

import threading
import time

from tasks.project.packages.maneuvers import (
    ramp_speed, turn_left, turn_right, straight_through,
)
from conftest import FakeWheels


# Small ticks/seconds so a maneuver finishes in a couple of fake-encoder reads.
TIMINGS = {
    "turn_left_inner_speed": 0.39, "turn_left_outer_speed": 0.45,
    "turn_left_ticks": 4, "turn_left_seconds": 0.05,
    "turn_right_inner_speed": 0.36, "turn_right_outer_speed": 0.45,
    "turn_right_ticks": 4, "turn_right_seconds": 0.05,
    "turn_inner_speed": 0.05, "turn_outer_speed": 0.45, "turn_ticks": 4,
    "turn_seconds": 0.05,
    "straight_speed": 0.30, "straight_ticks": 4, "straight_seconds": 0.05,
    "turn_exit_ticks": 0, "turn_exit_seconds": 0.05,
}


# ------------------------------------------------------------------ ramp
def test_ramp_steps_toward_target():
    assert ramp_speed(0.0, 0.5, 0.05) == 0.05


def test_ramp_snaps_when_within_step():
    assert ramp_speed(0.48, 0.5, 0.05) == 0.5


def test_ramp_decreases():
    assert abs(ramp_speed(0.5, 0.0, 0.05) - 0.45) < 1e-9


def test_ramp_already_at_target():
    assert ramp_speed(0.3, 0.3, 0.05) == 0.3


# ------------------------------------------------------------------ turns (encoder path)
def test_turn_left_commands_and_stops():
    w = FakeWheels(with_encoders=True)
    turn_left(w, threading.Event(), TIMINGS)
    # A left arc OPENS with the inner/outer pair (0.39, 0.45) — assert it's the
    # FIRST command issued (the maneuver actually drives the arc), not merely that
    # the pair appears somewhere in the history by coincidence.
    assert w.history[0] == (0.39, 0.45)
    assert w.last == (0.0, 0.0)        # always parks the wheels at the end


def test_turn_right_commands_and_stops():
    w = FakeWheels(with_encoders=True)
    turn_right(w, threading.Event(), TIMINGS)
    # turn_right OPENS with _sw(outer, inner) = (0.45, 0.36): left wheel faster -> right.
    assert w.history[0] == (0.45, 0.36)
    assert w.last == (0.0, 0.0)


def test_straight_through_commands_and_stops():
    w = FakeWheels(with_encoders=True)
    straight_through(w, threading.Event(), TIMINGS)
    assert w.history[0] == (0.30, 0.30)
    assert w.last == (0.0, 0.0)


# ------------------------------------------------------------------ time fallback (dead encoders)
def test_turn_left_time_fallback_still_completes():
    w = FakeWheels(with_encoders=False)     # encoders is None -> seconds path
    turn_left(w, threading.Event(), TIMINGS)
    assert w.history[0] == (0.39, 0.45)
    assert w.last == (0.0, 0.0)


def test_straight_time_fallback_still_completes():
    w = FakeWheels(with_encoders=False)
    straight_through(w, threading.Event(), TIMINGS)
    assert w.history[0] == (0.30, 0.30)
    assert w.last == (0.0, 0.0)


def test_stop_event_aborts_turn_early_and_parks():
    """A set stop_event must SHORT-CIRCUIT the maneuver, not just leave the wheels
    parked by luck. Dead encoders (no ticks) + a long turn duration mean a normal
    turn would block for ~seconds; with stop_event set it must return near-instantly.
    Proven by timing the call, so the test fails if the abort is removed."""
    long_timings = dict(TIMINGS)
    long_timings.update({"turn_left_seconds": 5.0, "turn_seconds": 5.0,
                         "turn_exit_seconds": 5.0})
    w = FakeWheels(with_encoders=True, encoder_step=0)   # encoders never advance
    ev = threading.Event(); ev.set()                     # already asked to stop
    t0 = time.time()
    turn_left(w, ev, long_timings)
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"stop_event did not abort the turn (took {elapsed:.2f}s)"
    assert w.history[0] == (0.39, 0.45)    # it still issued the arc command...
    assert w.last == (0.0, 0.0)            # ...then parked the wheels


def test_running_turn_without_stop_blocks_until_timeout():
    """Counterpart to the abort test: with dead encoders and NO stop_event, the same
    long-duration turn DOES run for its time budget (so the abort test above is
    proving the stop_event, not a no-op fast path)."""
    long_timings = dict(TIMINGS)
    long_timings.update({"turn_left_seconds": 0.5, "turn_seconds": 0.5,
                         "turn_exit_ticks": 0})
    w = FakeWheels(with_encoders=True, encoder_step=0)
    t0 = time.time()
    turn_left(w, threading.Event(), long_timings)
    elapsed = time.time() - t0
    assert elapsed >= 0.4, f"turn returned too fast ({elapsed:.2f}s) — time path not taken"
    assert w.last == (0.0, 0.0)
