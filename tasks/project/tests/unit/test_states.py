"""Unit tests — the FSM transition table (states.py), the "brain wiring".

Covers every transition the agent relies on plus the full stop-sign-to-turn happy
path, and that unknown events are no-ops (so a stray event can't derail the FSM)."""

import pytest

from tasks.project.packages.states import State, next_state, TRANSITIONS


def test_drive_reacts_to_each_trigger():
    assert next_state(State.DRIVE, "see_stop_or_yield") == State.APPROACH
    assert next_state(State.DRIVE, "see_light") == State.APPROACH
    assert next_state(State.DRIVE, "see_intersection") == State.APPROACH
    assert next_state(State.DRIVE, "obstacle") == State.SOFT_STOP


def test_approach_to_stopped_and_obstacle():
    assert next_state(State.APPROACH, "at_stop_line") == State.STOPPED
    assert next_state(State.APPROACH, "obstacle") == State.SOFT_STOP


@pytest.mark.parametrize("event,expected", [
    ("choose_turn_left", State.TURN_LEFT),
    ("choose_turn_right", State.TURN_RIGHT),
    ("choose_straight", State.STRAIGHT_THROUGH),
    ("wait", State.WAIT),
])
def test_stopped_branches(event, expected):
    assert next_state(State.STOPPED, event) == expected


def test_wait_clears_back_to_stopped():
    assert next_state(State.WAIT, "cleared") == State.STOPPED


@pytest.mark.parametrize("turn_state", [
    State.TURN_LEFT, State.TURN_RIGHT, State.STRAIGHT_THROUGH])
def test_turn_done_returns_to_drive(turn_state):
    assert next_state(turn_state, "turn_done") == State.DRIVE


def test_soft_stop_clears_to_drive():
    assert next_state(State.SOFT_STOP, "obstacle_cleared") == State.DRIVE


def test_unknown_event_is_noop():
    # A defensive table: no edge for (state, event) => stay put.
    assert next_state(State.DRIVE, "nonsense") == State.DRIVE
    assert next_state(State.STOPPED, "obstacle") == State.STOPPED  # no such edge
    assert next_state(State.TURN_LEFT, "wait") == State.TURN_LEFT


def test_full_stop_sign_happy_path():
    """DRIVE -> APPROACH -> STOPPED -> TURN -> DRIVE, the canonical sign cycle."""
    s = State.DRIVE
    s = next_state(s, "see_stop_or_yield"); assert s == State.APPROACH
    s = next_state(s, "at_stop_line");      assert s == State.STOPPED
    s = next_state(s, "choose_turn_left");  assert s == State.TURN_LEFT
    s = next_state(s, "turn_done");         assert s == State.DRIVE


def test_red_light_wait_path():
    """At a light: STOPPED -> WAIT (red) -> STOPPED (green) -> go straight."""
    s = State.APPROACH
    s = next_state(s, "at_stop_line"); assert s == State.STOPPED
    s = next_state(s, "wait");         assert s == State.WAIT
    s = next_state(s, "cleared");      assert s == State.STOPPED
    s = next_state(s, "choose_straight"); assert s == State.STRAIGHT_THROUGH


def test_transition_table_has_no_self_loops_or_bad_targets():
    for (src, event), dst in TRANSITIONS.items():
        assert isinstance(src, State) and isinstance(dst, State)
        assert isinstance(event, str) and event
        assert src != dst, f"{src} --{event}--> itself is a meaningless edge"
