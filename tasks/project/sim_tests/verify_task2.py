"""Deterministic checks for the Task-2 (traffic light) + Task-3 (precedence) logic.

No Godot / no camera needed -- this proves the decision logic in isolation:
  * t-light-ahead now drives DRIVE -> APPROACH -> STOPPED (the bot stops at a light)
  * the intersection-entry gate: stop on red, go on green, hold on yellow,
    yield to a vehicle with precedence, never enter with an obstacle in the box
  * who-goes-first (we_go_first) both branches

Run:  .venv311/Scripts/python.exe -m tasks.project.sim_tests.verify_task2
  (or: .venv311/Scripts/python.exe tasks/project/sim_tests/verify_task2.py)
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from tasks.project.packages.states import State, next_state
from tasks.project.packages.agent import _derive_event, _clear_to_enter
from tasks.project.packages.sign_registry import SignSemantic
from tasks.project.packages.precedence import we_go_first

PASS, FAIL = "PASS", "FAIL"
_failures = []


def check(name, got, want):
    ok = got == want
    print(f"  [{PASS if ok else FAIL}] {name}: got={got!r} want={want!r}")
    if not ok:
        _failures.append(name)


def _sign(kind, tag_type="TrafficSign", vehicle_name=None):
    return (None, SignSemantic(kind=kind, tag_type=tag_type, vehicle_name=vehicle_name))


def _close_duckie():
    # class 0 (duckie), y2 deep in the lower frame -> "in the box"
    return [((100, 300, 320, 470), 0.9, 0)]


HOST = "duckiebot00"          # a fixed host name for deterministic precedence
NOW = time.time()
FRAME_H = 480


print("FSM transitions")
check("DRIVE + see_light -> APPROACH",        next_state(State.DRIVE, 'see_light'),        State.APPROACH)
check("DRIVE + see_stop_or_yield -> APPROACH", next_state(State.DRIVE, 'see_stop_or_yield'), State.APPROACH)
check("APPROACH + at_stop_line -> STOPPED",   next_state(State.APPROACH, 'at_stop_line'),  State.STOPPED)
check("STOPPED + choose_straight -> STRAIGHT", next_state(State.STOPPED, 'choose_straight'), State.STRAIGHT_THROUGH)

print("\n_derive_event (DRIVE)")
check("t-light-ahead -> see_light", _derive_event(State.DRIVE, [_sign('t-light-ahead')], [], FRAME_H, None), 'see_light')
check("stop sign -> see_stop_or_yield", _derive_event(State.DRIVE, [_sign('stop')], [], FRAME_H, None), 'see_stop_or_yield')
check("stop beats light", _derive_event(State.DRIVE, [_sign('stop'), _sign('t-light-ahead')], [], FRAME_H, None), 'see_stop_or_yield')
check("obstacle beats light", _derive_event(State.DRIVE, [_sign('t-light-ahead')], _close_duckie(), FRAME_H, None), 'obstacle')

print("\n_clear_to_enter (intersection-entry gate)")
# red light -> stop
check("red light -> wait", _clear_to_enter(True, 'red', None, NOW, HOST, [], [], FRAME_H), False)
# green light -> go
check("green light -> go", _clear_to_enter(True, 'green', None, NOW, HOST, [], [], FRAME_H), True)
# settled yellow -> hold (can't guarantee clearing)
check("settled yellow -> wait", _clear_to_enter(True, 'yellow', NOW - 1.0, NOW, HOST, [], [], FRAME_H), False)
# obstacle in the box -> never enter (even with no light)
check("obstacle -> wait", _clear_to_enter(False, None, None, NOW, HOST, [], _close_duckie(), FRAME_H), False)
# plain stop sign (no light, no vehicle, no obstacle) -> proceed
check("clean stop sign -> go", _clear_to_enter(False, None, None, NOW, HOST, [], [], FRAME_H), True)
# vehicle with precedence (name sorts before host) -> yield
veh_before = [SignSemantic(kind='', tag_type='Vehicle', vehicle_name='aaa_bot')]
check("vehicle has precedence -> wait", _clear_to_enter(False, None, None, NOW, HOST, veh_before, [], FRAME_H), False)
# vehicle without precedence (name sorts after host) -> go
veh_after = [SignSemantic(kind='', tag_type='Vehicle', vehicle_name='zzz_bot')]
check("we have precedence -> go", _clear_to_enter(False, None, None, NOW, HOST, veh_after, [], FRAME_H), True)

print("\nwe_go_first (who goes first)")
check("yield to lower-sorting name", we_go_first(HOST, [SignSemantic('', 'Vehicle', 'aaa_bot')]), False)
check("go before higher-sorting name", we_go_first(HOST, [SignSemantic('', 'Vehicle', 'zzz_bot')]), True)
check("no vehicles -> go", we_go_first(HOST, []), True)

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL TASK-2/3 LOGIC CHECKS PASSED")
