"""Integration tests — the runnable reference sign-detection agent (agent_signs.main).

Drives the real loop (LaneServoingAgentWithSigns + SignBehaviorFSM + duckie soft-stop
+ manual_cmd) against fakes, with no Godot/hardware. Confirms: it lane-follows with no
signs, parks the wheels on shutdown, the duckie soft-stop overrides, and the bot-safe
manual override drives through the agent's single writer."""

import threading
import time

import numpy as np
import pytest

from conftest import FakeCamera, FakeWheels, FakeLEDs, blank_frame

import tasks.project.packages.agent_signs as agent_signs


def _run(camera, wheels, leds, *, timings=None, manual_cmd=None,
         on_update=None, max_seconds=3.0):
    snaps = []
    lock = threading.Lock()

    def _obs(s):
        with lock:
            snaps.append(s)

    stop = threading.Event()
    th = threading.Thread(
        target=agent_signs.main, args=(camera, wheels, leds, stop),
        kwargs=dict(observer=_obs, manual_cmd=manual_cmd,
                    timings=timings or {'object_detection': False, 'bot_hsv': False, 'detect_every': 1}),
        daemon=True)
    th.start()
    t0 = time.time()
    try:
        while time.time() - t0 < max_seconds:
            time.sleep(0.02)
            if on_update:
                with lock:
                    s = list(snaps)
                if on_update(s, camera, wheels):
                    break
    finally:
        stop.set(); th.join(timeout=4)
    with lock:
        return list(snaps)


def test_lane_follows_with_no_signs():
    cam = FakeCamera(blank_frame())
    wheels, leds = FakeWheels(), FakeLEDs()
    snaps = _run(cam, wheels, leds, max_seconds=1.5)
    assert len(snaps) > 5
    assert all(s["state"] == "MOVING" for s in snaps)      # no sign -> pure lane-follow
    assert wheels.moved()
    assert snaps[-1]["agent"] == "signs"


def test_shutdown_parks_wheels_and_clears_leds():
    cam = FakeCamera(blank_frame())
    wheels, leds = FakeWheels(), FakeLEDs()
    _run(cam, wheels, leds, max_seconds=1.0)
    assert wheels.last == (0.0, 0.0)
    assert leds.all_off_calls >= 1


def _run_gated(camera, wheels, leds, gate, *, max_seconds=1.2):
    stop = threading.Event()
    th = threading.Thread(
        target=agent_signs.main, args=(camera, wheels, leds, stop),
        kwargs=dict(drive_gate=lambda: gate["open"],
                    timings={'object_detection': False, 'bot_hsv': False, 'detect_every': 1}),
        daemon=True)
    th.start()
    time.sleep(max_seconds)
    stop.set(); th.join(timeout=3)


def test_drive_gate_suppresses_writes_for_sim_manual_drive():
    """The sim uses drive_gate (a separate server loop owns the wheels). While the
    gate is CLOSED the agent must NOT write a driving command, so it can't fight the
    sim's manual driver; when OPEN it drives normally."""
    # gate CLOSED the whole run -> the only writes are the shutdown park (0,0)
    w = FakeWheels()
    _run_gated(FakeCamera(blank_frame()), w, FakeLEDs(), {"open": False}, max_seconds=1.2)
    assert all(l == 0.0 and r == 0.0 for l, r in w.history), \
        "agent wrote wheels while drive_gate was closed (would fight sim manual drive)"

    # gate OPEN -> the agent drives
    w2 = FakeWheels()
    _run_gated(FakeCamera(blank_frame()), w2, FakeLEDs(), {"open": True}, max_seconds=1.2)
    assert w2.moved(), "agent did not drive with drive_gate open"


def test_manual_cmd_overrides_through_single_writer():
    cam = FakeCamera(blank_frame())
    wheels, leds = FakeWheels(), FakeLEDs()

    def manual_cmd():
        return (0.22, -0.22)

    def _saw(snaps, c, w):
        return (0.22, -0.22) in w.history

    _run(cam, wheels, leds, manual_cmd=manual_cmd, on_update=_saw, max_seconds=3)
    assert (0.22, -0.22) in wheels.history, "manual command not applied by the agent"


def test_duckie_soft_stop_overrides():
    # a close duckie (low + large box) must halt the bot regardless of lane/sign.
    def provider(t):
        f = blank_frame()
        if t < 1.2:
            # a yellow CIRCLE low-centre that duck_hsv reads as a close duck (a circle's
            # solidity ~0.785 passes the anti-dash solidity gate; kept off the bottom
            # edge so the anti-chunky-dash gate doesn't reject it).
            import cv2
            cv2.circle(f, (325, 360), 70, (0, 220, 240), -1)
        return f

    cam = FakeCamera(provider)
    wheels, leds = FakeWheels(), FakeLEDs()
    timings = {'object_detection': True, 'duck_hsv': True, 'bot_hsv': False,
               'detect_every': 1, 'obstacle_clear_grace_s': 0.2}
    snaps = _run(cam, wheels, leds, timings=timings,
                 on_update=lambda s, c, w: any(x.get("obstacle_stop") for x in s),
                 max_seconds=4)
    assert any(s.get("obstacle_stop") for s in snaps), "never detected the duckie obstacle"
    # while the obstacle is present the wheels are parked
    stopped = [s for s in snaps if s.get("obstacle_stop")]
    assert any(s["wheels"]["left"] == 0.0 and s["wheels"]["right"] == 0.0 for s in stopped)


# ---------------------------------------------------------------------------
#  Other-bot soft-stop + obstacle debounce — the real-bot-footage fixes.
#
#  From the dashboard recording: a blue Duckiebot filled the frame for seconds while
#  the bot kept MOVING and drove into it. Cause: on this (agent_signs) path vehicle
#  detections only fed the FSM's CHECKPATH yield, so a bot DEAD AHEAD was ignored
#  unless at an intersection. These tests pin that an in-path bot now soft-stops,
#  an off-side bot does not, and a one-cycle yellow-marking flicker can't phantom-stop.
def _blue_bot_provider(until=1.2, box=(270, 300, 380, 440)):
    import cv2

    def provider(t):
        f = blank_frame()
        if t < until:
            # vivid blue (BGR 180,40,10 ~ HSV 114,241,180) = another Duckiebot body
            cv2.rectangle(f, (box[0], box[1]), (box[2], box[3]), (180, 40, 10), -1)
        return f
    return provider


def test_other_bot_in_path_soft_stops():
    cam = FakeCamera(_blue_bot_provider(until=1.2))
    wheels, leds = FakeWheels(), FakeLEDs()
    timings = {'object_detection': False, 'duck_hsv': False, 'bot_hsv': True,
               'detect_every': 1, 'obstacle_clear_grace_s': 0.2}
    snaps = _run(cam, wheels, leds, timings=timings,
                 on_update=lambda s, c, w: any(x.get("obstacle_stop") for x in s),
                 max_seconds=4)
    assert any(s.get("obstacle_stop") for s in snaps), \
        "never soft-stopped for an in-path Duckiebot (it would drive into it)"
    stopped = [s for s in snaps if s.get("obstacle_stop")]
    assert any(s["wheels"]["left"] == 0.0 and s["wheels"]["right"] == 0.0 for s in stopped)


def test_other_bot_off_to_the_side_does_not_stop():
    # a blue bot hugging the LEFT edge is not in our lane -> the detector's lateral
    # gate drops it, so no phantom soft-stop and the bot keeps lane-following.
    cam = FakeCamera(_blue_bot_provider(until=3.0, box=(10, 300, 110, 440)))
    wheels, leds = FakeWheels(), FakeLEDs()
    timings = {'object_detection': False, 'duck_hsv': False, 'bot_hsv': True,
               'detect_every': 1, 'obstacle_clear_grace_s': 0.2}
    snaps = _run(cam, wheels, leds, timings=timings, max_seconds=1.5)
    assert not any(s.get("obstacle_stop") for s in snaps), \
        "soft-stopped for a bot off to the side (not in our path)"
    assert wheels.moved()


def test_obstacle_confirm_debounces_a_one_cycle_flicker():
    """A brief yellow glint (a road-marking flicker) must NOT latch a stop when
    obstacle_confirm_frames > 1 — the fix for false stops on an empty road."""
    import cv2
    state = {"reads": 0}

    def provider(_t):
        f = blank_frame()
        state["reads"] += 1
        if state["reads"] <= 2:                 # duck visible for only 2 detect cycles
            cv2.circle(f, (325, 360), 70, (0, 220, 240), -1)
        return f

    cam = FakeCamera(provider)
    wheels, leds = FakeWheels(), FakeLEDs()
    timings = {'object_detection': True, 'duck_hsv': True, 'bot_hsv': False,
               'detect_every': 1, 'obstacle_confirm_frames': 6, 'obstacle_clear_grace_s': 0.2}
    snaps = _run(cam, wheels, leds, timings=timings, max_seconds=1.5)
    assert not any(s.get("obstacle_stop") for s in snaps), \
        "a 2-cycle flicker latched a stop despite obstacle_confirm_frames=6"


def test_obstacle_confirm_still_stops_for_a_persistent_duck():
    # the debounce must not BREAK real stops: a duck seen every frame satisfies the
    # confirm streak and stops, even with a raised threshold.
    import cv2

    def provider(t):
        f = blank_frame()
        if t < 2.0:
            cv2.circle(f, (325, 360), 70, (0, 220, 240), -1)
        return f

    cam = FakeCamera(provider)
    wheels, leds = FakeWheels(), FakeLEDs()
    timings = {'object_detection': True, 'duck_hsv': True, 'bot_hsv': False,
               'detect_every': 1, 'obstacle_confirm_frames': 3, 'obstacle_clear_grace_s': 0.2}
    snaps = _run(cam, wheels, leds, timings=timings,
                 on_update=lambda s, c, w: any(x.get("obstacle_stop") for x in s),
                 max_seconds=4)
    assert any(s.get("obstacle_stop") for s in snaps), \
        "a persistent duck never stopped despite the confirm streak being satisfiable"
