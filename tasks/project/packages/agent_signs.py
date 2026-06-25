"""Runnable agent that uses the REFERENCE's working sign detection.

This is an ALTERNATIVE to tasks/project/packages/agent.main: same
main(camera, wheels, leds, stop_event, ...) interface and the same dashboard hooks
(observer / frame_observer / manual_cmd), but the perception->decision->motor path is
the KvatiTown design:

    lane = LaneServoingAgentWithSigns(...)         # lane servoing + SignBehaviorFSM
    left, right, sign_state = lane.compute_commands(rgb, vehicle_detections)

The SignBehaviorFSM (tasks/project/packages/sign_detection/) watches for AprilTags and
a painted red STOP LINE. A red line ALONE does nothing; only a red line reached WITH a
saved sign tag triggers behaviour:
  * stop/yield tag  -> brake to a full stop, hold, check for cross traffic, then go;
  * intersection tag -> pick a legal turn and EXECUTE it OPEN-LOOP (fixed wheel speeds
    for a fixed number of frames, LANE IGNORED), then resume lane-following.
That open-loop turn is what makes the sign decision authoritative — the lane can't pull
the bot the wrong way through the junction.

We add two things the bare FSM doesn't: a duckie obstacle soft-stop (so the bot still
brakes for ducks) and the bot-safe manual_cmd override (single I2C writer). The project
agent.main is left completely untouched; select this one with `sign_agent: true` in the
timings (real_server) — see servers/project/real_server.py.
"""

import sys
import threading
import time
import traceback
from typing import Optional

import cv2

from tasks.project.packages.sign_detection.agent_with_signs import LaneServoingAgentWithSigns
from tasks.project.packages.perception.duck_hsv import detect_duckies_hsv, detect_vehicles_hsv
from tasks.project.packages.obstacles import should_stop_for_obstacle

# LED indices (match agent.py)
_LED_FRONT_LEFT, _LED_FRONT_RIGHT = 0, 2
_LED_BACK_LEFT,  _LED_BACK_RIGHT  = 3, 4
_RED = [1.0, 0.0, 0.0]
_OFF = [0.0, 0.0, 0.0]

# Sign-FSM states where the bot is braking/standing -> back LEDs red.
_BRAKE_STATES = {"SLOWING", "STOPPED", "CHECKPATH"}
# ... and where it's turning -> (we just light the back red too; blinker optional).
_TURN_STATES = {"INTERSECT", "PRE_TURN", "TURNING", "EXITING", "APPROACHING"}


def main(camera, wheels, leds, stop_event, *, observer=None, frame_observer=None,
         manual_cmd=None, drive_gate=None, lane_config_path=None, sign_config=None,
         timings=None):
    # manual_cmd (bot real_server) and drive_gate (sim virtual_server) are the two
    # manual-drive styles; both are honoured in _safe_wheels so this agent is a
    # drop-in for either server. manual_cmd takes precedence if both are given.
    timings = timings or {}
    lane = LaneServoingAgentWithSigns(config_path=lane_config_path, sign_config=sign_config)
    print("[agent_signs] reference sign-detection agent running "
          "(lane servoing + SignBehaviorFSM)", flush=True)

    _io_lock = threading.Lock()
    detect_every = max(1, int(timings.get('detect_every', 3)))
    duck_cfg = timings.get('duck_hsv_cfg') or None
    veh_cfg = timings.get('bot_hsv_cfg') or None
    obstacle_grace = float(timings.get('obstacle_clear_grace_s', 1.0))
    # Obstacle debounce + lateral gate (parity with the project agent.main, which the
    # bot path was missing): require an obstacle on this many CONSECUTIVE fresh-detection
    # cycles before braking (kills one-frame yellow-marking / glare phantom stops), and
    # ignore duck blobs outside the central band (off-side ROAD MARKINGS, not in our path).
    obstacle_confirm = max(1, int(timings.get('obstacle_confirm_frames', 1)))
    obstacle_cx_margin = float(timings.get('obstacle_cx_margin_frac', 0.0))
    duck_enabled = bool(timings.get('object_detection', True)) and bool(timings.get('duck_hsv', True))
    veh_enabled = bool(timings.get('bot_hsv', True))

    _last_err = [0.0]

    def _log_err(where):
        nowp = time.time()
        if nowp - _last_err[0] > 2.0:
            _last_err[0] = nowp
            print(f"[agent_signs] {where} error (continuing):", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()

    def _safe_wheels(left, right):
        # Manual drive — two styles, both honoured so this agent is a drop-in for the
        # bot server (manual_cmd) and the sim server (drive_gate):
        #   * manual_cmd() -> (l, r): the AGENT writes that instead of its own (single
        #     I2C writer, bot-safe). Takes precedence.
        #   * drive_gate() False: a separate server loop owns the wheels, so we skip
        #     our write entirely (don't fight it).
        if manual_cmd is not None:
            try:
                mc = manual_cmd()
            except Exception:
                mc = None
            if mc is not None:
                try:
                    left, right = float(mc[0]), float(mc[1])
                except Exception:
                    pass
        elif drive_gate is not None:
            try:
                if not drive_gate():
                    return
            except Exception:
                pass
        try:
            with _io_lock:
                wheels.set_wheels_speed(left, right)
        except Exception:
            _log_err('wheels.set_wheels_speed')

    def _set_back(color):
        if leds is None:
            return
        try:
            with _io_lock:
                leds.set_rgb(_LED_BACK_LEFT, color)
                leds.set_rgb(_LED_BACK_RIGHT, color)
        except Exception:
            pass

    frame_count = 0
    raw_dets = []
    veh_dets = []
    obstacle_last_seen = -1e9
    obstacle_streak = 0

    try:
        while not stop_event.is_set():
            try:
                ok, bgr = camera.read()
            except Exception:
                _log_err('camera.read'); _safe_wheels(0.0, 0.0); time.sleep(0.05); continue
            if not ok or bgr is None:
                _safe_wheels(0.0, 0.0); time.sleep(0.03); continue
            if frame_observer is not None:
                try:
                    frame_observer(bgr)
                except Exception:
                    pass
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_h = bgr.shape[0]
            now = time.time()
            frame_count += 1

            # Detections (HSV, no model — works on the bot's OpenCV). Ducks (class 0)
            # for the obstacle stop; vehicles (class 1) feed the FSM's CHECKPATH yield.
            if frame_count % detect_every == 0:
                if duck_enabled:
                    try:
                        raw_dets = detect_duckies_hsv(bgr, duck_cfg) or []
                    except Exception:
                        _log_err('duck_hsv'); raw_dets = []
                if veh_enabled:
                    try:
                        veh_dets = detect_vehicles_hsv(bgr, veh_cfg) or []
                    except Exception:
                        _log_err('vehicle_hsv'); veh_dets = []

            # Lane servoing + sign behaviour (the FSM may OVERRIDE the lane command).
            try:
                left, right, sign_state = lane.compute_commands(rgb, veh_dets)
            except Exception:
                _log_err('lane.compute_commands'); left, right = 0.0, 0.0
                sign_state = getattr(lane, 'sign_state', '?')
            state_name = sign_state.name if hasattr(sign_state, 'name') else str(sign_state)

            # Obstacle soft-stop (debounced) layered on top of the FSM/lane command:
            # the bot brakes and holds until the obstacle clears, whatever the lane
            # wanted. TWO sources feed it (matching the project agent.main):
            #   * a close duckie ahead (should_stop_for_obstacle) — now with the same
            #     lateral "in the way" gate, so yellow ROAD MARKINGS off to the side
            #     don't brake the bot, and
            #   * another Duckiebot in our path: detect_vehicles_hsv only returns blobs
            #     that are already low + roughly-centred + close, so any return is an
            #     in-path bot. Without this the bot only ever reacted to other bots when
            #     a yellow wheel happened to trip the DUCK detector — it drove into them.
            # The confirm streak is advanced only on fresh-detection cycles (detections
            # are stale between them), so a one-cycle yellow-marking / glare flicker can't
            # trigger a phantom stop; the grace window then keeps a real stop latched
            # after the obstacle dips out of the down-tilted view.
            duck_stop = bool(raw_dets) and should_stop_for_obstacle(
                raw_dets, frame_h, bgr.shape[1], obstacle_cx_margin)[0]
            veh_stop = bool(veh_dets)
            if frame_count % detect_every == 0:
                obstacle_streak = obstacle_streak + 1 if (duck_stop or veh_stop) else 0
            if obstacle_streak >= obstacle_confirm:
                obstacle_last_seen = now
            obstacle_present = (now - obstacle_last_seen) < obstacle_grace
            if obstacle_present:
                left, right = 0.0, 0.0

            _safe_wheels(left, right)
            _set_back(_RED if (obstacle_present or state_name in _BRAKE_STATES) else _OFF)

            if observer is not None:
                dbg = lane.sign_debug or {}
                ldi = lane.last_debug_info or {}
                try:
                    observer({
                        't': now,
                        'state': state_name,
                        'wheels': {'left': round(float(left), 4), 'right': round(float(right), 4)},
                        'tags': [{'id': int(t), 'meaning': '?'} for t in dbg.get('confirmed_tags', [])],
                        'saved_tag': dbg.get('saved_tag'),
                        'red_line': bool(dbg.get('red_line', False)),
                        'legal_turns': dbg.get('possible_turns') or None,
                        'chosen_turn': dbg.get('chosen_turn'),
                        'obstacle_stop': obstacle_present,
                        'lane': {
                            'error': round(float(ldi.get('lateral_error', 0.0)), 4),
                            'detected': bool(ldi.get('lane_detected', False)),
                            'total_px': int(ldi.get('total_lane_pixels', 0)),
                        },
                        'sign_debug': dbg,
                        'agent': 'signs',
                    })
                except Exception:
                    pass

            time.sleep(0.02)
    finally:
        _safe_wheels(0.0, 0.0)
        if leds is not None:
            try:
                with _io_lock:
                    leds.all_off()
            except Exception:
                pass
