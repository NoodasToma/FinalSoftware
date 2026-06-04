from __future__ import annotations

import os
import random
import socket
import time
from collections import deque
from typing import Optional

import cv2
import yaml

from tasks.project.packages.perception import (
    TrafficLightDetector,
    AprilTagDetector,
    is_at_stop_line,
    merge_turn_constraints,
    should_brake_for_yellow,
    detect_red_line,
)
from tasks.project.packages.sign_registry import lookup
from tasks.project.packages.precedence import we_go_first

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.object_detection.packages.agent import ObjectDetectionAgent
from tasks.project.packages.states import State, next_state
from tasks.project.packages.maneuvers import (
    ramp_speed, turn_left, turn_right, straight_through,
)
from tasks.project.packages.obstacles import should_stop_for_obstacle


_TIMINGS_PATH = os.path.join(
    os.path.dirname(__file__), 'config', 'maneuver_timings.yaml'
)
_HSV_PATH = os.path.join(
    os.path.dirname(__file__), 'config', 'traffic_light_hsv.yaml'
)

_STOP_OR_YIELD = {'stop', 'yield'}
_T_LIGHT_AHEAD = 't-light-ahead'

_LED_FRONT_LEFT, _LED_FRONT_RIGHT = 0, 2
_LED_BACK_LEFT,  _LED_BACK_RIGHT  = 3, 4

_RED    = [1.0, 0.0, 0.0]
_YELLOW = [1.0, 0.6, 0.0]
_OFF    = [0.0, 0.0, 0.0]


def _load_timings() -> dict:
    with open(_TIMINGS_PATH) as fh:
        return yaml.safe_load(fh)


def _load_hsv_cfg() -> dict:
    """Red ranges (+ line_* knobs) for the stop-line detector; shares the
    traffic-light HSV YAML so red is tuned in ONE place per environment."""
    try:
        with open(_HSV_PATH) as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _set_brake(leds, on: bool) -> None:
    if leds is None:
        return
    color = _RED if on else _OFF
    leds.set_rgb(_LED_BACK_LEFT,  color)
    leds.set_rgb(_LED_BACK_RIGHT, color)


def _set_blinker(leds, direction: Optional[str], t_now: float = 0.0) -> None:
    """direction: 'left' | 'right' | 'off' | None. 2 Hz yellow blink when set."""
    if leds is None:
        return
    if direction in (None, 'off'):
        leds.all_off()
        return
    color = _YELLOW if (int(t_now * 4) % 2) == 0 else _OFF
    if direction == 'left':
        leds.set_rgb(_LED_FRONT_LEFT, color)
        leds.set_rgb(_LED_BACK_LEFT,  color)
    else:
        leds.set_rgb(_LED_FRONT_RIGHT, color)
        leds.set_rgb(_LED_BACK_RIGHT,  color)


def _direction_of(state: State) -> str:
    return {
        State.TURN_LEFT:        'left',
        State.TURN_RIGHT:       'right',
        State.STRAIGHT_THROUGH: 'off',
    }[state]


def _derive_event(state, signs, obstacles, frame_h, lane_mask,
                  ignore_signs=False, stop_distance_m=0.25,
                  at_red_line=False, react_distance_m=1.5) -> Optional[str]:
    if state in (State.DRIVE, State.APPROACH):
        stop, _ = should_stop_for_obstacle(obstacles, frame_h)
        if stop:
            return 'obstacle'

    if state == State.DRIVE:
        # ignore_signs is the post-intersection cooldown: after we stop+turn we
        # are still right next to the sign we just obeyed, so without this we'd
        # re-trigger on it and stop/turn over and over (a 360 spin in place).
        if not ignore_signs:
            # React only to signs governing the junction we're ARRIVING at
            # (within react_distance_m). Tags can decode from 3+ m away, and
            # starting the slow APPROACH creep that early both crawls and reacts
            # to intersections beyond the next one. est inf (uncalibrated
            # camera) keeps the old react-on-sight behavior.
            def near(o):
                if o is None:          # offline/logic tests pass bare semantics
                    return True
                return (o.est_distance_m < react_distance_m
                        or o.est_distance_m == float('inf'))
            kinds = {sem.kind for o, sem in signs if near(o)}
            if kinds & _STOP_OR_YIELD:
                return 'see_stop_or_yield'
            if _T_LIGHT_AHEAD in kinds:
                # A traffic-light intersection also makes us approach + stop at the
                # line, where STOPPED waits for green. (Previously only stop/yield
                # signs triggered APPROACH, so the bot drove straight through lights.)
                return 'see_light'
        return None

    if state == State.APPROACH:
        # PRIMARY stop trigger: the painted red stop line reaching the bottom of
        # the frame (we are physically AT the line). This is how real Duckiebots
        # stop at intersections; the tag-distance check below is the backup for
        # lines that are missing/washed out.
        if at_red_line:
            return 'at_stop_line'
        for obs, sem in signs:
            if sem.kind in _STOP_OR_YIELD or sem.kind == _T_LIGHT_AHEAD:
                if is_at_stop_line(obs, lane_mask, stop_distance_m):
                    return 'at_stop_line'
        return None

    if state == State.SOFT_STOP:
        stop, _ = should_stop_for_obstacle(obstacles, frame_h)
        if not stop:
            return 'obstacle_cleared'

    return None


def _clear_to_enter(light_was_red, light_color, yellow_started_at, now,
                    my_name, recent_vehicle_signs, obstacles, frame_h) -> bool:
    """Whether it is safe AND legal to enter the intersection from STOPPED/WAIT.

    Enforces the Task-2/3 rules in one place:
      * stop on red, go on green  (light_ok)
      * never *start* crossing on a settled yellow — we can't guarantee clearing
        the box in the remaining time  (yellow_hold)
      * yield to a vehicle that has precedence  (prec_ok)
      * never enter while an obstacle blocks the box  (obstacle_present)

    When no light is in play (stop/yield sign), light_was_red stays False and
    light_color is None, so light_ok=True / yellow_hold=False and the decision
    falls through to precedence + obstacle, preserving plain stop-sign behaviour.
    """
    light_ok = (not light_was_red) or (light_color == 'green')
    yellow_hold = should_brake_for_yellow(light_color, yellow_started_at, now)
    prec_ok = we_go_first(my_name, list(recent_vehicle_signs))
    obstacle_present, _ = should_stop_for_obstacle(obstacles, frame_h)
    return light_ok and not yellow_hold and prec_ok and not obstacle_present


def main(camera, wheels, leds, stop_event, *, observer=None,
         apriltag_intrinsics=None, apriltag_tag_size=None,
         timings_override=None, lane_config_path=None):
    """Main perception -> decision -> motor loop (same on bot and in sim).

    The keyword-only args are PLATFORM hooks with no-op defaults, so a bare
    main(camera, wheels, leds, stop_event) behaves exactly as before:
      * observer(snapshot):   if given, called once per loop with a dict of what
                              the agent perceived + decided (pose-less; the sim
                              telemetry logger enriches it with Godot pose).
                              None -> zero overhead.   [sim]
      * apriltag_intrinsics:  (fx, fy, cx, cy) for the AprilTag detector so the
                              sim computes a real est_distance_m. None -> the
                              detector's normal file search (the real bot reads
                              config/camera_intrinsics.yaml once calibrated).  [sim]
      * apriltag_tag_size:    physical sim tag size (m) matching the intrinsics.
                              None -> 0.065 (the real Duckietown tag size). [sim]
      * timings_override:     a complete timings dict to use INSTEAD of
                              maneuver_timings.yaml — real_server passes the base
                              file merged with maneuver_timings_bot.yaml so the
                              robot gets hardware-corrected maneuver values
                              (pwm_min compresses speed ratios; see the overlay
                              file). None -> load the YAML as always.   [bot]
      * lane_config_path:     alternate lane-servoing config YAML — real_server
                              passes config/lane_servoing_config_bot.yaml so the
                              robot starts from gentle hardware gains instead of
                              the sim-tuned ones. None -> the default file. [bot]
    """
    lane  = LaneServoingAgent(config_path=lane_config_path)
    obj   = ObjectDetectionAgent()
    light = TrafficLightDetector()
    tags  = AprilTagDetector(
        tag_size_m=apriltag_tag_size or 0.065,
        intrinsics=apriltag_intrinsics,
    )
    timings = timings_override if timings_override is not None else _load_timings()
    hsv_cfg = _load_hsv_cfg()

    state = State.DRIVE
    current_speed = 0.0
    my_name = socket.gethostname()
    yellow_started_at: Optional[float] = None
    light_was_red = False
    recent_vehicle_signs: deque = deque(maxlen=10)
    # Signs accumulated over the WHOLE approach (keyed by tag id). The legal-turn
    # decision used to read only the final pre-stop frame, but by the stop line
    # the intersection sign has usually scrolled out of view — so remember every
    # sign seen while approaching this intersection.
    intersection_signs: dict = {}
    signs_at_intersection: list = []
    ignore_signs_until = 0.0
    approach_closest = float('inf')   # closest a stop/yield/light sign got this APPROACH
    relevant_lost_at: Optional[float] = None  # when the approached sign left the frame
    stopped_at: Optional[float] = None  # when we entered STOPPED (for the full-stop pause)

    try:
        while not stop_event.is_set():
            ok, bgr = camera.read()
            if not ok:
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_h = bgr.shape[0]
            now = time.time()

            tag_obs = tags.detect(bgr) or []
            signs = [(o, lookup(o.id)) for o in tag_obs if lookup(o.id) is not None]
            obstacles = obj.detect(rgb) or []
            light_color = light.detect(bgr)

            for _, sem in signs:
                if sem.tag_type == 'Vehicle':
                    recent_vehicle_signs.append(sem)

            if light_color == 'yellow':
                if yellow_started_at is None:
                    yellow_started_at = now
            else:
                yellow_started_at = None
            if light_color == 'red':
                light_was_red = True
            elif light_color == 'green':
                light_was_red = False

            # Arm the light detector only when a t-light tag is CLOSE — i.e. the
            # light governs the junction we are arriving at. The sim render
            # decodes tags out to ~3 m, and arming on a far-off tag made the bot
            # obey a light 3+ m down the road while standing at a stop sign.
            # est inf (uncalibrated camera) keeps the old arm-on-sight behavior.
            arm_dist = timings.get('light_arm_distance_m', 1.5)
            if any(sem.kind == _T_LIGHT_AHEAD
                   and (o.est_distance_m < arm_dist or o.est_distance_m == float('inf'))
                   for o, sem in signs):
                light.arm()
            elif state == State.DRIVE and light_color is None:
                light.disarm()

            lane_mask = lane.last_debug_info.get('lane_mask')
            ignore_signs = now < ignore_signs_until

            # Commit latch: track how close a stop/yield/light sign got while we
            # were approaching. If it then scrolls out of view (a big tag rolls off
            # the top of the frame as we reach the line, or a momentary mis-detect)
            # we still stop -- once the intersection is identified a real bot
            # commits to stopping; it doesn't drive through just because the sign
            # left the camera frame. Reset whenever we're not approaching.
            red_line = detect_red_line(bgr, hsv_cfg)

            relevant_visible = False
            if state == State.APPROACH:
                for o, sem in signs:
                    intersection_signs[o.id] = sem    # remember every sign this approach
                    if sem.kind in _STOP_OR_YIELD or sem.kind == _T_LIGHT_AHEAD:
                        relevant_visible = True
                        approach_closest = min(approach_closest, o.est_distance_m)
                if relevant_visible:
                    relevant_lost_at = None
                elif relevant_lost_at is None:
                    relevant_lost_at = now
            else:
                approach_closest = float('inf')
                relevant_lost_at = None

            event = _derive_event(state, signs, obstacles, frame_h, lane_mask,
                                  ignore_signs=ignore_signs,
                                  stop_distance_m=timings.get('stop_distance_m', 0.25),
                                  at_red_line=red_line,
                                  react_distance_m=timings.get('sign_react_distance_m', 1.5))
            # Backup commit: the sign we were approaching left the frame after
            # getting close AND stayed gone for a grace period AND no red line has
            # appeared. The grace keeps the painted line the PRIMARY trigger — the
            # tag usually decodes its last ~0.3 m before the line, and the line
            # enters the bottom ROI moments later; without the grace this commit
            # fired the instant the tag dropped and stopped the bot short of the
            # line. Only if the line truly never shows (missing/washed out) does
            # this stop the bot near where the sign was.
            if (state == State.APPROACH and not event and not relevant_visible
                    and approach_closest < timings.get('stop_commit_distance_m', 0.5)
                    and relevant_lost_at is not None
                    and now - relevant_lost_at > timings.get('stop_commit_grace_s', 1.0)):
                event = 'at_stop_line'
            if event:
                new_state = next_state(state, event)
                if new_state == State.STOPPED and state in (State.APPROACH, State.WAIT):
                    if state == State.APPROACH:
                        signs_at_intersection = list(intersection_signs.values())
                    stopped_at = now      # full-stop pause starts now
                state = new_state

            if state == State.DRIVE:
                left, right = lane.compute_commands(rgb)
                current_speed = ramp_speed(
                    current_speed, timings['base_speed'], timings['ramp_max_step']
                )
                wheels.set_wheels_speed(
                    left  * current_speed * 2,
                    right * current_speed * 2,
                )
                _set_blinker(leds, 'off')

            elif state == State.APPROACH:
                # Creep toward the line (don't ramp to a dead stop) until we're
                # actually AT the stop line. Ramping to 0 made the bot halt the
                # instant a still-distant sign was first seen, stranding it short
                # of the line in APPROACH forever. STOPPED does the full stop.
                current_speed = ramp_speed(
                    current_speed, timings.get('approach_creep_speed', 0.12),
                    timings['ramp_max_step']
                )
                if approach_closest < timings.get('line_straight_distance_m', 0.6):
                    # Final stretch: the lane markings end at the intersection box,
                    # so lane-steering here yanks the heading right when we want to
                    # roll straight up to the painted line. Creep dead straight.
                    lane.compute_commands(rgb)   # keep masks/debug fresh
                    wheels.set_wheels_speed(current_speed, current_speed)
                else:
                    left, right = lane.compute_commands(rgb)
                    wheels.set_wheels_speed(
                        left  * current_speed * 2,
                        right * current_speed * 2,
                    )
                _set_brake(leds, on=True)

            elif state == State.STOPPED:
                wheels.set_wheels_speed(0.0, 0.0)
                current_speed = 0.0
                _set_brake(leds, on=True)
                legal_turns = merge_turn_constraints(signs_at_intersection)
                # A stop sign means a FULL stop: hold for stop_wait_seconds FIRST,
                # like a real car at a stop line — only then check right-of-way and
                # go. (Pause-first also debounces one-frame "not clear" blips that
                # used to bounce us through WAIT and skip the pause.)
                paused_enough = (stopped_at is None or
                                 now - stopped_at >= timings.get('stop_wait_seconds', 1.5))
                if paused_enough:
                    clear = _clear_to_enter(light_was_red, light_color, yellow_started_at,
                                            now, my_name, recent_vehicle_signs, obstacles, frame_h)
                    if not clear:
                        state = next_state(state, 'wait')
                    elif legal_turns:
                        choice = random.choice(sorted(legal_turns))
                        state = next_state(state, {
                            'left':     'choose_turn_left',
                            'right':    'choose_turn_right',
                            'straight': 'choose_straight',
                        }[choice])

            elif state == State.WAIT:
                wheels.set_wheels_speed(0.0, 0.0)
                _set_brake(leds, on=True)
                if _clear_to_enter(light_was_red, light_color, yellow_started_at,
                                   now, my_name, recent_vehicle_signs, obstacles, frame_h):
                    state = next_state(state, 'cleared')
                    # We already held at the line while waiting (red light etc.) —
                    # don't add ANOTHER full-stop pause on top; go when clear.
                    stopped_at = now - timings.get('stop_wait_seconds', 1.5)

            elif state in (State.TURN_LEFT, State.TURN_RIGHT, State.STRAIGHT_THROUGH):
                _set_blinker(leds, _direction_of(state), now)
                {
                    State.TURN_LEFT:        turn_left,
                    State.TURN_RIGHT:       turn_right,
                    State.STRAIGHT_THROUGH: straight_through,
                }[state](wheels, stop_event, timings)
                signs_at_intersection = []
                intersection_signs = {}
                light_was_red = False
                current_speed = 0.0
                # Cooldown: drive clear of the intersection before signs can fire
                # again, so we don't immediately re-stop on the sign we just obeyed.
                ignore_signs_until = time.time() + timings.get('sign_cooldown', 4.0)
                state = next_state(state, 'turn_done')

            elif state == State.SOFT_STOP:
                # Hold until the obstacle clears (a duckie is a pedestrian:
                # driving through it is a collision on hardware, and game-over
                # in sim). The SOFT_STOP -> 'obstacle_cleared' event above
                # resumes us automatically once the path is clear.
                wheels.set_wheels_speed(0.0, 0.0)
                current_speed = 0.0
                _set_brake(leds, on=True)

            if observer is not None:
                # One snapshot of exactly what the agent perceived and decided
                # this loop. Pose-less here (platform-agnostic); the sim
                # telemetry logger adds Godot pose. No-op on hardware (None).
                observer({
                    't': now,
                    'state': state.name,
                    'event': event,
                    'current_speed': round(current_speed, 4),
                    'wheels': {
                        'left':  round(float(getattr(wheels, 'left_pwm', 0.0)), 4),
                        'right': round(float(getattr(wheels, 'right_pwm', 0.0)), 4),
                    },
                    'tags': [
                        {'id': o.id,
                         'meaning': (sem.kind or sem.tag_type or '?'),
                         'center_xy': list(o.center_xy),
                         'side_px': o.side_length_px,
                         'est_distance_m': (None if o.est_distance_m == float('inf')
                                            else round(o.est_distance_m, 4))}
                        for o, sem in signs
                    ],
                    'light': {'color': light_color, 'armed': light.armed},
                    'red_line': red_line,
                    'obstacle_stop': should_stop_for_obstacle(obstacles, frame_h)[0],
                    'lane': {
                        'error':    round(float(lane.last_debug_info.get('lateral_error', 0.0)), 4),
                        'detected': bool(lane.last_debug_info.get('lane_detected', False)),
                        'total_px': int(lane.last_debug_info.get('total_lane_pixels', 0)),
                    },
                    'legal_turns': (sorted(merge_turn_constraints(signs_at_intersection))
                                    if state in (State.STOPPED, State.WAIT) else None),
                })

            time.sleep(0.02)
    finally:
        try:
            wheels.set_wheels_speed(0.0, 0.0)
        except Exception:
            pass
        if leds is not None:
            try:
                leds.all_off()
            except Exception:
                pass
