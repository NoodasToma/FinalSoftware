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


def _derive_event(state, signs, obstacles, frame_h, lane_mask) -> Optional[str]:
    if state in (State.DRIVE, State.APPROACH):
        stop, _ = should_stop_for_obstacle(obstacles, frame_h)
        if stop:
            return 'obstacle'

    if state == State.DRIVE:
        kinds = {sem.kind for _, sem in signs}
        if kinds & _STOP_OR_YIELD:
            return 'see_stop_or_yield'
        return None

    if state == State.APPROACH:
        for obs, sem in signs:
            if sem.kind in _STOP_OR_YIELD or sem.kind == _T_LIGHT_AHEAD:
                if is_at_stop_line(obs, lane_mask):
                    return 'at_stop_line'
        return None

    if state == State.SOFT_STOP:
        stop, _ = should_stop_for_obstacle(obstacles, frame_h)
        if not stop:
            return 'obstacle_cleared'

    return None


def main(camera, wheels, leds, stop_event):
    lane  = LaneServoingAgent()
    obj   = ObjectDetectionAgent()
    light = TrafficLightDetector()
    tags  = AprilTagDetector()
    timings = _load_timings()

    state = State.DRIVE
    current_speed = 0.0
    my_name = socket.gethostname()
    yellow_started_at: Optional[float] = None
    light_was_red = False
    recent_vehicle_signs: deque = deque(maxlen=10)
    signs_at_intersection: list = []

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

            if any(sem.kind == _T_LIGHT_AHEAD for _, sem in signs):
                light.arm()
            elif state == State.DRIVE and light_color is None:
                light.disarm()

            lane_mask = lane.last_debug_info.get('lane_mask')
            event = _derive_event(state, signs, obstacles, frame_h, lane_mask)
            if event:
                new_state = next_state(state, event)
                if new_state == State.STOPPED and state == State.APPROACH:
                    signs_at_intersection = [s for _, s in signs]
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
                left, right = lane.compute_commands(rgb)
                current_speed = ramp_speed(
                    current_speed, 0.0, timings['ramp_max_step']
                )
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
                if light_was_red and light_color != 'green':
                    state = next_state(state, 'wait')
                elif not we_go_first(my_name, list(recent_vehicle_signs)):
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
                light_ok = (not light_was_red) or (light_color == 'green')
                prec_ok  = we_go_first(my_name, list(recent_vehicle_signs))
                if light_ok and prec_ok:
                    state = next_state(state, 'cleared')

            elif state in (State.TURN_LEFT, State.TURN_RIGHT, State.STRAIGHT_THROUGH):
                _set_blinker(leds, _direction_of(state), now)
                {
                    State.TURN_LEFT:        turn_left,
                    State.TURN_RIGHT:       turn_right,
                    State.STRAIGHT_THROUGH: straight_through,
                }[state](wheels, stop_event, timings)
                signs_at_intersection = []
                light_was_red = False
                current_speed = 0.0
                state = next_state(state, 'turn_done')

            elif state == State.SOFT_STOP:
                wheels.set_wheels_speed(0.0, 0.0)
                current_speed = 0.0
                _set_brake(leds, on=True)

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
