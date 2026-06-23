"""
sign_behavior.py
"""

import random
import time
from typing import Dict, List, Optional

from tasks.project.packages.sign_detection.red_line_detection import detect_red_line
from tasks.project.packages.sign_detection.april_tag import detect_tags, confirm_tags
from tasks.project.packages.sign_detection.vehicle import vehicle_detected
from tasks.project.packages.sign_detection.sign_behavior_config import (
    TagID,
    _TAG_TURNS,
    SignBehaviorConfig,
    State,
    resolve_tag,
)


class SignBehaviorFSM:
    FPS = 24.0

    def __init__(self, config=None):
        self.cfg = config or SignBehaviorConfig()
        self.state = State.MOVING

        print("[SignBehavior] initialised — detector ready")

        self._saved_tag_time = None
        self._saved_tag_timeout = 4.0  # seconds 

        self._approach_start_time = None

        self._tag_buffer = {}       # type: Dict[int, int]
        self._saved_tag = None      # type: Optional[TagID]

        self._hold_counter = 0.0
        self._turn_counter = 0.0
        self._check_counter = 0.0

        self._vehicle_seen_left = False
        self._vehicle_seen_right = False

        self._possible_turns = []   # type: List[str]
        self._chosen_turn = None    # type: Optional[str]

        self._slow_factor = 1.0
        self._slow_target = 0.0

        self._red_line_locked = False
        self._ignore_red_counter = 0.0

        self.debug = {}

        self._frame_rgb = None
        self._vehicle_detected = vehicle_detected

        self._left_offsets = []
        self._right_offsets = []

        self._active_sign_op = None
        self._ignored_red_print_cooldown = 0.0

        self._state_started_at = time.monotonic()
        self._last_step_time = None
        self._dt = 1.0 / self.FPS

    def reset(self):
        self.state = State.MOVING

        self._tag_buffer.clear()
        self._saved_tag = None

        self._saved_tag_time = None

        self._hold_counter = 0.0
        self._turn_counter = 0.0
        self._check_counter = 0.0

        self._vehicle_seen_left = False
        self._vehicle_seen_right = False

        self._possible_turns = []
        self._chosen_turn = None

        self._slow_factor = 1.0
        self._slow_target = 0.0

        self._red_line_locked = False
        self._ignore_red_counter = 0.0

        self._approach_start_time = None

        self._left_offsets.clear()
        self._right_offsets.clear()

        self._active_sign_op = None
        self._ignored_red_print_cooldown = 0.0

        self._state_started_at = time.monotonic()
        self._last_step_time = None
        self._dt = 1.0 / self.FPS

        self.debug = {}

        print("[SignBehavior] FSM reset")

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    def _frames(self, attr_name, default=0.0):
        value = getattr(self.cfg, attr_name, default)
        return max(0.0, self._safe_float(value, default))

    def _speed(self, attr_name, default=0.0):
        value = getattr(self.cfg, attr_name, default)
        return self._safe_float(value, default)

    def _pair(self, attr_name, default):
        value = getattr(self.cfg, attr_name, default)

        try:
            left, right = value
            return self._safe_float(left, default[0]), self._safe_float(right, default[1])
        except Exception:
            return self._safe_float(default[0]), self._safe_float(default[1])

    def _frames_to_seconds(self, frames):
        return self._safe_float(frames, 0.0) / self.FPS

    def _seconds_to_frames(self, seconds):
        return self._safe_float(seconds, 0.0) * self.FPS

    def _elapsed_seconds(self):
        return max(0.0, time.monotonic() - self._state_started_at)

    def _elapsed_frames(self):
        return self._elapsed_seconds() * self.FPS

    def _set_state(self, new_state):
        self.state = new_state
        self._state_started_at = time.monotonic()

        self._hold_counter = 0.0
        self._turn_counter = 0.0
        self._check_counter = 0.0

    def _restart_current_state_timer(self):
        self._state_started_at = time.monotonic()

        self._hold_counter = 0.0
        self._turn_counter = 0.0
        self._check_counter = 0.0

    def _update_dt(self):
        now = time.monotonic()

        if self._last_step_time is None:
            self._dt = 1.0 / self.FPS
        else:
            self._dt = max(0.0, now - self._last_step_time)

        self._last_step_time = now

    def _update_time_counters(self):
        frames = self._elapsed_frames()

        if self.state == State.STOPPED:
            self._hold_counter = frames
        elif self.state == State.CHECKPATH:
            self._check_counter = frames
        elif self.state in (State.POST_STOP, State.INTERSECT, State.PRE_TURN, State.TURNING, State.EXITING):
            self._turn_counter = frames
        else:
            self._hold_counter = 0.0
            self._turn_counter = 0.0
            self._check_counter = 0.0

    def _tick_timers(self):
        if self._ignore_red_counter > 0.0:
            self._ignore_red_counter = max(0.0, self._ignore_red_counter - self._dt)

        if self._ignored_red_print_cooldown > 0.0:
            self._ignored_red_print_cooldown = max(0.0, self._ignored_red_print_cooldown - self._dt)

    def step(self, frame_rgb, base_left, base_right, detections):
        self._update_dt()
        self._tick_timers()

        self._frame_rgb = frame_rgb
        detections = detections or []

        tags = detect_tags(self, frame_rgb)
        confirmed = confirm_tags(self, tags)

        if self._ignore_red_counter > 0.0:
            red_line = False
        elif self._red_line_locked:
            red_line = False
        else:
            red_line = bool(detect_red_line(self, frame_rgb))

        left, right = self._fsm_step(
            confirmed,
            detections,
            red_line,
            base_left,
            base_right,
            frame_rgb,
        )

        self._update_time_counters()

        self.debug = {
            "state": self.state.name,
            "confirmed_tags": [int(t) for t in confirmed],
            "saved_tag": int(self._saved_tag) if self._saved_tag is not None else None,
            "red_line": bool(red_line),
            "red_locked": bool(self._red_line_locked),
            "ignore_red_counter": int(round(self._seconds_to_frames(self._ignore_red_counter))),
            "ignore_red_seconds": round(float(self._ignore_red_counter), 3),
            "possible_turns": list(self._possible_turns),
            "chosen_turn": self._chosen_turn,
            "slow_factor": round(float(self._slow_factor), 3),
            "hold_counter": int(round(self._hold_counter)),
            "check_counter": int(round(self._check_counter)),
            "turn_counter": int(round(self._turn_counter)),
            "state_elapsed_seconds": round(float(self._elapsed_seconds()), 3),
            "state_elapsed_frames_24fps": int(round(self._elapsed_frames())),
            "dt_seconds": round(float(self._dt), 4),
        }

        return left, right, self.state

    @property
    def state_name(self):
        return self.state.name

    def _save_confirmed_sign(self, confirmed_tags):
        # Prefer intersection signs over stop/yield if both are visible.
        for raw_tag_id in confirmed_tags:
            tag_id = resolve_tag(int(raw_tag_id))

            if tag_id is None:
                continue

            if tag_id in _TAG_TURNS:
                if self._saved_tag != tag_id:
                    print(
                        f"[SignBehavior] sign saved: intersection tag "
                        f"raw={raw_tag_id}, resolved={int(tag_id)} "
                        f"-> {_TAG_TURNS[tag_id]}"
                    )

                self._saved_tag = tag_id
                self._saved_tag_time = time.monotonic()
                return

        for raw_tag_id in confirmed_tags:
            tag_id = resolve_tag(int(raw_tag_id))

            if tag_id is None:
                continue

            if tag_id in (TagID.STOP, TagID.YIELD):
                if self._saved_tag != tag_id:
                    label = "STOP" if tag_id == TagID.STOP else "YIELD"
                    print(
                        f"[SignBehavior] sign saved: {label} "
                        f"(raw={raw_tag_id}, resolved={int(tag_id)})"
                    )

                self._saved_tag = tag_id
                self._saved_tag_time = time.monotonic()
                return

    def _fsm_step(self, confirmed_tags, detections, red_line, base_left, base_right, frame_rgb):
        # Expire stale sign
        if (
            self._saved_tag is not None
            and self._saved_tag_time is not None
        ):
            age = time.monotonic() - self._saved_tag_time

            if age > self._saved_tag_timeout:
                print(
                    f"[SignBehavior] sign expired "
                    f"(tag={self._saved_tag}, age={age:.1f}s)"
                )

                self._saved_tag = None
                self._saved_tag_time = None

        # MOVING
        if self.state == State.MOVING:
            self._save_confirmed_sign(confirmed_tags)

            if red_line:
                tag = self._saved_tag

                # Important fix:
                # Red line alone must NOT trigger countdown/intersection behavior.
                # Otherwise false red-line detection causes move-stop-move-stop.
                if tag is None and not getattr(self.cfg, "default_forward_on_red_without_tag", False):
                    if self._ignored_red_print_cooldown <= 0.0:
                        print("[SignBehavior] red line ignored — no saved sign/tag")
                        self._ignored_red_print_cooldown = self._frames_to_seconds(30)

                    return base_left, base_right

                self._red_line_locked = True

                if tag in _TAG_TURNS:
                    self._possible_turns = list(_TAG_TURNS[tag])
                    self._chosen_turn = self._pick_turn()
                    self._active_sign_op = f"intersection turn '{self._chosen_turn}'"
                    self._saved_tag = None
                    self._saved_tag_time = None
                    self.state = State.APPROACHING
                    self._approach_start_time = time.monotonic()

                    print(
                        "[SignBehavior] red line reached -> APPROACHING"
                    )

                    return base_left, base_right

                if tag == TagID.STOP or tag == TagID.YIELD:
                    label = "STOP" if tag == TagID.STOP else "YIELD"
                    self._active_sign_op = f"{label} sign"
                    self._slow_target = 0.0
                    self._set_state(State.SLOWING)

                    print(
                        f"[SignBehavior] >>> {label} — red line close, braking to full stop "
                        f"(saved_tag={tag})"
                    )

                    return base_left, base_right

                # Optional fallback, normally disabled by config.
                self._chosen_turn = "forward"
                self._active_sign_op = "intersection turn 'forward' without tag"
                self._saved_tag = None
                self._saved_tag_time = None
                self._set_state(State.INTERSECT)

                print("[SignBehavior] >>> INTERSECT — default forward")
                return base_left, base_right

            # Sign detected but red line not close yet: slow approach.
            if self._saved_tag is not None:
                return base_left, base_right

            self._slow_factor = 1.0
            return base_left, base_right

        # SLOWING
        if self.state == State.SLOWING:
            dt_frames = max(0.0, self._dt * self.FPS)

            slow_ramp_factor = self._safe_float(
                getattr(self.cfg, "slow_ramp_factor", 0.84),
                0.84,
            )

            stopped_speed_threshold = self._safe_float(
                getattr(self.cfg, "stopped_speed_threshold", 0.06),
                0.06,
            )

            self._slow_factor *= slow_ramp_factor ** dt_frames

            factor = max(self._slow_factor, self._slow_target)

            left = base_left * factor
            right = base_right * factor

            if self._slow_factor < stopped_speed_threshold:
                self._slow_factor = 0.0
                self._saved_tag = None
                self._saved_tag_time = None
                self._set_state(State.STOPPED)
                print("[SignBehavior] >>> STOPPED at red line")
                return 0.0, 0.0

            return left, right
        

        # STOPPED
        if self.state == State.STOPPED:
            self._hold_counter = self._elapsed_frames()

            if self._hold_counter < self._frames("stop_hold_frames", 0):
                return 0.0, 0.0

            self._vehicle_seen_left = False
            self._vehicle_seen_right = False
            self._left_offsets.clear()
            self._right_offsets.clear()

            self._set_state(State.CHECKPATH)
            print("[SignBehavior] >>> CHECKPATH — checking for vehicles")
            return 0.0, 0.0
        
        if self.state == State.APPROACHING:

            # _approach_start_time may be None in some code paths; guard against that
            if self._approach_start_time is None:
                self._approach_start_time = time.monotonic()

            elapsed = time.monotonic() - self._approach_start_time

            if elapsed < self.cfg.approach_duration:

                s = self.cfg.approach_speed

                return (
                    s,
                    s
                )

            print(
                "[SignBehavior] approach complete"
            )


            self._set_state(State.INTERSECT)

            return base_left, base_right

        # CHECKPATH
        if self.state == State.CHECKPATH:
            return self._checkpath_step(detections)

        # POST_STOP
        if self.state == State.POST_STOP:
            self._turn_counter = self._elapsed_frames()

            if self._turn_counter < self._frames("post_stop_frames", 25):
                return self._speed("post_stop_speed", 0.20), self._speed("post_stop_speed", 0.20)

            self._finish_behavior()
            return base_left, base_right

        # INTERSECT
        if self.state == State.INTERSECT:
            return self._intersect_step(base_left, base_right)

        # PRE_TURN
        if self.state == State.PRE_TURN:
            return self._preturn_step(base_left, base_right)

        # TURNING
        if self.state == State.TURNING:
            return self._turning_step(base_left, base_right)

        # EXITING
        if self.state == State.EXITING:
            return self._exiting_step(base_left, base_right)

        print("servoing")
        return base_left, base_right


    def _checkpath_step(self, detections):
        spd = self._speed("check_turn_speed", 0.04)
        cl = self._frames("check_left_frames", 10)
        cr = self._frames("check_right_frames", 4)
        cs = self._frames("check_settle_frames", 5)

        c = self._elapsed_frames()
        self._check_counter = c

        phase_a_end = cl
        phase_a_settle = cl + cs
        phase_b_end = phase_a_settle + cl + cr
        phase_b_settle = phase_b_end + cs
        phase_c_end = phase_b_settle + cl

        def is_stationary(offsets, threshold=0.05):
            if len(offsets) < 2:
                return True
            return (max(offsets) - min(offsets)) < threshold

        # Look left
        if c < phase_a_end:
            return -spd, spd

        # Hold left and detect
        if c < phase_a_settle:
            if c > phase_a_end + 1:
                seen, offset = self._vehicle_detected(detections)
                if seen:
                    self._vehicle_seen_left = True
                    self._left_offsets.append(offset)

            return 0.0, 0.0

        # Sweep to right
        if c < phase_b_end:
            return spd, -spd

        # Hold right and detect
        if c < phase_b_settle:
            if c > phase_b_end + 1:
                seen, offset = self._vehicle_detected(detections)
                if seen:
                    self._vehicle_seen_right = True
                    self._right_offsets.append(offset)

            return 0.0, 0.0

        # Re-center
        if c < phase_c_end + 10:
            return -spd, spd

        # Decision
        right_stationary = is_stationary(self._right_offsets)

        # Project rule:
        # Vehicle on right => wait.
        # Vehicle on left => we have priority.
        if self._vehicle_seen_right:
            if right_stationary:
                print("[SignBehavior] vehicle on right — yielding")
            else:
                print("[SignBehavior] moving vehicle on right — yielding")

            self._restart_current_state_timer()
            self._vehicle_seen_left = False
            self._vehicle_seen_right = False
            self._left_offsets.clear()
            self._right_offsets.clear()
            return 0.0, 0.0

        if self._vehicle_seen_left:
            print("[SignBehavior] vehicle on left — priority, continuing")

        self._vehicle_seen_left = False
        self._vehicle_seen_right = False
        self._left_offsets.clear()
        self._right_offsets.clear()

        self._set_state(State.POST_STOP)

        print("[SignBehavior] >>> path clear — POST_STOP")
        return 0.0, 0.0


    def _intersect_step(self, base_left, base_right):
        turn = self._chosen_turn or "forward"

        if turn in ("left", "right"):
            self._set_state(State.PRE_TURN)
            print(f"[SignBehavior] >>> PRE_TURN for '{turn}'")
            return self._speed("preturn_speed", 0.20), self._speed("preturn_speed", 0.20)

        self._set_state(State.TURNING)
        print("[SignBehavior] >>> TURNING 'forward'")
        return self._pair("intersect_forward_speed", (0.20, 0.20))

    def _preturn_step(self, base_left, base_right):
        turn = self._chosen_turn or "forward"

        if turn == "right":
            total = self._frames("preturn_right_frames", 11)
        else:
            total = self._frames("preturn_left_frames", 20)

        self._turn_counter = self._elapsed_frames()

        if self._turn_counter < total:
            spd = self._speed("preturn_speed", 0.20)
            return spd, spd

        self._set_state(State.TURNING)

        print(f"[SignBehavior] >>> PRE_TURN done — TURNING '{turn}'")
        return 0.0, 0.0

    def _turning_step(self, base_left, base_right):
        turn = self._chosen_turn or "forward"

        speeds_map = {
            "forward": self._pair("intersect_forward_speed", (0.20, 0.20)),
            "left": self._pair("intersect_left_speed", (0.15, 0.22)),
            "right": self._pair("intersect_right_speed", (0.22, 0.15)),
        }

        frames_map = {
            "forward": self._frames("intersect_forward_frames", 28),
            "left": self._frames("intersect_left_frames", 18),
            "right": self._frames("intersect_right_frames", 18),
        }

        total = frames_map.get(turn, self._frames("intersect_forward_frames", 28))

        self._turn_counter = self._elapsed_frames()

        if self._turn_counter < total:
            return speeds_map.get(turn, self._pair("intersect_forward_speed", (0.20, 0.20)))

        self._set_state(State.EXITING)

        print(f"[SignBehavior] >>> TURNING '{turn}' done — EXITING")
        return base_left, base_right

    def _exiting_step(self, base_left, base_right):
        self._turn_counter = self._elapsed_frames()

        if self._turn_counter >= self._frames("exit_timeout_frames", 5):
            self._finish_behavior()
            return base_left, base_right

        spd = self._speed("exit_speed", 0.20)
        return spd, spd

    def _finish_behavior(self):
        print(f"[SignBehavior] COMPLETE: {self._active_sign_op} — robot resumed driving")

        self._red_line_locked = False
        self._ignore_red_counter = self._frames_to_seconds(
            getattr(self.cfg, "red_ignore_after_frames", 24)
        )

        self._possible_turns = []
        self._chosen_turn = None
        self._saved_tag = None
        self._saved_tag_time = None
        self._active_sign_op = None

        self._slow_factor = 1.0
        self._hold_counter = 0.0
        self._turn_counter = 0.0
        self._check_counter = 0.0

        self._set_state(State.MOVING)

    def _pick_turn(self):
        return random.choice(self._possible_turns) if self._possible_turns else "forward"