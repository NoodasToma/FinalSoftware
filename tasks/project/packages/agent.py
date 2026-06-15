import os
import random
import socket
import threading
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

# A synthetic "duckie fills the frame" detection. While the obstacle debounce
# (see main loop) says an obstacle is present, the decision logic is fed THIS so
# should_stop_for_obstacle() returns True on the frames where YOLO didn't run.
_OBSTACLE_BLOCK = [((0, 0, 1, 1_000_000), 1.0, 0)]


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


# Single lock serialising ALL I2C traffic from this process. The motor HAT and
# the LED board share ONE physical I2C bus, and the agent touches it from TWO
# threads — the main control loop AND the camera watchdog (which zeros the wheels
# on a stall). smbus2 transactions are not atomic across threads, so concurrent
# access corrupts a transfer -> OSError(121) 'Remote I/O error', which crashed
# the agent thread mid-drive. Every wheel/LED write goes through this lock, and
# each is wrapped so a single transient bus glitch is survived, not fatal.
_IO_LOCK = threading.Lock()


def _set_brake(leds, on: bool) -> None:
    if leds is None:
        return
    color = _RED if on else _OFF
    try:
        with _IO_LOCK:
            leds.set_rgb(_LED_BACK_LEFT,  color)
            leds.set_rgb(_LED_BACK_RIGHT, color)
    except Exception:
        pass


def _set_blinker(leds, direction: Optional[str], t_now: float = 0.0) -> None:
    """direction: 'left' | 'right' | 'off' | None. 2 Hz yellow blink when set."""
    if leds is None:
        return
    try:
        with _IO_LOCK:
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
    except Exception:
        pass


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


def _wait_timed_out(intersection_since, now, timings, obstacles, frame_h) -> bool:
    """Failsafe to leave a WAIT that never resolves (a light that never turns green
    or drops out of view while latched red, a give-way that never clears): True
    once we've been stopped at the junction longer than light_wait_timeout_s AND
    nothing physically blocks the box — an obstacle ALWAYS overrides the timer, so
    the bot never drives into a duckie just because time elapsed."""
    if intersection_since is None:
        return False
    if (now - intersection_since) <= timings.get('light_wait_timeout_s', 12.0):
        return False
    return not should_stop_for_obstacle(obstacles, frame_h)[0]


def main(camera, wheels, leds, stop_event, *, observer=None, frame_observer=None,
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
    timings = timings_override if timings_override is not None else _load_timings()
    hsv_cfg = _load_hsv_cfg()

    lane  = LaneServoingAgent(config_path=lane_config_path)
    light = TrafficLightDetector()
    tags  = AprilTagDetector(
        tag_size_m=apriltag_tag_size or 0.065,
        intrinsics=apriltag_intrinsics,
    )
    # Build the YOLO object detector ONLY if it will actually run. With
    # yolo_every=0 (the bot default — the Nano can't run YOLO alongside real-time
    # control) obj.detect() is never called, yet constructing it still loads the
    # ONNX model (hundreds of MB) into the 4 GB Nano. Skipping that unused load
    # cuts the memory pressure that aggravates camera stalls / I2C glitches.
    obj = ObjectDetectionAgent() if int(timings.get('yolo_every', 0)) > 0 else None

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
    obstacle_last_seen = -1e9   # wall-clock of the last positive obstacle detection (debounce)

    # SINGLE-THREADED inline loop (like the working reference). Earlier I ran the
    # camera read and the detectors in BACKGROUND threads; the AprilTag/YOLO
    # C-extensions hold Python's GIL for long stretches, which starved the camera
    # read -> the nvargus pipeline stalled -> the image froze and the bot stopped.
    # So everything runs in THIS one thread: read the camera + run lane EVERY
    # frame (smooth steering), and run the heavy detectors (apriltags, YOLO) only
    # every Nth frame to keep the average loop rate up. No cross-thread GIL fight.
    _detectors_enabled = bool(timings.get('detectors_enabled', True))
    _detect_every = max(1, int(timings.get('detect_every', 3)))   # apriltags every Nth frame
    _yolo_every   = int(timings.get('yolo_every', 0))             # YOLO every Nth detect pass (0=off)
    tag_obs = []          # last apriltag detections (persist between detect frames)
    raw_dets = None       # last YOLO detections
    frame_count = 0
    intersection_since: Optional[float] = None  # first STOPPED at this junction (wait-timeout anchor)

    # CAMERA WATCHDOG. On the Jetson, cv2.VideoCapture.read() BLOCKS FOREVER when
    # nvargus stalls -> a single-threaded loop hard-freezes (the failure we kept
    # hitting). This watchdog thread mostly sleeps (no GIL contention) and, if no
    # good frame has arrived for a few seconds, rebuilds the camera: camera.stop()
    # releases the device (which UNBLOCKS a stuck read()), camera.start() brings it
    # back -> the bot recovers from a stall on its own, no reboot. `restarting`
    # tells the main loop to keep its hands off the camera during the rebuild.
    # 'seen' gates the watchdog until the FIRST frame has ever arrived: at startup
    # the first read() can lag, and calling camera.stop()/release() from this
    # thread while the main loop is still inside that first read() wedges both
    # threads (cv2.VideoCapture is not safe to release mid-read) -> "freezes at
    # first frames". 'maneuvering' is set while a blocking turn runs, because the
    # loop then intentionally stops reading the camera for a few seconds and that
    # must NOT be mistaken for a camera stall.
    _cam_state = {'last_good': time.time(), 'restarting': False,
                  'seen': False, 'maneuvering': False, 'alive': True}

    # Throttled error logger. A single bad frame / transient I2C glitch must NOT
    # crash the loop (a daemon thread that dies just leaves the bot frozen with no
    # visible cause), but a persistent error must still be SEEN — so print the full
    # traceback at most once every 2 s and carry on with the previous value.
    import traceback as _tb
    _last_err_print = [0.0]

    def _log_perception_error(where):
        nowp = time.time()
        if nowp - _last_err_print[0] > 2.0:
            _last_err_print[0] = nowp
            print(f"[agent] {where} error (continuing):")
            _tb.print_exc()

    def _safe_set_wheels(left, right):
        """The ONLY way the agent writes wheel speeds: serialised with the LED
        writes on the shared I2C bus (so the watchdog thread and the main loop
        never transact at once) and tolerant of a transient bus glitch
        (OSError 121 'Remote I/O error') so one bad transaction is logged and
        skipped instead of crashing the whole agent thread."""
        try:
            with _IO_LOCK:
                wheels.set_wheels_speed(left, right)
        except Exception:
            _log_perception_error('wheels.set_wheels_speed')

    def _cam_watchdog():
        while not stop_event.is_set() and _cam_state['alive']:
            time.sleep(0.5)
            # In these the loop legitimately isn't reading the camera, so a stale
            # last_good does NOT mean the pipeline died — don't touch it.
            if (_cam_state['restarting'] or _cam_state['maneuvering']
                    or not _cam_state['seen']):
                continue
            gap = time.time() - _cam_state['last_good']
            # 4 s, not 2 s: a slow first detector pass (AprilTag / ONNX warm-up) can
            # stall the loop ~2 s while the camera is perfectly fine; a false rebuild
            # then fights the live camera AND (before the I2C lock) raced the motor
            # bus -> OSError(121) crash. Only a genuine multi-second stall trips it.
            if gap > 4.0:
                print("[watchdog] no camera frame for %.1fs - rebuilding pipeline..." % gap)
                _cam_state['restarting'] = True
                _safe_set_wheels(0.0, 0.0)
                try:
                    camera.stop()        # releases device -> unblocks a stuck read()
                except Exception:
                    pass
                time.sleep(0.4)
                try:
                    camera.start()       # rebuild the pipeline
                    print("[watchdog] camera restarted OK")
                    _cam_state['last_good'] = time.time()
                except Exception as e:
                    print("[watchdog] restart failed (%s) - retrying; reboot the "
                          "bot if this persists (nvargus wedged)." % e)
                    _cam_state['last_good'] = time.time() - 1.0   # retry in ~1.5s
                _cam_state['restarting'] = False

    # The camera watchdog is OFF by default. It adds a SECOND thread that writes
    # the wheels (I2C), and concurrent I2C with the main loop is exactly what
    # crashed the agent (OSError 121 'Remote I/O error'). The proven reference
    # design has no watchdog and a single I2C writer, so it never hit this. The
    # "camera.read() blocks forever" case the watchdog was meant to cover was
    # really the two-reader hang (agent + /video), already fixed by frame_observer.
    # Re-enable ONLY if a genuine mid-run camera stall is ever observed, via
    # `camera_watchdog: true` in maneuver_timings_bot.yaml — the I2C lock above
    # then keeps the second writer bus-safe.
    if bool(timings.get('camera_watchdog', False)):
        print("[agent] camera watchdog ENABLED (camera_watchdog: true in timings)")
        threading.Thread(target=_cam_watchdog, daemon=True, name='CamWatchdog').start()
    else:
        print("[agent] camera watchdog OFF (single-thread I2C; matches reference design)")

    try:
        while not stop_event.is_set():
            if _cam_state['restarting']:
                time.sleep(0.05)
                continue
            ok, bgr = camera.read()
            if not ok or bgr is None:
                # No frame right now. Don't drive blind; the watchdog rebuilds the
                # camera if this persists.
                _safe_set_wheels(0.0, 0.0)
                time.sleep(0.03)
                continue
            _cam_state['last_good'] = time.time()
            _cam_state['seen'] = True
            # Hand the frame to the server for the /video HUD.
            if frame_observer is not None:
                try:
                    frame_observer(bgr)
                except Exception:
                    pass
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_h = bgr.shape[0]
            now = time.time()
            frame_count += 1

            # Heavy detectors, throttled, INLINE (single thread -> no GIL fight
            # with the camera read). tag_obs/raw_dets persist between detect frames.
            if _detectors_enabled and (frame_count % _detect_every == 0):
                try:
                    tag_obs = tags.detect(bgr) or []
                except Exception:
                    pass
                if obj is not None and _yolo_every > 0 and (frame_count % (_detect_every * _yolo_every) == 0):
                    try:
                        d = obj.detect(rgb)
                        if d is not None:
                            raw_dets = d
                    except Exception:
                        pass

            signs = [(o, lookup(o.id)) for o in tag_obs if lookup(o.id) is not None]
            # Obstacle debounce. YOLO runs only every Nth frame (obj.detect returns
            # None in between) and a close duckie can dip below the down-tilted
            # camera; treating those blank frames as "clear" made SOFT_STOP flip
            # back to DRIVE every other frame, so the bot crept straight INTO the
            # duckie. Instead: remember the last frame a duckie was actually seen,
            # and consider one present for obstacle_clear_grace_s afterwards — so we
            # resume only once it has truly been gone for the whole grace window.
            if raw_dets is not None and should_stop_for_obstacle(raw_dets, frame_h)[0]:
                obstacle_last_seen = now
            obstacle_present = (now - obstacle_last_seen) < timings.get('obstacle_clear_grace_s', 1.0)
            obstacles = _OBSTACLE_BLOCK if obstacle_present else []
            try:
                light_color = light.detect(bgr)
            except Exception:
                _log_perception_error('light.detect')
                light_color = None

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
            try:
                red_line = detect_red_line(bgr, hsv_cfg)
            except Exception:
                _log_perception_error('detect_red_line')
                red_line = False

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
                        intersection_since = now   # first arrival; anchors the wait timeout
                    stopped_at = now      # full-stop pause starts now
                state = new_state

            if state == State.DRIVE:
                left, right = lane.compute_commands(rgb)
                current_speed = ramp_speed(
                    current_speed, timings['base_speed'], timings['ramp_max_step']
                )
                _safe_set_wheels(
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
                    _safe_set_wheels(current_speed, current_speed)
                else:
                    left, right = lane.compute_commands(rgb)
                    _safe_set_wheels(
                        left  * current_speed * 2,
                        right * current_speed * 2,
                    )
                _set_brake(leds, on=True)

            elif state == State.STOPPED:
                _safe_set_wheels(0.0, 0.0)
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
                    # Failsafe: if we've been stopped at this junction longer than
                    # light_wait_timeout_s (a light that never goes green / drops out
                    # of view while latched red, a give-way that never resolves),
                    # proceed anyway so the bot can't be stranded forever. Anchored
                    # to FIRST arrival (intersection_since), so the STOPPED<->WAIT
                    # flicker can't keep resetting it. An obstacle in the box STILL
                    # blocks — we never drive into a duckie just because time passed.
                    timed_out = _wait_timed_out(intersection_since, now, timings,
                                                obstacles, frame_h)
                    if (clear or timed_out) and legal_turns:
                        choice = random.choice(sorted(legal_turns))
                        state = next_state(state, {
                            'left':     'choose_turn_left',
                            'right':    'choose_turn_right',
                            'straight': 'choose_straight',
                        }[choice])
                    elif not clear and not timed_out:
                        state = next_state(state, 'wait')

            elif state == State.WAIT:
                _safe_set_wheels(0.0, 0.0)
                _set_brake(leds, on=True)
                if (_clear_to_enter(light_was_red, light_color, yellow_started_at,
                                    now, my_name, recent_vehicle_signs, obstacles, frame_h)
                        or _wait_timed_out(intersection_since, now, timings, obstacles, frame_h)):
                    state = next_state(state, 'cleared')
                    # We already held at the line while waiting (red light etc.) —
                    # don't add ANOTHER full-stop pause on top; STOPPED then proceeds.
                    stopped_at = now - timings.get('stop_wait_seconds', 1.5)

            elif state in (State.TURN_LEFT, State.TURN_RIGHT, State.STRAIGHT_THROUGH):
                _set_blinker(leds, _direction_of(state), now)
                # The turn BLOCKS this loop (no camera reads) for a few seconds by
                # design; flag it so the watchdog doesn't read the resulting stale
                # last_good as a camera stall and abort the turn mid-intersection.
                _cam_state['maneuvering'] = True
                try:
                    {
                        State.TURN_LEFT:        turn_left,
                        State.TURN_RIGHT:       turn_right,
                        State.STRAIGHT_THROUGH: straight_through,
                    }[state](wheels, stop_event, timings)
                except Exception:
                    # A transient I2C glitch mid-turn aborts THIS maneuver, but must
                    # not crash the agent thread; next_state below still advances.
                    _log_perception_error('maneuver')
                finally:
                    _cam_state['maneuvering'] = False
                    _cam_state['last_good'] = time.time()
                signs_at_intersection = []
                intersection_signs = {}
                light_was_red = False
                current_speed = 0.0
                intersection_since = None
                # Cooldown: drive clear of the intersection before signs can fire
                # again, so we don't immediately re-stop on the sign we just obeyed.
                ignore_signs_until = time.time() + timings.get('sign_cooldown', 4.0)
                state = next_state(state, 'turn_done')

            elif state == State.SOFT_STOP:
                # Hold until the obstacle clears (a duckie is a pedestrian:
                # driving through it is a collision on hardware, and game-over
                # in sim). The SOFT_STOP -> 'obstacle_cleared' event above
                # resumes us automatically once the path is clear.
                _safe_set_wheels(0.0, 0.0)
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
                    'obstacles': [
                        {'bbox': [int(v) for v in bbox],
                         'cls': int(cls), 'score': round(float(score), 2)}
                        for bbox, score, cls in (obstacles or [])
                    ],
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
        # Loop ending (shutdown OR a crash that escaped the guards): stop the
        # watchdog so it can't zombie-rebuild the camera forever, and leave the bot
        # safe — wheels stopped, LEDs off — through the same I2C lock.
        _cam_state['alive'] = False
        _safe_set_wheels(0.0, 0.0)
        if leds is not None:
            try:
                with _IO_LOCK:
                    leds.all_off()
            except Exception:
                pass
