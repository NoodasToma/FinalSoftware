import os
import queue
import random
import socket
import sys
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
    red_line_pixel_count,
    red_line_threshold,
    red_band_metrics,
)
from tasks.project.packages.perception.duck_hsv import detect_duckies_hsv, detect_vehicles_hsv
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
# Junction signs that, like a stop/yield sign, mean "an intersection is ahead":
# approach, stop at the red line, then take a legal turn. WHICH turns are legal
# comes from the sign itself via merge_turn_constraints (e.g. right-T-intersect ->
# {straight, right}). Previously only stop/yield/light signs triggered the
# APPROACH->STOPPED->TURN flow, so at a junction marked ONLY by a T/4-way sign the
# bot drove straight through without stopping or turning. (oneway-*/do-not-enter
# are turn DIRECTIVES, not junction-stop markers: they constrain the chosen turn
# when present but don't by themselves trigger the stop.)
_INTERSECTION_KINDS = {'4-way-intersect', 'T-intersection',
                       'right-T-intersect', 'left-T-intersect'}


def _triggers_approach(kind) -> bool:
    """A sign whose presence means we should run the intersection flow
    (APPROACH -> stop at the line -> turn): a stop/yield sign, a junction sign,
    or a traffic light ahead."""
    return (kind in _STOP_OR_YIELD or kind in _INTERSECTION_KINDS
            or kind == _T_LIGHT_AHEAD)

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
                  at_red_line=False, react_distance_m=1.5,
                  react_min_px=0, stop_min_px=0, red_band=False) -> Optional[str]:
    if state in (State.DRIVE, State.APPROACH):
        stop, _ = should_stop_for_obstacle(obstacles, frame_h)
        if stop:
            return 'obstacle'

    if state == State.DRIVE:
        # ignore_signs is the post-intersection cooldown: after we stop+turn we
        # are still right next to the sign we just obeyed, so without this we'd
        # re-trigger on it and stop/turn over and over (a 360 spin in place).
        if not ignore_signs:
            # React only to signs governing the junction we're ARRIVING at.
            # Tags can decode from 3+ m away, and starting the slow APPROACH
            # creep that early both crawls forever and reacts to intersections
            # beyond the next one.
            #   * Calibrated camera (metric est_distance): gate on react_distance_m.
            #   * Uncalibrated camera (est_distance == inf, the bot's default
            #     mode): there is NO metric distance, so gate on the tag's
            #     apparent SIZE instead — a far sign is a tiny tag; only once it
            #     grows past react_min_px is it close enough to act on. Without
            #     this the bot reacted to a 0.065 m tag the instant it decoded
            #     (metres away) and stopped/turned far short of the line.
            #     react_min_px <= 0 restores the old react-on-sight behaviour.
            def near(o):
                if o is None:          # offline/logic tests pass bare semantics
                    return True
                if o.est_distance_m != float('inf'):
                    return o.est_distance_m < react_distance_m
                return react_min_px <= 0 or o.side_length_px >= react_min_px
            kinds = {sem.kind for o, sem in signs if near(o)}
            if kinds & _STOP_OR_YIELD:
                return 'see_stop_or_yield'
            if _T_LIGHT_AHEAD in kinds:
                # A traffic-light intersection also makes us approach + stop at the
                # line, where STOPPED waits for green. (Previously only stop/yield
                # signs triggered APPROACH, so the bot drove straight through lights.)
                return 'see_light'
            if kinds & _INTERSECTION_KINDS:
                # A junction sign (T / 4-way) with no stop/yield/light still marks an
                # intersection: approach, stop at the red line, then take a legal turn
                # (the sign supplies which turns are legal in STOPPED).
                return 'see_intersection'
        return None

    if state == State.APPROACH:
        # Apparent size of the closest relevant (stop/junction/light) sign in view.
        vis_px = max((o.side_length_px for o, sem in signs
                      if _triggers_approach(sem.kind)), default=0)
        if stop_min_px > 0:
            # UNCALIBRATED PROXIMITY MODE (the bot's default — no camera
            # intrinsics). Two ways to know we're AT the line, whichever first:
            #  1) the ROBUST red BAND (a wide horizontal red bar across the lane)
            #     once the sign is near enough to be THIS junction's (react gate).
            #     This is the preferred trigger — it stops the bot ON the painted
            #     line. The raw red-pixel count is NOT used (the venue's red/orange
            #     markings make it spike far from any line); the band's structure
            #     test rejects that noise.
            #  2) FAILSAFE: the tag grew past stop_min_px (very close) but no band
            #     was seen — stop anyway so a washed-out/!missing line can't make
            #     the bot drive through the intersection.
            if vis_px >= react_min_px and red_band:
                return 'at_stop_line'
            if vis_px >= stop_min_px:
                return 'at_stop_line'
            return None
        # CALIBRATED / CLEAN-LINE MODE (the sim, or a bot with intrinsics +
        # reliable red line): the painted red line is the PRIMARY trigger (we are
        # physically AT the line), with the metric tag-distance proxy as backup.
        # Accept either the raw count or the structural band (a clean sim line
        # satisfies both).
        if at_red_line or red_band:
            return 'at_stop_line'
        for obs, sem in signs:
            if _triggers_approach(sem.kind):
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
    # Per-venue red overrides (band thresholds + tighter red) live in the bot
    # timings so the shared traffic_light_hsv.yaml (sim) stays untouched.
    hsv_cfg.update(timings.get('red_line_cfg') or {})
    _red_line_thr = red_line_threshold(hsv_cfg)
    # Uncalibrated-camera react gate: minimum tag side length (px) for a sign to
    # count as "close enough to act on" when there is no metric distance. 0 keeps
    # the legacy react-on-sight behaviour. Tuned on the bot (see BOT_BEHAVIOR §1b).
    _sign_react_min_px = float(timings.get('sign_react_min_px', 0))
    # Uncalibrated-camera STOP gate: tag side length (px) at which the bot is AT
    # the stop line and should stop. >0 switches APPROACH->STOPPED to proximity
    # (tag size) instead of the red-line detector, which is unreliable in venues
    # with red/orange road markings. 0 keeps the red-line/metric stop path.
    _sign_stop_min_px = float(timings.get('sign_stop_px', 0))
    # Uncalibrated-camera STRAIGHT-CREEP gate: once the relevant sign's tag is at
    # least this many px (i.e. we're near the junction), creep DEAD STRAIGHT to the
    # line instead of lane-following — the lane markings curve away at the junction
    # box and following them drifts the bot off the sign (and the tag then never
    # grows to the stop size, so it never stops). The metric equivalent is
    # line_straight_distance_m, which is dead on an inf-distance camera. 0 disables.
    _line_straight_px = float(timings.get('line_straight_px', 0))

    lane  = LaneServoingAgent(config_path=lane_config_path)
    light = TrafficLightDetector()
    tags  = AprilTagDetector(
        tag_size_m=apriltag_tag_size or 0.065,
        intrinsics=apriltag_intrinsics,
    )
    # Flushed so it actually lands in the bot's task log (the agent runs on a
    # thread whose default block-buffered stdout otherwise hides these prints).
    print("[agent] AprilTag backend=%r (None => sign/light/other-bot-tag detection OFF)"
          % tags._backend, flush=True)
    # Object (duckie) detector. It runs in a DECOUPLED background thread (see
    # _detection_worker below), NOT inline — that is what makes it safe on the
    # Nano now (the old inline obj.detect() blocked the control loop ~0.3 s/frame
    # and froze it). The model is small (~7 MB ONNX) and on the bot real_server
    # forces CPU inference (OBJDET_CPU=1), so there is no TensorRT-build OOM.
    # Disable entirely with `object_detection: false` in the timings.
    _obj_enabled = bool(timings.get('object_detection', True))
    obj = ObjectDetectionAgent() if _obj_enabled else None
    # Pick the duckie-detection backend. The trained model is best, but it needs
    # onnxruntime or a recent OpenCV — the real bot has neither (OpenCV 4.1.1's
    # cv2.dnn can't parse the YOLOv5 ONNX). So when the model can't load we fall
    # back to the dependency-free HSV+shape detector (perception/duck_hsv.py),
    # which runs on the bot's stock OpenCV. Both emit [(bbox, score, 0)] in
    # full-frame pixels, so the rest of the obstacle/SOFT_STOP path is identical.
    if obj is not None and obj.model_loaded:
        _duck_mode = 'model'
    elif _obj_enabled and bool(timings.get('duck_hsv', True)):
        _duck_mode = 'hsv'
    else:
        _duck_mode = 'off'
    _duck_hsv_cfg = timings.get('duck_hsv_cfg') or None
    # Other-bot detection by colour (the local bots are blue, with no AprilTag
    # plate, so the tag path can't see them). Runs inline alongside the duck
    # detector; cheap OpenCV. Disable with bot_hsv: false.
    _bot_hsv = bool(timings.get('bot_hsv', True))
    _bot_hsv_cfg = timings.get('bot_hsv_cfg') or None

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
    approach_max_px = 0               # biggest a stop/yield/light tag got this APPROACH (px proxy)
    relevant_lost_at: Optional[float] = None  # when the approached sign left the frame
    stopped_at: Optional[float] = None  # when we entered STOPPED (for the full-stop pause)
    obstacle_last_seen = -1e9   # wall-clock of the last positive obstacle detection (debounce)

    # The CONTROL path is single-threaded: this loop reads the camera (single
    # nvargus reader), runs lane EVERY frame (smooth steering) and the AprilTag
    # detector every Nth frame (cheap C, fast). The ONLY work that is off-loaded
    # to another thread is the heavy duckie model (see _detection_worker): it ran
    # ~0.3 s/frame on the Nano CPU and, inline, dropped the loop rate until the
    # camera pipeline stalled (a freeze). The worker is safe because (a) it never
    # touches I2C — the main loop stays the lone wheel/LED writer, so it can't
    # recreate the OSError(121) bus crash; (b) onnxruntime/cv2.dnn RELEASE the GIL
    # during inference, so a slow detection does not block this loop's camera
    # read / steering; (c) the camera is still read by THIS thread only — the
    # worker consumes already-captured frames from a queue, never the device.
    _detectors_enabled = bool(timings.get('detectors_enabled', True))
    _detect_every = max(1, int(timings.get('detect_every', 3)))   # apriltags + duck-enqueue every Nth frame
    # Other-bot soft-stop thresholds: a Vehicle AprilTag (another Duckiebot)
    # that is close and roughly centred ahead. Calibration-free — est_distance_m
    # is inf without camera intrinsics, so we gate primarily on apparent tag size
    # (a near bot's plate is big) plus lateral position. Tune on the bot.
    _bot_tag_min_px     = float(timings.get('bot_stop_tag_px', 55))
    _bot_center_frac    = float(timings.get('bot_stop_center_frac', 0.33))
    _bot_stop_dist_m    = float(timings.get('bot_stop_distance_m', 0.45))
    tag_obs = []          # last apriltag detections (persist between detect frames)
    raw_dets = None       # last duckie detections (from the worker; persists between updates)
    veh_dets = []         # last other-bot (blue) detections; persists between detect frames
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
            print(f"[agent] {where} error (continuing):", flush=True)
            _tb.print_exc(file=sys.stdout)
            sys.stdout.flush()

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

    # DECOUPLED duckie detector (the reference's proven real-hardware pattern).
    # The main loop hands the latest frame to _det_q (non-blocking, maxsize=1 so a
    # backlog can't build); this worker pulls it, runs the model, and publishes the
    # result to _det_store. The control loop only READS _det_store — it never calls
    # detect(), so heavy inference never blocks steering, and this thread issues no
    # I2C (the watchdog's I2C writes are what crashed us before; this one has none).
    _det_q: "queue.Queue" = queue.Queue(maxsize=1)
    _det_store = {'dets': None}
    _det_lock = threading.Lock()

    def _detection_worker():
        while not stop_event.is_set() and _cam_state['alive']:
            try:
                frame_rgb = _det_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                d = obj.detect(frame_rgb)
            except Exception:
                _log_perception_error('obj.detect')
                continue
            if d is not None:           # None = internally skipped frame
                with _det_lock:
                    _det_store['dets'] = d

    def _bot_ahead(signs, frame_w) -> bool:
        """True if another Duckiebot (a Vehicle AprilTag) is close and roughly
        centred ahead — i.e. directly in our path, so we soft-stop behind it."""
        half = frame_w / 2.0
        for o, sem in signs:
            if sem.tag_type != 'Vehicle':
                continue
            if abs(o.center_xy[0] - half) > frame_w * _bot_center_frac:
                continue   # off to the side, not in our lane
            if o.side_length_px >= _bot_tag_min_px or o.est_distance_m < _bot_stop_dist_m:
                return True
        return False

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

    # Start the duckie detector worker only if the model actually loaded. If it
    # didn't (no onnxruntime/cv2.dnn, missing model), the bot still lane-follows
    # and obeys signs; it just won't brake for duckies — no crash, no freeze.
    if _duck_mode == 'model':
        print("[agent] duckie detection: decoupled model thread (backend=%s)" % obj._backend)
        threading.Thread(target=_detection_worker, daemon=True, name='DuckDetect').start()
    elif _duck_mode == 'hsv':
        err = f" (model unavailable: {obj.load_error})" if obj is not None else ""
        print("[agent] duckie detection: inline HSV+shape, no model on this platform" + err)
    else:
        print("[agent] duckie detection OFF; other-bot stop via Vehicle tags still active")

    # Per-iteration callback used DURING a blocking maneuver (turn/straight) so the
    # bot keeps SEEING instead of driving the intersection blind: read a frame,
    # refresh the /video HUD, run the object detector (throttled), and return True
    # if an obstacle is in the box (the maneuver then HOLDS, not abandons, and
    # resumes when clear). Runs in THIS (main) thread => still the single I2C
    # writer. Never raises (a bad frame must not freeze or crash the turn).
    _tick_st = {'n': 0, 'obstacle': False}

    def _maneuver_tick():
        try:
            ok2, bgr2 = camera.read()
        except Exception:
            return _tick_st['obstacle']
        if not ok2 or bgr2 is None:
            return _tick_st['obstacle']
        _cam_state['last_good'] = time.time()
        if frame_observer is not None:
            try:
                frame_observer(bgr2)
            except Exception:
                pass
        _tick_st['n'] += 1
        if _tick_st['n'] % _detect_every == 0:
            dets, veh = [], []
            if _duck_mode == 'model':
                try:
                    _det_q.put_nowait(cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB))
                except queue.Full:
                    pass
                with _det_lock:
                    dets = _det_store['dets'] or []
            elif _duck_mode == 'hsv':
                try:
                    dets = detect_duckies_hsv(bgr2, _duck_hsv_cfg) or []
                except Exception:
                    dets = []
            if _bot_hsv:
                try:
                    veh = detect_vehicles_hsv(bgr2, _bot_hsv_cfg) or []
                except Exception:
                    veh = []
            duck_stop = bool(dets) and should_stop_for_obstacle(dets, bgr2.shape[0])[0]
            _tick_st['obstacle'] = bool(duck_stop or veh)
            if observer is not None:
                try:
                    observer({
                        'state': state.name,
                        'obstacle_stop': _tick_st['obstacle'],
                        'obstacles': [
                            {'bbox': [int(v) for v in bb], 'cls': int(cl),
                             'score': round(float(sc), 2)}
                            for bb, sc, cl in (list(dets) + list(veh))
                        ],
                    })
                except Exception:
                    pass
        return _tick_st['obstacle']

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

            # AprilTags run INLINE every Nth frame (cheap C, fast); the heavy
            # duckie model runs in the worker thread instead. On the same cadence
            # we hand the latest frame to the worker (non-blocking — drop if it's
            # still busy on the previous one). tag_obs persists between detect frames.
            if _detectors_enabled and (frame_count % _detect_every == 0):
                try:
                    tag_obs = tags.detect(bgr) or []
                except Exception:
                    _log_perception_error('tags.detect')
                if _duck_mode == 'model':
                    try:
                        _det_q.put_nowait(rgb)   # worker resizes + infers off-thread
                    except queue.Full:
                        pass
                elif _duck_mode == 'hsv':
                    try:
                        # Pass the detected yellow lane-line x's so the detector can
                        # reject yellow DASHES (on the line) vs real ducks (off it).
                        # last_debug_info is from the prior frame's lane pass — the
                        # centre line barely moves frame-to-frame, so that's fine.
                        _lane_xs = lane.last_debug_info.get('yellow_xs') or []
                        raw_dets = detect_duckies_hsv(bgr, _duck_hsv_cfg, lane_xs=_lane_xs)
                    except Exception:
                        _log_perception_error('duck_hsv')
                if _bot_hsv:
                    try:
                        veh_dets = detect_vehicles_hsv(bgr, _bot_hsv_cfg)  # blue other-bot, inline
                    except Exception:
                        _log_perception_error('vehicle_hsv')
            # Model path: pull the worker's most recent detections (non-blocking).
            # Stays the previous value between updates; the debounce adds hysteresis.
            if _duck_mode == 'model':
                with _det_lock:
                    _latest_dets = _det_store['dets']
                if _latest_dets is not None:
                    raw_dets = _latest_dets

            signs = [(o, lookup(o.id)) for o in tag_obs if lookup(o.id) is not None]
            # Obstacle debounce. Detection updates asynchronously and a close
            # duckie can dip below the down-tilted camera; treating those blank
            # frames as "clear" made SOFT_STOP flip back to DRIVE every other frame,
            # so the bot crept straight INTO the duckie. Instead: remember the last
            # frame an obstacle was actually seen and consider one present for
            # obstacle_clear_grace_s afterwards — resume only once it has truly been
            # gone for the whole grace window. TWO obstacle sources feed this:
            #   * a duckie from the model (should_stop_for_obstacle), and
            #   * another Duckiebot close + centred ahead (_bot_ahead, via its
            #     Vehicle AprilTag) — "soft-stop when it sees a bot in its path".
            duck_stop = bool(raw_dets) and should_stop_for_obstacle(raw_dets, frame_h)[0]
            bot_stop  = _bot_ahead(signs, bgr.shape[1]) or bool(veh_dets)
            if duck_stop or bot_stop:
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
            # Red stop-line detector (PRIMARY stop trigger on the bot). Compute
            # the raw red-pixel count so it can be surfaced in telemetry for live
            # tuning — on an uncalibrated camera this is how we tell "stopped
            # short of the line" (count never reached the threshold) from a false
            # stop, without re-running the HSV pass.
            try:
                red_px = red_line_pixel_count(bgr, hsv_cfg)
                red_line = red_px >= _red_line_thr
            except Exception:
                _log_perception_error('detect_red_line')
                red_px, red_line = 0, False
            # Robust red-line: a wide HORIZONTAL red BAND across the lane (rejects
            # the venue's scattered red/orange markings that fool the raw count).
            # This is what lets the bot drive all the way TO the painted line on an
            # uncalibrated camera instead of stopping early on red noise.
            try:
                red_band, red_band_diag = red_band_metrics(bgr, hsv_cfg)
            except Exception:
                _log_perception_error('red_band_metrics')
                red_band, red_band_diag = False, {'rows': 0, 'rows_frac': 0.0, 'row_max': 0.0}

            relevant_visible = False
            if state == State.APPROACH:
                for o, sem in signs:
                    intersection_signs[o.id] = sem    # remember every sign this approach
                    if _triggers_approach(sem.kind):
                        relevant_visible = True
                        approach_closest = min(approach_closest, o.est_distance_m)
                        approach_max_px = max(approach_max_px, o.side_length_px)
                if relevant_visible:
                    relevant_lost_at = None
                elif relevant_lost_at is None:
                    relevant_lost_at = now
            else:
                approach_closest = float('inf')
                approach_max_px = 0
                relevant_lost_at = None

            event = _derive_event(state, signs, obstacles, frame_h, lane_mask,
                                  ignore_signs=ignore_signs,
                                  stop_distance_m=timings.get('stop_distance_m', 0.25),
                                  at_red_line=red_line,
                                  react_distance_m=timings.get('sign_react_distance_m', 1.5),
                                  react_min_px=_sign_react_min_px,
                                  stop_min_px=_sign_stop_min_px,
                                  red_band=red_band)
            # Backup commit — also the ROBUST "at the line" path on the bot. As the
            # bot reaches the stop line the sign goes OVERHEAD: its big tag rises
            # and clips off the TOP of the frame, so the tag disappears right when
            # we're at the line. Once the intersection is identified a real bot
            # commits to stopping; it doesn't coast through just because the tag
            # left view. "Got close enough that disappearing means we're there" is:
            #   * metric distance (calibrated): approach_closest < stop_commit_distance_m
            #   * tag pixels (uncalibrated bot): the tag grew past the REACT size
            #     (react_min_px) before vanishing — i.e. it went overhead, not a
            #     momentary far mis-detect. (Using react, not the full stop size, so
            #     a sign that clips out before reaching stop_px still stops us.)
            committed = (approach_closest < timings.get('stop_commit_distance_m', 0.5)
                         or (_sign_react_min_px > 0
                             and approach_max_px >= _sign_react_min_px))
            if (state == State.APPROACH and not event and not relevant_visible
                    and committed
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
                _near_line = (approach_closest < timings.get('line_straight_distance_m', 0.6)
                              or (_line_straight_px > 0 and approach_max_px >= _line_straight_px))
                if _near_line:
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
                # The turn no longer drives BLIND: _maneuver_tick keeps the camera +
                # object detector live during the maneuver and reports obstacles so
                # the turn HOLDS (not abandons) for a duckie/bot in the box and
                # resumes when clear. Still flag 'maneuvering' so the watchdog
                # doesn't misread the (now tick-refreshed) last_good as a stall.
                _cam_state['maneuvering'] = True
                try:
                    {
                        State.TURN_LEFT:        turn_left,
                        State.TURN_RIGHT:       turn_right,
                        State.STRAIGHT_THROUGH: straight_through,
                    }[state](wheels, stop_event, timings,
                             tick=_maneuver_tick, set_wheels=_safe_set_wheels,
                             hold_max_s=timings.get('maneuver_hold_max_s', 3.0))
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
                    'red_line_px': red_px,
                    'red_line_thr': _red_line_thr,
                    'red_band': red_band,
                    'red_band_rows': red_band_diag.get('rows'),
                    'red_band_rows_frac': red_band_diag.get('rows_frac'),
                    'red_band_row_max': red_band_diag.get('row_max'),
                    'obstacle_stop': obstacle_present,
                    'bot_ahead': bot_stop,
                    'obstacles': [
                        {'bbox': [int(v) for v in bbox],
                         'cls': int(cls), 'score': round(float(score), 2)}
                        for bbox, score, cls in ((raw_dets or []) + (veh_dets or []))
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
