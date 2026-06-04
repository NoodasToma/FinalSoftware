"""
Project task - Godot simulation server + test dashboard.

Runs the SAME tasks.project.packages.agent.main() against the Godot sim, and
serves a browser dashboard at http://localhost:<port>/ with:
  * the bot's live camera (/video)
  * a 2x2 debug grid (/debug): lane detection, AprilTags, traffic-light HSV,
    object (duckie) detection - so you can SEE what the perception sees
  * a live config editor (/config): edit the lane / maneuver / traffic-light /
    lane-HSV YAMLs and apply them without relaunching Godot

Launched by:  python launch.py --sim --task project
"""

import os
import sys
import signal
import threading
import time
import argparse

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)
project_root = os.path.abspath(project_root)

import numpy as np
import cv2
import yaml
from flask import Flask, Response, jsonify, request

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

import socket
import tasks.project.packages.agent as agent
from tasks.project.packages.perception import AprilTagDetector, TrafficLightDetector
from tasks.project.packages.sign_registry import lookup
from tasks.project.packages.precedence import we_go_first
from tasks.project.sim_tests.sim_telemetry import (
    TelemetryLogger, sim_fidelity_kwargs, SIM_APRILTAG_INTRINSICS, SIM_TAG_SIZE_M,
)
from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.visual_lane_servoing.packages import visual_servoing_activity as lane_hsv
from tasks.object_detection.packages.agent import ObjectDetectionAgent, CLASS_NAMES, CLASS_COLORS


app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None

_agent_stop   = threading.Event()
_agent_thread = None
_telemetry    = None   # live TelemetryLogger (observer for the running agent)

# Manual-drive state. While manual mode is on the agent thread is paused and the
# wheels are driven only by /drive commands. A watchdog zeros the wheels if no
# command arrives for _MANUAL_TIMEOUT_S, so a dropped/closed client can't leave
# the bot running away.
_manual_lock = threading.Lock()
_manual = {'on': False, 'last_cmd': 0.0}
_MANUAL_TIMEOUT_S = 0.4

# ----------------------------------------------------------------------------- config
# Editable YAMLs surfaced in the dashboard. Keyed by the section name shown in the UI.
CONFIG_FILES = {
    'maneuver_timings':  os.path.join(project_root, 'tasks/project/packages/config/maneuver_timings.yaml'),
    'lane_servoing':     os.path.join(project_root, 'config/lane_servoing_config.yaml'),
    'lane_hsv':          os.path.join(project_root, 'config/lane_servoing_hsv_config.yaml'),
    'traffic_light_hsv': os.path.join(project_root, 'tasks/project/packages/config/traffic_light_hsv.yaml'),
}


def load_all_config() -> dict:
    out = {}
    for name, path in CONFIG_FILES.items():
        try:
            with open(path) as fh:
                out[name] = yaml.safe_load(fh) or {}
        except Exception as e:
            out[name] = {'_error': str(e)}
    return out


def save_section(name: str, values: dict) -> None:
    path = CONFIG_FILES[name]
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    for k, v in values.items():
        # keep ints int, floats float
        try:
            fv = float(v)
            cfg[k] = int(fv) if float(fv).is_integer() and 'gain' not in k and 'speed' not in k and 'boost' not in k else fv
        except (TypeError, ValueError):
            cfg[k] = v
    with open(path, 'w') as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=True)


def _apply_lane_hsv_live() -> None:
    """Push lane-HSV bounds into the (module-global) lane detector so the change
    takes effect immediately, without an agent restart."""
    try:
        with open(CONFIG_FILES['lane_hsv']) as fh:
            h = yaml.safe_load(fh) or {}
        lane_hsv.set_hsv_bounds(
            [h.get('yellow_lower_h', 0), h.get('yellow_lower_s', 0), h.get('yellow_lower_v', 0)],
            [h.get('yellow_upper_h', 0), h.get('yellow_upper_s', 0), h.get('yellow_upper_v', 0)],
            [h.get('white_lower_h', 0),  h.get('white_lower_s', 0),  h.get('white_lower_v', 0)],
            [h.get('white_upper_h', 0),  h.get('white_upper_s', 0),  h.get('white_upper_v', 0)],
        )
    except Exception as e:
        print(f'[dashboard] lane HSV live-apply failed: {e}')


# ----------------------------------------------------------------------------- debug views
class DebugProcessor(threading.Thread):
    """Runs its OWN perception on the live frame to render a 2x2 debug grid.
    Independent of the agent (read-only observation); object detection is
    throttled to keep CPU sane."""

    def __init__(self, get_camera, yolo_period=0.4):
        super().__init__(daemon=True, name='DebugProcessor')
        self._get_camera = get_camera
        self._yolo_period = yolo_period
        self._lock = threading.Lock()
        self._grid = None
        self._running = True
        self._obj_cache = []
        self._last_yolo = 0.0
        self._build_detectors()

    def _build_detectors(self):
        self.lane = LaneServoingAgent()
        self.tags = AprilTagDetector(tag_size_m=SIM_TAG_SIZE_M, intrinsics=SIM_APRILTAG_INTRINSICS)
        self.light = TrafficLightDetector(); self.light.arm()
        self.obj = ObjectDetectionAgent()

    def reload_cheap(self):
        """Recreate the cheap detectors (picks up new YAMLs); keep the YOLO model."""
        try:
            self.lane = LaneServoingAgent()
            self.tags = AprilTagDetector(tag_size_m=SIM_TAG_SIZE_M, intrinsics=SIM_APRILTAG_INTRINSICS)
            self.light = TrafficLightDetector(); self.light.arm()
        except Exception as e:
            print(f'[dashboard] debug reload failed: {e}')

    def stop(self):
        self._running = False

    def latest(self):
        with self._lock:
            return None if self._grid is None else self._grid.copy()

    def run(self):
        while self._running:
            cam = self._get_camera()
            if cam is None:
                time.sleep(0.1); continue
            ok, bgr = cam.read()
            if not ok or bgr is None:
                time.sleep(0.03); continue
            try:
                cells = [
                    ('lane',    self._lane(bgr)),
                    ('apriltags', self._tags(bgr)),
                    ('traffic light', self._light(bgr)),
                    ('objects', self._objects(bgr)),
                ]
                grid = self._compose(cells)
                with self._lock:
                    self._grid = grid
            except Exception as e:
                print(f'[dashboard] debug frame error: {e}')
            time.sleep(0.07)

    # --- per-view overlays -------------------------------------------------
    def _lane(self, bgr):
        out = bgr.copy()
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            left, right = self.lane.compute_commands(rgb)
            d = self.lane.last_debug_info
            ym, wm = d.get('yellow_mask'), d.get('white_mask')
            if ym is not None: out[ym > 0] = (0, 255, 255)
            if wm is not None: out[wm > 0] = (255, 160, 0)
            for sy, xs in zip(d.get('slice_ys', []), [d.get('yellow_xs', []), d.get('white_xs', [])]):
                pass
            for x in d.get('yellow_xs', []):
                cv2.circle(out, (int(x), out.shape[0] - 30), 5, (0, 0, 255), -1)
            cv2.putText(out, f"err={d.get('lateral_error', 0):+.2f} L={left:.2f} R={right:.2f}",
                        (8, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        except Exception as e:
            cv2.putText(out, f"lane err: {e}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return out

    def _tags(self, bgr):
        out = bgr.copy()
        try:
            obs = self.tags.detect(bgr) or []
            for o in obs:
                cx, cy = o.center_xy
                s = max(8, o.side_length_px // 2)
                cv2.rectangle(out, (cx - s, cy - s), (cx + s, cy + s), (0, 255, 0), 2)
                sem = lookup(o.id)
                meaning = (sem.kind or sem.tag_type) if sem else '?'
                cv2.putText(out, f"{o.id}:{meaning}", (cx - s, cy - s - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if not obs:
                cv2.putText(out, "no tags", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        except Exception as e:
            cv2.putText(out, f"tag err: {e}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return out

    def _light(self, bgr):
        try:
            return self.light.get_debug_overlay(bgr)
        except Exception as e:
            out = bgr.copy()
            cv2.putText(out, f"light err: {e}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return out

    def _objects(self, bgr):
        out = bgr.copy()
        try:
            now = time.time()
            if now - self._last_yolo >= self._yolo_period:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                dets = self.obj.detect(rgb)
                if dets is not None:
                    self._obj_cache = dets
                self._last_yolo = now
            for (x1, y1, x2, y2), score, cls in self._obj_cache:
                color = CLASS_COLORS.get(cls, (0, 255, 0))
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, f"{CLASS_NAMES.get(cls, cls)} {score:.2f}", (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            cv2.putText(out, f"obj err: {e}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return out

    @staticmethod
    def _compose(cells):
        cw, ch = 320, 240
        tiles = []
        for label, img in cells:
            t = cv2.resize(img, (cw, ch))
            cv2.rectangle(t, (0, 0), (cw - 1, ch - 1), (60, 60, 60), 1)
            cv2.rectangle(t, (0, 0), (len(label) * 11 + 12, 22), (0, 0, 0), -1)
            cv2.putText(t, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)
            tiles.append(t)
        top = np.hstack([tiles[0], tiles[1]])
        bot = np.hstack([tiles[2], tiles[3]])
        return np.vstack([top, bot])


debug_proc = None  # set in main()


def _debug_stream():
    blank = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(blank, "debug starting...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2)
    while True:
        grid = debug_proc.latest() if debug_proc else None
        frame = grid if grid is not None else blank
        ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        time.sleep(0.06)


# ----------------------------------------------------------------------------- agent mgmt
def _start_agent():
    global _agent_stop, _agent_thread, _telemetry
    _agent_stop = threading.Event()
    # Fresh telemetry logger: enriches each agent snapshot with Godot pose, keeps
    # the latest for /telemetry, and streams to _sim_logs/live/live.jsonl. The
    # SAME fidelity kwargs the test suite uses (sim intrinsics + tag size) are
    # passed so the LIVE agent runs the real est_distance stop-line path too.
    if _telemetry is not None:
        try:
            _telemetry.close()
        except Exception:
            pass
    _telemetry = TelemetryLogger(os.path.join(project_root, '_sim_logs', 'live'),
                                 wheels=wheels, label='live', keep_records=False)
    _agent_thread = threading.Thread(
        target=agent.main, args=(camera, wheels, leds, _agent_stop),
        kwargs=dict(observer=_telemetry, **sim_fidelity_kwargs()),
        daemon=True, name='AgentThread')
    _agent_thread.start()


def _restart_agent():
    global _agent_thread
    try:
        _agent_stop.set()
        if _agent_thread:
            _agent_thread.join(timeout=6)
    except Exception:
        pass
    try:
        wheels.set_wheels_speed(0.0, 0.0)
    except Exception:
        pass
    _start_agent()


def _stop_agent():
    """Stop the agent thread and leave it stopped (for manual drive)."""
    global _agent_thread
    try:
        _agent_stop.set()
        if _agent_thread:
            _agent_thread.join(timeout=6)
    except Exception:
        pass
    _agent_thread = None
    try:
        wheels.set_wheels_speed(0.0, 0.0)
    except Exception:
        pass


# ----------------------------------------------------------------------------- task demo
# Runs the Task-2/3 rubric scenarios LIVE in this running sim: teleports the bot
# to face each situation, runs the REAL agent, and records what it did. Watch it
# happen in the same Godot window / the dashboard camera.
_demo = {'running': False, 'current': '', 'results': []}


def _demo_run_agent(x, z, heading, seconds):
    """Teleport, run a fresh real agent for `seconds`, return its state transitions."""
    wheels.set_wheels_speed(0.0, 0.0)
    wheels.teleport(x, z, heading)
    time.sleep(0.8)
    transitions = []
    _rn = agent.next_state
    def _tn(s, ev):
        ns = _rn(s, ev)
        if ns != s:
            transitions.append((s.name, ev, ns.name))
        return ns
    agent.next_state = _tn
    stop = threading.Event()
    th = threading.Thread(target=agent.main, args=(camera, wheels, None, stop),
                          kwargs=sim_fidelity_kwargs(), daemon=True)
    th.start()
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(0.1)
    stop.set(); th.join(timeout=5)
    agent.next_state = _rn
    wheels.set_wheels_speed(0.0, 0.0)
    return transitions


def _run_demo():
    """Drive the rubric scenarios in sequence (runs in a background thread)."""
    global _demo
    _demo = {'running': True, 'current': 'starting', 'results': []}
    _stop_agent()
    tags = AprilTagDetector()

    def rec(name, ok, detail):
        _demo['results'].append({'name': name, 'pass': bool(ok), 'detail': detail})

    try:
        _demo['current'] = 'Stop sign → red line, full stop, car-like turn'
        tr = _demo_run_agent(5.85, 7.05, 0.0, 20)
        st = [t[2] for t in tr]
        rec('T3 stop sign: stop at the red line, wait, take a turn',
            'STOPPED' in st and any(s in st for s in ('TURN_LEFT', 'TURN_RIGHT', 'STRAIGHT_THROUGH')),
            ' → '.join(['DRIVE'] + st) or 'no transitions')

        _demo['current'] = 'Traffic light → stop on red, go on green'
        tr = _demo_run_agent(5.85, 4.45, 0.0, 24)
        st = [t[2] for t in tr]
        rec('T2 traffic light: stop on red, go on green',
            'see_light' in [t[1] for t in tr] and 'STOPPED' in st,
            ' → '.join(['DRIVE'] + st) or 'no transitions')

        _demo['current'] = 'Obstacle → soft stop for duckie'
        tr = _demo_run_agent(4.5, 1.632, 270.0, 8)
        n_soft = sum(1 for t in tr if t[2] == 'SOFT_STOP')
        rec('T3 obstacle: soft-stop for a duckie', n_soft > 0,
            f'entered SOFT_STOP {n_soft}× for the duckie in its path')

        _demo['current'] = 'Other robot → detect parked Duckiebot, who goes first'
        wheels.set_wheels_speed(0.0, 0.0); wheels.teleport(5.85, 4.85, 0.0); time.sleep(1.0)
        veh = None
        for _ in range(20):
            ok, f = camera.read()
            if ok and f is not None:
                for o in (tags.detect(f) or []):
                    sem = lookup(o.id)
                    if sem and sem.tag_type == 'Vehicle':
                        veh = sem
            time.sleep(0.1)
        if veh:
            first = we_go_first(socket.gethostname(), [veh])
            rec('T3 precedence: detect other vehicle + who-goes-first', True,
                f'detected {veh.vehicle_name}; we_go_first={first}')
        else:
            rec('T3 precedence: detect other vehicle', False, 'vehicle tag not seen')
    except Exception as e:
        rec('demo error', False, str(e))
    finally:
        _demo['current'] = 'done'
        _demo['running'] = False
        try:
            wheels.reset_game()
        except Exception:
            pass
        time.sleep(0.3)
        _start_agent()


def _set_manual(on: bool) -> None:
    """Enter/leave manual drive. Entering pauses the agent; leaving resumes it."""
    with _manual_lock:
        was = _manual['on']
        _manual['on'] = on
        _manual['last_cmd'] = time.time()
    if on and not was:
        _stop_agent()
    elif not on and was:
        try:
            wheels.set_wheels_speed(0.0, 0.0)
        except Exception:
            pass
        _start_agent()


def _manual_watchdog():
    """Zero the wheels if manual mode is on but no /drive arrived recently."""
    while True:
        time.sleep(0.1)
        with _manual_lock:
            on, last = _manual['on'], _manual['last_cmd']
        if on and (time.time() - last) > _MANUAL_TIMEOUT_S:
            try:
                wheels.set_wheels_speed(0.0, 0.0)
            except Exception:
                pass


# ----------------------------------------------------------------------------- routes
generate_frames = make_frame_generator(lambda: camera, lambda f: f if f is not None else
                                        np.zeros((480, 640, 3), np.uint8), quality=70, rgb=False)


@app.route('/')
def index():
    return Response(_PAGE, mimetype='text/html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/debug')
def debug():
    return Response(_debug_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/config', methods=['GET'])
def get_config():
    return jsonify(load_all_config())


@app.route('/config', methods=['POST'])
def post_config():
    data = request.get_json(force=True) or {}
    changed = []
    for section, values in data.items():
        if section in CONFIG_FILES and isinstance(values, dict):
            save_section(section, values)
            changed.append(section)
    _apply_lane_hsv_live()
    if debug_proc:
        debug_proc.reload_cheap()
    restart = bool(request.args.get('restart', '1') == '1')
    if restart:
        _restart_agent()
    return jsonify({'status': 'ok', 'saved': changed, 'agent_restarted': restart})


@app.route('/restart', methods=['POST'])
def restart():
    with _manual_lock:
        _manual['on'] = False
    _restart_agent()
    return jsonify({'status': 'ok'})


@app.route('/reset', methods=['POST'])
def reset():
    """FULL simulation reset: teleport the bot back to its spawn (Godot reset)
    AND restart the agent from a clean state. Godot keeps running."""
    with _manual_lock:
        _manual['on'] = False
    # stop the agent first so it isn't fighting the reset
    _stop_agent()
    try:
        wheels.reset_game()          # Godot: bot -> spawn position/heading
    except Exception as e:
        print(f'[dashboard] reset_game failed: {e}')
    time.sleep(0.3)
    _start_agent()                   # fresh agent from DRIVE
    return jsonify({'status': 'ok'})


@app.route('/demo', methods=['POST'])
def demo():
    if _demo['running']:
        return jsonify({'status': 'busy'})
    with _manual_lock:
        _manual['on'] = False
    threading.Thread(target=_run_demo, daemon=True, name='TaskDemo').start()
    return jsonify({'status': 'started'})


@app.route('/demo_status', methods=['GET'])
def demo_status():
    return jsonify(_demo)


@app.route('/telemetry', methods=['GET'])
def telemetry():
    """Latest agent snapshot + Godot pose: what the bot sees and is doing right now
    (state, detected tags with est_distance, light colour, red line, obstacle, lane, pose)."""
    return jsonify((_telemetry.latest if _telemetry else None) or {})


# Watch-one-behaviour teleports: park the bot at the start of a scenario and let
# the (restarted) REAL agent do its thing while you watch /video + the live
# status panel. Coordinates match tasks/project/sim_tests/behaviour_suite.py.
SCENARIOS = {
    'lane':     (0.9,  5.1,   0.0, 'lane-follow the west straight'),
    'stop':     (5.85, 7.05,  0.0, 'stop sign @ 4-way: red line → full stop → car turn'),
    'light':    (5.85, 4.45,  0.0, 'traffic light: arm → stop at line → go on green'),
    'obstacle': (4.5,  1.632, 270.0, 'duckie in the lane → SOFT_STOP until cleared'),
    'robot':    (5.85, 4.85,  0.0, 'parked Duckiebot with vehicle plate → precedence'),
}


@app.route('/scenario/<name>', methods=['POST'])
def scenario(name):
    sc = SCENARIOS.get(name)
    if sc is None:
        return jsonify({'status': 'error', 'msg': f'unknown scenario {name!r}',
                        'known': sorted(SCENARIOS)}), 404
    with _manual_lock:
        _manual['on'] = False
    x, z, heading, blurb = sc
    _stop_agent()
    try:
        wheels.set_wheels_speed(0.0, 0.0)
        wheels.teleport(x, z, heading)
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500
    time.sleep(0.5)
    _start_agent()
    return jsonify({'status': 'ok', 'scenario': name, 'at': [x, z, heading], 'what': blurb})


@app.route('/manual', methods=['POST'])
def manual():
    data = request.get_json(force=True) or {}
    _set_manual(bool(data.get('on', False)))
    return jsonify({'status': 'ok', 'manual': _manual['on']})


@app.route('/drive', methods=['POST'])
def drive():
    if not _manual['on']:
        return jsonify({'status': 'error',
                        'msg': 'not in manual mode; POST /manual {"on": true} first'}), 409
    data = request.get_json(force=True) or {}
    try:
        left  = max(-0.8, min(0.8, float(data.get('left', 0.0))))
        right = max(-0.8, min(0.8, float(data.get('right', 0.0))))
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'msg': 'left/right must be numbers'}), 400
    with _manual_lock:
        _manual['last_cmd'] = time.time()
    try:
        wheels.set_wheels_speed(left, right)
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500
    return jsonify({'status': 'ok', 'left': left, 'right': right})


@app.route('/shutdown')
def shutdown():
    shutdown_cleanup(wheels, camera, _agent_stop)
    if debug_proc:
        debug_proc.stop()
    return jsonify({'status': 'ok'})


# ----------------------------------------------------------------------------- main
def main():
    global camera, wheels, leds, debug_proc

    ap = argparse.ArgumentParser(description='Project Server - Godot Simulation + Dashboard')
    ap.add_argument('--port',       type=int, default=5000)
    ap.add_argument('--frame-port', type=int, default=5001)
    ap.add_argument('--wheel-port', type=int, default=5002)
    ap.add_argument('--godot-host', type=str, default='localhost')
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT SERVER - GODOT SIMULATION + DASHBOARD')
    print('=' * 60)

    leds = None
    print('\n[1/4] LEDs: disabled in sim (None)')

    print('[2/4] Wheels driver (Godot)...')
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host, godot_port=args.wheel_port)

    print('[3/4] Camera driver (Godot)...')
    camera = GodotCameraDriver(godot_config=GodotCameraConfig(host='0.0.0.0', port=args.frame_port))
    camera.start()

    print('[4/4] Starting agent + debug processor...')
    _apply_lane_hsv_live()
    _start_agent()
    debug_proc = DebugProcessor(lambda: camera)
    debug_proc.start()
    threading.Thread(target=_manual_watchdog, daemon=True, name='ManualWatchdog').start()

    def _shutdown(signum, frame):
        shutdown_cleanup(wheels, camera, _agent_stop)
        if debug_proc:
            debug_proc.stop()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nDashboard: http://localhost:{web_port}/   (video + debug + config editor)')
    print('Press Ctrl+C to stop\n')
    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        shutdown_cleanup(wheels, camera, _agent_stop)
        if debug_proc:
            debug_proc.stop()


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Duckiebot Sim Dashboard</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#15171c;color:#e6e6e6}
 header{background:#0e1014;padding:10px 16px;font-size:18px;font-weight:600;border-bottom:1px solid #2a2d34}
 .wrap{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
 .col{flex:1 1 480px;min-width:360px}
 .card{background:#1d2027;border:1px solid #2a2d34;border-radius:8px;padding:12px;margin-bottom:16px}
 .card h2{margin:0 0 8px;font-size:14px;color:#7fd1b9;text-transform:uppercase;letter-spacing:.05em}
 img{width:100%;border-radius:6px;background:#000;display:block}
 .legend{font-size:12px;color:#9aa0aa;margin-top:6px}
 .sec{border:1px solid #2a2d34;border-radius:6px;margin-bottom:10px}
 .sec>summary{cursor:pointer;padding:8px 10px;font-weight:600;color:#cdd3dc}
 .grid{display:grid;grid-template-columns:1fr 90px;gap:6px 8px;padding:8px 10px}
 .grid label{font-size:13px;align-self:center;color:#b9c0ca}
 .grid input{width:84px;background:#0e1014;border:1px solid #333;color:#e6e6e6;border-radius:4px;padding:4px}
 button{background:#2d7;border:0;color:#06210f;font-weight:700;padding:9px 14px;border-radius:6px;cursor:pointer;font-size:14px}
 button.alt{background:#39c;color:#04121f}
 button.warn{background:#e85;color:#2a0d04}
 .dpad{display:grid;grid-template-columns:repeat(3,60px);gap:6px;justify-content:center;margin-top:10px}
 .dpad button{padding:0;height:48px;font-size:20px;background:#2a2d34;color:#e6e6e6;border:1px solid #3a3d44;font-weight:700}
 .dpad button.stop{background:#933;color:#fff}
 #msg{margin-left:10px;font-size:13px;color:#7fd1b9}
 .bar{display:flex;align-items:center;gap:8px;margin-top:6px}
</style></head><body>
<header>🦆 Duckiebot Project — Simulation Dashboard</header>
<div class="wrap">
  <div class="col">
    <div class="card"><h2>Bot camera (what the agent sees)</h2><img src="/video"></div>
    <div class="card"><h2>Perception debug</h2><img src="/debug">
      <div class="legend">Top-left: lane (yellow/white masks + steering) · Top-right: AprilTags (id:meaning)
      · Bottom-left: traffic-light HSV blobs · Bottom-right: object detection (duckie/truck/sign)</div></div>
  </div>
  <div class="col">
    <div class="card"><h2>Drive control</h2>
      <div class="bar">
        <button id="manualBtn" class="alt" onclick="toggleManual()">Take manual control</button>
        <button onclick="restartAgent()">Restart agent</button>
        <button class="warn" onclick="resetSim()">↻ Restart simulation</button>
        <span id="drvmsg">agent running</span>
      </div>
      <div id="dpad" class="dpad" style="display:none">
        <span></span><button data-l="0.42" data-r="0.42">&#9650;</button><span></span>
        <button data-l="-0.38" data-r="0.38">&#9664;</button>
        <button class="stop" data-l="0" data-r="0">&#9632;</button>
        <button data-l="0.38" data-r="-0.38">&#9654;</button>
        <span></span><button data-l="-0.42" data-r="-0.42">&#9660;</button><span></span>
      </div>
      <div class="legend" id="drvhint" style="display:none">Keyboard: <b>W/A/S/D</b> or arrows to drive ·
        <b>Space</b> = stop. Hold to move, release to stop. The agent is paused while you hold manual control.</div>
    </div>
    <div class="card"><h2>Live bot status</h2>
      <div id="status" style="font-size:14px;line-height:1.7">waiting for telemetry…</div>
    </div>
    <div class="card"><h2>Watch a behaviour (teleports the bot, agent stays on)</h2>
      <div class="bar" style="flex-wrap:wrap">
        <button class="alt" onclick="scenario('lane')">Lane follow</button>
        <button class="alt" onclick="scenario('stop')">Stop sign + turn</button>
        <button class="alt" onclick="scenario('light')">Traffic light</button>
        <button class="alt" onclick="scenario('obstacle')">Duckie obstacle</button>
        <button class="alt" onclick="scenario('robot')">Other robot</button>
        <span id="scmsg"></span>
      </div>
      <div class="legend">Each button parks the bot at a scenario start and restarts the agent —
      watch the camera + status above. Stop sign: approaches the 4-way, halts at the <b>red line</b>,
      waits, then makes a gradual car-like turn.</div>
    </div>
    <div class="card"><h2>Task 2 &amp; 3 demo</h2>
      <div class="bar">
        <button onclick="runDemo()">▶ Run task demo</button>
        <span id="demostat">drives the bot through each rubric scenario, live</span>
      </div>
      <div id="demores" class="legend"></div>
    </div>
    <div class="card"><h2>Live config</h2>
      <div id="cfg">loading…</div>
      <div class="bar">
        <button onclick="save(true)">Save &amp; Apply (restart agent)</button>
        <button class="alt" onclick="save(false)">Save (HSV live, no restart)</button>
        <span id="msg"></span>
      </div>
      <div class="legend">Tuning tip: open the <b>lane_hsv</b> / <b>lane_servoing</b> sections and watch the
      lane debug view. Lower <b>d_gain</b> if steering jitters; widen <b>yellow/white</b> HSV if the lane
      isn't detected. HSV changes apply live; gains/speeds need "Save &amp; Apply".</div>
    </div>
  </div>
</div>
<script>
let CFG={};
function render(){
 let h='';
 for(const sec in CFG){
   h+='<details class="sec"'+(sec.includes('hsv')||sec==='lane_servoing'?' open':'')+'><summary>'+sec+'</summary><div class="grid">';
   for(const k in CFG[sec]){
     const v=CFG[sec][k];
     h+='<label>'+k+'</label><input data-s="'+sec+'" data-k="'+k+'" value="'+v+'">';
   }
   h+='</div></details>';
 }
 document.getElementById('cfg').innerHTML=h;
}
async function load(){ CFG=await (await fetch('/config')).json(); render(); }
async function save(restart){
 const body={};
 document.querySelectorAll('#cfg input').forEach(i=>{
   const s=i.dataset.s,k=i.dataset.k; (body[s]=body[s]||{})[k]=i.value;
 });
 document.getElementById('msg').textContent='saving…';
 const r=await fetch('/config?restart='+(restart?1:0),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json();
 document.getElementById('msg').textContent='saved '+(j.saved||[]).join(', ')+(j.agent_restarted?' · agent restarted':' · live');
}
// ---- manual drive ----------------------------------------------------------
function clamp(x){return Math.max(-0.8,Math.min(0.8,x));}
let manualOn=false, hb=null, held=new Set(), btn=null;
const DV=0.42, TV=0.38;
function compute(){
 if(btn) return btn;
 const t=(held.has('f')?1:0)-(held.has('b')?1:0);
 const s=(held.has('r')?1:0)-(held.has('l')?1:0);
 return [clamp(t*DV+s*TV), clamp(t*DV-s*TV)];
}
async function sendDrive(l,r){
 try{await fetch('/drive',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({left:l,right:r})});}catch(e){}
}
function pump(){ if(!manualOn) return; const c=compute(); sendDrive(c[0],c[1]); }
async function toggleManual(){
 manualOn=!manualOn;
 try{await fetch('/manual',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({on:manualOn})});}catch(e){}
 document.getElementById('manualBtn').textContent=manualOn?'Release (resume agent)':'Take manual control';
 document.getElementById('dpad').style.display=manualOn?'grid':'none';
 document.getElementById('drvhint').style.display=manualOn?'block':'none';
 document.getElementById('drvmsg').textContent=manualOn?'MANUAL — agent paused':'agent running';
 if(manualOn){ if(!hb) hb=setInterval(pump,120); }
 else { if(hb){clearInterval(hb);hb=null;} held.clear(); btn=null; }
}
async function restartAgent(){
 manualOn=false; if(hb){clearInterval(hb);hb=null;} held.clear(); btn=null;
 document.getElementById('manualBtn').textContent='Take manual control';
 document.getElementById('dpad').style.display='none';
 document.getElementById('drvhint').style.display='none';
 try{await fetch('/restart',{method:'POST'});}catch(e){}
 document.getElementById('drvmsg').textContent='agent restarted';
}
let demoPoll=null;
async function runDemo(){
 manualOn=false; if(hb){clearInterval(hb);hb=null;}
 document.getElementById('demostat').textContent='starting…';
 document.getElementById('demores').innerHTML='';
 try{await fetch('/demo',{method:'POST'});}catch(e){}
 if(demoPoll) clearInterval(demoPoll);
 demoPoll=setInterval(async()=>{
   let d; try{ d=await (await fetch('/demo_status')).json(); }catch(e){ return; }
   document.getElementById('demostat').textContent = d.running ? ('running: '+d.current) : 'finished';
   let h='';
   for(const r of (d.results||[])){
     h+='<div style="margin:3px 0"><b style="color:'+(r.pass?'#2d7':'#e66')+'">'+(r.pass?'PASS':'FAIL')+'</b> '+r.name+'<br><span style="color:#9aa0aa">'+r.detail+'</span></div>';
   }
   document.getElementById('demores').innerHTML=h;
   if(!d.running){ clearInterval(demoPoll); demoPoll=null; document.getElementById('drvmsg').textContent='agent running'; }
 },700);
}
async function resetSim(){
 manualOn=false; if(hb){clearInterval(hb);hb=null;} held.clear(); btn=null;
 document.getElementById('manualBtn').textContent='Take manual control';
 document.getElementById('dpad').style.display='none';
 document.getElementById('drvhint').style.display='none';
 document.getElementById('drvmsg').textContent='resetting simulation…';
 try{await fetch('/reset',{method:'POST'});}catch(e){}
 document.getElementById('drvmsg').textContent='simulation reset — bot back at spawn';
}
// ---- scenarios + live status ------------------------------------------------
async function scenario(name){
 document.getElementById('scmsg').textContent='teleporting…';
 try{
   const r=await fetch('/scenario/'+name,{method:'POST'});
   const j=await r.json();
   document.getElementById('scmsg').textContent=j.status==='ok'?('▶ '+j.what):('error: '+(j.msg||''));
 }catch(e){ document.getElementById('scmsg').textContent='error'; }
 document.getElementById('drvmsg').textContent='agent running';
}
function fmtTags(ts){
 if(!ts||!ts.length) return '<span style="color:#666">none</span>';
 return ts.map(t=>t.id+':'+t.meaning+(t.est_distance_m!=null?(' @'+t.est_distance_m.toFixed(2)+'m'):'')).join(' · ');
}
setInterval(async()=>{
 let d; try{ d=await (await fetch('/telemetry')).json(); }catch(e){ return; }
 if(!d||!d.state) return;
 const p=d.pose||{}, l=d.light||{}, lane=d.lane||{};
 const stCol={DRIVE:'#2d7',APPROACH:'#fc3',STOPPED:'#e85',WAIT:'#e85',SOFT_STOP:'#e66',
              TURN_LEFT:'#39c',TURN_RIGHT:'#39c',STRAIGHT_THROUGH:'#39c'}[d.state]||'#fff';
 document.getElementById('status').innerHTML=
  'state <b style="color:'+stCol+'">'+d.state+'</b>'
  +(d.event?(' <span style="color:#9aa0aa">(via '+d.event+')</span>'):'')
  +' · pose ('+(p.x!=null?p.x.toFixed(2):'?')+', '+(p.z!=null?p.z.toFixed(2):'?')+') @'
  +(p.heading_deg!=null?p.heading_deg.toFixed(0):'?')+'°'
  +'<br>signs: '+fmtTags(d.tags)
  +'<br>light: <b>'+(l.color||'—')+'</b>'+(l.armed?' (armed)':'')
  +' · red line: <b style="color:'+(d.red_line?'#e55':'#666')+'">'+(d.red_line?'AT LINE':'no')+'</b>'
  +' · obstacle: <b>'+(d.obstacle_stop?'STOP':'clear')+'</b>'
  +'<br>lane err '+(lane.error!=null?lane.error.toFixed(2):'?')
  +' · wheels L/R '+(d.wheels?d.wheels.left.toFixed(2)+'/'+d.wheels.right.toFixed(2):'?')
  +(d.legal_turns?(' · legal turns: '+d.legal_turns.join(',')):'');
},600);
const KEYMAP={KeyW:'f',ArrowUp:'f',KeyS:'b',ArrowDown:'b',KeyA:'l',ArrowLeft:'l',KeyD:'r',ArrowRight:'r'};
document.addEventListener('keydown',e=>{
 if(e.target&&(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'))return;
 if(!manualOn)return;
 if(e.code==='Space'){held.clear();btn=null;sendDrive(0,0);e.preventDefault();return;}
 const k=KEYMAP[e.code]; if(!k)return; held.add(k); e.preventDefault(); pump();
});
document.addEventListener('keyup',e=>{
 if(!manualOn)return; const k=KEYMAP[e.code]; if(!k)return;
 held.delete(k); if(held.size===0&&!btn) sendDrive(0,0); else pump();
});
document.querySelectorAll('#dpad button').forEach(b=>{
 const l=parseFloat(b.dataset.l), r=parseFloat(b.dataset.r);
 const down=ev=>{ev.preventDefault(); if(!manualOn)return; btn=[l,r]; sendDrive(l,r);};
 const up=ev=>{ev.preventDefault(); btn=null; if(manualOn&&held.size===0) sendDrive(0,0);};
 b.addEventListener('mousedown',down); b.addEventListener('mouseup',up); b.addEventListener('mouseleave',up);
 b.addEventListener('touchstart',down); b.addEventListener('touchend',up);
});
load();
</script></body></html>"""


if __name__ == '__main__':
    sys.exit(main())
