import sys
import os
import signal
import threading
import time
import argparse
import traceback

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

# Force CPU object detection on the Jetson. Building the YOLO TensorRT engine
# runs the Nano out of device memory ("Device memory insufficient") and thrashes
# the whole bot into a freeze (agent stops mid-lane). CPU inference is slower but
# reliable; the project agent runs detection in a throttled background thread so
# it never blocks driving. Must be set BEFORE ObjectDetectionAgent is imported.
os.environ.setdefault('OBJDET_CPU', '1')

# numpy 1.19.5 (installed so onnxruntime can run the YOLO model on the bot) crashes
# with "Illegal instruction" on this Jetson unless OpenBLAS is pinned to the ARMv8
# kernel — its CPU autodetection picks an unsupported kernel intermittently. Must
# be set BEFORE the first `import numpy` in the process (so before cv2 too).
os.environ.setdefault('OPENBLAS_CORETYPE', 'ARMV8')

from flask import Flask, Response, jsonify, request
import numpy as np
import cv2
import yaml

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.led_driver import LEDDriver
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

import tasks.project.packages.agent as agent

# ---------------------------------------------------------------- bot configs
# The base YAMLs hold the SIM-verified values (the behaviour suite runs against
# them). The real robot is NOT the sim — in particular DaguWheelsDriver uses
# pwm_min=60 (stiction floor), which compresses wheel-speed ratios and would
# make the sim's turn arcs far wider on hardware — so the hardware starting
# values live in small *_bot.yaml overlays that only this server applies.
_PKG_CONFIG = os.path.join(project_root, 'tasks', 'project', 'packages', 'config')


def _load_bot_timings():
    """maneuver_timings.yaml merged with maneuver_timings_bot.yaml (if present)."""
    with open(os.path.join(_PKG_CONFIG, 'maneuver_timings.yaml')) as fh:
        timings = yaml.safe_load(fh) or {}
    overlay_path = os.path.join(_PKG_CONFIG, 'maneuver_timings_bot.yaml')
    overlay = {}
    try:
        with open(overlay_path) as fh:
            overlay = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        pass
    timings.update(overlay)
    return timings, sorted(overlay)


def _bot_lane_config_path():
    """config/lane_servoing_config_bot.yaml if it exists, else None (default)."""
    path = os.path.join(project_root, 'config', 'lane_servoing_config_bot.yaml')
    return path if os.path.isfile(path) else None


def _apply_bot_lane_hsv():
    """Apply the BOT-only lane HSV overlay (config/lane_servoing_hsv_config_bot.yaml)
    over the shared lane HSV, by calling the lane detector's set_hsv_bounds(). The
    lane HSV is module-global in visual_servoing_activity (loaded once from the base
    file), and that base is the SIM's calibration — so a per-venue bot override must
    be applied here, at the hardware entry point, not by editing the shared file.
    Returns the keys applied (for the startup log), or None if no overlay file."""
    path = os.path.join(project_root, 'config', 'lane_servoing_hsv_config_bot.yaml')
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        ov = yaml.safe_load(fh) or {}
    if not ov:
        return None
    from tasks.visual_lane_servoing.packages import visual_servoing_activity as _vsa
    cur = _vsa.get_hsv_bounds()
    cur.update(ov)
    _vsa.set_hsv_bounds(
        [cur['yellow_lower_h'], cur['yellow_lower_s'], cur['yellow_lower_v']],
        [cur['yellow_upper_h'], cur['yellow_upper_s'], cur['yellow_upper_v']],
        [cur['white_lower_h'],  cur['white_lower_s'],  cur['white_lower_v']],
        [cur['white_upper_h'],  cur['white_upper_s'],  cur['white_upper_v']],
    )
    # Fill chunky-dash interiors on the bot (this venue's centre dashes are fat
    # squares the edge-AND gate would otherwise gut). Sim leaves this off.
    _vsa.set_chunky_fill(True)
    return sorted(ov)

app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None
stop_event = threading.Event()

# Latest agent snapshot (set by the observer in the agent thread, read by the
# video/HUD + /telemetry in the Flask thread). Plain reference swap = thread-safe
# enough for a live debug view. This is what makes the bot URL show the SAME info
# as the sim dashboard: state, every detection, and WHY it's stopped.
_latest_snap = None


_latest_frame = None   # most recent BGR frame, handed over by the agent thread


def _observe(snap):
    global _latest_snap
    _latest_snap = snap


def _frame_observe(bgr):
    # The agent gives us each frame it read. We do NOT read the camera ourselves
    # (cv2.VideoCapture is single-reader; a second reader hangs the agent and the
    # bot goes idle). So /video is built purely from these frames.
    global _latest_frame
    _latest_frame = bgr


# ============================================================ manual drive (BOT-safe)
# The bot shares ONE I2C bus for wheels + LEDs; a SECOND thread writing the wheels
# is exactly what crashed this project with OSError(121). So manual drive here does
# NOT write the wheels from a Flask thread. Instead the dashboard stores the latest
# command, and the AGENT (the single I2C writer) applies it via the `manual_cmd`
# hook — perception/telemetry keep running, and there is never a second writer.
# A stale command (dropped client) decays to (0, 0) so the bot can't run away.
_manual_lock = threading.Lock()
_manual = {'on': False, 'left': 0.0, 'right': 0.0, 'last_cmd': 0.0}
_MANUAL_TIMEOUT_S = 0.4
_MANUAL_MAX = 0.6           # clamp manual wheel speed (gentler than full throttle)


def _manual_cmd():
    """Polled by the agent every loop. Returns (left, right) to drive, or None to
    let the agent drive normally. Bot-safe: the AGENT does the actual wheel write."""
    with _manual_lock:
        if not _manual['on']:
            return None
        if (time.time() - _manual['last_cmd']) > _MANUAL_TIMEOUT_S:
            return (0.0, 0.0)               # dropped client -> stop, but stay in manual
        return (_manual['left'], _manual['right'])


# ---------------------------------------------------------------- live config editor
# YAMLs surfaced in the dashboard so you can tune ON THE BOT without redeploying.
# These are the BOT-specific files real_server already applies; editing + applying
# re-reads them (lane HSV applies live; the rest restart the agent thread).
_ROOT = project_root
CONFIG_FILES = {
    'maneuver_timings_bot': os.path.join(_PKG_CONFIG, 'maneuver_timings_bot.yaml'),
    'lane_servoing_bot':    os.path.join(_ROOT, 'config', 'lane_servoing_config_bot.yaml'),
    'lane_hsv_bot':         os.path.join(_ROOT, 'config', 'lane_servoing_hsv_config_bot.yaml'),
    'traffic_light_hsv':    os.path.join(_PKG_CONFIG, 'traffic_light_hsv.yaml'),
}


def _load_all_config() -> dict:
    out = {}
    for name, path in CONFIG_FILES.items():
        try:
            with open(path) as fh:
                out[name] = yaml.safe_load(fh) or {}
        except Exception as e:
            out[name] = {'_error': str(e)}
    return out


def _save_config_section(name: str, values: dict) -> None:
    path = CONFIG_FILES[name]
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    for k, v in values.items():
        # keep ints int, floats float, leave strings/bools alone
        if isinstance(v, str):
            s = v.strip()
            low = s.lower()
            if low in ('true', 'false'):
                cfg[k] = (low == 'true')
                continue
            try:
                fv = float(s)
                cfg[k] = int(fv) if (fv.is_integer() and 'gain' not in k
                                     and 'speed' not in k and 'bias' not in k
                                     and 'boost' not in k) else fv
                continue
            except (TypeError, ValueError):
                cfg[k] = v
                continue
        cfg[k] = v
    with open(path, 'w') as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=True)


# ---------------------------------------------------------------- restartable agent
_agent_thread = None


def _start_agent():
    """(Re)start agent.main in a daemon thread with freshly-loaded bot config. The
    camera/wheels/LEDs are created once in main() and reused, so a restart only
    re-reads config + rebuilds the perception objects — it never re-inits hardware."""
    global _agent_thread, stop_event
    stop_event = threading.Event()
    timings, overlay_keys = _load_bot_timings()
    lane_cfg = _bot_lane_config_path()
    _apply_bot_lane_hsv()

    # Agent selection: `sign_agent: true` in the bot timings runs the REFERENCE's
    # sign-detection agent (lane servoing + SignBehaviorFSM — KvatiTown design, the
    # one that executes the turn open-loop so the sign decision wins over the lane);
    # otherwise the project's default agent.main runs. Toggle it live in the dashboard
    # config editor (maneuver_timings_bot.yaml) + Restart.
    use_sign_agent = bool(timings.get('sign_agent', False))
    sign_config = timings.get('sign_config') or None

    def _run_agent():
        try:
            if use_sign_agent:
                import tasks.project.packages.agent_signs as agent_signs
                print('[server] using REFERENCE sign-detection agent (sign_agent: true)', flush=True)
                agent_signs.main(camera, wheels, leds, stop_event,
                                 observer=_observe, frame_observer=_frame_observe,
                                 manual_cmd=_manual_cmd, lane_config_path=lane_cfg,
                                 sign_config=sign_config, timings=timings)
            else:
                agent.main(camera, wheels, leds, stop_event,
                           timings_override=timings, lane_config_path=lane_cfg,
                           observer=_observe, frame_observer=_frame_observe,
                           manual_cmd=_manual_cmd)
        except Exception as e:
            print('=' * 60, flush=True)
            print(f'AGENT THREAD CRASHED: {e!r}', flush=True)
            traceback.print_exc()
            print('=' * 60, flush=True)
            try:
                wheels.set_wheels_speed(0.0, 0.0)
            except Exception:
                pass

    _agent_thread = threading.Thread(target=_run_agent, daemon=True, name='AgentThread')
    _agent_thread.start()
    return overlay_keys


def _restart_agent():
    """Stop the running agent, then start a fresh one (picks up edited config)."""
    global _agent_thread
    with _manual_lock:                      # leave manual mode on restart
        _manual['on'] = False
        _manual['left'] = _manual['right'] = 0.0
    try:
        stop_event.set()
        if _agent_thread:
            _agent_thread.join(timeout=6)
    except Exception:
        pass
    try:
        wheels.set_wheels_speed(0.0, 0.0)
    except Exception:
        pass
    _start_agent()


# YOLO object_detection class ids -> label/colour (BGR), matching the model.
_OBJ = {0: ('duckie', (0, 215, 255)), 1: ('truck', (180, 100, 220)), 2: ('sign', (50, 205, 50))}

# Why the FSM is holding still, by state — shown big on the HUD so "drives then
# stands idle" is self-explanatory instead of a mystery.
_IDLE_REASON = {
    'STOPPED':   'stopped at line (pausing / deciding turn)',
    'WAIT':      'waiting (red light or yielding to a robot)',
    'SOFT_STOP': 'OBSTACLE ahead - holding until it clears',
    # No approach slowdown on the bot (approach_creep:false): a sign is in view but
    # the bot keeps lane-following normally until the commit point.
    'APPROACH':  'sign in view - following the lane until the line',
}


_STATE_COLOR = {
    'DRIVE': (80, 230, 80), 'APPROACH': (60, 200, 255), 'STOPPED': (60, 140, 255),
    'WAIT': (60, 140, 255), 'SOFT_STOP': (60, 60, 255),
    'TURN_LEFT': (255, 180, 60), 'TURN_RIGHT': (255, 180, 60), 'STRAIGHT_THROUGH': (255, 180, 60),
}


def _draw_hud(frame, snap):
    """Overlay the agent's live perception + decision on the camera frame, so the
    bot URL is a debug screen: boxes for detected signs/objects, current FSM
    state, lane error, light/red-line/obstacle flags, and WHY it is stopped."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- detections drawn on the scene ---
    for t in snap.get('tags', []):
        cx, cy = t.get('center_xy', [w // 2, h // 2])
        s = max(12, int(t.get('side_px', 30))) // 2
        cv2.rectangle(frame, (cx - s, cy - s), (cx + s, cy + s), (0, 255, 255), 2)
        d = t.get('est_distance_m')
        label = f"{t['id']}:{t.get('meaning', '?')}" + (f" {d:.2f}m" if d else "")
        cv2.putText(frame, label, (cx - s, cy - s - 6), font, 0.5, (0, 255, 255), 1)
    for o in snap.get('obstacles', []):
        x1, y1, x2, y2 = o['bbox']
        name, col = _OBJ.get(o.get('cls', -1), ('obj', (200, 200, 200)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"{name} {o.get('score', 0):.2f}", (x1, max(12, y1 - 5)), font, 0.5, col, 1)

    # --- top banner: STATE + reason ---
    st = snap.get('state', '?')
    col = _STATE_COLOR.get(st, (230, 230, 230))
    cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(frame, st, (8, 22), font, 0.7, col, 2)
    reason = _IDLE_REASON.get(st, '')
    if reason:
        cv2.putText(frame, reason, (120, 21), font, 0.5, (200, 200, 200), 1)

    # --- bottom strip: lane error + flags ---
    cv2.rectangle(frame, (0, h - 56), (w, h), (0, 0, 0), -1)
    lane = snap.get('lane', {})
    err = float(lane.get('error', 0.0))
    bx, bw, by = 70, 180, h - 44
    cv2.putText(frame, "lane", (8, by + 12), font, 0.5, (220, 220, 220), 1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 14), (60, 60, 60), -1)
    mid = bx + bw // 2
    ex = int(np.clip(mid + err * bw / 2, bx, bx + bw))
    ecol = (80, 230, 80) if abs(err) < 0.15 else (60, 200, 255) if abs(err) < 0.4 else (60, 60, 255)
    cv2.line(frame, (mid, by), (mid, by + 14), (130, 130, 130), 1)
    cv2.circle(frame, (ex, by + 7), 6, ecol, -1)
    cv2.putText(frame, f"{err:+.2f}{'' if lane.get('detected') else ' NO-LANE'}",
                (bx + bw + 8, by + 12), font, 0.45, ecol, 1)

    lt = snap.get('light', {}) or {}
    flags = []
    if lt.get('color'):
        flags.append((f"light:{lt['color']}", (0, 0, 255) if lt['color'] == 'red' else
                      (0, 200, 200) if lt['color'] == 'yellow' else (0, 200, 0)))
    if snap.get('red_line'):
        flags.append(("RED LINE", (0, 0, 255)))
    if snap.get('obstacle_stop'):
        flags.append(("OBSTACLE", (0, 0, 255)))
    if snap.get('legal_turns'):
        flags.append(("turns:" + ",".join(snap['legal_turns']), (200, 200, 60)))
    fx = 8
    for text, fcol in flags:
        cv2.putText(frame, text, (fx, h - 8), font, 0.5, fcol, 2)
        fx += 12 + len(text) * 11
    return frame


def _waiting_frame(msg="Waiting for agent camera..."):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, msg, (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2)
    return blank


def generate_frames():
    """MJPEG stream built ONLY from the agent's frames (_latest_frame) — this
    server never reads the camera itself, so it can't contend with / hang the
    agent's camera.read(). HUD overlaid from the latest agent snapshot."""
    while True:
        frame = _latest_frame
        if frame is None:
            disp = _waiting_frame()
        else:
            disp = frame.copy()
            snap = _latest_snap
            if snap is not None:
                try:
                    disp = _draw_hud(disp, snap)
                except Exception as e:
                    cv2.putText(disp, f"HUD err: {e}", (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        ok, jpeg = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(1.0 / 15)   # ~15 fps view; agent loop is unaffected


@app.route('/')
def index():
    return Response(_PAGE, mimetype='text/html')


@app.route('/telemetry')
def telemetry():
    """Latest agent perception + decision (state, tags, light, red line, obstacle,
    lane, legal turns). Empty until the agent's first loop."""
    return jsonify(_latest_snap or {})


# ----------------------------------------------------------------- control routes
@app.route('/manual', methods=['POST'])
def manual():
    """Enter/leave manual drive. The agent keeps running (perceiving + logging); its
    wheel writes are simply overridden by the manual command while on."""
    data = request.get_json(force=True, silent=True) or {}
    with _manual_lock:
        _manual['on'] = bool(data.get('on', False))
        _manual['left'] = _manual['right'] = 0.0
        _manual['last_cmd'] = time.time()
    return jsonify({'status': 'ok', 'manual': _manual['on']})


@app.route('/drive', methods=['POST'])
def drive():
    """Store the latest manual wheel command (the agent applies it — see _manual_cmd).
    409 if not in manual mode."""
    if not _manual['on']:
        return jsonify({'status': 'error', 'msg': 'not in manual mode; POST /manual {"on":true} first'}), 409
    data = request.get_json(force=True, silent=True) or {}
    try:
        left  = max(-_MANUAL_MAX, min(_MANUAL_MAX, float(data.get('left', 0.0))))
        right = max(-_MANUAL_MAX, min(_MANUAL_MAX, float(data.get('right', 0.0))))
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'msg': 'left/right must be numbers'}), 400
    with _manual_lock:
        _manual['left'], _manual['right'] = left, right
        _manual['last_cmd'] = time.time()
    return jsonify({'status': 'ok', 'left': left, 'right': right})


@app.route('/config', methods=['GET'])
def get_config():
    return jsonify(_load_all_config())


@app.route('/config', methods=['POST'])
def post_config():
    """Save edited values to the bot YAMLs. ?restart=1 (default) restarts the agent
    so timings/gains take effect; ?restart=0 only writes + applies lane HSV live."""
    data = request.get_json(force=True, silent=True) or {}
    changed = []
    for section, values in data.items():
        if section in CONFIG_FILES and isinstance(values, dict):
            try:
                _save_config_section(section, values)
                changed.append(section)
            except Exception as e:
                return jsonify({'status': 'error', 'msg': f'{section}: {e}'}), 500
    # lane HSV can apply live without a restart (module-global bounds)
    try:
        _apply_bot_lane_hsv()
    except Exception:
        pass
    restart = request.args.get('restart', '1') == '1'
    if restart:
        _restart_agent()
    return jsonify({'status': 'ok', 'saved': changed, 'agent_restarted': restart})


@app.route('/restart', methods=['POST'])
def restart_route():
    _restart_agent()
    return jsonify({'status': 'ok'})


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/raw')
def raw():
    """The most recent RAW camera frame, NO HUD overlay — for off-board perception
    tuning (lane/duck/red-line HSV) where the HUD boxes + blacked-out strips would
    pollute the pixels. Single JPEG, not a stream."""
    frame = _latest_frame
    if frame is None:
        frame = _waiting_frame()
    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return ('', 503)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/shutdown')
def shutdown():
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({'status': 'ok'})


# ============================================================ dashboard page (bot)
_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Duckiebot — project (real bot)</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#15171c;color:#e6e6e6}
 header{background:#0e1014;padding:10px 16px;font-size:17px;font-weight:600;border-bottom:1px solid #2a2d34}
 .wrap{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
 .col{flex:1 1 460px;min-width:340px}
 .card{background:#1d2027;border:1px solid #2a2d34;border-radius:8px;padding:12px;margin-bottom:16px}
 .card h2{margin:0 0 8px;font-size:13px;color:#7fd1b9;text-transform:uppercase;letter-spacing:.05em}
 img{width:100%;border-radius:6px;background:#000;display:block}
 .legend{font-size:12px;color:#9aa0aa;margin-top:6px}
 .sec{border:1px solid #2a2d34;border-radius:6px;margin-bottom:10px}
 .sec>summary{cursor:pointer;padding:8px 10px;font-weight:600;color:#cdd3dc}
 .grid{display:grid;grid-template-columns:1fr 92px;gap:6px 8px;padding:8px 10px}
 .grid label{font-size:13px;align-self:center;color:#b9c0ca;word-break:break-word}
 .grid input{width:86px;background:#0e1014;border:1px solid #333;color:#e6e6e6;border-radius:4px;padding:4px}
 button{background:#2d7;border:0;color:#06210f;font-weight:700;padding:9px 14px;border-radius:6px;cursor:pointer;font-size:14px}
 button.alt{background:#39c;color:#04121f}
 button.warn{background:#e85;color:#2a0d04}
 .dpad{display:grid;grid-template-columns:repeat(3,62px);gap:6px;justify-content:center;margin-top:10px}
 .dpad button{padding:0;height:50px;font-size:20px;background:#2a2d34;color:#e6e6e6;border:1px solid #3a3d44}
 .dpad button.stop{background:#933;color:#fff}
 .bar{display:flex;align-items:center;gap:8px;margin-top:6px;flex-wrap:wrap}
 #msg,#drvmsg{font-size:13px;color:#7fd1b9}
 b.k{color:#cdd3dc;font-weight:600}
</style></head><body>
<header>🦆 Duckiebot — project task (REAL BOT) — live debug + manual drive + tuning</header>
<div class="wrap">
  <div class="col">
    <div class="card"><h2>Camera (agent view + HUD)</h2><img src="/video">
      <div class="legend">Boxes = detected signs/objects · top banner = FSM state + why it's stopped ·
        bottom strip = lane error + flags. <a href="/raw" style="color:#39c">/raw</a> = clean frame for HSV tuning.</div></div>
    <div class="card"><h2>Drive control</h2>
      <div class="bar">
        <button id="manualBtn" class="alt" onclick="toggleManual()">Take manual control</button>
        <button onclick="restartAgent()">Restart agent</button>
        <span id="drvmsg">agent driving</span>
      </div>
      <div id="dpad" class="dpad" style="display:none">
        <span></span><button data-l="0.4" data-r="0.4">&#9650;</button><span></span>
        <button data-l="-0.36" data-r="0.36">&#9664;</button>
        <button class="stop" data-l="0" data-r="0">&#9632;</button>
        <button data-l="0.36" data-r="-0.36">&#9654;</button>
        <span></span><button data-l="-0.4" data-r="-0.4">&#9660;</button><span></span>
      </div>
      <div class="legend" id="drvhint" style="display:none">Keyboard <b class="k">W/A/S/D</b> or arrows · <b class="k">Space</b>=stop.
        Hold to move, release to stop. The agent is paused (wheels only) while you hold manual control.</div>
    </div>
  </div>
  <div class="col">
    <div class="card"><h2>Live bot status</h2>
      <div id="status" style="font-size:14px;line-height:1.8">waiting for telemetry…</div>
    </div>
    <div class="card"><h2>Live config (tune on the bot)</h2>
      <div id="cfg">loading…</div>
      <div class="bar">
        <button onclick="save(true)">Save &amp; Apply (restart agent)</button>
        <button class="alt" onclick="save(false)">Save (lane-HSV live, no restart)</button>
        <span id="msg"></span>
      </div>
      <div class="legend">Open <b class="k">lane_hsv_bot</b> / <b class="k">lane_servoing_bot</b> and watch the camera.
        HSV changes apply live; gains/speeds/sign-gates need "Save &amp; Apply". Edits persist to the bot's YAMLs.</div>
    </div>
  </div>
</div>
<script>
let CFG={};
function render(){let h='';for(const sec in CFG){h+='<details class="sec"'+(sec.indexOf('hsv')>=0||sec.indexOf('servoing')>=0?' open':'')+'><summary>'+sec+'</summary><div class="grid">';for(const k in CFG[sec]){if(k==='_error'){h+='<label style="color:#e66">'+CFG[sec][k]+'</label><span></span>';continue;}h+='<label>'+k+'</label><input data-s="'+sec+'" data-k="'+k+'" value="'+CFG[sec][k]+'">';}h+='</div></details>';}document.getElementById('cfg').innerHTML=h;}
async function load(){try{CFG=await(await fetch('/config')).json();render();}catch(e){}}
async function save(restart){const body={};document.querySelectorAll('#cfg input').forEach(i=>{const s=i.dataset.s,k=i.dataset.k;(body[s]=body[s]||{})[k]=i.value;});document.getElementById('msg').textContent='saving…';try{const r=await fetch('/config?restart='+(restart?1:0),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();document.getElementById('msg').textContent='saved '+(j.saved||[]).join(', ')+(j.agent_restarted?' · agent restarted':' · live');}catch(e){document.getElementById('msg').textContent='save failed';}}
// ---- manual drive (bot-safe: agent applies the command) ----
let manualOn=false,hb=null,held=new Set(),btn=null;
function clamp(x){return Math.max(-0.6,Math.min(0.6,x));}
function compute(){if(btn)return btn;const t=(held.has('f')?1:0)-(held.has('b')?1:0);const s=(held.has('r')?1:0)-(held.has('l')?1:0);return [clamp(t*0.4+s*0.36),clamp(t*0.4-s*0.36)];}
async function sendDrive(l,r){try{await fetch('/drive',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({left:l,right:r})});}catch(e){}}
function pump(){if(!manualOn)return;const c=compute();sendDrive(c[0],c[1]);}
async function toggleManual(){manualOn=!manualOn;try{await fetch('/manual',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({on:manualOn})});}catch(e){}
 document.getElementById('manualBtn').textContent=manualOn?'Release (resume agent)':'Take manual control';
 document.getElementById('dpad').style.display=manualOn?'grid':'none';
 document.getElementById('drvhint').style.display=manualOn?'block':'none';
 document.getElementById('drvmsg').textContent=manualOn?'MANUAL — agent paused (wheels)':'agent driving';
 if(manualOn){if(!hb)hb=setInterval(pump,120);}else{if(hb){clearInterval(hb);hb=null;}held.clear();btn=null;}}
async function restartAgent(){manualOn=false;if(hb){clearInterval(hb);hb=null;}held.clear();btn=null;
 document.getElementById('manualBtn').textContent='Take manual control';document.getElementById('dpad').style.display='none';document.getElementById('drvhint').style.display='none';
 document.getElementById('drvmsg').textContent='restarting agent…';try{await fetch('/restart',{method:'POST'});}catch(e){}
 setTimeout(()=>{document.getElementById('drvmsg').textContent='agent driving';load();},1500);}
const KEYMAP={KeyW:'f',ArrowUp:'f',KeyS:'b',ArrowDown:'b',KeyA:'l',ArrowLeft:'l',KeyD:'r',ArrowRight:'r'};
document.addEventListener('keydown',e=>{if(e.target&&(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'))return;if(!manualOn)return;if(e.code==='Space'){held.clear();btn=null;sendDrive(0,0);e.preventDefault();return;}const k=KEYMAP[e.code];if(!k)return;held.add(k);e.preventDefault();pump();});
document.addEventListener('keyup',e=>{if(!manualOn)return;const k=KEYMAP[e.code];if(!k)return;held.delete(k);if(held.size===0&&!btn)sendDrive(0,0);else pump();});
document.querySelectorAll('#dpad button').forEach(b=>{const l=parseFloat(b.dataset.l),r=parseFloat(b.dataset.r);const dn=ev=>{ev.preventDefault();if(!manualOn)return;btn=[l,r];sendDrive(l,r);};const up=ev=>{ev.preventDefault();btn=null;if(manualOn&&held.size===0)sendDrive(0,0);};b.addEventListener('mousedown',dn);b.addEventListener('mouseup',up);b.addEventListener('mouseleave',up);b.addEventListener('touchstart',dn);b.addEventListener('touchend',up);});
// ---- live status ----
function fmtTags(ts){if(!ts||!ts.length)return '<span style="color:#666">none</span>';return ts.map(t=>t.id+':'+t.meaning+(t.side_px?(' '+t.side_px+'px'):'')+(t.est_distance_m!=null?(' @'+t.est_distance_m.toFixed(2)+'m'):'')).join(' · ');}
const SC={DRIVE:'#2d7',APPROACH:'#fc3',STOPPED:'#e85',WAIT:'#e85',SOFT_STOP:'#e66',TURN_LEFT:'#39c',TURN_RIGHT:'#39c',STRAIGHT_THROUGH:'#39c'};
setInterval(async()=>{let d;try{d=await(await fetch('/telemetry')).json();}catch(e){return;}if(!d||!d.state)return;const l=d.light||{},lane=d.lane||{};
 document.getElementById('status').innerHTML=
  'state <b style="color:'+(SC[d.state]||'#fff')+'">'+d.state+'</b>'+(d.event?(' <span style="color:#9aa0aa">(via '+d.event+')</span>'):'')
  +' · speed '+(d.current_speed!=null?d.current_speed.toFixed(2):'?')
  +'<br>signs: '+fmtTags(d.tags)
  +'<br>lane err <b class="k">'+(lane.error!=null?lane.error.toFixed(2):'?')+'</b>'+(lane.detected?'':' <span style="color:#e66">NO-LANE</span>')
  +' · wheels L/R '+(d.wheels?d.wheels.left.toFixed(2)+'/'+d.wheels.right.toFixed(2):'?')
  +'<br>red line <b style="color:'+(d.red_line?'#e55':'#666')+'">'+(d.red_line?'AT LINE':'no')+'</b>'
  +' ('+(d.red_line_px!=null?d.red_line_px:'?')+'/'+(d.red_line_thr!=null?d.red_line_thr:'?')+'px)'
  +' · band '+(d.red_band?'<b style="color:#e55">yes</b>':'no')
  +'<br>obstacle <b style="color:'+(d.obstacle_stop?'#e55':'#666')+'">'+(d.obstacle_stop?'STOP':'clear')+'</b>'
  +(d.bot_ahead?' · <b style="color:#e55">bot ahead</b>':'')
  +' · light '+(l.color?('<b>'+l.color+'</b>'+(l.armed?' (armed)':'')):'—')
  +(d.legal_turns?('<br>legal turns: <b class="k">'+d.legal_turns.join(', ')+'</b>'):'')
  +(d.chosen_turn?(' · chose: <b class="k">'+d.chosen_turn+'</b>'):'')
  +(d.junction_cross?(' · crossing: <b style="color:#39c">'+d.junction_cross+'</b>'):'');
},500);
load();
</script></body></html>"""


def main():
    global camera, wheels, leds, stop_event

    ap = argparse.ArgumentParser(description='Project Server — Real Hardware')
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT SERVER — REAL HARDWARE')
    print('=' * 60)

    print('\n[1/4] Initializing LED driver...')
    try:
        leds = LEDDriver()
        leds.all_off()
        print('  LEDs: ok')
    except Exception as e:
        print(f'  LEDs: not available ({e})')
        leds = None

    print('\n[2/4] Initializing wheels driver...', flush=True)
    try:
        wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
    except Exception as e:
        print(f'  FATAL: wheels driver init failed: {e!r}', flush=True)
        traceback.print_exc()
        return 1
    print('  Wheels: ok', flush=True)

    print('\n[3/4] Initializing camera driver...', flush=True)
    # If this is the LAST line in the task log, startup hung inside camera.start()
    # (nvargus not ready / a stale capture session). Fix on the bot:
    # `sudo systemctl restart nvargus-daemon` (or reboot), then start ONCE.
    try:
        camera = CameraDriver()
        camera.start()
    except Exception as e:
        print(f'  FATAL: camera init failed: {e!r}', flush=True)
        traceback.print_exc()
        return 1
    print('  Camera: ok', flush=True)

    print('\n[4/4] Starting agent...', flush=True)
    # _start_agent loads bot config + applies lane HSV + spawns the agent thread
    # (with the bot-safe manual_cmd hook). It's the SAME path the dashboard's
    # Restart/Save-&-Apply use, so a restart picks up edited config identically.
    overlay_keys = _start_agent()
    print(f"  bot timing overrides: {overlay_keys or 'none'}", flush=True)
    print(f"  bot lane config: {_bot_lane_config_path() or 'default (sim-tuned!)'}", flush=True)
    print('  agent.main() running (manual drive + live config available on the dashboard)', flush=True)

    def _shutdown(signum, frame):
        print('\nShutting down...')
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nVideo stream: http://localhost:{web_port}/video')
    print('Press Ctrl+C to stop\n')

    try:
        app.run(host='0.0.0.0', port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
