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

from flask import Flask, Response, jsonify
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


# YOLO object_detection class ids -> label/colour (BGR), matching the model.
_OBJ = {0: ('duckie', (0, 215, 255)), 1: ('truck', (180, 100, 220)), 2: ('sign', (50, 205, 50))}

# Why the FSM is holding still, by state — shown big on the HUD so "drives then
# stands idle" is self-explanatory instead of a mystery.
_IDLE_REASON = {
    'STOPPED':   'stopped at line (pausing / deciding turn)',
    'WAIT':      'waiting (red light or yielding to a robot)',
    'SOFT_STOP': 'OBSTACLE ahead - holding until it clears',
    'APPROACH':  'approaching a sign - slowing to the line',
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
    # Debug homepage: the live camera with the agent's perception + decision
    # overlaid (same info as the sim dashboard) so you can SEE what it detects
    # and why it stops. Auto-refreshing /telemetry text below the image.
    return ("<!doctype html><title>project - DuckieBot</title>"
            "<body style='margin:0;background:#111;color:#ddd;font-family:sans-serif;text-align:center'>"
            "<h3 style='padding:6px;margin:0'>project task - live debug view</h3>"
            "<img src='/video' style='max-width:100%;height:auto'>"
            "<pre id='t' style='text-align:left;max-width:680px;margin:8px auto;color:#9fe'></pre>"
            "<script>setInterval(async()=>{try{let d=await(await fetch('/telemetry')).json();"
            "document.getElementById('t').textContent="
            "'state: '+d.state+'   lane err: '+(d.lane?d.lane.error:'?')+'   light: '+((d.light||{}).color||'-')"
            "+'   red_line: '+d.red_line+'   obstacle: '+d.obstacle_stop+'\\n'"
            "+'signs: '+JSON.stringify((d.tags||[]).map(t=>t.id+':'+t.meaning))"
            "+'\\nobjects: '+JSON.stringify((d.obstacles||[]).map(o=>o.cls));}catch(e){}},500);</script>"
            "<p style='opacity:.6'>/video annotated stream &middot; /telemetry json &middot; /shutdown to stop</p>"
            "</body>")


@app.route('/telemetry')
def telemetry():
    """Latest agent perception + decision (state, tags, light, red line, obstacle,
    lane, legal turns). Empty until the agent's first loop."""
    return jsonify(_latest_snap or {})


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
    timings, overlay_keys = _load_bot_timings()
    lane_cfg = _bot_lane_config_path()
    hsv_keys = _apply_bot_lane_hsv()
    print(f"  bot timing overrides: {overlay_keys or 'none'}", flush=True)
    print(f"  bot lane config: {lane_cfg or 'default (sim-tuned!)'}", flush=True)
    print(f"  bot lane HSV overrides: {hsv_keys or 'none'}", flush=True)
    stop_event.clear()

    def _run_agent():
        # agent.main runs in a daemon thread; if it raises (a setup error such as
        # a failed model load, or an unhandled per-frame error) the thread would
        # otherwise die SILENTLY — wheels left at zero and Flask still serving
        # stale telemetry, so the bot just looks "frozen at startup" with no error
        # anywhere visible. Catch + print the full traceback (it lands in the bot
        # daemon's task log) and zero the wheels, so the failure is diagnosable.
        try:
            agent.main(camera, wheels, leds, stop_event,
                       timings_override=timings, lane_config_path=lane_cfg,
                       observer=_observe, frame_observer=_frame_observe)
        except Exception as e:
            print('=' * 60, flush=True)
            print(f'AGENT THREAD CRASHED: {e!r}', flush=True)
            traceback.print_exc()
            print('=' * 60, flush=True)
            try:
                wheels.set_wheels_speed(0.0, 0.0)
            except Exception:
                pass

    threading.Thread(target=_run_agent, daemon=True, name='AgentThread').start()
    print('  agent.main() running', flush=True)

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
