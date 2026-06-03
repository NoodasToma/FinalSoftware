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

import tasks.project.packages.agent as agent
from tasks.project.packages.perception import AprilTagDetector, TrafficLightDetector
from tasks.project.packages.sign_registry import lookup
from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.visual_lane_servoing.packages import visual_servoing_activity as lane_hsv
from tasks.object_detection.packages.agent import ObjectDetectionAgent, CLASS_NAMES, CLASS_COLORS


app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None

_agent_stop   = threading.Event()
_agent_thread = None

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
        self.tags = AprilTagDetector()
        self.light = TrafficLightDetector(); self.light.arm()
        self.obj = ObjectDetectionAgent()

    def reload_cheap(self):
        """Recreate the cheap detectors (picks up new YAMLs); keep the YOLO model."""
        try:
            self.lane = LaneServoingAgent()
            self.tags = AprilTagDetector()
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
    global _agent_stop, _agent_thread
    _agent_stop = threading.Event()
    _agent_thread = threading.Thread(
        target=agent.main, args=(camera, wheels, leds, _agent_stop),
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
    _restart_agent()
    return jsonify({'status': 'ok'})


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
load();
</script></body></html>"""


if __name__ == '__main__':
    sys.exit(main())
