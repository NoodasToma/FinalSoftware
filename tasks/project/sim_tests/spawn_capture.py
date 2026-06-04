"""TEMP: capture the SPAWN moment. Launch Godot, run the REAL agent from t=0,
save a camera frame every 0.2s AND log heading+state, so we SEE exactly what the
bot does in the first seconds (the part every prior harness missed)."""
from __future__ import annotations
import json, os, struct, subprocess, sys, tempfile, threading, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)
import cv2
from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
GODOT = os.path.join(os.path.expanduser("~"), ".cache", "duckietown", "godot", "Godot_v4.6-stable_win64.exe")
GP = os.path.join(_ROOT, "GodotSimulation", "ducky-bot")
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0
SAVE = os.path.join(_ROOT, "_sim_logs", "spawn"); os.makedirs(SAVE, exist_ok=True)
for f in os.listdir(SAVE):
    try: os.remove(os.path.join(SAVE, f))
    except OSError: pass

tmp = tempfile.mkdtemp(prefix="spawn_"); pf = os.path.join(tmp, "ports.json")
g = subprocess.Popen([GODOT, "--path", GP, "res://scenes/maps/project.tscn", "--",
                      "--camera-port=5001", "--wheel-port=5002", f"--port-file={pf}"],
                     cwd=GP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    wp = 5002; dl = time.time() + 20
    while time.time() < dl:
        if os.path.isfile(pf):
            try: wp = int(json.load(open(pf)).get("wheel_port", 5002)); break
            except Exception: pass
        time.sleep(0.3)
    cam = GodotCameraDriver(godot_config=GodotCameraConfig(host="0.0.0.0", port=5001)); cam.start()
    wheels = GodotWheelsDriver(WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0), godot_host="localhost", godot_port=wp)

    cur = {"state": "DRIVE"}
    import tasks.project.packages.agent as agent
    _rn = agent.next_state
    def _tn(s, ev):
        ns = _rn(s, ev)
        if ns != s: cur["state"] = ns.name; print(f"  {time.time()-t0:5.2f}s {s.name} --{ev}--> {ns.name}")
        return ns
    agent.next_state = _tn

    se = threading.Event()
    t0 = time.time()
    th = threading.Thread(target=agent.main, args=(cam, wheels, None, se), daemon=True); th.start()
    print("SPAWN CAPTURE (state transitions):")
    i = 0; nxt = 0.0
    while time.time() - t0 < SECONDS:
        el = time.time() - t0
        if el >= nxt:
            ok, f = cam.read()
            if ok and f is not None:
                cv2.imwrite(os.path.join(SAVE, f"s{i:03d}_{el:04.1f}s_{cur['state']}.png"), f); i += 1
            nxt = el + 0.2
        time.sleep(0.03)
    se.set(); th.join(timeout=5)
    print(f"saved {i} frames -> {SAVE}")
finally:
    try: g.terminate(); g.wait(timeout=4)
    except Exception: g.kill()
