"""Calibrate the sim camera intrinsics (fx) AND probe where each sign decodes.

Why: in sim we want a REAL est_distance_m so the bot runs the hardware
`est_distance < 0.25 m` stop-line path. est_distance scales linearly with the
focal length fx, so we measure it: place the bot a KNOWN distance from a tag,
read the detected side_px, and set fx so est_distance == true distance.

It also sweeps the bot in a circle around a sign to find which BEARING the tag
actually faces (the roadside signs are rotated 45 deg, so the bot can see the
*back*). Output tells us (a) the facing/approach side, (b) the calibrated fx.

Run:  PYTHONUTF8=1 .venv311/Scripts/python.exe tasks/project/sim_tests/calibrate_intrinsics.py
"""
from __future__ import annotations
import json, math, os, subprocess, sys, tempfile, time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)
from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from tasks.project.packages.perception import AprilTagDetector

GODOT = os.path.join(os.path.expanduser("~"), ".cache", "duckietown", "godot", "Godot_v4.6-stable_win64.exe")
GP = os.path.join(_ROOT, "GodotSimulation", "ducky-bot")

TAG_SIZE = 0.20
TRIAL_FX = 313.0
CAM_FWD_OFFSET = 0.08            # camera sits ~8 cm forward of the bot origin
# Scene tag (x, z) positions, from project.tscn (col9 east roadside, north-facing):
TAGS = {1: (6.0, 4.5), 8: (6.0, 4.0), 400: (6.0, 3.6), 74: (6.0, 3.2)}


def heading_facing(px, pz, sx, sz):
    """Heading (deg) so the bot's forward (-Z) points from (px,pz) to (sx,sz)."""
    dx, dz = sx - px, sz - pz
    n = math.hypot(dx, dz) or 1.0
    return math.degrees(math.atan2(-dx / n, -dz / n))


def cam_xz(px, pz, heading_deg):
    a = math.radians(heading_deg)
    fwd = (-math.sin(a), -math.cos(a))           # bot forward in world
    return (px + CAM_FWD_OFFSET * fwd[0], pz + CAM_FWD_OFFSET * fwd[1])


def grab(cam, n=6):
    f = None
    for _ in range(n):
        ok, fr = cam.read()
        if ok and fr is not None:
            f = fr
        time.sleep(0.05)
    return f


def detect_tag(det, frame, want_id):
    for o in (det.detect(frame) or []):
        if o.id == want_id:
            return o
    return None


def main():
    tmp = tempfile.mkdtemp(prefix="calib_"); pf = os.path.join(tmp, "ports.json")
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
        wheels = GodotWheelsDriver(WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
                                   godot_host="localhost", godot_port=wp)
        time.sleep(1.0)
        det = AprilTagDetector(tag_size_m=TAG_SIZE, intrinsics=(TRIAL_FX, TRIAL_FX, 320.0, 240.0))

        # ---- 1) BEARING SWEEP around the stop sign: where does tag 1 face? ----
        tid = 1; sx, sz = TAGS[tid]; R = 0.8
        print(f"\n=== BEARING SWEEP around tag {tid} @({sx},{sz}), R={R} ===")
        print(f"{'bearing':>7} {'botX':>6} {'botZ':>6} {'head':>6} {'side_px':>7} {'est_m':>6} {'true_m':>6}")
        best = None
        for deg in range(0, 360, 30):
            a = math.radians(deg)
            px, pz = sx + R * math.sin(a), sz + R * math.cos(a)
            head = heading_facing(px, pz, sx, sz)
            wheels.set_wheels_speed(0.0, 0.0); wheels.teleport(px, pz, head); time.sleep(0.6)
            o = detect_tag(det, grab(cam), tid)
            cx, cz = cam_xz(px, pz, head); true_m = math.hypot(sx - cx, sz - cz)
            if o:
                print(f"{deg:7d} {px:6.2f} {pz:6.2f} {head:6.0f} {o.side_length_px:7d} {o.est_distance_m:6.2f} {true_m:6.2f}")
                if best is None or o.side_length_px > best[1]:
                    best = (deg, o.side_length_px)
            else:
                print(f"{deg:7d} {px:6.2f} {pz:6.2f} {head:6.0f} {'--':>7} {'--':>6} {true_m:6.2f}")
        if best is None:
            print("!! tag 1 never decoded on the bearing sweep"); return 1
        best_deg = best[0]
        print(f"-> best bearing = {best_deg} deg (largest side_px). Calibrating fx there.")

        # ---- 2) RADIUS SWEEP at best bearing: calibrate fx ----
        print(f"\n=== RADIUS SWEEP at bearing {best_deg} deg ===")
        print(f"{'R':>5} {'side_px':>7} {'est_trial':>9} {'true_m':>6} {'fx_fit':>7}")
        a = math.radians(best_deg); fxs = []
        for R in (0.4, 0.55, 0.7, 0.9, 1.1):
            px, pz = sx + R * math.sin(a), sz + R * math.cos(a)
            head = heading_facing(px, pz, sx, sz)
            wheels.set_wheels_speed(0.0, 0.0); wheels.teleport(px, pz, head); time.sleep(0.6)
            o = detect_tag(det, grab(cam), tid)
            cx, cz = cam_xz(px, pz, head); true_m = math.hypot(sx - cx, sz - cz)
            if o and o.est_distance_m not in (0.0, float("inf")):
                fx_fit = TRIAL_FX * (true_m / o.est_distance_m)
                fxs.append(fx_fit)
                print(f"{R:5.2f} {o.side_length_px:7d} {o.est_distance_m:9.3f} {true_m:6.3f} {fx_fit:7.1f}")
            else:
                print(f"{R:5.2f} {'--':>7} {'--':>9} {true_m:6.3f} {'--':>7}")
        if fxs:
            fx = sum(fxs) / len(fxs)
            print(f"\n>>> CALIBRATED fx = {fx:.1f}  (mean of {len(fxs)} fits)")
            print(f">>> SIM_APRILTAG_INTRINSICS = ({fx:.1f}, {fx:.1f}, 320.0, 240.0)")
            # verification: est with calibrated fx vs true
            det2 = AprilTagDetector(tag_size_m=TAG_SIZE, intrinsics=(fx, fx, 320.0, 240.0))
            print(f"\n=== VERIFY with fx={fx:.1f} ===")
            print(f"{'R':>5} {'est_m':>6} {'true_m':>6} {'err%':>6}")
            for R in (0.45, 0.65, 0.85):
                px, pz = sx + R * math.sin(a), sz + R * math.cos(a)
                head = heading_facing(px, pz, sx, sz)
                wheels.set_wheels_speed(0.0, 0.0); wheels.teleport(px, pz, head); time.sleep(0.6)
                o = detect_tag(det2, grab(cam), tid)
                cx, cz = cam_xz(px, pz, head); true_m = math.hypot(sx - cx, sz - cz)
                if o:
                    err = 100.0 * (o.est_distance_m - true_m) / true_m
                    print(f"{R:5.2f} {o.est_distance_m:6.3f} {true_m:6.3f} {err:6.1f}")
        return 0
    finally:
        try: g.terminate(); g.wait(timeout=4)
        except Exception: g.kill()


if __name__ == "__main__":
    sys.exit(main())
