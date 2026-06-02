"""
Manual-drive diagnostic for the project simulation.

Launches Godot ALONE on res://scenes/maps/project.tscn, becomes the sole wheel +
camera client (no agent in the loop), commands a fixed forward speed for a few
seconds, and measures whether the simulated bot actually moves. This isolates
"can the Godot bot move from wheel commands?" from "does the agent drive it?".

    .venv311/Scripts/python.exe tasks/project/sim_tests/manual_drive.py \
        --speed 0.5 --drive-seconds 6

Prints the start->end camera frame difference (large => the bot moved) and the
bot's reported distance_traveled, and saves start/end frames under _sim_logs/.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration

GODOT = os.path.join(os.path.expanduser("~"), ".cache", "duckietown", "godot",
                     "Godot_v4.6-stable_win64.exe")
GODOT_PROJECT = os.path.join(_PROJECT_ROOT, "GodotSimulation", "ducky-bot")
SCENE = "res://scenes/maps/project.tscn"


def _wait_for_wheel_port(port_file, hint, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.isfile(port_file):
            try:
                with open(port_file) as fh:
                    return int(json.load(fh).get("wheel_port", hint))
            except Exception:
                pass
        time.sleep(0.3)
    return hint


def _request_distance(wheels):
    """Send a get_state request and read back distance_traveled (best effort)."""
    t = wheels.transport
    if t._sock is None:
        t._ensure_connected()
    try:
        payload = json.dumps({"type": "get_state"}).encode()
        t._sock.sendall(struct.pack("!I", len(payload)) + payload)
        time.sleep(0.2)
        t._check_incoming()
    except Exception as e:
        print(f"  (get_state failed: {e})")
    return t.game_state.distance_traveled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--drive-seconds", type=float, default=6.0)
    ap.add_argument("--camera-port", type=int, default=5001)
    ap.add_argument("--wheel-port", type=int, default=5002)
    ap.add_argument("--save-dir", default=os.path.join(_PROJECT_ROOT, "_sim_logs"))
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="ducky_manual_")
    port_file = os.path.join(tmp, "ports.json")

    cmd = [GODOT, "--path", GODOT_PROJECT, SCENE, "--",
           f"--camera-port={args.camera_port}",
           f"--wheel-port={args.wheel_port}",
           f"--port-file={port_file}"]
    print("[manual] launching Godot:", " ".join(cmd))
    godot = subprocess.Popen(cmd, cwd=GODOT_PROJECT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wheel_port = _wait_for_wheel_port(port_file, args.wheel_port)
        print(f"[manual] wheel port = {wheel_port}")

        cam = GodotCameraDriver(godot_config=GodotCameraConfig(host="0.0.0.0", port=args.camera_port))
        cam.start()
        wheels = GodotWheelsDriver(WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
                                   godot_host="localhost", godot_port=wheel_port)

        # Let the scene settle and grab a stable starting frame.
        start_frame = None
        for _ in range(60):
            ok, f = cam.read()
            if ok and f is not None:
                start_frame = f.copy()
            time.sleep(0.03)
        dist0 = _request_distance(wheels)
        print(f"[manual] start distance={dist0:.3f}m; driving fwd={args.speed} for {args.drive_seconds}s")

        end_frame = start_frame
        t_end = time.time() + args.drive_seconds
        while time.time() < t_end:
            wheels.set_wheels_speed(args.speed, args.speed)   # straight forward
            ok, f = cam.read()
            if ok and f is not None:
                end_frame = f.copy()
            time.sleep(0.03)
        wheels.set_wheels_speed(0.0, 0.0)
        dist1 = _request_distance(wheels)

        diff = float(np.mean(cv2.absdiff(start_frame, end_frame))) if start_frame is not None else -1.0
        cv2.imwrite(os.path.join(args.save_dir, "manual_start.png"), start_frame)
        cv2.imwrite(os.path.join(args.save_dir, "manual_end.png"), end_frame)
        print("\n" + "=" * 56)
        print("MANUAL DRIVE RESULT")
        print("=" * 56)
        print(f"  distance_traveled: {dist0:.3f}m -> {dist1:.3f}m  (delta {dist1-dist0:+.3f}m)")
        print(f"  start->end frame mean abs diff: {diff:.2f}  (0=identical, >5 => view changed)")
        print(f"  verdict: {'BOT MOVED' if (dist1-dist0) > 0.05 or diff > 5 else 'BOT DID NOT MOVE'}")
        print("=" * 56)
        return 0
    finally:
        try:
            godot.terminate(); godot.wait(timeout=4)
        except Exception:
            godot.kill()


if __name__ == "__main__":
    sys.exit(main())
