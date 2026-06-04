"""Measure the ACTUAL rotation (degrees) a turn maneuver produces in the sim.

Why this exists: a turn that over-rotates by >360 degrees looks like a small net
heading change in a single before/after sample (515 deg aliases to 155 deg), and
it makes the bot "spin uncontrollably". Sparse camera frames hid this. This
harness owns the wheel socket alone (no agent, no contention) and INTEGRATES the
Godot bot's heading at ~50 Hz across each maneuver, so multi-revolution spins are
caught. It calls the REAL maneuvers.turn_left/right/straight_through.

    .venv311/Scripts/python.exe tasks/project/sim_tests/measure_turns.py

Pass = each 'turn' is ~90 deg (a quarter turn), not a multi-spin.
"""
from __future__ import annotations
import json, os, struct, subprocess, sys, tempfile, threading, time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from tasks.project.packages.maneuvers import turn_left, turn_right, straight_through

GODOT = os.path.join(os.path.expanduser("~"), ".cache", "duckietown", "godot", "Godot_v4.6-stable_win64.exe")
GP = os.path.join(_ROOT, "GodotSimulation", "ducky-bot")


def _norm(d):  # normalize a heading step into [-180, 180]
    while d > 180: d -= 360
    while d < -180: d += 360
    return d


def _read_heading(wheels):
    """Send get_state and return Godot heading_deg (single-thread, no contention)."""
    t = wheels.transport
    if t._sock is None:
        t._ensure_connected()
    try:
        payload = json.dumps({"type": "get_state"}).encode()
        t._sock.sendall(struct.pack("!I", len(payload)) + payload)
        time.sleep(0.015)
        t._check_incoming()
    except Exception:
        pass
    return wheels.transport.game_state.heading_deg


def _run_maneuver(wheels, fn, timings, label):
    """Run fn(wheels, stop_event, timings) in a thread while we integrate heading."""
    stop = threading.Event()
    th = threading.Thread(target=fn, args=(wheels, stop, timings), daemon=True)
    h_prev = _read_heading(wheels)
    total = 0.0
    th.start()
    t0 = time.time()
    while th.is_alive() and time.time() - t0 < 8.0:
        h = _read_heading(wheels)
        total += _norm(h - h_prev)
        h_prev = h
        time.sleep(0.02)
    th.join(timeout=2)
    dur = time.time() - t0
    print(f"  {label:16}: total rotation = {total:+7.1f} deg over {dur:4.2f}s")
    return total


def main():
    tmp = tempfile.mkdtemp(prefix="measure_"); pf = os.path.join(tmp, "ports.json")
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
        wheels = GodotWheelsDriver(WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
                                   godot_host="localhost", godot_port=wp)
        time.sleep(1.0)
        timings = {"turn_inner_speed": 0.05, "turn_outer_speed": 0.45, "turn_ticks": 90,
                   "turn_seconds": 0.45, "straight_speed": 0.3, "straight_ticks": 60,
                   "straight_seconds": 2.2}
        print(f"\nMEASURED MANEUVERS (turn_seconds={timings['turn_seconds']}, omega~4 rad/s):")
        results = {}
        for label, fn in [("turn_left", turn_left), ("straight_through", straight_through),
                          ("turn_right", turn_right)]:
            wheels.set_wheels_speed(0.0, 0.0); time.sleep(0.6)
            results[label] = _run_maneuver(wheels, fn, timings, label)
        wheels.set_wheels_speed(0.0, 0.0)

        print("\nVERDICT:")
        ok = True
        for label in ("turn_left", "turn_right"):
            mag = abs(results[label])
            good = 60 <= mag <= 140
            ok = ok and good
            print(f"  {label}: {mag:.0f} deg -> {'OK (~quarter turn)' if good else 'BAD (not ~90 deg!)'}")
        smag = abs(results["straight_through"])
        sgood = smag < 45
        ok = ok and sgood
        print(f"  straight_through: {smag:.0f} deg drift -> {'OK (goes straight)' if sgood else 'BAD (curving!)'}")
        print("\n" + ("ALL TURNS SANE" if ok else "TURNS STILL WRONG"))
        return 0 if ok else 1
    finally:
        try: g.terminate(); g.wait(timeout=4)
        except Exception: g.kill()


if __name__ == "__main__":
    sys.exit(main())
