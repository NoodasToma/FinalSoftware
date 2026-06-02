"""
Agent state/command tracer for the project simulation.

Launches Godot ALONE, then runs the REAL tasks.project.packages.agent.main()
against it with two lightweight instruments:
  * agent.next_state is wrapped to log every (state, event) -> new_state
    transition, which reveals exactly what triggers a state change.
  * the wheels driver is wrapped to record every set_wheels_speed() command.

The bot froze under the full sim even though manual_drive proves the Godot bot
moves at agent-magnitude speeds, so the agent must be leaving DRIVE. This shows
which event causes it and what it commands afterwards.

    .venv311/Scripts/python.exe tasks/project/sim_tests/agent_trace.py --seconds 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter

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


class WheelLogger:
    """Delegates to the real driver but records every command."""
    def __init__(self, inner):
        self._inner = inner
        self.encoders = getattr(inner, "encoders", None)
        self.cmds = []
        self.t0 = time.time()
    def set_wheels_speed(self, left, right):
        self.cmds.append((time.time() - self.t0, round(left, 4), round(right, 4)))
        return self._inner.set_wheels_speed(left, right)
    def __getattr__(self, name):
        return getattr(self._inner, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--camera-port", type=int, default=5001)
    ap.add_argument("--wheel-port", type=int, default=5002)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="ducky_trace_")
    port_file = os.path.join(tmp, "ports.json")
    cmd = [GODOT, "--path", GODOT_PROJECT, SCENE, "--",
           f"--camera-port={args.camera_port}", f"--wheel-port={args.wheel_port}",
           f"--port-file={port_file}"]
    print("[trace] launching Godot...")
    godot = subprocess.Popen(cmd, cwd=GODOT_PROJECT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wheel_port = _wait_for_wheel_port(port_file, args.wheel_port)
        cam = GodotCameraDriver(godot_config=GodotCameraConfig(host="0.0.0.0", port=args.camera_port))
        cam.start()
        wheels = WheelLogger(GodotWheelsDriver(
            WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
            godot_host="localhost", godot_port=wheel_port))

        import tasks.project.packages.agent as agent

        transitions = []
        _real_next = agent.next_state

        def _traced_next(state, event):
            ns = _real_next(state, event)
            if ns != state:
                transitions.append((round(time.time() - wheels.t0, 2),
                                    state.name, event, ns.name))
            return ns
        agent.next_state = _traced_next

        stop_event = threading.Event()
        th = threading.Thread(target=agent.main, args=(cam, wheels, None, stop_event),
                              daemon=True, name="TracedAgent")
        th.start()
        time.sleep(args.seconds)
        stop_event.set()
        th.join(timeout=5)

        print("\n" + "=" * 60)
        print("AGENT TRACE RESULT")
        print("=" * 60)
        print(f"observed {len(wheels.cmds)} wheel commands over {args.seconds:.0f}s")
        print("\nstate transitions (t, from, event, to):")
        if transitions:
            for t, frm, ev, to in transitions[:40]:
                print(f"  {t:6.2f}s  {frm:>16} --{ev}--> {to}")
        else:
            print("  (none - stayed in initial DRIVE state the whole time)")
        # Command fingerprint: how often each rounded (l,r) was issued.
        fp = Counter((l, r) for _, l, r in wheels.cmds)
        print("\nmost common commands (left,right : count):")
        for (l, r), c in fp.most_common(8):
            print(f"  ({l:+.3f},{r:+.3f}) x{c}")
        if wheels.cmds:
            print("\nfirst 12 commands:")
            for t, l, r in wheels.cmds[:12]:
                print(f"  {t:6.2f}s  ({l:+.3f},{r:+.3f})")
            mx = max(abs(l) for _, l, r in wheels.cmds) if wheels.cmds else 0
            my = max(abs(r) for _, l, r in wheels.cmds) if wheels.cmds else 0
            print(f"\nmax |left|={mx:.3f}  max |right|={my:.3f}")
        print("=" * 60)
        return 0
    finally:
        try:
            godot.terminate(); godot.wait(timeout=4)
        except Exception:
            godot.kill()


if __name__ == "__main__":
    sys.exit(main())
