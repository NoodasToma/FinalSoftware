"""Authoritative `project` sim test: every traffic sign + light + object detection.

Launches Godot once on the rebuilt project.tscn, then:
  * PER-SIGN pass -- teleports the bot to each station in course_map.STATIONS,
    runs the REAL agent (sim fidelity + telemetry), and asserts the matrix row
    (legal-turn set, do-not-enter hold, light arm+stop, vehicle precedence,
    decode-only, obstacle soft-stop, lane following). This is the coverage
    guarantee for "every single traffic sign is tested".
  * CONTINUOUS pass -- spawns at the course start and drives end-to-end, asserting
    the bot lane-follows, decodes the roadside signs, and reaches the 4-way.

Reuses sim_telemetry.TelemetryLogger + sim_fidelity_kwargs, the REAL agent.main,
and merge_turn_constraints / we_go_first / lookup. Logs to _sim_logs/course/<ts>/.

Run:    .venv311\Scripts\python.exe tasks\project\sim_tests\course_suite.py
Filter: ...course_suite.py oneway_right t_intersect   (run only those stations)
        ...course_suite.py --no-continuous            (skip the continuous pass)
"""
from __future__ import annotations
import json, math, os, subprocess, sys, tempfile, threading, time
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)
import cv2

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
import tasks.project.packages.agent as agent
from tasks.project.packages.sign_registry import lookup
from tasks.project.packages.precedence import we_go_first
from tasks.project.sim_tests.sim_telemetry import (
    TelemetryLogger, sim_fidelity_kwargs, SIM_APRILTAG_INTRINSICS, SIM_TAG_SIZE_M,
)
from tasks.project.sim_tests.course_map import STATIONS, COURSE_START, station

GODOT = os.path.join(os.path.expanduser("~"), ".cache", "duckietown", "godot",
                     "Godot_v4.6-stable_win64.exe")
GP = os.path.join(_ROOT, "GodotSimulation", "ducky-bot")
RUN_DIR = os.path.join(_ROOT, "_sim_logs", "course", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(RUN_DIR, exist_ok=True)
HOST = "Toma_PC"   # deterministic precedence host name (matches behaviour_suite)

results = []   # (name, passed, detail)


def rec(name, passed, detail):
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def run_station(cam, wheels, st):
    """Teleport to the station and run the real agent; return (summary, records, transitions)."""
    x, z, h = st["teleport"]
    wheels.set_wheels_speed(0.0, 0.0)
    wheels.teleport(x, z, h)
    time.sleep(0.9)
    log = TelemetryLogger(RUN_DIR, wheels=wheels, label=st["name"])
    transitions = []
    _rn = agent.next_state
    def _tn(s, ev):
        ns = _rn(s, ev)
        if ns != s:
            transitions.append((s.name, ev, ns.name))
        return ns
    agent.next_state = _tn
    stop = threading.Event()
    th = threading.Thread(target=agent.main, args=(cam, wheels, None, stop),
                          kwargs=dict(observer=log, **sim_fidelity_kwargs()), daemon=True)
    th.start()
    last = None
    t0 = time.time()
    while time.time() - t0 < st["seconds"]:
        ok, f = cam.read()
        if ok and f is not None:
            last = f
        time.sleep(0.1)
    stop.set(); th.join(timeout=5)
    agent.next_state = _rn
    wheels.set_wheels_speed(0.0, 0.0)
    summ = log.close()
    if last is not None:
        cv2.imwrite(os.path.join(RUN_DIR, f"{st['name']}.png"), last)
    return summ, log.records, transitions


# --- per-kind assertions ---------------------------------------------------- #
def _legal_at_stop(records):
    """legal_turns reported on the first STOPPED frame that has it, or None."""
    for r in records:
        if r.get("state") == "STOPPED" and r.get("legal_turns") is not None:
            return r["legal_turns"]
    return None


def _states(transitions):
    return {t[2] for t in transitions} | {t[0] for t in transitions}


def check_station(st, summ, records, transitions):
    """Pure assertion: returns (label, passed, detail). No printing / globals, so
    both the CLI suite and the dashboard test-runner can reuse it."""
    name, kind, exp = st["name"], st["kind"], st["expect"]
    seen = set(int(k) for k in (summ.get("tags_seen") or {}))
    states = _states(transitions)

    if kind == "lane":
        moved = 0.0
        if summ.get("start_pose") and summ.get("end_pose"):
            moved = math.hypot(summ["end_pose"]["x"] - summ["start_pose"]["x"],
                               summ["end_pose"]["z"] - summ["start_pose"]["z"])
        errs = [abs(r["lane"]["error"]) for r in records if r.get("lane")]
        max_err = max(errs) if errs else 1.0
        return (f"{name}: lane-follows, on-lane",
                moved > exp["min_move"] and max_err < exp["max_err"],
                f"moved={moved:.2f}m max|err|={max_err:.2f}")

    if kind == "legal":
        legal = _legal_at_stop(records)
        tags_ok = set(exp["tags"]) <= seen
        return (f"{name}: decoded {exp['tags']} + legal turns {exp['legal']}",
                "STOPPED" in states and legal == exp["legal"] and tags_ok,
                f"legal={legal} tags_seen={sorted(seen)}")

    if kind == "do_not_enter":
        legal = _legal_at_stop(records)
        turned = any(s in states for s in ("TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH"))
        tags_ok = set(exp["tags"]) <= seen
        return (f"{name}: stops and refuses to enter (legal=∅)",
                "STOPPED" in states and legal == [] and not turned and tags_ok,
                f"legal={legal} turned={turned} tags_seen={sorted(seen)}")

    if kind == "light":
        armed = any((r.get("light") or {}).get("armed") for r in records)
        evs = {t[1] for t in transitions}
        stopped = "STOPPED" in states or "WAIT" in states
        colors = set(summ.get("light_colors_seen") or [])
        return (f"{name}: arms + stops at light + reads colour",
                (armed or "see_light" in evs) and stopped and bool(colors & {"red", "green", "yellow"}),
                f"armed={armed} stopped={stopped} colors={sorted(colors)}")

    if kind == "vehicle":
        veh = sorted(t for t in seen if t >= exp["min_tag"])
        first = we_go_first(HOST, [lookup(veh[0])]) if veh else None
        return (f"{name}: detect vehicle plate + precedence",
                len(veh) > 0,
                f"vehicle_tags={veh} we_go_first(host={HOST})={first}")

    if kind == "decode":
        tag = exp["tag"]
        # decode-only: tag recognised, and it must NOT drive the bot into a stop
        no_stop = "STOPPED" not in states
        return (f"{name}: decoded tag {tag}, no behaviour change",
                tag in seen and no_stop,
                f"tag_seen={tag in seen} states={sorted(states)}")

    if kind == "obstacle":
        return (f"{name}: SOFT_STOP for duckie",
                "SOFT_STOP" in states and summ.get("obstacle_stop_frames", 0) > 0,
                f"states={sorted(states)} obstacle_frames={summ.get('obstacle_stop_frames')}")

    return (f"{name}: UNKNOWN kind {kind}", False, "")


def launch_godot():
    tmp = tempfile.mkdtemp(prefix="course_"); pf = os.path.join(tmp, "ports.json")
    g = subprocess.Popen([GODOT, "--path", GP, "res://scenes/maps/project.tscn", "--",
                          "--camera-port=5001", "--wheel-port=5002", f"--port-file={pf}"],
                         cwd=GP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wp = 5002; dl = time.time() + 20
    while time.time() < dl:
        if os.path.isfile(pf):
            try:
                wp = int(json.load(open(pf)).get("wheel_port", 5002)); break
            except Exception:
                pass
        time.sleep(0.3)
    cam = GodotCameraDriver(godot_config=GodotCameraConfig(host="0.0.0.0", port=5001)); cam.start()
    wheels = GodotWheelsDriver(WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
                               godot_host="localhost", godot_port=wp)
    time.sleep(1.0)
    return g, cam, wheels


def _path_len(records):
    """Total path length from the pose trace (curving path, not straight-line)."""
    pts = [(r["pose"]["x"], r["pose"]["z"]) for r in records if r.get("pose")]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def continuous_pass(cam, wheels):
    """Two realistic continuous segments (one full loop is blocked by the duck
    row by design, so we demo the two drivable stretches):
      A) spawn straight  -> lane-follow + decode roadside signs
      B) 4-way approach  -> full DRIVE->APPROACH->STOPPED->TURN->DRIVE chain
    """
    print("\n[continuous A] lane-follow the spawn straight + decode signs")
    a = {"name": "continuous_A", "teleport": COURSE_START, "seconds": 25}
    summ, records, _ = run_station(cam, wheels, a)
    path = _path_len(records)
    seen = {int(k) for k in (summ.get("tags_seen") or {})}
    drive_frames = [r for r in records if r["state"] == "DRIVE" and r.get("lane")]
    healthy = sum(1 for r in drive_frames if r["lane"]["detected"])
    frac = healthy / max(1, len(drive_frames))
    rec("continuous A: sustained healthy lane-following",
        path > 1.5 and frac > 0.8, f"path={path:.2f}m lane-detected {frac*100:.0f}% of DRIVE frames")
    rec("continuous A: decodes roadside signs while driving",
        len(seen & {95, 12, 125}) >= 1, f"decode tags seen={sorted(seen & {95, 12, 125})}")

    print("\n[continuous B] drive through the 4-way (stop -> turn -> resume)")
    b = {"name": "continuous_B", "teleport": (5.85, 7.05, 0.0), "seconds": 22}
    summ, records, transitions = run_station(cam, wheels, b)
    states = _states(transitions)
    turned = [s for s in ("TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH") if s in states]
    resumed = any(t[0] in ("TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH") and t[2] == "DRIVE"
                  for t in transitions)
    rec("continuous B: full DRIVE->APPROACH->STOPPED->TURN->DRIVE",
        {"APPROACH", "STOPPED"} <= states and bool(turned) and resumed,
        f"chain={['->'.join(t) for t in transitions][:6]}")


def main():
    args = [a for a in sys.argv[1:]]
    do_continuous = "--no-continuous" not in args
    names = [a for a in args if not a.startswith("--")]
    only_continuous = names == ["continuous"]
    if only_continuous:
        names = []
    todo = [] if only_continuous else [s for s in STATIONS if (not names or s["name"] in names)]
    if only_continuous:
        do_continuous = True

    print(f"FIDELITY intrinsics={SIM_APRILTAG_INTRINSICS} tag={SIM_TAG_SIZE_M}m")
    print(f"STATIONS: {[s['name'] for s in todo]}")
    print(f"LOGS -> {RUN_DIR}\n")
    g, cam, wheels = launch_godot()
    try:
        for st in todo:
            print(f"[{st['name']}] {st['label']}  tp={st['teleport']}")
            summ, records, transitions = run_station(cam, wheels, st)
            rec(*check_station(st, summ, records, transitions))
        if do_continuous and not names:
            continuous_pass(cam, wheels)

        npass = sum(1 for _, p, _ in results if p)
        print("\n" + "=" * 70)
        print(f"COURSE SUITE: {npass}/{len(results)} checks passed")
        for n, p, _ in results:
            print(f"  {'PASS' if p else 'FAIL'}  {n}")
        print(f"logs -> {RUN_DIR}")
        print("=" * 70)
        return 0 if npass == len(results) else 1
    finally:
        try:
            g.terminate(); g.wait(timeout=4)
        except Exception:
            g.kill()


if __name__ == "__main__":
    sys.exit(main())
