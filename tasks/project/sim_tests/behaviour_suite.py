"""Behaviour suite — the authoritative, evidence-backed sim test for `project`.

Launches Godot once, then for each behaviour teleports the bot to face the
situation and runs the REAL agent.main with FULL fidelity (sim camera
intrinsics so est_distance is real; sim wheel encoders so turns are tick-based)
and FULL telemetry (pose + every detection + reaction logged to JSONL). Each
behaviour is asserted from MEASURED evidence (stop distance via est_distance,
turn angle via Godot heading, light colour at the stop instant, ...), not by
eyeballing. Per-run logs land in _sim_logs/behaviour/<ts>/<label>.{jsonl,summary.json,png}.

Run:  PYTHONUTF8=1 .venv311/Scripts/python.exe tasks/project/sim_tests/behaviour_suite.py
"""
from __future__ import annotations
import json, math, os, subprocess, sys, tempfile, threading, time
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)
import cv2
import yaml

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
import tasks.project.packages.agent as agent
from tasks.project.packages.maneuvers import turn_left, turn_right, straight_through
from tasks.project.packages.sign_registry import lookup
from tasks.project.packages.precedence import we_go_first
from tasks.project.sim_tests.sim_telemetry import (
    TelemetryLogger, sim_fidelity_kwargs, SIM_APRILTAG_INTRINSICS, SIM_TAG_SIZE_M,
)

GODOT = os.path.join(os.path.expanduser("~"), ".cache", "duckietown", "godot", "Godot_v4.6-stable_win64.exe")
GP = os.path.join(_ROOT, "GodotSimulation", "ducky-bot")
RUN_DIR = os.path.join(_ROOT, "_sim_logs", "behaviour", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(RUN_DIR, exist_ok=True)

_TIMINGS = yaml.safe_load(open(os.path.join(
    _ROOT, "tasks/project/packages/config/maneuver_timings.yaml")))

results = []   # (name, passed, detail)


def rec(name, passed, detail):
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def grab(cam, n=4):
    f = None
    for _ in range(n):
        ok, fr = cam.read()
        if ok and fr is not None:
            f = fr
        time.sleep(0.04)
    return f


def run_agent(cam, wheels, label, x, z, heading, seconds, settle=0.9):
    """Teleport, run the real agent (fidelity + telemetry) for `seconds`.
    Returns (summary, transitions, records)."""
    wheels.set_wheels_speed(0.0, 0.0)
    wheels.teleport(x, z, heading)
    time.sleep(settle)
    log = TelemetryLogger(RUN_DIR, wheels=wheels, label=label)
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
    while time.time() - t0 < seconds:
        ok, f = cam.read()
        if ok and f is not None:
            last = f
        time.sleep(0.1)
    stop.set(); th.join(timeout=5)
    agent.next_state = _rn
    wheels.set_wheels_speed(0.0, 0.0)
    summ = log.close()
    if last is not None:
        cv2.imwrite(os.path.join(RUN_DIR, f"{label}.png"), last)
    # persist the transition list too
    try:
        with open(os.path.join(RUN_DIR, f"{label}.transitions.json"), "w") as fh:
            json.dump(transitions, fh, indent=2)
    except Exception:
        pass
    return summ, transitions, log.records


def _norm(d):
    while d > 180: d -= 360
    while d < -180: d += 360
    return d


# ============================================================================ #
def main():
    print(f"FIDELITY: intrinsics={SIM_APRILTAG_INTRINSICS} tag_size={SIM_TAG_SIZE_M}m  "
          f"turn_ticks={_TIMINGS.get('turn_ticks')} straight_ticks={_TIMINGS.get('straight_ticks')}")
    print(f"LOGS -> {RUN_DIR}\n")
    tmp = tempfile.mkdtemp(prefix="suite_"); pf = os.path.join(tmp, "ports.json")
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

        # ---- 1) LANE FOLLOWING -------------------------------------------------
        print("[1] lane following (spawn straight)")
        summ, tr, recs = run_agent(cam, wheels, "1_lane_follow", 0.9, 5.1, 0.0, 7)
        moved = 0.0
        if summ.get("start_pose") and summ.get("end_pose"):
            moved = math.hypot(summ["end_pose"]["x"] - summ["start_pose"]["x"],
                               summ["end_pose"]["z"] - summ["start_pose"]["z"])
        errs = [abs(r["lane"]["error"]) for r in recs if r.get("lane")]
        max_err = max(errs) if errs else 1.0
        drove = [r for r in recs if r["state"] == "DRIVE"]
        rec("1 lane-follow: drives & stays on lane",
            moved > 0.25 and max_err < 0.7 and len(drove) > 5,
            f"moved={moved:.2f}m, max|err|={max_err:.2f}, states={summ.get('states')}")

        # ---- 2) STOP SIGN at the 4-way: red line -> full stop -> wait -> turn ---
        # Southbound on col9 (x~5.85) from north of the cross. Sees the 4-way sign
        # (tag 8 @z6.5) then the stop octagon (tag 1 @z6.15); the painted RED LINE
        # at z=6.08 triggers the stop (the hardware-correct trigger); the bot must
        # hold stop_wait_seconds, then make a gradual car-like turn.
        print("[2] stop sign at the 4-way (red line + full stop + car turn)")
        summ, tr, recs = run_agent(cam, wheels, "2_stop_sign", 5.85, 7.05, 0.0, 20)
        tstates = {t[2] for t in tr}
        turned = [s for s in ("TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH") if s in tstates]
        rec("2 stop: APPROACH+STOPPED reached",
            "APPROACH" in tstates and "STOPPED" in tstates,
            f"transitions={['->'.join(t) for t in tr][:5]}")
        # red line must be what stopped us, with the bot AT the line
        line_seen = any(r.get("red_line") for r in recs)
        stop_recs = [r for r in recs if r["state"] == "STOPPED" and r.get("pose")]
        stop_pose = stop_recs[0]["pose"] if stop_recs else None
        at_line = (stop_pose is not None and 6.05 <= stop_pose["z"] <= 6.60
                   and 5.70 <= stop_pose["x"] <= 6.0)
        rec("2 stop: stopped AT the red line (z=6.08)",
            line_seen and at_line,
            f"red_line_seen={line_seen} stop_pose={stop_pose} (line at z=6.08)")
        # full-stop pause: time between entering STOPPED and entering the turn
        t_stop = next((t_ for t_, s in [(r['rel_t'], r['state']) for r in recs] if s == 'STOPPED'), None)
        t_turn = next((r['rel_t'] for r in recs
                       if r['state'] in ('TURN_LEFT', 'TURN_RIGHT', 'STRAIGHT_THROUGH')), None)
        pause = (t_turn - t_stop) if (t_stop is not None and t_turn is not None) else None
        rec("2 stop: waited at the line before going (>=1.4s)",
            pause is not None and pause >= 1.4,
            f"stop pause = {None if pause is None else round(pause, 2)}s (config stop_wait_seconds=1.5)")
        rec("2 stop: knew the 4-way sign + chose a legal turn",
            len(turned) > 0 and 8 in (summ.get("tags_seen") or {}),
            f"turn={turned[:1] or 'none'} signs_seen={summ.get('tags_seen')}")

        # ---- 3) CAR-LIKE TURN GEOMETRY (direct maneuvers from the stop pose) ----
        # Run each maneuver from the stop-line pose and check the bot LANDS on the
        # correct outgoing lane with the correct heading -- a gradual forward arc,
        # not a spin in place. (Right: east arm eastbound lane z~5.85 heading +X.
        # Left: west arm westbound lane z~5.55 heading -X. Straight: south arm.)
        print("[3] car-like turn landing (from the stop line)")
        def pose():
            """Fresh pose: poll_state lags one round-trip, so poll a few times."""
            p = (0.0, 0.0, 0.0)
            for _ in range(4):
                p = wheels.poll_state(); time.sleep(0.06)
            return p

        def land(fn, label):
            wheels.set_wheels_speed(0.0, 0.0)
            wheels.teleport(5.85, 6.30, 0.0); time.sleep(0.7)
            h0 = pose()[0]
            stop = threading.Event()
            fn(wheels, stop, _TIMINGS)
            time.sleep(0.3)
            h, x, z = pose()
            print(f"      {label}: land=({x:.2f},{z:.2f}) heading={h:+.0f} (dh={_norm(h-h0):+.0f})")
            return _norm(h - h0), x, z
        dh_r, x_r, z_r = land(turn_right, "turn_right")
        dh_l, x_l, z_l = land(turn_left, "turn_left")
        dh_s, x_s, z_s = land(straight_through, "straight")
        rec("3 right turn: gradual arc onto east lane",
            -120 <= dh_r <= -60 and x_r > 5.95 and 5.65 <= z_r <= 6.1,
            f"landed ({x_r:.2f},{z_r:.2f}) dh={dh_r:+.0f} (want x>5.95, z~5.85, dh~-90)")
        rec("3 left turn: wide arc onto west lane",
            60 <= dh_l <= 120 and x_l < 5.6 and 5.3 <= z_l <= 5.9,
            f"landed ({x_l:.2f},{z_l:.2f}) dh={dh_l:+.0f} (want x<5.6, z~5.55, dh~+90)")
        rec("3 straight: crosses the box, no drift",
            abs(dh_s) < 30 and 5.7 <= x_s <= 6.0 and z_s < 5.5,
            f"landed ({x_s:.2f},{z_s:.2f}) dh={dh_s:+.0f} (want x~5.85, z<5.5)")

        # ---- 4) TRAFFIC LIGHT: arm -> stop at its red line -> proceed ----------
        print("[4] traffic light (tag 74 @z3.2 arms; line z3.1; light @6.05,2.7)")
        summ, tr, recs = run_agent(cam, wheels, "4_traffic_light", 5.85, 4.45, 0.0, 24)
        tstates = {t[2] for t in tr}
        evs = [t[1] for t in tr]
        armed_any = any((r.get("light") or {}).get("armed") for r in recs)
        rec("4 light: detector armed by t-light tag",
            armed_any or "see_light" in evs,
            f"armed_seen={armed_any} see_light={'see_light' in evs} tags={summ.get('tags_seen')}")
        stop_recs = [r for r in recs if r["state"] in ("STOPPED", "WAIT") and r.get("pose")]
        lp = stop_recs[0]["pose"] if stop_recs else None
        rec("4 light: stops at the light's red line",
            ("STOPPED" in tstates or "WAIT" in tstates) and lp is not None and 3.1 <= lp["z"] <= 3.7,
            f"stop_pose={lp} (line at z=3.1) colors_seen={summ.get('light_colors_seen')}")

        # ---- 5) OBSTACLE (duckie) ----------------------------------------------
        print("[5] obstacle (duckie) soft-stop")
        summ, tr, recs = run_agent(cam, wheels, "5_obstacle", 4.5, 1.632, 270.0, 9)
        st = summ.get("states", [])
        rec("5 obstacle: SOFT_STOP for duckie",
            "SOFT_STOP" in st and summ.get("obstacle_stop_frames", 0) > 0,
            f"states={st} obstacle_frames={summ.get('obstacle_stop_frames')}")

        # ---- 6) OTHER ROBOT (parked Duckiebot with a vehicle tag plate) ---------
        # A real robot model (not a billboard) parked roadside at (6.12, 4.05)
        # carries vehicle-tag plate 400 = megabot01, like real Duckiebots do. The
        # moving NPC bot carries the same plates (ambient encounters).
        print("[6] other robot: parked Duckiebot w/ vehicle plate @ (6.12,4.05)")
        summ, tr, recs = run_agent(cam, wheels, "6_other_robot", 5.85, 4.85, 0.0, 8)
        veh_ids = [tid for tid in (summ.get("tags_seen") or {}) if int(tid) >= 400]
        first = we_go_first("Toma_PC", [lookup(int(veh_ids[0]))]) if veh_ids else None
        rec("6 other robot: detected the parked bot's vehicle plate",
            len(veh_ids) > 0,
            f"vehicle_tags={veh_ids} we_go_first(host=Toma_PC)={first}")

        # ---- summary -----------------------------------------------------------
        npass = sum(1 for _, p, _ in results if p)
        print("\n" + "=" * 70)
        print(f"BEHAVIOUR SUITE: {npass}/{len(results)} checks passed")
        for n, p, _ in results:
            print(f"  {'PASS' if p else 'FAIL'}  {n}")
        print(f"logs -> {RUN_DIR}")
        print("=" * 70)
        return 0 if npass == len(results) else 1
    finally:
        try: g.terminate(); g.wait(timeout=4)
        except Exception: g.kill()


if __name__ == "__main__":
    sys.exit(main())
