"""Sweep turn_left_exit_ticks at the real 4-way (ducks OFF) and print each
post-turn trajectory, to pick the value that lands the LEFT turn cleanly on the
short west-arm straight tile (col8, x~4.8-5.4) and then stays on the curving arm
instead of drifting off.

The west arm is only ONE straight tile (col8) before it curves (col7/col6), so a
too-long exit overshoots the straight into the curve; too-short re-engages the PD
at the box edge. We want it to land ~col8 heading ~+90 and follow the road.
"""
import os, sys, random, threading, time, copy
import yaml

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
import tasks.project.sim_tests.course_suite as cs
import tasks.project.packages.agent as agent
from tasks.project.sim_tests.sim_telemetry import TelemetryLogger, sim_fidelity_kwargs

_orig_choice = random.choice
random.choice = lambda seq: 'left' if 'left' in list(seq) else _orig_choice(list(seq))

with open(os.path.join(_ROOT, "tasks", "project", "packages", "config", "maneuver_timings.yaml")) as fh:
    BASE = yaml.safe_load(fh)
BASE["object_detection"] = False

SWEEP = [int(a) for a in sys.argv[1:]] or [0, 100, 200, 300]


def run(cam, wheels, left_exit, seconds=18):
    tim = copy.deepcopy(BASE)
    tim["turn_left_exit_ticks"] = left_exit
    wheels.set_wheels_speed(0.0, 0.0); wheels.teleport(5.85, 7.05, 0.0); time.sleep(0.9)
    log = TelemetryLogger(cs.RUN_DIR, wheels=wheels, label="lexit_%d" % left_exit)
    stop = threading.Event()
    kw = dict(observer=log, timings_override=tim, **sim_fidelity_kwargs())
    th = threading.Thread(target=agent.main, args=(cam, wheels, None, stop), kwargs=kw, daemon=True)
    th.start()
    t0 = time.time()
    while time.time() - t0 < seconds:
        cam.read(); time.sleep(0.1)
    stop.set(); th.join(timeout=5); wheels.set_wheels_speed(0.0, 0.0); log.close()

    turned = False; post = []
    for r in log.records:
        st = r.get("state")
        if st in ("TURN_LEFT", "TURN_RIGHT", "STRAIGHT_THROUGH"):
            turned = True
        elif turned and st == "DRIVE":
            p = r.get("pose") or {}
            post.append((p.get("heading_deg"), p.get("x"), p.get("z"), (r.get("lane") or {}).get("error")))
    clean = [f for f in post[2:] if f[0] is not None]
    print("\n=== turn_left_exit_ticks = %d ===" % left_exit)
    if not clean:
        print("  (no post-turn frames)"); return
    land = clean[0]
    errs = [abs(f[3]) for f in clean if f[3] is not None]
    print("  land: head=%.0f x=%.2f z=%.2f  (col8 straight = x4.8-5.4)" % (land[0], land[1], land[2]))
    print("  |error| mean=%.3f max=%.3f   end: x=%.2f z=%.2f head=%.0f" %
          (sum(errs)/len(errs), max(errs), clean[-1][1], clean[-1][2], clean[-1][0]))
    for i, f in enumerate(clean):
        if i % 4 == 0:
            print("     head=%6.1f x=%5.2f z=%5.2f err=%6.3f" % (f[0], f[1], f[2], f[3]))


g, cam, wheels = cs.launch_godot()
try:
    for v in SWEEP:
        run(cam, wheels, v)
finally:
    try:
        g.terminate(); g.wait(timeout=4)
    except Exception:
        g.kill()
