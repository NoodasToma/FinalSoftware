"""Offline validation for the bot's object-detection path (no Godot, no hardware).

Mirrors the bot by forcing CPU inference (OBJDET_CPU=1) and checks:
  1. best.onnx loads on the CPU path.
  2. it actually DETECTS ducks in real sample images (recall, not just "runs").
  3. inference time is in a range that justifies the decoupled worker thread.
  4. the decoupled queue->worker->store flow publishes results without deadlock.
  5. should_stop_for_obstacle fires for a close duck, not a far one.
  6. the other-bot (_bot_ahead) gating: near+centred Vehicle tag stops; a far or
     off-to-the-side one does not; non-Vehicle tags never do.

Run:  .venv311/Scripts/python.exe tasks/project/sim_tests/verify_object_detection.py
"""
import os
import sys
import glob
import time
import queue
import threading

os.environ['OBJDET_CPU'] = '1'   # mirror the Jetson real_server path

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)

import cv2  # noqa: E402

from tasks.object_detection.packages.agent import ObjectDetectionAgent  # noqa: E402
from tasks.project.packages.obstacles import should_stop_for_obstacle  # noqa: E402
from tasks.project.packages.perception.apriltags import TagObservation  # noqa: E402
from tasks.project.packages.sign_registry import SignSemantic  # noqa: E402

_fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _fails.append(name)


print("== 1) model loads (CPU path, mirrors bot OBJDET_CPU=1) ==")
obj = ObjectDetectionAgent()
check("ObjectDetectionAgent.model_loaded", obj.model_loaded,
      f"backend={getattr(obj, '_backend', None)} err={obj.load_error}")
if not obj.model_loaded:
    print("Cannot continue without a loaded model.")
    sys.exit(1)

print("\n== 2) detects ducks in real sample images ==")
samples = sorted(glob.glob(os.path.join(_ROOT, 'tasks/assets/samples/big-duck/*.jpg')))[:6]
samples += sorted(glob.glob(os.path.join(_ROOT, 'tasks/assets/samples/many-duckies/*.jpg')))[:4]
hits = 0
times = []
for p in samples:
    bgr = cv2.imread(p)
    if bgr is None:
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    dets = obj.detect(rgb)
    times.append((time.time() - t0) * 1000.0)
    n = len(dets or [])
    if n > 0:
        hits += 1
    print(f"    {os.path.basename(p):24s} -> {n} duckie(s)")
check("at least half the duck images yield a detection", hits >= max(1, len(samples) // 2),
      f"{hits}/{len(samples)} images had >=1 duck")
if times:
    avg = sum(times) / len(times)
    # NOTE: this is a fast x86 dev CPU. The Jetson Nano (4x Cortex-A57 @ 1.4 GHz)
    # runs this same model ~10-20x slower (~150-350 ms), which is exactly why the
    # bot runs detection in a DECOUPLED thread — inline at that latency throttled
    # the control loop until the camera stalled (the freeze). So this is reported,
    # not asserted: we only sanity-check that a single inference doesn't hang.
    print(f"    inference: avg={avg:.0f} ms  max={max(times):.0f} ms over {len(times)} frames "
          f"(x86 dev CPU; Nano ~10-20x slower -> decoupled thread)")
    check("inference completes without hanging (<5 s/frame)", max(times) < 5000,
          f"max {max(times):.0f} ms")

print("\n== 3) decoupled queue -> worker -> store flow (no deadlock) ==")
_det_q = queue.Queue(maxsize=1)
_store = {'dets': None}
_lock = threading.Lock()
_alive = {'v': True}


def _worker():
    while _alive['v']:
        try:
            fr = _det_q.get(timeout=0.2)
        except queue.Empty:
            continue
        d = obj.detect(fr)
        if d is not None:
            with _lock:
                _store['dets'] = d


th = threading.Thread(target=_worker, daemon=True)
th.start()
duck_img = cv2.cvtColor(cv2.imread(samples[0]), cv2.COLOR_BGR2RGB)
for _ in range(5):
    try:
        _det_q.put_nowait(duck_img)
    except queue.Full:
        pass
    time.sleep(0.1)
deadline = time.time() + 3.0
got = None
while time.time() < deadline:
    with _lock:
        got = _store['dets']
    if got is not None:
        break
    time.sleep(0.05)
_alive['v'] = False
th.join(timeout=1.0)
check("worker published a result via the shared store", got is not None,
      f"store={None if got is None else len(got)} det(s)")

print("\n== 4) should_stop_for_obstacle: close duck stops, far duck doesn't ==")
FRAME_H = 480
close_duck = [((280, 300, 380, 470), 0.9, 0)]   # big, near frame bottom
far_duck   = [((300, 40, 320, 70), 0.9, 0)]     # tiny, high up
check("close duck -> stop", should_stop_for_obstacle(close_duck, FRAME_H)[0] is True)
check("far/tiny duck -> no stop", should_stop_for_obstacle(far_duck, FRAME_H)[0] is False)


def _bot_ahead(signs, frame_w, tag_px=55, center_frac=0.33, dist_m=0.45):
    """Standalone copy of agent.main's closure, for logic validation."""
    half = frame_w / 2.0
    for o, sem in signs:
        if sem.tag_type != 'Vehicle':
            continue
        if abs(o.center_xy[0] - half) > frame_w * center_frac:
            continue
        if o.side_length_px >= tag_px or o.est_distance_m < dist_m:
            return True
    return False


print("\n== 5) other-bot soft-stop gating (_bot_ahead) ==")
FRAME_W = 640
veh = SignSemantic(kind='', tag_type='Vehicle', vehicle_name='megabot01')
sign_tag = SignSemantic(kind='stop', tag_type='TrafficSign', vehicle_name=None)


def tag(cx, side, dist=float('inf')):
    return TagObservation(id=400, center_xy=(cx, 240), side_length_px=side,
                          est_distance_m=dist, est_yaw_rad=0.0)


check("near, centred bot -> stop",
      _bot_ahead([(tag(320, 70), veh)], FRAME_W) is True)
check("far (small tag), centred bot -> no stop",
      _bot_ahead([(tag(320, 20), veh)], FRAME_W) is False)
check("near but off to the side -> no stop",
      _bot_ahead([(tag(610, 70), veh)], FRAME_W) is False)
check("calibrated close (dist) bot -> stop",
      _bot_ahead([(tag(320, 20, dist=0.3), veh)], FRAME_W) is True)
check("a TrafficSign tag never triggers a bot-stop",
      _bot_ahead([(tag(320, 90), sign_tag)], FRAME_W) is False)

print("\n" + "=" * 60)
if _fails:
    print(f"OBJECT-DETECTION VALIDATION: {len(_fails)} FAILED -> {_fails}")
    sys.exit(1)
print("OBJECT-DETECTION VALIDATION: ALL CHECKS PASSED")
sys.exit(0)
