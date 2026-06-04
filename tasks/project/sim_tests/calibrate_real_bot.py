"""Calibrate the REAL Duckiebot's camera + verify HSV — from the laptop.

Runs against the robot's live MJPEG stream while the task is running
(`python launch.py --run --bot <name> --task project`), so nothing extra is
installed on the bot. Three modes:

  # 1) Measure fx (place ONE printed 6.5 cm tag exactly --distance metres
  #    from the camera lens, facing it squarely):
  .venv311\\Scripts\\python.exe tasks\\project\\sim_tests\\calibrate_real_bot.py ^
      --url http://<bot>.local:5000/video --measure-fx --distance 0.50
  #    -> prints the intrinsics block to paste into config/camera_intrinsics.yaml

  # 2) Verify est_distance after filling that file in:
  ...calibrate_real_bot.py --url ... --verify --distance 0.75

  # 3) On-site HSV check (red stop line at the bot's feet / light in view):
  ...calibrate_real_bot.py --url ... --hsv

Works with --image <file.jpg> instead of --url for a saved frame.
"""
from __future__ import annotations
import argparse, os, sys, time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)
import cv2
import numpy as np

from tasks.project.packages.perception import AprilTagDetector, TrafficLightDetector, detect_red_line
from tasks.project.packages.agent import _load_hsv_cfg


def frames_from_mjpeg(url, n, timeout=15.0):
    """Yield up to n decoded BGR frames from an MJPEG stream."""
    import requests
    buf = b""
    got = 0
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        t0 = time.time()
        for chunk in r.iter_content(chunk_size=16384):
            buf += chunk
            while True:
                s = buf.find(b"\xff\xd8")
                e = buf.find(b"\xff\xd9", s + 2) if s >= 0 else -1
                if s < 0 or e < 0:
                    break
                jpg, buf = buf[s:e + 2], buf[e + 2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield img
                    got += 1
                    if got >= n:
                        return
            if time.time() - t0 > timeout:
                return


def get_frames(args, n=12):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"could not read --image {args.image}")
        return [img] * 1
    if not args.url:
        sys.exit("need --url http://<bot>.local:5000/video or --image <file>")
    frames = list(frames_from_mjpeg(args.url, n))
    if not frames:
        sys.exit("no frames received — is the task running on the bot? (launch.py --run)")
    return frames


_TRIAL_FX = 310.0   # arbitrary; est_distance is exactly linear in fx, so we fit it out


def detect_tag(frames, tag_size, intrinsics, want_id=None):
    """Per frame: the wanted tag (or the largest one). Returns observations."""
    det = AprilTagDetector(tag_size_m=tag_size, intrinsics=intrinsics)
    obs = []
    for f in frames:
        found = det.detect(f) or []
        if want_id is not None:
            found = [o for o in found if o.id == want_id]
        if found:
            obs.append(max(found, key=lambda o: o.side_length_px))
    return obs


def cmd_measure_fx(args):
    frames = get_frames(args)
    h, w = frames[0].shape[:2]
    trial = (_TRIAL_FX, _TRIAL_FX, w / 2.0, h / 2.0)
    obs = detect_tag(frames, args.tag_size, trial, want_id=args.tag_id)
    if not obs:
        sys.exit("tag not detected — bigger print / better light / closer? (or wrong --tag-id)")
    ids = sorted({o.id for o in obs})
    if len(ids) > 1:
        sys.exit(f"multiple tags seen across frames ({ids}) — keep ONE tag in view "
                 f"or pin it with --tag-id <id>")
    # Fit fx through the detector's OWN pose estimate: est is linear in fx, so
    # fx = trial * true/est is exact and immune to tag-border/pose conventions
    # (a naive pinhole side*D/size disagrees with the pose estimate by ~25-50%).
    ests = [o.est_distance_m for o in obs]
    est = float(np.median(ests))
    fx = _TRIAL_FX * args.distance / est
    sides = [o.side_length_px for o in obs]
    print(f"tag id {ids[0]}  frames used: {len(obs)}  side_px median={np.median(sides):.0f} "
          f"(spread {min(sides)}-{max(sides)})")
    print(f"true {args.distance:.2f} m, est@trial {est:.3f} m  ->  fx = {fx:.1f}")
    print("\nPaste into config/camera_intrinsics.yaml (then redeploy with launch.py --run):\n")
    print("intrinsics:")
    print(f"  fx: {fx:.1f}")
    print(f"  fy: {fx:.1f}")
    print(f"  cx: {w/2:.1f}")
    print(f"  cy: {h/2:.1f}")
    print("\nThen verify:  --verify --distance 0.75   (expect ~<10% error)")


def cmd_verify(args):
    import yaml
    path = os.path.join(_ROOT, "config", "camera_intrinsics.yaml")
    try:
        src = (yaml.safe_load(open(path)) or {})
        src = src.get("intrinsics", src)
        intr = (float(src["fx"]), float(src["fy"]), float(src["cx"]), float(src["cy"]))
    except Exception as e:
        sys.exit(f"could not read intrinsics from {path}: {e} — run --measure-fx first")
    frames = get_frames(args)
    obs = detect_tag(frames, args.tag_size, intr, want_id=args.tag_id)
    if not obs:
        sys.exit("tag not detected")
    ids = sorted({o.id for o in obs})
    est = float(np.median([o.est_distance_m for o in obs]))
    err = 100.0 * (est - args.distance) / args.distance
    print(f"tag id {ids}: intrinsics fx={intr[0]:.1f}: est_distance = {est:.3f} m vs true "
          f"{args.distance:.2f} m ({err:+.1f}%)  ->  {'OK' if abs(err) < 10 else 'RE-MEASURE fx'}")


def cmd_hsv(args):
    frames = get_frames(args, n=6)
    cfg = _load_hsv_cfg()
    light = TrafficLightDetector(); light.arm()
    print(f"{'frame':>5} {'red_line':>8} {'light':>7}")
    for i, f in enumerate(frames):
        print(f"{i:5d} {str(detect_red_line(f, cfg)):>8} {str(light.detect(f)):>7}")
    out = os.path.join(_ROOT, "_sim_logs", "bot_hsv_check.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, light.get_debug_overlay(frames[-1]))
    print(f"\nlight-debug overlay saved -> {out}")
    print("Expect: red_line True with a line at the bot's feet; light colour correct at ~0.5-1 m.")
    print("If not: tune red_*/line_* (line) and colour bands (light) in "
          "tasks/project/packages/config/traffic_light_hsv.yaml, redeploy, re-check.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="bot MJPEG stream, e.g. http://<bot>.local:5000/video")
    ap.add_argument("--image", help="use a saved frame instead of the stream")
    ap.add_argument("--tag-size", type=float, default=0.065, help="printed tag side (m), default 0.065")
    ap.add_argument("--distance", type=float, help="true camera->tag distance (m)")
    ap.add_argument("--tag-id", type=int, help="only use this tag id (when several are visible)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--measure-fx", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--hsv", action="store_true")
    args = ap.parse_args()

    if (args.measure_fx or args.verify) and not args.distance:
        ap.error("--measure-fx/--verify need --distance <metres>")
    if args.measure_fx:
        cmd_measure_fx(args)
    elif args.verify:
        cmd_verify(args)
    else:
        cmd_hsv(args)


if __name__ == "__main__":
    main()
